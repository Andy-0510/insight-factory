import os
import json
import glob
import time
import hashlib
import re
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
import trafilatura
from trafilatura.settings import use_config
from src.utils import load_json, save_json, latest, clean_text
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import threading
import requests
from collections import defaultdict

# 본문 최소 길이(환경변수로 조절). 기본 120자
MIN_LEN = int(os.environ.get("BODY_MIN_LEN", "120"))
MAX_WORKERS = int(os.environ.get("FETCH_MAX_WORKERS", "8"))
PER_DOMAIN_LIMIT = int(os.environ.get("FETCH_PER_DOMAIN", "3"))

# 전역: 도메인별 세마포어 관리 (defaultdict으로 간단화)
_domain_locks: Dict[str, threading.Semaphore] = defaultdict(lambda: threading.Semaphore(PER_DOMAIN_LIMIT))
# trafilatura config 캐시
_CONFIG_CACHE = None
# requests.Session 재사용
_SESSION: Optional[requests.Session] = None

# ----------------------------
# 정규식(사전 컴파일)
# ----------------------------
_WS_RE = re.compile(r"\s+")
_EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
_URL_RE = re.compile(r"https?://\S+")
_PRICE_RE = re.compile(r"\b[0-9]{1,3}(?:,[0-9]{3})+(?:\s원(?:부터)?)?\b")
_PHOTO_CAP_RE = re.compile(r"^\s사진\s*=\s*.*$", flags=re.M)
_SPECIAL_ICONS_RE = re.compile(r"[│•▪◆■※☆★▷▶▸▹◀◁◾◼︎]+")
_NAVER_HOST_RE = re.compile(r"^https?://([^/]+).")
_HOST_EXTRACT_RE = re.compile(r"^https?://([^/]+)/?.*$")

# anchors & noise patterns (확장된 패턴들)
_ANCHORS = [
    re.compile(p) for p in [
        r"^앱토 한마디\b",
        r"^BEST댓글\b",
        r"^댓글\b",
        r"^이 기사를 공유합니다\b",
        r"^제원표\b",
        r"^POINT\b",
        r"^관련기사\b",
        r"^기사원문\b",
        r"^관련 기사\b",
        r"^주요뉴스\b",
        r"^많이 본 뉴스\b",
        r"^추천영상\b",
        r"^이 기사와 함께 보면\b",
    ]
]

_NOISE_SUBS = [
    # (pattern, flags)
    (re.compile(r"\[[^\]]{0,120}기자\]", flags=re.I), 0),
    (re.compile(r"[-–—]\s*기자명?\s*[^\s]*기자", flags=re.I), 0),
    (re.compile(r"(무단전재\s*및\s*재배포\s*금지|무단전재|저작권\s*ⓒ[^,\n]+)", flags=re.I), 0),
    (re.compile(r"(BEST댓글|댓글삭제|댓글수정|이 기사를 공유합니다|구독해주세요|좋아요)", flags=re.I), 0),
    (re.compile(r"페이스북\s*공유|트위터\s*공유|공유하기", flags=re.I), 0),
    (re.compile(r"읽어볼만한 기사|관련기사|추천기사", flags=re.I), 0),
    (re.compile(r"사진=[^\n]+", flags=re.I), 0),
]


def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def pick_url(it: Dict[str, Any]) -> str:
    """
    URL 우선순위: url > canonical > link > origin_url
    네이버 도메인 우선 + AMP 제거 + n.news 표준화
    """
    cand = [it.get("url"), it.get("canonical"), it.get("link"), it.get("origin_url")]
    cand = [c.strip() for c in cand if c]
    if not cand:
        return ""

    def _naver_score(u: str) -> int:
        # 원래 구현의 의도를 유지: 호스트 부분에 naver 관련 도메인이 있으면 우선
        h = re.sub(r"^https?://([^/]+).", r"\1", (u or "").lower())
        return 2 if ("n.news.naver.com" in h or "news.naver.com" in h or "m.news.naver.com" in h) else 1

    cand.sort(key=_naver_score, reverse=True)
    u = cand[0]
    # AMP 경로 제거(원본 구현과 동일 형태 유지)
    u = re.sub(r"/amp(/|).", "/", u)
    u = u.replace("m.news.naver.com", "n.news.naver.com")
    u = u.replace("news.naver.com", "n.news.naver.com")
    return u


def _make_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        cfg = use_config()
        cfg.set("DEFAULT", "user_agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
        cfg.set("DEFAULT", "timeout", "12")
        _CONFIG_CACHE = cfg
    return _CONFIG_CACHE


def _domain_semaphore(url: str) -> threading.Semaphore:
    # defaultdict을 사용했으므로 단순히 호스트로 lookup하면 semaphore가 생성됨
    host = urlparse(url).netloc
    return _domain_locks[host]


# -------- 정제 유틸(노이즈 컷) --------
def _remove_after_anchors(text: str) -> str:
    """ 특정 앵커가 나오면 그 이후는 노이즈로 간주하고 컷. """
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    for ln in lines:
        s = ln.strip()
        if any(p.search(s) for p in _ANCHORS):
            break
        out.append(ln)
    return "\n".join(out)


def _strip_common_noise(text: str) -> str:
    """ 기자/저작권/이메일/URL/가격/캡션/UX 문구 제거(보수적). """
    if not text:
        return ""
    t = text
    for pat, _f in _NOISE_SUBS:
        t = pat.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = PRICE_RE.sub(" ", t)
    t = t.replace("▲", " ")
    t = _PHOTO_CAP_RE.sub(" ", t)
    t = _SPECIAL_ICONS_RE.sub(" ", t)
    # 여러 공백은 한 번에 정리 (최종 정리는 sanitize_article에서 한 번 더 수행)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _dedup_sentences(text: str) -> str:
    """ 완전 동일 문장/문단 중복 제거(순서 유지). """
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?다])\s+", text)
    seen, out = set(), []
    for s in parts:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return " ".join(out)


def _remove_low_korean_density(text: str, threshold: float = 0.25) -> str:
    """문장 단위로 한글 비율이 너무 낮으면 제거 (URL/영문/코드 조각 제거에 도움)."""
    if not text:
        return ""
    lines = re.split(r"\n+", text)
    out = []
    for ln in lines:
        ln_s = ln.strip()
        if not ln_s:
            continue
        han = len(re.findall(r"[가-힣]", ln_s))
        ratio = han / max(len(ln_s), 1)
        if ratio < threshold:
            # 단문(짧음)은 보수적으로 유지: 길이 기준으로 짧은 문장은 버리지 않음
            if len(ln_s) < 30:
                out.append(ln_s)
            else:
                # 제거
                continue
        else:
            out.append(ln_s)
    return "\n".join(out)


def sanitize_article(text: str) -> str:
    """
    전체 정제 파이프라인: 앵커 컷 → 공통 노이즈 제거 → '제원/스펙' 표 이전 컷
    → 중복 제거 → 저한글 비율 문장 제거 → 공백 정리.
    """
    if not text:
        return ""
    t = _remove_after_anchors(text)
    t = _strip_common_noise(t)
    m = re.search(r"(제원|스펙|사양)\s*표", t)
    if m:
        t = t[:m.start()].strip()
    t = _dedup_sentences(t)
    t = _remove_low_korean_density(t)
    t = _WS_RE.sub(" ", t).strip()
    return t


# -------- 네이버 전용 파서 --------
def _strip_html_tags(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def extract_naver_body(html: str) -> str:
    if not html:
        return ""
    m = re.search(r'<div[^>]+id=["\']dic_area["\'][^>]>(.*?)', html, flags=re.I | re.S)
    if not m:
        m = re.search(r'<div[^>]+id=["\']articleBodyContents["\'][^>]>(.*?)', html, flags=re.I | re.S)

    txt = _strip_html_tags(m.group(1)) if m else ""
    txt = re.sub(r"\s*ⓒ.?무단전재.?$", "", txt, flags=re.I)
    txt = EMAIL_RE.sub("", txt)
    txt = _WS_RE.sub(" ", txt).strip()
    return txt


# -------- 본문 수집 --------
def fetch_body(url: str, timeout: int = 12) -> Tuple[str, str]:
    """
    반환: (sanitized_text, raw_text)
    - sanitized_text: 정제된 본문(캐시 저장 기준)
    - raw_text: 정제 전 추출결과(신호 추출용)
    """
    if not url:
        return "", ""

    os.makedirs("data/article_cache", exist_ok=True)
    key = sha1(url)
    cache_path = os.path.join("data/article_cache", key + ".txt")
    cache_raw_path = os.path.join("data/article_cache", key + ".raw.txt")

    # 캐시 히트(정제본 우선). _read_file 대신 빠른 존재/열기 체크
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = f.read().strip()
            if len(cached) >= MIN_LEN:
                raw = ""
                if os.path.exists(cache_raw_path):
                    try:
                        with open(cache_raw_path, "r", encoding="utf-8") as rf:
                            raw = rf.read().strip()
                    except Exception:
                        raw = ""
                return cached, raw
        except Exception:
            # 캐시 읽기 실패 시 무시하고 새로 수집
            pass

    cfg = _make_config()
    text_raw = ""

    # 1차: trafilatura.fetch_url → extract
    try:
        downloaded = trafilatura.fetch_url(url, config=cfg, timeout=timeout, no_ssl=True)
        if downloaded:
            t = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_formatting=False,
                favor_recall=True,
                with_metadata=False
            ) or ""
            text_raw = clean_text(t)
    except Exception:
        text_raw = ""

    # 2차: requests로 HTML → 네이버 전용 파서 → trafilatura(HTML)
    if len(text_raw) < MIN_LEN:
        try:
            global _SESSION
            if _SESSION is None:
                _SESSION = requests.Session()
                _SESSION.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
            r = _SESSION.get(url, timeout=timeout, allow_redirects=True)
            if r.ok and r.text:
                host = re.sub(r"^https?://([^/]+).*$", r"\1", (url or "").lower())
                t1 = extract_naver_body(r.text) if "naver.com" in host else ""
                t2 = trafilatura.extract(
                    r.text,
                    include_comments=False,
                    include_formatting=False,
                    favor_recall=True,
                    with_metadata=False
                ) or ""
                t1 = clean_text(t1)
                t2 = clean_text(t2)
                cand = t1 if len(t1) >= len(t2) else t2
                if len(cand) > len(text_raw):
                    text_raw = cand
        except Exception:
            # 네트워크/파싱 실패 시 그냥 넘어감
            pass

    # 3) 정제는 항상 최종에서 1회 적용
    text_san = sanitize_article(text_raw)

    # 캐시 저장(정제본 + raw)
    if len(text_san) >= MIN_LEN:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text_san)
        except Exception:
            pass
        try:
            with open(cache_raw_path, "w", encoding="utf-8") as rf:
                rf.write(text_raw)
        except Exception:
            pass

    return text_san, text_raw


def _process_one(it: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    개별 기사 처리.
    성공 시 (업데이트된 item, domain) 반환, 실패/스킵 시 (None, domain) 반환.
    """
    url = pick_url(it)
    domain = re.sub(r"^https?://([^/]+)/?.*$", r"\1", url) if url else "-"
    if not url:
        return None, domain
    body_now = (it.get("body") or "").strip()
    if len(body_now) >= MIN_LEN:
        # raw_body/description_short 보강만 수행
        if not it.get("raw_body"):
            it["raw_body"] = it.get("raw_body") or ""
        if not it.get("description_short"):
            it["description_short"] = make_description_short(body_now)
        return it, domain
    try:
        sem = _domain_semaphore(url)
        with sem:
            body_san, body_raw = fetch_body(url)
        if len(body_san) >= MIN_LEN:
            it["raw_body"] = body_raw or ""
            it["body"] = body_san
            it["description"] = body_san
            it["description_short"] = make_description_short(body_san)
            return it, domain
        return None, domain
    except Exception:
        return None, domain


def make_description_short(text: str, target_min=400, target_max=600) -> str:
    """
    본문 첫 문단/문장 기준으로 400~600자 요약 생성(문장 경계에서 자르기 시도).
    """
    if not text:
        return ""
    txt = text.strip()
    # 문단 기준
    para = txt.split("\n")[0].strip()
    base = para if len(para) >= target_min else txt

    if len(base) <= target_max:
        return base

    # 문장 경계에서 컷
    cut = base[:target_max]
    # 뒤에서부터 첫 종결부호 찾기 (원래 로직과 동일)
    m = re.search(r"[.!?다]\s", cut[::-1])
    if m:
        idx = len(cut) - m.start()
        return cut[:idx].strip()

    return cut.strip()


def main() -> int:
    print("[INFO] [fetch_article_bodies] KICK-OFF: 기사 본문 수집을 시작합니다.")  # 시작 로그
    is_monthly_run = os.getenv("MONTHLY_RUN", "false").lower() == "true"

    if is_monthly_run:
        meta_path = "outputs/debug/monthly_meta_agg.json"
        print(f"[INFO] [fetch_article_bodies] 월간 실행 모드: 집계된 메타 파일 사용")
    else:
        meta_path = "outputs/debug/news_meta_latest.json"
        if not os.path.exists(meta_path):
            meta_path = latest("data/news_meta_*.json")

    if not meta_path or not os.path.exists(meta_path):
        print("[ERROR] [fetch_article_bodies] 입력 메타 파일을 찾을 수 없습니다.")  # 에러 로그 강화
        return 1

    print(f"[INFO] [fetch_article_bodies] 메타 데이터 로드: {meta_path}")  # 입력 파일 로그
    items: List[Dict[str, Any]] = load_json(meta_path, [])
    if not items:
        print("[WARN] [fetch_article_bodies] 처리할 기사가 없습니다. 종료합니다.")
        return 0

    print(f"[INFO] [fetch_article_bodies] 총 {len(items)}개 기사에 대한 본문 수집 시작... (병렬 워커 수: {MAX_WORKERS})")

    tried, updated = 0, 0
    per_domain: Dict[str, Dict[str, int]] = {}
    indices = list(range(len(items)))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_map = {ex.submit(_process_one, items[i]): i for i in indices}
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            try:
                res_item, domain = fut.result()
            except Exception as e:
                res_item, domain = None, "-"
                print(f"[WARN] [fetch_article_bodies] 기사 처리 중 오류 발생 (index={i}): {e}")

            tried += 1
            if domain not in per_domain:
                per_domain[domain] = {"ok": 0, "fail": 0}

            if res_item is not None:
                items[i] = res_item
                updated += 1
                per_domain[domain]["ok"] += 1
            else:
                per_domain[domain]["fail"] += 1

            if tried % 50 == 0:
                print(f"[INFO] [fetch_article_bodies] 진행률: {tried}/{len(items)} (성공: {updated})")

    print(f"[INFO] [fetch_article_bodies] 본문 수집 완료. 변경된 메타 데이터를 파일에 다시 저장합니다: {meta_path}")
    save_json(meta_path, items)  # save_json 유틸리티 사용

    # 최종 결과 로그
    print(f"[SUCCESS] [fetch_article_bodies] 본문 수집 작업 완료 | 시도={tried}, 업데이트={updated}, 파일={os.path.basename(meta_path)}")
    if per_domain:
        stats = ", ".join(f"{d}: ok={v['ok']}, fail={v['fail']}" for d, v in list(per_domain.items())[:15])
        print(f"[DEBUG] [fetch_article_bodies] 도메인별 수집 현황: {stats}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
