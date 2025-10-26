# File: scripts/fetch_article_bodies.py
import os # Make sure os is imported
import json # Make sure json is imported
import glob # Make sure glob is imported
import time # Make sure time is imported
import hashlib # Make sure hashlib is imported
import re # Make sure re is imported
import unicodedata # Make sure unicodedata is imported
from typing import List, Dict, Any, Tuple, Optional # Make sure typing imports are present
import trafilatura # Make sure trafilatura is imported
from trafilatura.settings import use_config # Make sure use_config is imported
from src.utils import load_json, save_json, latest, clean_text # Make sure utils imports are present
from concurrent.futures import ThreadPoolExecutor, as_completed # Make sure concurrent imports are present
from urllib.parse import urlparse # Make sure urlparse is imported
import threading # Make sure threading is imported
import requests # Make sure requests is imported
from collections import defaultdict # Make sure defaultdict is imported
from bs4 import BeautifulSoup # Make sure BeautifulSoup is imported

# ----------------------------
# 설정값
# ----------------------------
MIN_LEN = int(os.environ.get("BODY_MIN_LEN", "120"))
MAX_WORKERS = int(os.environ.get("FETCH_MAX_WORKERS", "8"))
PER_DOMAIN_LIMIT = int(os.environ.get("FETCH_PER_DOMAIN", "3"))

# 전역 객체
_domain_locks: Dict[str, threading.Semaphore] = defaultdict(lambda: threading.Semaphore(PER_DOMAIN_LIMIT))
_CONFIG_CACHE = None
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
_ANCHORS = [re.compile(p) for p in [
    r"^앱토 한마디", r"^BEST댓글", r"^댓글", r"^이 기사를 공유합니다", r"^제원표",
    r"^POINT", r"^관련기사", r"^기사원문", r"^관련 기사", r"^주요뉴스",
    r"^많이 본 뉴스", r"^추천영상", r"^이 기사와 함께 보면",
]]
_NOISE_SUBS = [
    (re.compile(r"\[[^\]]{0,120}기자\]", flags=re.I), 0),
    (re.compile(r"[-–—]\s*기자명?\s*[^\s]*기자", flags=re.I), 0),
    (re.compile(r"(무단전재\s*및\s*재배포\s*금지|무단전재|저작권\s*ⓒ[^,\n]+)", flags=re.I), 0),
    (re.compile(r"(BEST댓글|댓글삭제|댓글수정|이 기사를 공유합니다|구독해주세요|좋아요)", flags=re.I), 0),
    (re.compile(r"페이스북\s*공유|트위터\s*공유|공유하기", flags=re.I), 0),
    (re.compile(r"읽어볼만한 기사|관련기사|추천기사", flags=re.I), 0),
    (re.compile(r"사진=[^\n]+", flags=re.I), 0),
]

# ----------------------------
# 유틸 함수
# ----------------------------
def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def pick_url(it: Dict[str, Any]) -> str:
    cand = [it.get("url"), it.get("canonical"), it.get("link"), it.get("origin_url")]
    cand = [c.strip() for c in cand if c]
    if not cand:
        return ""
    def _naver_score(u: str) -> int:
        h = re.sub(r"^https?://([^/]+).", r"\1", (u or "").lower())
        return 2 if ("naver.com" in h) else 1
    cand.sort(key=_naver_score, reverse=True)
    u = cand[0]
    u = re.sub(r"/amp(/|)?", "/", u)
    u = u.replace("m.news.naver.com", "n.news.naver.com")
    return u

def _make_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        cfg = use_config()
        cfg.set("DEFAULT", "user_agent", "Mozilla/5.0 (X11; Linux x86_64)")
        cfg.set("DEFAULT", "timeout", "12")
        _CONFIG_CACHE = cfg
    return _CONFIG_CACHE

def _domain_semaphore(url: str) -> threading.Semaphore:
    host = urlparse(url).netloc
    return _domain_locks[host]

# ----------------------------
# 정제 파이프라인
# ----------------------------
def _remove_after_anchors(text: str) -> str:
    if not text:
        return ""
    out = []
    for ln in text.splitlines():
        if any(p.search(ln.strip()) for p in _ANCHORS):
            break
        out.append(ln)
    return "\n".join(out)

def _strip_common_noise(text: str) -> str:
    if not text:
        return ""
    t = text
    for pat, _ in _NOISE_SUBS:
        t = pat.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = _URL_RE.sub(" ", t)
    t = _PRICE_RE.sub(" ", t)
    t = _PHOTO_CAP_RE.sub(" ", t)
    t = _SPECIAL_ICONS_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()

def _dedup_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?다])\s+", text)
    seen, out = set(), []
    for s in parts:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return " ".join(out)

def _remove_low_korean_density(text: str, threshold=0.25) -> str:
    lines = re.split(r"\n+", text)
    out = []
    for ln in lines:
        han = len(re.findall(r"[가-힣]", ln))
        if han / max(len(ln), 1) >= threshold or len(ln) < 30:
            out.append(ln.strip())
    return "\n".join(out)

def sanitize_article(text: str) -> str:
    t = _remove_after_anchors(text)
    t = _strip_common_noise(t)
    m = re.search(r"(제원|스펙|사양)\s*표", t)
    if m:
        t = t[:m.start()]
    t = _dedup_sentences(t)
    t = _remove_low_korean_density(t)
    return _WS_RE.sub(" ", t).strip()

# ----------------------------
# 메타데이터 파서
# ----------------------------
def extract_meta_from_html(html: str) -> Dict[str, str]:
    """BeautifulSoup 기반 보조 메타데이터 추출"""
    meta = {"site_name": "", "title_og": "", "description_og": "", "published_time": ""}
    if not html:
        return meta
    try:
        soup = BeautifulSoup(html, "html.parser")
        def _get(attr, key):
            el = soup.find("meta", attrs={attr: key})
            return el["content"].strip() if el and el.has_attr("content") else ""
        meta["site_name"] = _get("property", "og:site_name") or _get("name", "site_name")
        meta["title_og"] = _get("property", "og:title") or (soup.title.string.strip() if soup.title else "")
        meta["description_og"] = _get("property", "og:description") or _get("name", "description")
        meta["published_time"] = _get("property", "article:published_time") or \
                                 _get("name", "pubdate") or _get("name", "date")
    except Exception:
        pass
    return meta

# ----------------------------
# 본문 추출
# ----------------------------
def fetch_body(url: str, timeout=12) -> Tuple[str, str, Dict[str, str]]:
    if not url:
        return "", "", {}
    os.makedirs("data/article_cache", exist_ok=True)
    key = sha1(url)
    cache_path = f"data/article_cache/{key}.txt"

    # 캐시 우선
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = f.read().strip()
            if len(cached) >= MIN_LEN:
                return cached, "", {}

    cfg = _make_config()
    text_raw, html, meta = "", "", {}

    try:
        downloaded = trafilatura.fetch_url(url, config=cfg, timeout=timeout, no_ssl=True)
        if downloaded:
            t = trafilatura.extract(downloaded, include_comments=False, favor_recall=True, with_metadata=False)
            text_raw = clean_text(t or "")
    except Exception:
        text_raw = ""

    if len(text_raw) < MIN_LEN:
        global _SESSION
        if _SESSION is None:
            _SESSION = requests.Session()
            _SESSION.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
        try:
            r = _SESSION.get(url, timeout=timeout, verify=False)
            if r.ok and r.text:
                html = r.text
                meta = extract_meta_from_html(html)
                t2 = trafilatura.extract(html, include_comments=False, favor_recall=True, with_metadata=False)
                text_raw = clean_text(t2 or text_raw)
        except Exception:
            pass

    text_san = sanitize_article(text_raw)
    if len(text_san) >= MIN_LEN:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text_san)
    return text_san, text_raw, meta

# ----------------------------
# 처리 함수
# ----------------------------
def make_description_short(text: str, target_min=400, target_max=600) -> str:
    if not text:
        return ""
    txt = text.strip()
    para = txt.split("\n")[0].strip()
    base = para if len(para) >= target_min else txt
    if len(base) <= target_max:
        return base
    cut = base[:target_max]
    m = re.search(r"[.!?다]\s", cut[::-1])
    if m:
        idx = len(cut) - m.start()
        return cut[:idx].strip()
    return cut.strip()

def _process_one(it: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = pick_url(it)
    domain = urlparse(url).netloc if url else "-"
    if not url:
        return None, domain

    if len(it.get("body", "")) >= MIN_LEN:
        return it, domain

    try:
        sem = _domain_semaphore(url)
        with sem:
            body_san, body_raw, meta = fetch_body(url)
        if len(body_san) >= MIN_LEN:
            it.update(meta)
            it["raw_body"] = body_raw
            it["body"] = body_san
            it["description"] = body_san
            it["description_short"] = make_description_short(body_san)
            return it, domain
    except Exception:
        pass
    return None, domain

# ----------------------------
# 메인 실행
# ----------------------------
def main() -> int:
    print("[INFO] 기사 본문 수집 시작")
    is_monthly_run = os.getenv("MONTHLY_RUN", "false").lower() == "true"

    if is_monthly_run:
        meta_path = "outputs/debug/monthly_meta_agg.json"
        print(f"[INFO] Monthly Run: Using aggregated meta file.") # Simplified log
    else: # 일간 실행
        # data/ 디렉토리의 최신 news_meta_*.json 파일을 직접 사용
        meta_path = latest("data/news_meta_*.json")
        print(f"[INFO] Daily Run: Using latest meta file from data/ directory.") # Simplified log

    # --- ▼▼▼ 추가: 선택된 입력 파일 경로 로그 ▼▼▼ ---
    print(f"Selected Input File: {meta_path}")
    # --- ▲▲▲ 추가 완료 ▲▲▲ ---

    if not meta_path or not os.path.exists(meta_path):
        print(f"[ERROR] 입력 파일 없음. 경로 확인: {meta_path}")
        return 1

    items = load_json(meta_path, [])
    print(f"[INFO] 기사 {len(items)}개 처리 중... (MAX_WORKERS={MAX_WORKERS})")
    tried, updated, per_domain = 0, 0, {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_map = {ex.submit(_process_one, it): i for i, it in enumerate(items)}
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            try:
                res_item, domain = fut.result()
            except Exception:
                res_item, domain = None, "-"
            tried += 1
            if domain not in per_domain:
                per_domain[domain] = {"ok": 0, "fail": 0}
            if res_item:
                items[i] = res_item
                updated += 1
                per_domain[domain]["ok"] += 1
            else:
                per_domain[domain]["fail"] += 1
            if tried % 50 == 0:
                print(f"[INFO] 진행률 {tried}/{len(items)} (성공 {updated})")

    # --- ▼▼▼ 추가: 저장될 출력 파일 경로 로그 ▼▼▼ ---
    print(f"Saving updated data back to: {meta_path}")
    # --- ▲▲▲ 추가 완료 ▲▲▲ ---
    save_json(meta_path, items) # Save back to the *same* file it read from
    # --- ▼▼▼ 수정: 최종 성공 메시지에 출력 파일 경로 포함 ▼▼▼ ---
    print(f"[SUCCESS] 완료 | 시도={tried}, 업데이트={updated} | Output File: {meta_path}")
    # --- ▲▲▲ 수정 완료 ▲▲▲ ---
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
