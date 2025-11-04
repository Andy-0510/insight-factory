import os
import re
import glob
import json
import csv
import time
import datetime
import unicodedata
from collections import defaultdict, Counter
from src.timeutil import to_date, kst_date_str, kst_run_suffix
from src.utils import latest, load_json
from typing import List, Dict, Any, Tuple, Optional

# --- ▼▼▼ 1. 라이브러리 임포트 추가 ▼▼▼ ---
from src.config import load_config
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] sklearn 라이브러리를 찾을 수 없습니다. 이벤트 중복 제거(코사인 유사도)를 건너뜁니다.")
# --- ▲▲▲ 1. 추가 완료 ▲▲▲ ---

# =================== 설정 ===================
DICT_DIR = "data/dictionaries"

# --- ▼▼▼ 2. config 로드 및 스코어링 어휘집 전역 변수 추가 ▼▼▼ ---
CFG = load_config()

def _load_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []

STOP_EXT = set(_load_lines(os.path.join(DICT_DIR, "stopwords_ext.txt")))

# =================== 정규화/토큰화 ===================
def norm_tok(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(t):
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", t or "")
    toks = [norm_tok(x) for x in toks if x and x not in STOP_EXT]
    return toks

def _load_scoring_vocab():
    """제안 1: Relevance 스코어링을 위한 어휘집 로드"""
    print("[INFO] [signal_export] Loading scoring vocabulary (domain_hints, brands, entities_org)...")
    vocab = set(CFG.get("domain_hints", [])) # [cite: 4-10]
    vocab.update(_load_lines(os.path.join(DICT_DIR, "brands.txt")))
    vocab.update(_load_lines(os.path.join(DICT_DIR, "entities_org.txt")))
    # 검색을 위해 소문자로 정규화
    return {norm_tok(v) for v in vocab if v and len(v) > 1}

# 스크립트 로드 시 한 번만 실행
SCORING_VOCAB = _load_scoring_vocab()
# --- ▲▲▲ 2. 추가 완료 ▲▲▲ ---

# =================== 어휘집 로드/정제 ===================
def load_signal_vocabulary():
    cfg = {}
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        print("[WARN] signal_export: config.json을 찾을 수 없습니다.")
    
    vocab = set(cfg.get("domain_hints", []))
    vocab.update(_load_lines(os.path.join(DICT_DIR, "brands.txt")))
    vocab.update(_load_lines(os.path.join(DICT_DIR, "entities_org.txt")))
    vocab = {norm_tok(v) for v in vocab if v}

    # 1차 정제: 범주형/숫자·단위/기간 제거
    _GENERIC_STOP = {"미국","유럽","산업","업계","공급","시장","관련","분야","글로벌","국내","해외","업체"}
    _RE_UNIT   = re.compile(r"^\d+(hz|w|mah|nm|mm|cm|kg|g|gb|tb|mhz|ghz|nit|nits)\$", re.I)
    _RE_PERIOD = re.compile(r"^\d{1,4}(년|월|분기)\$")
    _RE_COUNT  = re.compile(r"^\d+(위|종|개국|명|가지)\$")
    _RE_MIXED  = re.compile(r"^\d+[a-z가-힣]+\$", re.I)

    def _vocab_noise(x: str) -> bool:
        if x in _GENERIC_STOP: return True
        if x.isdigit(): return True
        if _RE_UNIT.match(x) or _RE_PERIOD.match(x) or _RE_COUNT.match(x) or _RE_MIXED.match(x):
            return True
        return False

    vocab = {t for t in vocab if not _vocab_noise(t)}
    return vocab

# =================== 데이터 로딩(안정) ===================
def select_latest_files_per_day(glob_pattern: str):
    all_files = sorted(glob.glob(glob_pattern))
    daily_files = defaultdict(list)
    for f in all_files:
        date_key = os.path.basename(f)[:10]
        daily_files[date_key].append(f)
    latest_daily_files = []
    for date_key in sorted(daily_files.keys()):
        latest_file_for_day = sorted(daily_files[date_key])[-1]
        latest_daily_files.append(latest_file_for_day)
    return latest_daily_files

def load_stable_warehouse_data(days: int = 30):
    files = select_latest_files_per_day("data/warehouse/*.jsonl")
    file_map = {os.path.basename(f)[:10]: f for f in files}
    if not file_map:
        return []
    sorted_dates = sorted(file_map.keys())
    start_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=days)
    end_date   = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=1)

    rows = []
    current_date = start_date
    while current_date <= end_date:
        d0 = current_date.strftime("%Y-%m-%d")
        d1 = (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        files_to_check = []
        if d0 in file_map: files_to_check.append(file_map[d0])
        if d1 in file_map: files_to_check.append(file_map[d1])

        for fp in files_to_check:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            d_raw = obj.get("published") or obj.get("created_at") or os.path.basename(fp)[:10]
                            published_date = (d_raw or "")[:10]
                            if published_date == d0:
                                title = (obj.get("title") or "").strip()
                                toks = tokenize(title)
                                rows.append((d0, toks))
                        except Exception:
                            continue
            except Exception:
                continue
        current_date += datetime.timedelta(days=1)
    return rows

# =================== 통계/지표 ===================
def daily_counts(rows):
    by_day = defaultdict(Counter)
    for d, toks in rows:
        for t in toks:
            by_day[d][t] += 1
    return dict(sorted(by_day.items()))

def moving_avg(vals, w=7):
    out = []
    for i in range(len(vals)):
        s = max(0, i - w + 1)
        seg = vals[s:i+1]
        out.append(sum(seg) / max(1, len(seg)))
    return out

def z_like(vals, ma):
    z = []
    for v, m in zip(vals, ma):
        zv = (v - m) / ((m**0.5) + 1.0)
        z.append(zv)
    return z

def to_rows(dc):
    terms = set()
    for d, c in dc.items():
        terms.update(c.keys())
    dates = sorted(dc.keys())
    rows = []
    for t in sorted(terms):
        counts = [dc[d].get(t, 0) for d in dates]
        ma7 = moving_avg(counts, 7)
        z = z_like(counts, ma7)
        cur = counts[-1] if counts else 0
        prev = counts[-2] if len(counts) >= 2 else 0
        diff = cur - prev
        rows.append({
            "term": t,
            "dates": dates,
            "counts": counts,
            "cur": cur, "prev": prev, "diff": diff,
            "ma7": ma7[-1] if ma7 else 0.0,
            "z_like": z[-1] if z else 0.0,
            "total": sum(counts)
        })
    return rows

# --- ▼▼▼ 3. 신규 헬퍼 함수 3개 추가 ▼▼▼ ---

def _score_article_relevance(item: dict) -> int:
    """제안 1: 기사별 Relevance 스코어 계산"""
    text = (item.get("title", "") + " " + (item.get("body") or item.get("description") or "")).lower()
    if not text:
        return 0
    
    # 미리 로드된 SCORING_VOCAB (소문자 셋)을 사용해 점수 계산
    score = 0
    for keyword in SCORING_VOCAB:
        if keyword in text:
            score += 1
    return score

def _deduplicate_by_content(articles: list, threshold=0.9) -> list:
    """제안 2: 코사인 유사도 기반 중복 기사 필터링"""
    if not SKLEARN_AVAILABLE or len(articles) < 2:
        return articles

    docs = []
    for item in articles:
        # 본문(body) > 요약(description) > 제목(title) 순으로 텍스트 확보
        body = item.get("body")
        if not body or len(body) < 100: # 본문이 너무 짧으면
            body = item.get("description")
        if not body or len(body) < 100:
            body = item.get("title", "")
        docs.append(body or "")

    if not any(docs): # 모든 문서가 비어있으면
        return articles

    try:
        vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb', ngram_range=(4, 7))
        X = vectorizer.fit_transform(docs)
        sim_matrix = cosine_similarity(X)
        
        keep_indices = []
        removed_indices = set()
        
        # 점수(score)가 이미 매겨져 있다고 가정하고,
        # 점수가 높은 기사를 원본으로 삼기 위해 정렬 (select_top_articles와 동일 로직)
        # (여기서는 score가 없으므로 index 순서대로 진행)
        for i in range(len(articles)):
            if i in removed_indices:
                continue
            keep_indices.append(i)
            for j in range(i + 1, len(articles)):
                if j in removed_indices:
                    continue
                if sim_matrix[i, j] >= threshold:
                    removed_indices.add(j)
        
        print(f"[INFO] [signal_export] Content deduplication: {len(articles)} -> {len(keep_indices)} articles.")
        return [articles[i] for i in keep_indices]
    except Exception as e:
        print(f"[WARN] [signal_export] Content deduplication failed: {e}. Skipping.")
        return articles

def _extract_org_from_text(text: str, whitelist: set, alias_map: Dict, brand_to_company: Dict) -> str:
    """[보너스] 텍스트에서 'org'를 추출하는 로직 (일간 리포트 호환용)"""
    raw_toks = re.findall(r"[가-힣A-Za-z0-9\-\+\.]{2,}", text)
    mentioned_orgs = set()
    for t in raw_toks:
        norm_t = alias_map.get(t.lower(), t.lower())
        mapped_org = brand_to_company.get(norm_t, norm_t)
        if mapped_org in whitelist:
            mentioned_orgs.add(mapped_org)
    
    # whitelist에 "lg디스플레이"가 있고, mapped_org도 "lg디스플레이"여야 함
    
    # 대소문자 보정: 소문자 셋(whitelist)에 있다면, 원본(ent_org)에서 대소문자 맞는 이름 찾기
    final_orgs = set()
    ent_org_list = _load_lines(os.path.join(DICT_DIR, "entities_org.txt"))
    for org_lower in mentioned_orgs:
        found = False
        for org_proper in ent_org_list:
            if org_proper.lower() == org_lower:
                final_orgs.add(org_proper)
                found = True
                break
        if not found:
             final_orgs.add(org_lower) # 원본 리스트에 없으면 그냥 소문자 이름 사용

    return ", ".join(sorted(final_orgs)) if final_orgs else "Unknown"

# --- ▲▲▲ 3. 추가 완료 ▲▲▲ ---

# =================== 이벤트 규칙 ===================
EVENT_MAP = {
    "LAUNCH":      [r"출시", r"론칭", r"발표", r"선보이", r"공개"],
    "PARTNERSHIP": [r"제휴", r"파트너십", r"업무협약", r"\bMOU\b", r"맞손"],
    "INVEST":      [r"투자", r"유치", r"라운드", r"시리즈 [ABCD]"],
    "ORDER":       [r"수주", r"계약 체결", r"납품 계약", r"공급 계약", r"수의 계약"],
    "CERT":        [r"인증", r"허가", r"승인", r"적합성 평가", r"CE ?인증", r"FDA ?승인"],
    "REGUL":       [r"규제", r"가이드라인", r"행정예고", r"고시", r"지침", r"제정", r"개정"],
}

def _latest(path_glob: str):
    files = sorted(glob.glob(path_glob))
    return files[-1] if files else None

def _pick_meta_path():
    """실행 주기(일간/주간/월간)에 맞는 메타 데이터 파일 경로를 반환합니다."""
    is_monthly_run = os.getenv("MONTHLY_RUN", "false").lower() == "true"
    is_weekly_run = os.getenv("WEEKLY_RUN", "false").lower() == "true"

    if is_monthly_run:
        path = "outputs/debug/monthly_meta_agg.json"
        print(f"[INFO] signal_export: Using monthly aggregated meta file: {path}")
        return path if os.path.exists(path) else None
    
    if is_weekly_run:
        path = "outputs/debug/weekly_meta_agg.json"
        print(f"[INFO] signal_export: Using weekly aggregated meta file: {path}")
        return path if os.path.exists(path) else None

    # 일간 실행 (기존 로직)
    p1 = "outputs/debug/news_meta_latest.json"
    if os.path.exists(p1):
        return p1
    return latest("data/news_meta_*.json")

def _detect_events_from_items(items: list, whitelist: set, alias_map: Dict, brand_to_company: Dict) -> list:
    rows = []
    for it in items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        body  = (it.get("body") or it.get("description") or it.get("description_og") or "").strip()
        text  = f"{title}\n{body}"
        date_raw = it.get("published_time") or it.get("pubDate_raw") or ""
        date = to_date(date_raw)
        url = it.get("url") or ""

        detected_types = []
        for etype, pats in EVENT_MAP.items():
            for pat in pats:
                if re.search(pat, text, flags=re.IGNORECASE):
                    detected_types.append(etype)
                    break

        if detected_types:
            # [신규] Org 추출 로직 호출
            org_str = _extract_org_from_text(text, whitelist, alias_map, brand_to_company)

            rows.append({
                "date": date or "",
                "types": ",".join(sorted(detected_types)),
                "title": title[:300],
                "url": url,
                "org": org_str # [신규] Org 컬럼 추가
            })
    return rows

def _dedup_events(rows: list) -> list:
    seen_titles, seen_urls, out = set(), set(), []
    for r in rows:
        title = r.get("title", "")
        url = r.get("url", "")
        if not title or not url or title in seen_titles or url in seen_urls:
            continue
        seen_titles.add(title)
        seen_urls.add(url)
        out.append(r)
    return out

# =================== CSV Export ===================
def export_trend_strength(rows):
    os.makedirs("outputs/export", exist_ok=True)
    final_path = "outputs/export/trend_strength.csv"
    tmp_path = final_path + ".tmp"

    bad_generic = {"공급","산업","업계","시장","관련","분야"}
    filtered = [r for r in rows if r["term"] not in bad_generic and r["cur"] >= 1]
    filtered.sort(key=lambda x: (x["z_like"], x["diff"], x["cur"]), reverse=True)
    topk = filtered[:300]

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in topk:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(float(r["ma7"]),3), round(float(r["z_like"]),3), r["total"]])
    os.replace(tmp_path, final_path)
    print(f"[INFO] [signal_export] -> trend_strength.csv 생성 완료 ({len(topk)}개 행)")

def export_weak_signals(rows):
    os.makedirs("outputs/export", exist_ok=True)
    final_path = "outputs/export/weak_signals.csv"
    tmp_path = final_path + ".tmp"

    generic_stop = {"미국","유럽","산업","업계","공급","시장","관련","분야","글로벌","국내","해외","업체"}
    cand = []
    pre_cand = []  # 1차 조건만 통과(디버그용)

    for r in rows:
        term = r["term"]
        if term in generic_stop:
            continue

        z = float(r["z_like"])
        tot = int(r["total"])

        # 최근성 필터(스테이징 전)
        if r["cur"] >= 1 and r["prev"] <= 3 and r["diff"] >= 0:
            pre_cand.append(r)

            # 스테이징 컷
            pass_cond = False
            if z > 1.0 and tot <= 40:
                pass_cond = True
            elif z > 0.9 and tot <= 25:
                pass_cond = True

            if pass_cond:
                cand.append(r)

    # 후보 과다 시 2단 컷(보수형)
    if len(cand) > 50:
        cand = [r for r in cand if float(r["z_like"]) > 1.4 and r["total"] <= 20]

    cand.sort(key=lambda x: (x["z_like"], x["cur"], x["diff"], -x["total"], -x["prev"]), reverse=True)

    # 디버그 덤프
    os.makedirs("outputs/debug", exist_ok=True)
    with open("outputs/debug/weak_candidates.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","total","ma7","z_like"])
        for r in sorted(pre_cand, key=lambda x: (x["z_like"], x["cur"], -x["total"]), reverse=True):
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], r["total"], round(float(r["ma7"]),3), round(float(r["z_like"]),3)])

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in cand[:50]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(float(r["ma7"]),3), round(float(r["z_like"]),3), r["total"]])
    os.replace(tmp_path, final_path)
    print(f"[INFO] weak_signals | pre={len(pre_cand)} final={len(cand)}")
    # 분포 힌트
    z_over_1 = sum(1 for r in pre_cand if float(r["z_like"]) > 1.0)
    z_09_10 = sum(1 for r in pre_cand if 0.9 < float(r["z_like"]) <= 1.0)
    z_085_09 = sum(1 for r in pre_cand if 0.85 < float(r["z_like"]) <= 0.9)
    print(f"[INFO] [signal_export] -> weak_signals.csv 생성 완료 (후보 {len(pre_cand)}개 중 {len(cand)}개 선정)")
    print(f"[DEBUG] z>1.0:{z_over_1} | 0.9<z<=1.0:{z_09_10} | 0.85<z<=0.9:{z_085_09}")

def export_events(out_path="outputs/export/events.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta_path = _pick_meta_path()
    if not meta_path:
        print("[INFO] events.csv skipped (no meta)")
        # 3단계를 위해 빈 파일 생성
        pd.DataFrame(columns=["date", "types", "title", "url", "org"]).to_csv(out_path, index=False, encoding="utf-8-sig")
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception as e:
        print(f"[WARN] events: meta load failed: {repr(e)}")
        items = []

    if not items:
        print("[WARN] events: No items to process.")
        pd.DataFrame(columns=["date", "types", "title", "url", "org"]).to_csv(out_path, index=False, encoding="utf-8-sig")
        return

    # --- ▼▼▼ 4. 제안 로직 실행 ▼▼▼ ---
    
    # 4-1. [신규] Org 추출에 필요한 자원 로드
    alias_map = CFG.get("alias", {})
    brand_to_company = load_json("data/dictionaries/brand_to_company.json", {})
    topic_like_entities = _load_lines(os.path.join(DICT_DIR, "topic_like_entities.txt"))
    ent_org = _load_lines(os.path.join(DICT_DIR, "entities_org.txt"))
    # (중요) whitelist는 소문자로 정규화
    whitelist_base = {w.lower() for w in ent_org} - {t.lower() for t in topic_like_entities}
    whitelist = {norm_tok(alias_map.get(w, w)) for w in whitelist_base}

    # 4-2. [제안 1] Relevance 스코어링 및 필터링
    RELEVANCE_THRESHOLD = 17
    relevant_items = []
    for item in items:
        score = _score_article_relevance(item)
        if score >= RELEVANCE_THRESHOLD:
            relevant_items.append(item)
    
    print(f"[INFO] [signal_export] Relevance filtering: {len(items)} -> {len(relevant_items)} articles (score >= {RELEVANCE_THRESHOLD})")

    # 4-3. [제안 2] 코사인 유사도 중복 필터링
    filtered_items = _deduplicate_by_content(relevant_items, threshold=0.1)

    # 4-4. [수정됨] 최종 필터링된 기사로 이벤트 감지 (Org 추출 포함)
    rows = _detect_events_from_items(filtered_items, whitelist, alias_map, brand_to_company)
    
    # 4-5. [삭제됨] 기존의 _dedup_events(rows)는 더 이상 필요 없음
    # rows = _dedup_events(rows) 

    # --- ▲▲▲ 4. 로직 적용 완료 ▲▲▲ ---

    tmp_path = out_path + ".tmp"
    
    # [수정] fieldnames에 'org' 추가
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "types", "title", "url", "org"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    
    os.replace(tmp_path, out_path)
    print(f"[INFO] [signal_export] -> events.csv 생성 완료 ({len(rows)}개 이벤트)")

# --- ▼▼▼▼▼ [신규 추가] 일일 급등 신호 추출 함수 ▼▼▼▼▼ ---
def export_daily_spikes(rows, out_path="outputs/export/daily_hot_signals.csv"):
    """
    당일 데이터(cur)가 전일(prev) 및 7일 평균(ma7)보다 높고,
    z_like 점수가 특정 임계치를 넘는 '오늘의 급등 신호'만 추출합니다.
    """
    spikes = []
    for r in rows:
        try:
            # 급등 조건 정의
            is_spike = (
                r["cur"] > r["prev"] and
                r["cur"] > r["ma7"] and
                r["z_like"] >= 1.5 and
                r["diff"] >= 2
            )
            if is_spike:
                spikes.append(r)
        except (TypeError, KeyError):
            continue
            
    spikes.sort(key=lambda x: (x["z_like"], x["diff"]), reverse=True)
    
    # CSV 파일로 저장
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "cur", "diff", "z_like"])
        for r in spikes[:10]: # 상위 10개만 저장
            writer.writerow([r["term"], r["cur"], r["diff"], f"{r['z_like']:.2f}"])
            
    print(f"[INFO] [signal_export] -> daily_hot_signals.csv 생성 완료 ({len(spikes)}개 후보 중 {min(len(spikes), 10)}개 선정)")

# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# =================== 메인 ===================
def main():
    t0 = time.time()
    print("[INFO] [signal_export] KICK-OFF: 트렌드 신호 및 이벤트 추출을 시작합니다.")

    # 1) 신호 어휘집
    signal_vocab = load_signal_vocabulary()
    print(f"[INFO] 신호 어휘집 로드/정제 완료: {len(signal_vocab)}개")

    # 2) 데이터 로드
    rows_raw = load_stable_warehouse_data(days=30)
    print(f"[INFO] [signal_export] 원본 기사 {len(rows_raw)}건 로드 완료.")

    # 3) 어휘집 기반 필터링
    filtered_rows = []
    for d, toks in rows_raw:
        qualified = [t for t in toks if t in signal_vocab]
        if qualified:
            filtered_rows.append((d, qualified))
    print(f"[INFO] 원본 기사수={len(rows_raw)} → 어휘집 필터 후={len(filtered_rows)}")

    # 4) 통계 변환
    dc = daily_counts(filtered_rows)
    rows_stat = to_rows(dc)
    print(f"[INFO] 통계 대상 term 수={len(rows_stat)}")

    # 5) 오늘자(cur>0) 디버그 덤프
    os.makedirs("outputs/debug", exist_ok=True)
    with open("outputs/debug/today_terms.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","total","ma7","z_like"])
        for r in sorted(rows_stat, key=lambda x: (x["cur"], x["z_like"], x["total"]), reverse=True):
            if r["cur"] > 0:
                w.writerow([r["term"], r["cur"], r["prev"], r["diff"], r["total"], round(float(r["ma7"]),3), round(float(r["z_like"]),3)])
    print(f"[SUCCESS] [signal_export] 모든 신호 추출 및 파일 생성 완료 | 소요시간: {round(time.time()-t0, 2)}초")

    # 6) Export
    export_trend_strength(rows_stat)
    export_weak_signals(rows_stat)
    export_events()
    # --- ▼▼▼▼▼ [추가] 신규 함수 호출 ▼▼▼▼▼ ---
    export_daily_spikes(rows_stat)
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

if __name__ == "__main__":
    main()
