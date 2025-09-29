import os
import re
import glob
import json
import csv
import datetime
import unicodedata
from collections import defaultdict, Counter

# =================== 설정 및 정규식 패턴 정의 ===================
DICT_DIR = "data/dictionaries"

def _load_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return {x.strip().lower() for x in f if x.strip()}
    except Exception:
        return set()

STOP_EXT = _load_lines(os.path.join(DICT_DIR, "stopwords_ext.txt"))
WHITELIST_KEYWORDS = _load_lines(os.path.join(DICT_DIR, "keyword_whitelist.txt"))
STOP_EXT |= {"news","press","corp","ltd","inc","co","group","update","daily","today",
             "기자","사진","자료","제공","종합","속보","단독","전문","영상","인터뷰",
             "리뷰","광고","pr","홍보","출처","보도","보도자료","이벤트","공지","알림"}

RE_UNIT = re.compile(r"^\d+(hz|w|mah|nm|mm|cm|kg|g|gb|인치|니트)$", re.I)
RE_PERIOD = re.compile(r"^\d{1,4}(년|월|분기|차)$")
RE_COUNT = re.compile(r"^\d+(위|종|개국|명|가지)$")
RE_FORM = re.compile(r"^\d+-in-\d+$", re.I)

def _looks_like_noise(tok: str) -> bool:
    if len(tok) < 2: return True
    if tok.isdigit(): return True
    if RE_UNIT.match(tok): return True
    if RE_PERIOD.match(tok): return True
    if RE_COUNT.match(tok): return True
    if RE_FORM.match(tok): return True
    if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
        # 화이트리스트에 없는 숫자+문자 조합은 노이즈로 간주 (예: 8gb)
        if tok not in WHITELIST_KEYWORDS:
            return True
    return False

def tokenize(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9\-]{2,}", text or "")
    out = []
    for x in toks:
        x_lower = x.lower()
        if x_lower in STOP_EXT:
            continue
        if x_lower in WHITELIST_KEYWORDS:
            out.append(x_lower)
            continue
        if _looks_like_noise(x_lower):
            continue
        out.append(x_lower)
    return out

# ================= 데이터 로더 (기존 안정화 버전) =================
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
    """
    D일자와 D+1일자 규칙을 적용하여 안정적인 시계열 데이터를 로드합니다.
    """
    warehouse_files = select_latest_files_per_day("data/warehouse/*.jsonl")
    file_map = {os.path.basename(f)[:10]: f for f in warehouse_files}
    
    if not file_map:
        return []
    
    sorted_dates = sorted(file_map.keys())
    # 분석 시작일은 요청일수(days)만큼 과거로 설정
    start_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=days)
    # 마지막 날짜는 D+1일이 없으므로, 그 전날까지만 계산합니다.
    end_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=1)
    
    rows = []
    current_date = start_date

    while current_date <= end_date:
        date_str_d = current_date.strftime("%Y-%m-%d")
        date_str_d_plus_1 = (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        files_to_check = []
        if date_str_d in file_map:
            files_to_check.append(file_map[date_str_d])
        if date_str_d_plus_1 in file_map:
            files_to_check.append(file_map[date_str_d_plus_1])
            
        for fp in files_to_check:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            d_raw = obj.get("published") or obj.get("created_at") or os.path.basename(fp)[:10]
                            published_date = d_raw[:10]
                            
                            if published_date == date_str_d:
                                title = (obj.get("title") or "").strip()
                                toks = tokenize(title)
                                rows.append((date_str_d, toks))
                        except Exception:
                            continue
            except Exception:
                continue
        current_date += datetime.timedelta(days=1)
        
    return rows

# ================= 통계 계산 (z_like 안정화 적용) =================
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
        denom = (m ** 0.5) + 1.0
        z.append((v - m) / denom)
    return [max(-4.0, min(4.0, float(x))) for x in z]

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
            "term": t, "dates": dates, "counts": counts, "cur": cur, "prev": prev,
            "diff": diff, "ma7": ma7[-1] if ma7 else 0.0,
            "z_like": z[-1] if z else 0.0, "total": sum(counts)
        })
    return rows

# ================= CSV 출력 (안정성 강화 최종 버전) =================
MIN_OBSERVED_DAYS = 7 # 최소 관측일수 조건

def export_trend_strength(rows):
    os.makedirs("outputs/export", exist_ok=True)
    final_path = "outputs/export/trend_strength.csv"
    tmp_path = "outputs/export/trend_strength_tmp.csv"
    
    filtered = []
    for r in rows:
        observed_days = sum(1 for c in r['counts'] if c > 0)
        if observed_days < MIN_OBSERVED_DAYS:
            continue

        if r["total"] >= 8 and r["cur"] >= 3 and r["diff"] >= 1:
            if _looks_like_noise(r["term"]):
                continue
            filtered.append(r)
            
    filtered.sort(key=lambda x: (x["z_like"], x["diff"], x["cur"]), reverse=True)
    
    with open(tmp_path,"w",encoding="utf-8",newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in filtered[:300]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])
    
    os.rename(tmp_path, final_path)

def export_weak_signals(rows):
    os.makedirs("outputs/export", exist_ok=True)
    final_path = "outputs/export/weak_signals.csv"
    tmp_path = "outputs/export/weak_signals_tmp.csv"

    cand = []
    for r in rows:
        observed_days = sum(1 for c in r['counts'] if c > 0)
        if observed_days < MIN_OBSERVED_DAYS:
            continue
        
        if r["total"] <= 40 and r["cur"] >= 2 and float(r["z_like"]) > 0.8:
            if not _looks_like_noise(r["term"]):
                cand.append(r)
    
    if not cand:
        backup_cand = []
        sorted_by_z = sorted(rows, key=lambda x: x.get("z_like", 0.0), reverse=True)
        for r in sorted_by_z:
            if r["total"] >= 2 and not _looks_like_noise(r["term"]):
                backup_cand.append(r)
                if len(backup_cand) >= 30:
                    break
        cand = backup_cand

    cand.sort(key=lambda x: (x["z_like"], x["cur"], -x["total"]), reverse=True)
    
    with open(tmp_path,"w",encoding="utf-8",newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in cand[:200]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(float(r["ma7"]),3), round(float(r["z_like"]),3), r["total"]])

    os.rename(tmp_path, final_path)

# ================= 이벤트 추출/저장 (기존과 동일) =================
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
    p1 = "outputs/debug/news_meta_latest.json"
    if os.path.exists(p1):
        return p1
    return _latest("data/news_meta_*.json")

def _detect_events_from_items(items: list) -> list:
    rows = []
    for it in items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        body  = (it.get("body") or it.get("description") or it.get("description_og") or "").strip()
        text  = f"{title}\n{body}"
        
        # 새로 추가한 to_date 함수를 사용하여 날짜를 정확하게 변환합니다.
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
            rows.append({
                "date": date or "",
                "types": ",".join(sorted(detected_types)),
                "title": title[:300],
                "url": url
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
    
def export_events(out_path="outputs/export/events.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta_path = _pick_meta_path()
    if not meta_path:
        print("[INFO] events.csv skipped (no meta)")
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception as e:
        print("[WARN] events: meta load failed:", repr(e))
        items = []

    rows = _detect_events_from_items(items)
    rows = _dedup_events(rows)

    # 1. 최종 파일과 임시 파일 경로를 준비합니다.
    final_path = out_path
    tmp_path = os.path.join(os.path.dirname(out_path), "events_tmp.csv")

    # 2. 임시 파일에 안전하게 내용을 모두 씁니다.
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "types", "title", "url"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
            
    # 3. 작업이 끝나면 임시 파일의 이름을 최종 파일 이름으로 변경합니다.
    os.rename(tmp_path, final_path)

    print(f"[INFO] events.csv exported | rows={len(rows)}")

# ================= 메인 =================
def main():
    rows = load_stable_warehouse_data(days=30)
    dc = daily_counts(rows)
    rows2 = to_rows(dc)
    export_trend_strength(rows2)
    export_weak_signals(rows2)
    export_events()
    print("[INFO] signal_export | terms=", len(rows2))

if __name__ == "__main__":
    main()
