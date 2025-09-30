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
MIN_OBSERVED_DAYS = 7  # 최소 관측일수 조건 강화

def _load_lines(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return {x.strip().lower() for x in f if x.strip()}
    except Exception:
        return set()

# 불용어, 화이트리스트, 필터링용 단어 목록 로드
STOP_EXT = _load_lines(os.path.join(DICT_DIR, "stopwords_ext.txt"))
WHITELIST_KEYWORDS = _load_lines(os.path.join(DICT_DIR, "keyword_whitelist.txt"))
ROOT_COMPANY_NAMES = {"삼성", "lg", "sk", "현대"}
COMMON_BUSINESS_VERBS = {"본격화", "확대", "강화", "전망", "출시", "발표", "계획", "지원"}
WEAK_SIGNAL_SPECIFIC_STOPWORDS = {"미국", "유럽", "산업", "공급", "업계", "tv", "it", "모바일"}

# 범용 불용어 목록 확장 (A)
STOP_EXT.update(ROOT_COMPANY_NAMES)
STOP_EXT.update(COMMON_BUSINESS_VERBS)

# 노이즈 패턴 정규식
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
    warehouse_files = select_latest_files_per_day("data/warehouse/*.jsonl")
    file_map = {os.path.basename(f)[:10]: f for f in warehouse_files}
    if not file_map: return []
    
    sorted_dates = sorted(file_map.keys())
    start_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=days)
    end_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=1)
    
    rows = []
    current_date = start_date
    while current_date <= end_date:
        date_str_d = current_date.strftime("%Y-%m-%d")
        date_str_d_plus_1 = (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        files_to_check = []
        if date_str_d in file_map: files_to_check.append(file_map[date_str_d])
        if date_str_d_plus_1 in file_map: files_to_check.append(file_map[date_str_d_plus_1])
            
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
                        except Exception: continue
            except Exception: continue
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
def export_trend_strength(rows):
    os.makedirs("outputs/export", exist_ok=True)
    final_path = "outputs/export/trend_strength.csv"
    tmp_path = os.path.join(os.path.dirname(final_path), "trend_strength_tmp.csv")
    
    filtered = []
    for r in rows:
        observed_days = sum(1 for c in r['counts'] if c > 0)
        if observed_days < MIN_OBSERVED_DAYS:
            continue

        if r["total"] >= 8 and r["cur"] >= 3 and r["diff"] >= 1:
            term = r["term"]
            # 최종 안전장치: 루트 기업명 및 노이즈 필터
            if term in ROOT_COMPANY_NAMES or _looks_like_noise(term):
                continue
            filtered.append(r)
            
    filtered.sort(key=lambda x: (x["z_like"], x["diff"], x["cur"], x["total"]/10), reverse=True)
    
    with open(tmp_path,"w",encoding="utf-8",newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in filtered[:300]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])
    
    os.replace(tmp_path, final_path) # (E) 파일 쓰기 안정성
    return filtered

def export_weak_signals(rows):
    os.makedirs("outputs/export", exist_ok=True)
    final_path = "outputs/export/weak_signals.csv"
    tmp_path = os.path.join(os.path.dirname(final_path), "weak_signals_tmp.csv")

    # (B) 1차 필터: 제안된 강화된 규칙 적용
    cand = []
    for r in rows:
        if r["total"] <= 25 and r["cur"] >= 2 and float(r["z_like"]) > 1.3 and r["diff"] >= 1 and r["prev"] <= 1:
             term = r["term"]
             if not _looks_like_noise(term) and term not in ROOT_COMPANY_NAMES and term not in WEAK_SIGNAL_SPECIFIC_STOPWORDS:
                cand.append(r)
    
    # (C) 2단 컷: 후보가 50개 이상이면 더 엄격한 기준으로 추가 필터링
    if len(cand) > 50:
        cand = [r for r in cand if float(r["z_like"]) > 1.5 and r["total"] <= 20]

    # 백업 규칙
    backup_fired = False
    if not cand:
        backup_fired = True
        backup_cand = []
        sorted_by_z = sorted(rows, key=lambda x: x.get("z_like", 0.0), reverse=True)
        for r in sorted_by_z:
            term = r["term"]
            if r["total"] >= 2 and not _looks_like_noise(term) and term not in ROOT_COMPANY_NAMES and term not in WEAK_SIGNAL_SPECIFIC_STOPWORDS:
                backup_cand.append(r)
                if len(backup_cand) >= 30:
                    break
        cand = backup_cand

    # (D) 최종 정렬
    cand.sort(key=lambda x: (x["z_like"], x["cur"], x["diff"], -x["total"], -x["prev"]), reverse=True)
    
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in cand[:80]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(float(r["ma7"]),3), round(float(r["z_like"]),3), r["total"]])

    os.replace(tmp_path, final_path) # (E) 파일 쓰기 안정성
    return cand, backup_fired

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
            
    os.replace(tmp_path, final_path) # (E) 파일 쓰기 안정성
    print(f"[INFO] events.csv exported | rows={len(rows)}")


# ================= 메인 =================
def main():
    rows_raw = load_stable_warehouse_data(days=30)
    dc = daily_counts(rows_raw)
    rows_stat = to_rows(dc)
    
    print(f"[INFO] signal_export | total_terms={len(rows_stat)}")
    
    trends = export_trend_strength(rows_stat)
    weaks, backup_fired = export_weak_signals(rows_stat)
    
    print(f"[INFO] > Trend Strength: {len(trends)} candidates found.")
    if trends:
        preview = trends[0]
        print(f"[INFO] > Top Trend: {preview['term']} (z={preview['z_like']:.2f}, cur={preview['cur']}, total={preview['total']})")

    print(f"[INFO] > Weak Signals: {len(weaks)} candidates found. (backup_fired={backup_fired})")
    if weaks:
        preview = weaks[0]
        print(f"[INFO] > Top Weak: {preview['term']} (z={preview['z_like']:.2f}, cur={preview['cur']}, total={preview['total']})")
        
    export_events()

if __name__ == "__main__":
    main()

    
