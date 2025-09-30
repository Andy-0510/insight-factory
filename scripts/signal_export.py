import os
import re
import glob
import json
import csv
import datetime
import unicodedata
from collections import defaultdict, Counter

## ⭐️ --- 어휘집 및 필터링 설정 ---
DICT_DIR = "data/dictionaries"

def _load_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []

STOP_EXT = set(_load_lines(os.path.join(DICT_DIR, "stopwords_ext.txt")))

def load_signal_vocabulary():
    """
    분석할 가치가 있는 핵심 단어 목록(신호 어휘집)을 동적으로 생성합니다.
    """
    cfg = {}
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        print("[WARN] signal_export: config.json을 찾을 수 없습니다.")

    vocab = set(cfg.get("domain_hints", []))
    vocab.update(_load_lines(os.path.join(DICT_DIR, "brands.txt")))
    vocab.update(_load_lines(os.path.join(DICT_DIR, "entities_org.txt")))

    return {norm_tok(v) for v in vocab if v}


## ⭐️ --- 텍스트 정규화 및 토큰화 ---
def norm_tok(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(t):
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", t or "")
    toks = [norm_tok(x) for x in toks if x and x not in STOP_EXT]
    return toks

def to_date(s: str) -> str:
    today = datetime.date.today()
    if not s or not isinstance(s, str): return today.strftime("%Y-%m-%d")
    s = s.strip()
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(iso)
        d = dt.date()
    except Exception:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(s)
            d = dt.date()
        except Exception:
            m = re.search(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})", s)
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try: d = datetime.date(y, mm, dd)
                except Exception: d = today
            else:
                d = today
    if d > today: d = today
    return d.strftime("%Y-%m-%d")


## ⭐️ --- 데이터 로딩 (안정성 규칙 적용) ---
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
    
    if not file_map:
        return []
    
    sorted_dates = sorted(file_map.keys())
    start_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=days)
    end_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=1)
    
    rows = []
    current_date = start_date

    while current_date <= end_date:
        date_str_d = current_date.strftime("%Y-%m-%d")
        files_to_check = []
        if date_str_d in file_map:
            files_to_check.append(file_map[date_str_d])
        if (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d") in file_map:
            files_to_check.append(file_map[(current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")])
            
        for fp in files_to_check:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            published_date = (obj.get("published") or obj.get("created_at") or os.path.basename(fp))[:10]
                            if published_date == date_str_d:
                                title = (obj.get("title") or "").strip()
                                toks = tokenize(title)
                                if toks:
                                    rows.append((date_str_d, toks))
                        except Exception:
                            continue
            except Exception:
                continue
        current_date += datetime.timedelta(days=1)
    return rows


## ⭐️ --- 통계 계산 및 CSV 출력 ---
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

## ⭐️ 2. z_like 클리핑 적용
def z_like(vals, ma):
    z = []
    for v, m in zip(vals, ma):
        zv = (v - m) / ((m**0.5) + 1.0)
        # 과도 스파이크 억제
        z.append(max(-4.0, min(4.0, float(zv))))
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
            "term": t, "counts": counts, "cur": cur, "prev": prev, "diff": diff,
            "ma7": ma7[-1] if ma7 else 0.0, "z_like": z[-1] if z else 0.0,
            "total": sum(counts)
        })
    return rows

def export_trend_strength(rows):
    os.makedirs("outputs/export", exist_ok=True)
    final_path = "outputs/export/trend_strength.csv"
    tmp_path = os.path.join(os.path.dirname(final_path), "trend_strength_tmp.csv")

    bad_generic = {"공급","산업","업계","시장","관련","분야"}

    # 세이프가드 + 최소 최근성
    filtered = []
    for r in rows:
        term = r["term"]
        if term in bad_generic:
            continue
        if r["cur"] < 1:  # 현재 노출 0이면 제외
            continue
        if r["diff"] < 0:  # 하락 항목은 제외
            continue
        filtered.append(r)
    
    filtered.sort(key=lambda x: (x["z_like"], x["diff"], x["cur"], x["total"]/10), reverse=True)
    topk = filtered[:300]
    
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in topk:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])
    os.replace(tmp_path, final_path)


## ⭐️ 3. weak_signals 필터 조정
def export_weak_signals(rows):
    os.makedirs("outputs/export", exist_ok=True)
    final_path = "outputs/export/weak_signals.csv"
    tmp_path = os.path.join(os.path.dirname(final_path), "weak_signals_tmp.csv")

    # 약신호 전용 범주형 금칙(어휘집 정제 이후에도 안전망)
    generic_stop = {"미국","유럽","산업","업계","공급","시장","관련","분야","글로벌","국내","해외","업체"}

    # 1차 필터: '작지만 가파른' 급등 신호
    cand = []
    for r in rows:
        term = r["term"]
        if term in generic_stop:
            continue
        # 살짝 완화: total≤30, z>1.02, prev≤1, diff≥1
        if r["total"] <= 30 and r["cur"] >= 2 and r["prev"] <= 1 and r["diff"] >= 1 and float(r["z_like"]) > 1.02:
            cand.append(r)

    # 2단 컷: 후보가 많을 때만 더 조여서 품질 유지
    if len(cand) > 50:
        cand = [r for r in cand if float(r["z_like"]) > 1.4 and r["total"] <= 20]

    # 정렬: 급등성(z), 현재강도(cur), 변화량(diff) 우선, 대형어는 하방(-total, -prev)
    cand.sort(key=lambda x: (x["z_like"], x["cur"], x["diff"], -x["total"], -x["prev"]), reverse=True)

    # 출력 상한 50
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in cand[:50]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(float(r["ma7"]),3), round(float(r["z_like"]),3), r["total"]])
    os.replace(tmp_path, final_path)
    return cand

## ⭐️ --- 이벤트 추출/저장 ---
EVENT_MAP = {
    "LAUNCH":      [r"출시", r"론칭", r"발표", r"선보이", r"공개"],
    "PARTNERSHIP": [r"제휴", r"파트너십", r"업무협약", r"\bMOU\b", r"맞손"],
    "INVEST":      [r"투자", r"유치", r"라운드", r"시리즈 [ABCD]"],
    "ORDER":       [r"수주", r"계약 체결", r"납품 계약", r"공급 계약"],
    "CERT":        [r"인증", r"허가", r"승인"],
    "REGUL":       [r"규제", r"가이드라인", r"행정예고"],
}

def _latest(path_glob: str):
    files = sorted(glob.glob(path_glob))
    return files[-1] if files else None

def _pick_meta_path():
    p1 = "outputs/debug/news_meta_latest.json"
    if os.path.exists(p1): return p1
    return _latest("data/news_meta_*.json")

def _detect_events_from_items(items: list) -> list:
    rows = []
    for it in items:
        text  = f"{(it.get('title') or '')}\n{(it.get('body') or it.get('description') or '')}"
        date = to_date(it.get("published_time") or it.get("pubDate_raw") or "")
        detected = {etype for etype, pats in EVENT_MAP.items() for pat in pats if re.search(pat, text, re.I)}
        if detected:
            rows.append({
                "date": date, "types": ",".join(sorted(detected)),
                "title": (it.get("title") or "")[:300], "url": it.get("url") or ""
            })
    return rows

def _dedup_events(rows: list) -> list:
    seen = set()
    out = []
    for r in rows:
        key = (r.get("title"), r.get("url"))
        if not all(key) or key in seen: continue
        seen.add(key)
        out.append(r)
    return out
    
def export_events():
    final_path = "outputs/export/events.csv"
    tmp_path = final_path + ".tmp"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    
    meta_path = _pick_meta_path()
    if not meta_path:
        print("[INFO] events.csv skipped (no meta)")
        return
        
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception as e:
        print(f"[WARN] events: meta load failed: {repr(e)}")
        items = []
    
    rows = _detect_events_from_items(items)
    rows = _dedup_events(rows)
    
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "types", "title", "url"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp_path, final_path)
    print(f"[INFO] events.csv exported | rows={len(rows)}")    

## ⭐️ --- 메인 실행 로직 ---
def main():
    # 1. '신호 어휘집' 동적 생성
    signal_vocab = load_signal_vocabulary()
    print(f"[INFO] 신호 어휘집 로드 완료: {len(signal_vocab)}개 단어")

    # ▼▼ 어휘집 1차 정제: 범주형 일반어·숫자/단위/기간 컷 ▼▼
    _GENERIC_STOP = {"미국","유럽","산업","업계","공급","시장","관련","분야","글로벌","국내","해외","업체"}
    _RE_UNIT   = re.compile(r"^\d+(hz|w|mah|nm|mm|cm|kg|g|gb|tb|mhz|ghz)$", re.I)
    _RE_PERIOD = re.compile(r"^\d{1,4}(년|월|분기)$")
    _RE_COUNT  = re.compile(r"^\d+(위|종|개국|명|가지)$")
    _RE_MIXED  = re.compile(r"^\d+[a-z가-힣]+$", re.I)

    def _vocab_noise(x: str) -> bool:
        if x in _GENERIC_STOP: return True
        if x.isdigit(): return True
        if _RE_UNIT.match(x) or _RE_PERIOD.match(x) or _RE_COUNT.match(x) or _RE_MIXED.match(x):
            return True
        return False

    signal_vocab = {t for t in signal_vocab if not _vocab_noise(t)}
    print(f"[INFO] 신호 어휘집 정제 후: {len(signal_vocab)}개 단어")
    
    # 2. 데이터 로드 및 '신호 어휘집' 기반 필터링
    rows = load_stable_warehouse_data(days=30)
    filtered_rows = []
    for d, toks in rows:
        qualified_toks = [t for t in toks if t in signal_vocab]
        if qualified_toks:
            filtered_rows.append((d, qualified_toks))
            
    print(f"[INFO] 원본 기사 제목: {len(rows)}개 -> 어휘집 필터링 후: {len(filtered_rows)}개")

    # 3. 통계 계산 및 CSV 파일 생성
    if not filtered_rows:
        print("[WARN] 분석할 유효 데이터가 없습니다. CSV 파일이 비어 있을 수 있습니다.")
        # 빈 파일이라도 생성하도록 처리
        open("outputs/export/trend_strength.csv", "w").close()
        open("outputs/export/weak_signals.csv", "w").close()
    else:
        dc = daily_counts(filtered_rows)
        rows2 = to_rows(dc)
        export_trend_strength(rows2)
        export_weak_signals(rows2)
    
    # 4. 이벤트 데이터는 별도 소스에서 추출
    export_events()
    
    print(f"[INFO] signal_export 완료")

if __name__ == "__main__":
    main()

