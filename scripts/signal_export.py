import os
import re
import glob
import json
import csv
import datetime
import unicodedata
from collections import defaultdict, Counter

# ⭐️ --- 어휘집 및 필터링 설정 ---
DICT_DIR = "data/dictionaries"

def _load_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []

# 기존 불용어 집합은 유지
STOP_EXT = set(_load_lines(os.path.join(DICT_DIR, "stopwords_ext.txt")))

def load_signal_vocabulary():
    """
    분석할 가치가 있는 핵심 단어 목록(신호 어휘집)을 동적으로 생성합니다.
    - config.json의 domain_hints
    - brands.txt, entities_org.txt
    """
    cfg = {}
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        print("[WARN] signal_export: config.json을 찾을 수 없습니다.")

    # 여러 소스에서 어휘를 수집
    vocab = set(cfg.get("domain_hints", []))
    vocab.update(_load_lines(os.path.join(DICT_DIR, "brands.txt")))
    vocab.update(_load_lines(os.path.join(DICT_DIR, "entities_org.txt")))

    # 정규화 (소문자, 공백제거) 후 반환
    return {norm_tok(v) for v in vocab if v}


# ⭐️ --- 텍스트 정규화 및 토큰화 ---
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


# ⭐️ --- 데이터 로딩 (안정성 규칙 적용) ---
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
    start_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=days)
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


# ⭐️ --- 통계 계산 및 CSV 출력 ---
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
        z.append((v - m) / ( (m**0.5) + 1.0 ))
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

def export_trend_strength(rows):
    os.makedirs("outputs/export", exist_ok=True)
    with open("outputs/export/trend_strength.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "cur", "prev", "diff", "ma7", "z_like", "total"])
        for r in sorted(rows, key=lambda x: (x["z_like"], x["diff"], x["cur"]), reverse=True)[:300]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])

def export_weak_signals(rows):
    cand = []
    for r in rows:
        # 이 후보군(r)은 이미 '신호 어휘집'으로 필터링된 결과
        if r["total"] <= 15 and r["cur"] >= 2 and r["z_like"] > 0.8:
            cand.append(r)
    with open("outputs/export/weak_signals.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "cur", "prev", "diff", "ma7", "z_like", "total"])
        for r in sorted(cand, key=lambda x: (x["z_like"], x["cur"]), reverse=True)[:200]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])


# ⭐️ --- 이벤트 추출/저장 ---
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
    
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "types", "title", "url"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
            
    print(f"[INFO] events.csv exported | rows={len(rows)}")


# ⭐️ --- 메인 실행 로직 ---
def main():
    # 1. '신호 어휘집'을 동적으로 생성
    signal_vocab = load_signal_vocabulary()
    print(f"[INFO] 신호 어휘집 로드 완료: {len(signal_vocab)}개 단어")
    
    # 2. 안정성 규칙이 적용된 데이터 로드
    rows = load_stable_warehouse_data(days=30)
    
    # 3. ⭐️ 제목 토큰 중 '신호 어휘집'에 포함된 단어만 필터링
    filtered_rows = []
    for d, toks in rows:
        qualified_toks = [t for t in toks if t in signal_vocab]
        if qualified_toks:
            filtered_rows.append((d, qualified_toks))
            
    print(f"[INFO] 원본 데이터: {len(rows)}개 기사 제목 -> 필터링 후: {len(filtered_rows)}개 기사 제목")

if __name__ == "__main__":
    main()
