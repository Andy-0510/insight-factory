import os
import re
import glob
import json
import csv
import datetime
import unicodedata
from collections import defaultdict, Counter

DICT_DIR = "data/dictionaries"
def _load_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []
STOP_EXT = set(_load_lines(os.path.join(DICT_DIR, "stopwords_ext.txt")))

def norm_tok(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

EN_STOP = set()
KO_FUNC = set()
EN_STOP |= {"news","press","corp","ltd","inc","co","group","update","daily","today"}
KO_FUNC |= {"기자","사진","자료","제공","종합","속보","단독","전문","영상","인터뷰","리뷰","광고","pr","홍보","출처","보도","보도자료","이벤트","공지","알림"}
STOP_EXT = set(EN_STOP) | set(KO_FUNC)


def _is_mostly_numeric(tok: str) -> bool:
    digits = sum(c.isdigit() for c in tok)
    return digits >= max(2, int(len(tok) * 0.6))

def _looks_noise(tok: str) -> bool:
    if len(tok) < 2:
        return True
    if _is_mostly_numeric(tok):
        return True
    # 숫자와 단위가 결합된 형태 (예: 7개국, 3분기)를 제거
    if re.match(r"^\d+(cm|mm|kg|g|m|km|년|월|일|분기|개국|위|종|가지)$", tok, flags=re.IGNORECASE):
        return True
    if tok in {"주식회사", "홀딩스", "그룹", "센터", "본부", "사업부"}:
        return True
    return False

def tokenize(t):
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", t or "")
    out = []
    for x in toks:
        x_norm = norm_tok(x)
        if x_norm in STOP_EXT:
            continue
        if _looks_noise(x_norm):
            continue
        out.append(x_norm)
    return out

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

# ================= 데이터 로더 (모듈 C와 로직 통일) =================
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

# ================= 통계 계산 함수들 =================
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
    # z_like 값을 -4.0 ~ 4.0 사이로 안정화(클리핑)
    return [max(-4.0, min(4.0, float(x))) for x in z]

def to_rows(dc):
    # ... (기존과 동일)
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

def export_trend_strength(rows):
    os.makedirs("outputs/export", exist_ok=True)
    filtered = []
    for r in rows:
        # 필터 기준 상향: total >= 8, cur >= 3, diff >= 1
        if r["total"] >= 8 and r["cur"] >= 3 and r["diff"] >= 1:
            # 랭킹/기간 패턴 단어는 강한 신호에서 제외
            if re.match(r"^\d+(위|개국|종|분기|월|년)$", r["term"]):
                continue
            filtered.append(r)
            
    filtered.sort(key=lambda x: (x["z_like"], x["diff"], x["cur"]), reverse=True)
    
    with open("outputs/export/trend_strength.csv","w",encoding="utf-8",newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in filtered[:300]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])

def export_weak_signals(rows):
    cand = []
    # 관측 일수가 14일 미만인 단어는 불안정하므로 제외
    min_observed_days = 14
    
    for r in rows:
        observed_days = sum(1 for c in r['counts'] if c > 0)
        if observed_days < min_observed_days:
            continue
            
        # 약한 신호 필터 기준 적용
        if r["total"] <= 30 and r["cur"] >= 2 and float(r["z_like"]) > 1.2:
            cand.append(r)
            
    cand.sort(key=lambda x: (x["z_like"], x["cur"], -x["total"]), reverse=True)
    
    with open("outputs/export/weak_signals.csv","w",encoding="utf-8",newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in cand[:200]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(float(r["ma7"]),3), round(float(r["z_like"]),3), r["total"]])
            
    
def export_weak_signals(rows):
    final_path = "outputs/export/weak_signals.csv"
    tmp_path = "outputs/export/weak_signals_tmp.csv"

    cand = []
    for r in rows:
        # 약한 신호의 기준: 전체 언급 20회 이하, 현재 언급 2회 이상, z_like 1.0 초과
        if r["total"] <= 20 and r["cur"] >= 2 and float(r["z_like"]) > 1.0:
            r["z_like"] = max(-5.0, min(5.0, float(r["z_like"]))) # z_like 안정화
            cand.append(r)

    cand.sort(key=lambda x: (x["z_like"], x["cur"], "total"), reverse=True)

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "cur", "prev", "diff", "ma7", "z_like", "total"])
        for r in cand[:200]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(float(r["ma7"]), 3), round(float(r["z_like"]), 3), r["total"]])

    os.rename(tmp_path, final_path)
    
# ===== 이벤트 추출/저장 =====
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
    # 하루 1파일 및 D+D+1 안정성 정책이 적용된 함수를 호출합니다.
    rows = load_stable_warehouse_data(days=30)
    
    dc = daily_counts(rows)
    rows2 = to_rows(dc)
    export_trend_strength(rows2)
    export_weak_signals(rows2)
    export_events()
    print("[INFO] signal_export | terms=", len(rows2))

if __name__ == "__main__":
    main()
