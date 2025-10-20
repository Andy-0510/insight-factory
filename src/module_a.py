import os
import json
import time
import random
import re
import html
import requests
from bs4 import BeautifulSoup
from src.config import load_config
from src.timeutil import kst_date_str, kst_run_suffix
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures


CFG = load_config()
NAVER_API = "https://openapi.naver.com/v1/search/news.json"

def naver_headers():
    return {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID", ""),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET", ""),
        "User-Agent": "Mozilla/5.0"
    }

def http_get(url, params=None, headers=None, timeout=10, max_retry=3):
    for i in range(max_retry):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait = 30 + i * 15
                print(f"[WARN] [module_a] 429 Too Many Requests, {wait}초 대기 후 재시도") # 로그 강화
                time.sleep(wait)
                continue  # 재시도
            if r.status_code >= 500:
                raise requests.HTTPError(f"5xx {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as e:
            if i == max_retry - 1:
                print(f"[ERROR] [module_a] HTTP GET 실패: {e}") # 로그 강화
                raise
            time.sleep(1.2 * (2 ** i) + random.random())

def prefer_link(item):
    return item.get("originallink") or item.get("link") or ""

def fetch_naver_news(query, display=30, pages=2):
    items = []
    for p in range(pages):
        start = 1 + p * display
        if start > 1000: break
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": "date"
        }
        r = http_get(NAVER_API, params=params, headers=naver_headers(), timeout=10, max_retry=3)
        data = r.json()
        batch = data.get("items", [])
        if not batch:
            break
        for it in batch:
            it["_query"] = query if query else "unknown"
        items.extend(batch)
        time.sleep(0.3)
    return items

def dedup_by_url(items):
    seen, out = set(),[]
    for it in items:
        url = prefer_link(it)
        if "_query" not in it or it["_query"] is None:
            it["_query"] = "unknown"
        if url and url not in seen:
            seen.add(url)
            out.append(it)
    return out

def expand_with_og(url):
    meta = {
        "url": url,
        "site_name": None,
        "title_og": None,
        "description_og": None,
        "published_time": None
    }
    try:
        # [수정] SSL 오류를 우회하기 위해 verify=False 옵션 추가
        r = http_get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, max_retry=2, verify=False)
        
        soup = BeautifulSoup(r.text, "lxml")
        def og(name):
            tag = soup.find("meta", property=name)
            return tag["content"].strip() if tag and tag.has_attr("content") else None
        meta["site_name"] = og("og:site_name")
        meta["title_og"] = og("og:title")
        meta["description_og"] = og("og:description")
        meta["published_time"] = og("article:published_time")
    except Exception:
        # 403 Forbidden 같은 오류는 여기서 처리되어 해당 URL만 건너뛰게 됩니다.
        pass
    return meta

def clean_html(s):
    if not s:
        return s
    s = re.sub(r"<.+?>", " ", s)
    s = html.unescape(s)
    return s.strip()

def main():
    t0 = time.time()
    print("[INFO] [module_a] KICK-OFF: 네이버 뉴스 API 데이터 수집을 시작합니다.")
    
    dry_run = (os.getenv("DRY_RUN", str(CFG.get("dry_run", True))).lower() == "true")
    q_raw = CFG.get("queries", ["unknown"])
    queries = q_raw if isinstance(q_raw, list) and q_raw else ["unknown"]
    display = int(CFG.get("per_query_display", 10))
    display = max(1, min(display, 100))
    pages = int(CFG.get("pages", 1))
    pages = max(1, pages)
    if not dry_run:
        pages = max(1, pages)
    
    print(f"[INFO] queries={len(queries)} dry_run={dry_run} display={display} pages={pages}")
    
    # --- ▼▼▼ [수정] 네이버 뉴스 API 호출 병렬 처리 ▼▼▼ ---
    print(f"[INFO] Starting Naver API calls for {len(queries)} queries in parallel...")
    all_items = []
    # API 서버에 과도한 부하를 주지 않도록 max_workers를 5 정도로 제한하는 것이 안정적입니다.
    with ThreadPoolExecutor(max_workers=5) as executor:
        # map 함수는 각 쿼리에 대해 fetch_naver_news를 실행하고, 그 결과(batch)들을 순서대로 반환합니다.
        results = executor.map(
            lambda q: fetch_naver_news(q, display=display, pages=pages),
            queries
        )
        
        # map이 반환한 결과(batch들의 이터레이터)를 하나의 리스트로 통합합니다.
        for batch in results:
            all_items.extend(batch)
            
    print(f"[INFO] Naver API calls finished. Total items fetched: {len(all_items)}")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
        
    clean_items = dedup_by_url(all_items)
    print(f"[INFO] Unique articles to process: {len(clean_items)}")

    # --- ▼▼▼ [수정] OG 메타데이터 수집 병렬 처리 ▼▼▼ ---
    meta_list = []
    # max_workers: 동시에 실행할 작업의 최대 개수 (네트워크 환경에 따라 조절 가능)
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 각 뉴스 아이템의 URL로 OG 메타데이터 수집 작업을 예약합니다.
        future_to_item = {executor.submit(expand_with_og, prefer_link(it)): it for it in clean_items}

        # 완료되는 작업부터 순서대로 결과를 처리합니다.
        for future in concurrent.futures.as_completed(future_to_item):
            it = future_to_item[future]
            try:
                # 작업 결과(OG 메타데이터)를 가져옵니다.
                meta = future.result()
                
                # 기존의 데이터 조합 로직은 그대로 유지합니다.
                meta["title"] = clean_html(it.get("title"))
                meta["description"] = clean_html(it.get("description"))
                meta["pubDate_raw"] = it.get("pubDate")
                meta["_query"] = it.get("_query")
                meta_list.append(meta)
            except Exception as e:
                print(f"[ERROR] OG data collection failed for url {prefer_link(it)}: {e}")

    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    os.makedirs("data", exist_ok=True)
    ts_str = f"{kst_date_str()}-{kst_run_suffix()}"
    raw_path = f"data/news_clean_{ts_str}.json"
    meta_path = f"data/news_meta_{ts_str}.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(clean_items, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 저장 완료: {raw_path}, {meta_path} | 총 수집(중복 제거 후): {len(clean_items)} | 경과(초): {round(time.time()-t0,2)}")
    
if __name__ == "__main__":
    main()
