# File: src/module_a.py
import os
import json
import time
import random
import re
import html
import requests
from bs4 import BeautifulSoup
from src.config import load_config
from src.timeutil import kst_date_str, kst_run_suffix, to_date, KST # Added KST and to_date
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures # Added for as_completed
from datetime import datetime, timedelta # Added for date filtering

CFG = load_config()
NAVER_API = "https://openapi.naver.com/v1/search/news.json"

def naver_headers():
    return {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID", ""),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET", ""),
        "User-Agent": "Mozilla/5.0"
    }

def http_get(url, params=None, headers=None, timeout=10, max_retry=3, verify=True): # Added verify parameter
    for i in range(max_retry):
        try:
            # Pass verify parameter to requests.get
            r = requests.get(url, params=params, headers=headers, timeout=timeout, verify=verify)
            if r.status_code == 429:
                 wait = 30 + i * 15
                 print(f"[WARN] [module_a] 429 Too Many Requests, {wait}초 대기 후 재시도")
                 time.sleep(wait)
                 continue
            if r.status_code >= 500:
                 raise requests.HTTPError(f"5xx {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as e:
            if i == max_retry - 1:
                print(f"[ERROR] [module_a] HTTP GET 실패: {e}")
                raise # Re-raise the exception on the last attempt
            # Use exponential backoff with jitter
            time.sleep(1.2 * (2 ** i) + random.random())
    # This part should ideally not be reached if max_retry >= 1
    # raise RuntimeError("HTTP GET failed after multiple retries")

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
        try: # Add try-except for the API call itself
            r = http_get(NAVER_API, params=params, headers=naver_headers(), timeout=10, max_retry=3)
            data = r.json()
            batch = data.get("items", [])
            if not batch:
                 break
            for it in batch:
                it["_query"] = query if query else "unknown"
            items.extend(batch)
        except Exception as api_err:
            print(f"[ERROR] [module_a] Failed fetching query '{query}' page {p+1}: {api_err}")
            # Optionally break or continue based on error handling strategy
            break
        time.sleep(0.3)
    return items

def dedup_by_url(items):
    seen, out = set(),[]
    for it in items:
        url = prefer_link(it)
        # Ensure _query exists and is not None before checking
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
    # if not dry_run: # This check seems redundant now, pages defaults to 1 or more
    #     pages = max(1, pages)

    print(f"[INFO] queries={len(queries)} dry_run={dry_run} display={display} pages={pages}")

    # --- ▼▼▼ 네이버 뉴스 API 호출 병렬 처리 (all_items 초기화 및 결과 수집) ▼▼▼ ---
    print(f"[INFO] Starting Naver API calls for {len(queries)} queries in parallel...")
    all_items = [] # Initialize all_items here
    # API 서버에 과도한 부하를 주지 않도록 max_workers를 5 정도로 제한하는 것이 안정적입니다.
    with ThreadPoolExecutor(max_workers=5) as executor:
        # map 함수는 각 쿼리에 대해 fetch_naver_news를 실행하고, 그 결과(batch)들을 순서대로 반환합니다.
        # Use list() to ensure all results are collected before proceeding
        results = list(executor.map(
            lambda q: fetch_naver_news(q, display=display, pages=pages),
            queries
        ))

        # map이 반환한 결과(batch들의 리스트)를 하나의 리스트로 통합합니다.
        for batch in results:
            all_items.extend(batch)

    print(f"[INFO] Naver API calls finished. Total items fetched: {len(all_items)}")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    # --- ▼▼▼ dedup_by_url 호출 위치 (all_items가 정의된 후) ▼▼▼ ---
    clean_items = dedup_by_url(all_items)
    print(f"[INFO] Unique articles fetched (after dedup): {len(clean_items)}")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    # --- ▼▼▼ 날짜 필터링 로직 추가 (clean_items 사용) ▼▼▼ ---
    three_days_ago = (datetime.now(KST) - timedelta(days=2)).date() # 3일 전 날짜 (KST 기준)
    filtered_items = []
    skipped_by_date = 0

    for item in clean_items: # Use clean_items here
        pub_date_str = item.get("pubDate") or item.get("pubDate_raw") # 네이버 API는 pubDate 제공, fallback 추가
        if pub_date_str:
            try:
                # to_date는 YYYY-MM-DD 문자열 반환, date 객체로 변환 필요
                article_date = datetime.strptime(to_date(pub_date_str), "%Y-%m-%d").date()
                if article_date >= three_days_ago:
                    filtered_items.append(item)
                else:
                    skipped_by_date += 1
            except Exception as e:
                # 날짜 파싱 실패 시 일단 포함 (혹은 로깅 후 제외 결정)
                # print(f"[WARN] Date parsing failed for item {prefer_link(item)}: {e}")
                filtered_items.append(item) # 파싱 실패 시 일단 포함하는 정책
        else:
            # 날짜 정보 없는 경우 포함 (혹은 제외 결정)
            # print(f"[WARN] No date found for item: {prefer_link(item)}")
            filtered_items.append(item) # 날짜 없으면 일단 포함

    print(f"[INFO] Filtered by date (last 3 days): {len(filtered_items)} items kept, {skipped_by_date} items skipped.")
    # --- ▲▲▲ 날짜 필터링 로직 끝 ▲▲▲ ---


    # --- ▼▼▼ OG 메타데이터 수집 병렬 처리 (filtered_items 사용) ▼▼▼ ---
    meta_list = []
    # max_workers: 동시에 실행할 작업의 최대 개수 (네트워크 환경에 따라 조절 가능)
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 각 뉴스 아이템의 URL로 OG 메타데이터 수집 작업을 예약합니다.
        # clean_items 대신 filtered_items 사용
        future_to_item = {executor.submit(expand_with_og, prefer_link(it)): it for it in filtered_items}

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
    # Save the original deduplicated list before date filtering if needed
    # raw_path = f"data/news_clean_dedup_{ts_str}.json"
    # with open(raw_path, "w", encoding="utf-8") as f:
    #     json.dump(clean_items, f, ensure_ascii=False, indent=2)

    # Save the date-filtered items that were processed for OG meta
    filtered_list_path = f"data/news_filtered_list_{ts_str}.json"
    with open(filtered_list_path, "w", encoding="utf-8") as f:
        json.dump(filtered_items, f, ensure_ascii=False, indent=2)

    # Save the final meta list (derived from filtered_items)
    meta_path = f"data/news_meta_{ts_str}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 저장 완료: {filtered_list_path}, {meta_path} | 최종 메타 기사 수: {len(meta_list)} | 경과(초): {round(time.time()-t0,2)}")

if __name__ == "__main__":
    main()
