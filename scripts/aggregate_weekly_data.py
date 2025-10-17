# 파일 경로: scripts/aggregate_weekly_data.py

import os
import json
import glob
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

ROOT_OUTPUT_DIR = "outputs"
DAILY_ARCHIVE_DIR = os.path.join(ROOT_OUTPUT_DIR, "daily")
DAYS_TO_AGGREGATE = 7

def aggregate_data():
    """지난 7일간의 모든 주요 데이터를 집계하여 메인 outputs 폴더에 저장합니다."""
    print(f"[INFO] Aggregating all weekly data from the last {DAYS_TO_AGGREGATE} days...")

    # 집계할 데이터 초기화
    all_keywords = []
    all_trends = pd.DataFrame()
    all_events = pd.DataFrame()
    all_weak_signals = pd.DataFrame()
    all_meta_articles = []
    seen_urls = set()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_AGGREGATE - 1)

    for i in range(DAYS_TO_AGGREGATE):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        
        date_folders = sorted(glob.glob(os.path.join(DAILY_ARCHIVE_DIR, date_str, "*")))
        if not date_folders:
            continue
        
        latest_daily_folder = date_folders[-1]
        
        # 각 데이터 파일 경로 정의
        kw_path = os.path.join(latest_daily_folder, "keywords.json")
        trends_path = os.path.join(latest_daily_folder, "export", "trend_strength.csv")
        events_path = os.path.join(latest_daily_folder, "export", "events.csv")
        weak_signals_path = os.path.join(latest_daily_folder, "export", "weak_signals.csv")
        meta_path = os.path.join(latest_daily_folder, "debug", "news_meta_latest.json") # debug 폴더의 메타 파일 활용

        # 데이터 로드 및 집계
        if os.path.exists(kw_path):
            with open(kw_path, 'r', encoding='utf-8') as f:
                all_keywords.extend(json.load(f).get("keywords", []))

        if os.path.exists(trends_path):
            df = pd.read_csv(trends_path); df['date'] = date_str
            all_trends = pd.concat([all_trends, df], ignore_index=True)

        if os.path.exists(events_path):
            all_events = pd.concat([all_events, pd.read_csv(events_path)], ignore_index=True)

        if os.path.exists(weak_signals_path):
            all_weak_signals = pd.concat([all_weak_signals, pd.read_csv(weak_signals_path)], ignore_index=True)
            
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                for article in articles:
                    url = article.get("url")
                    if url and url not in seen_urls:
                        all_meta_articles.append(article)
                        seen_urls.add(url)
    
    # --- 집계 후 최종 파일 저장 ---
    os.makedirs(os.path.join(ROOT_OUTPUT_DIR, "export"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_OUTPUT_DIR, "debug"), exist_ok=True)

    # 1. 주간 keywords.json 생성
    if all_keywords:
        weekly_scores = defaultdict(float)
        for k in all_keywords: weekly_scores[k['keyword']] += k.get('score', 0.0)
        sorted_kws = sorted(weekly_scores.items(), key=lambda item: item[1], reverse=True)
        final_kws = { "keywords": [{"keyword": k, "score": v} for k, v in sorted_kws] }
        with open(os.path.join(ROOT_OUTPUT_DIR, "keywords.json"), 'w', encoding='utf-8') as f:
            json.dump(final_kws, f, ensure_ascii=False, indent=2)
        print("[INFO] Weekly keywords.json created.")

    # 2. 주간 events.csv, weak_signals.csv, trend_strength.csv 생성
    if not all_events.empty:
        all_events.to_csv(os.path.join(ROOT_OUTPUT_DIR, "export", "events.csv"), index=False, encoding='utf-8-sig')
        print("[INFO] Weekly events.csv created.")
    if not all_weak_signals.empty:
        all_weak_signals.to_csv(os.path.join(ROOT_OUTPUT_DIR, "export", "weak_signals.csv"), index=False, encoding='utf-8-sig')
        print("[INFO] Weekly weak_signals.csv created.")
    if not all_trends.empty:
        all_trends.to_csv(os.path.join(ROOT_OUTPUT_DIR, "export", "trend_strength.csv"), index=False, encoding='utf-8-sig')
        print("[INFO] Weekly trend_strength.csv created.")

    # 3. 주간 news_meta_agg.json 생성 (module_c 용)
    with open(os.path.join(ROOT_OUTPUT_DIR, "debug", "weekly_meta_agg.json"), 'w', encoding='utf-8') as f:
        json.dump(all_meta_articles, f, ensure_ascii=False, indent=2)
    print("[INFO] Weekly weekly_meta_agg.json created.")

if __name__ == "__main__":
    aggregate_data()