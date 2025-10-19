# 파일 경로: scripts/aggregate_monthly_data.py

import os
import json
import glob
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
from src.utils import load_json

# --- 설정 ---
WAREHOUSE_META_DIR = "data/warehouse/meta"
DAILY_ARCHIVE_DIR = "outputs/daily"
OUTPUT_DIR = "outputs"
EXPORT_DIR = os.path.join(OUTPUT_DIR, "export")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
DAYS_TO_AGGREGATE = 30

def aggregate_monthly_data():
    """지난 30일간의 모든 주요 데이터를 집계하여 메인 outputs 폴더에 저장합니다."""
    print(f"[INFO] Aggregating all monthly data from the last {DAYS_TO_AGGREGATE} days...")

    # 집계할 데이터 초기화
    all_articles, all_keywords = [], []
    all_trends = pd.DataFrame()
    all_events = pd.DataFrame()
    all_weak_signals = pd.DataFrame()
    seen_urls = set()

    # 폴더 생성
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_AGGREGATE)

    # 1. 웨어하우스에서 30일치 news_meta 데이터 집계
    for i in range(DAYS_TO_AGGREGATE + 1):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        daily_folder = os.path.join(WAREHOUSE_META_DIR, date_str)
        if os.path.isdir(daily_folder):
            daily_meta_files = glob.glob(os.path.join(daily_folder, "news_meta_*.json"))
            if daily_meta_files:
                latest_file_for_day = sorted(daily_meta_files)[-1]
                try:
                    with open(latest_file_for_day, "r", encoding="utf-8") as fp:
                        articles = json.load(fp)
                        for article in articles:
                            url = article.get("url")
                            if url and url not in seen_urls:
                                all_articles.append(article)
                                seen_urls.add(url)
                except Exception as e:
                    print(f"[WARN] Failed to process {latest_file_for_day}: {e}")

    # 2. 일일 아카이브에서 30일치 분석 결과 데이터 집계
    for i in range(DAYS_TO_AGGREGATE + 1):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        date_folders = sorted(glob.glob(os.path.join(DAILY_ARCHIVE_DIR, date_str, "*")))
        if not date_folders: continue
        latest_daily_folder = date_folders[-1]

        # keywords.json 집계
        kw_path = os.path.join(latest_daily_folder, "keywords.json")
        if os.path.exists(kw_path):
            all_keywords.extend(load_json(kw_path, {"keywords": []}).get("keywords", []))

        # trend_strength.csv 집계
        trends_path = os.path.join(latest_daily_folder, "export", "trend_strength.csv")
        if os.path.exists(trends_path):
            df = pd.read_csv(trends_path)
            all_trends = pd.concat([all_trends, df], ignore_index=True)

        # events.csv 집계
        events_path = os.path.join(latest_daily_folder, "export", "events.csv")
        if os.path.exists(events_path):
            all_events = pd.concat([all_events, pd.read_csv(events_path)], ignore_index=True)

        # weak_signals.csv 집계
        weak_signals_path = os.path.join(latest_daily_folder, "export", "weak_signals.csv")
        if os.path.exists(weak_signals_path):
            all_weak_signals = pd.concat([all_weak_signals, pd.read_csv(weak_signals_path)], ignore_index=True)

    # --- 집계 후 최종 파일 저장 ---
    
    # monthly_meta_agg.json 저장
    with open(os.path.join(DEBUG_DIR, "monthly_meta_agg.json"), 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Monthly aggregated meta file created with {len(all_articles)} articles.")

    # 월간 keywords.json 생성
    if all_keywords:
        monthly_scores = defaultdict(float)
        for k in all_keywords: monthly_scores[k['keyword']] += k.get('score', 0.0)
        sorted_kws = sorted(monthly_scores.items(), key=lambda item: item[1], reverse=True)
        final_kws = { "keywords": [{"keyword": k, "score": v} for k, v in sorted_kws] }
        with open(os.path.join(OUTPUT_DIR, "keywords.json"), 'w', encoding='utf-8') as f:
            json.dump(final_kws, f, ensure_ascii=False, indent=2)
        print("[INFO] Monthly keywords.json created.")
        
    # 월간 trend_strength.csv 저장
    if not all_trends.empty:
        monthly_trends = all_trends.groupby('term').agg(
            total=('total', 'sum'), z_like=('z_like', 'mean'), cur=('cur', 'sum')
        ).reset_index().sort_values(by="total", ascending=False)
        monthly_trends.to_csv(os.path.join(EXPORT_DIR, "trend_strength.csv"), index=False, encoding='utf-8-sig')
        print(f"[INFO] Monthly trend_strength.csv created with {len(monthly_trends)} terms.")

    # 월간 events.csv, weak_signals.csv 저장
    if not all_events.empty:
        all_events.to_csv(os.path.join(EXPORT_DIR, "events.csv"), index=False, encoding='utf-8-sig')
        print(f"[INFO] Monthly events.csv created with {len(all_events)} records.")
    if not all_weak_signals.empty:
        all_weak_signals.to_csv(os.path.join(EXPORT_DIR, "weak_signals.csv"), index=False, encoding='utf-8-sig')
        print(f"[INFO] Monthly weak_signals.csv created with {len(all_weak_signals)} records.")
        
    print(f"[SUCCESS] Monthly data aggregation complete.")

if __name__ == "__main__":
    aggregate_monthly_data()