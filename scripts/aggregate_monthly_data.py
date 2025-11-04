# [scripts/aggregate_monthly_data.py]

import os
import json
import glob
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
from src.utils import load_json

# ... (파일 상단 정의는 동일) ...
DAILY_ARCHIVE_DIR = "outputs/daily"
OUTPUT_DIR = "outputs"
EXPORT_DIR = os.path.join(OUTPUT_DIR, "export")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
DAYS_TO_AGGREGATE = 30

def aggregate_monthly_data():
    """
    [수정] 지난 30일간의 '일일 아카이브(outputs/daily)'에서 모든 주요 데이터를 집계합니다.
    [개선] 단, daily_topic_sentiment.csv는 누적 파일이므로 최신 파일 1개만 사용합니다.
    """
    print(f"[INFO] Aggregating all monthly data from the last {DAYS_TO_AGGREGATE} days...")

    all_articles, all_keywords = [], []
    all_trends = pd.DataFrame()
    all_events = pd.DataFrame()
    all_weak_signals = pd.DataFrame()
    # all_sentiments = pd.DataFrame() # <-- 삭제 (더 이상 30일치 집계 안 함)
    seen_urls = set()

    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_AGGREGATE)

    # --- ▼▼▼ 1. [기존] news_meta 데이터 집계 (30일 순회) ▼▼▼ ---
    print("[INFO] Aggregating news_meta data from daily archives...")
    for i in range(DAYS_TO_AGGREGATE + 1):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        
        date_folders = sorted(glob.glob(os.path.join(DAILY_ARCHIVE_DIR, date_str, "*")))
        if not date_folders:
            continue
        
        latest_daily_folder = date_folders[-1]
        meta_path = os.path.join(latest_daily_folder, "debug", "news_meta_latest.json")
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fp:
                    articles = json.load(fp)
                    for article in articles:
                        url = article.get("url")
                        if url and url not in seen_urls:
                            all_articles.append(article)
                            seen_urls.add(url)
            except Exception as e:
                print(f"[WARN] Failed to process {meta_path}: {e}")
    # --- ▲▲▲ 1. [기존] news_meta 데이터 집계 완료 ▲▲▲ ---

    # --- ▼▼▼ 2. [기존] keywords, trends 등 집계 (30일 순회) ▼▼▼ ---
    print("[INFO] Aggregating keywords, trends, events, and signals from daily archives...")
    for i in range(DAYS_TO_AGGREGATE + 1):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        date_folders = sorted(glob.glob(os.path.join(DAILY_ARCHIVE_DIR, date_str, "*")))
        if not date_folders: continue
        latest_daily_folder = date_folders[-1]

        # keywords.json (기존)
        kw_path = os.path.join(latest_daily_folder, "keywords.json")
        if os.path.exists(kw_path):
            all_keywords.extend(load_json(kw_path, {"keywords": []}).get("keywords", []))

        # (trend_strength, events, weak_signals 집계 로직... 기존과 동일)
        trends_path = os.path.join(latest_daily_folder, "export", "trend_strength.csv")
        if os.path.exists(trends_path):
            df = pd.read_csv(trends_path)
            all_trends = pd.concat([all_trends, df], ignore_index=True)

        events_path = os.path.join(latest_daily_folder, "export", "events.csv")
        if os.path.exists(events_path):
            all_events = pd.concat([all_events, pd.read_csv(events_path)], ignore_index=True)

        weak_signals_path = os.path.join(latest_daily_folder, "export", "weak_signals.csv")
        if os.path.exists(weak_signals_path):
            all_weak_signals = pd.concat([all_weak_signals, pd.read_csv(weak_signals_path)], ignore_index=True)
    # --- ▲▲▲ 2. [기존] keywords, trends 등 집계 완료 ▲▲▲ ---

    # --- (중간 저장: meta, keywords, trends, events, weak_signals 저장 로직은 기존과 동일) ---
    # ... (monthly_meta_agg.json 저장) ...
    with open(os.path.join(DEBUG_DIR, "monthly_meta_agg.json"), 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Monthly aggregated meta file created with {len(all_articles)} articles.")
    # ... (월간 keywords.json 저장) ...
    if all_keywords:
        monthly_scores = defaultdict(float)
        for k in all_keywords: monthly_scores[k['keyword']] += k.get('score', 0.0)
        sorted_kws = sorted(monthly_scores.items(), key=lambda item: item[1], reverse=True)
        final_kws = { "keywords": [{"keyword": k, "score": v} for k, v in sorted_kws] }
        with open(os.path.join(OUTPUT_DIR, "keywords.json"), 'w', encoding='utf-8') as f:
            json.dump(final_kws, f, ensure_ascii=False, indent=2)
        print("[INFO] Monthly keywords.json created.")
    # ... (월간 trend_strength.csv 저장) ...
    if not all_trends.empty:
        monthly_trends = all_trends.groupby('term').agg(
            total=('total', 'sum'), z_like=('z_like', 'mean'), cur=('cur', 'sum')
        ).reset_index().sort_values(by="total", ascending=False)
        monthly_trends.to_csv(os.path.join(EXPORT_DIR, "trend_strength.csv"), index=False, encoding='utf-8-sig')
        print(f"[INFO] Monthly trend_strength.csv created with {len(monthly_trends)} terms.")
    # ... (월간 events.csv, weak_signals.csv 저장) ...
    if not all_events.empty:
        all_events.to_csv(os.path.join(EXPORT_DIR, "events.csv"), index=False, encoding='utf-8-sig')
        print(f"[INFO] Monthly events.csv created with {len(all_events)} records.")
    if not all_weak_signals.empty:
        all_weak_signals.to_csv(os.path.join(EXPORT_DIR, "weak_signals.csv"), index=False, encoding='utf-8-sig')
        print(f"[INFO] Monthly weak_signals.csv created with {len(all_weak_signals)} records.")
        
    # --- ▼▼▼ 3. [개선된] 월간 평균 감성 점수(positivity) 저장 ▼▼▼ ---
    monthly_sent_path = os.path.join(EXPORT_DIR, "monthly_sentiment_agg.csv")
    all_sentiments = pd.DataFrame() # 비어있는 DataFrame으로 시작
    
    # 가장 최신 아카이브 1개만 찾기
    all_daily_archives = sorted(glob.glob(os.path.join(DAILY_ARCHIVE_DIR, "*", "*")))
    if all_daily_archives:
        latest_archive_folder = all_daily_archives[-1]
        latest_sentiment_file = os.path.join(latest_archive_folder, "export", "daily_topic_sentiment.csv")
        
        if os.path.exists(latest_sentiment_file):
            print(f"[INFO] Loading latest cumulative sentiment file: {latest_sentiment_file}")
            try:
                # 이 파일 하나가 90일치 누적 데이터임
                all_sentiments = pd.read_csv(latest_sentiment_file)
            except Exception as e:
                print(f"[WARN] Failed to load {latest_sentiment_file}: {e}")
        else:
            print(f"[WARN] Latest sentiment file not found at: {latest_sentiment_file}")
    else:
        print("[WARN] No daily archives found to load sentiment data from.")

    if not all_sentiments.empty:
        # (선택) 90일치 전체가 아닌, 최근 30일치 평균만 낼 경우 필터링
        all_sentiments['date'] = pd.to_datetime(all_sentiments['date'])
        thirty_days_ago = datetime.now() - timedelta(days=30)
        all_sentiments = all_sentiments[all_sentiments['date'] >= thirty_days_ago]
        
        # 'semantic_key'별로 'avg_sentiment'의 월간 평균 계산
        monthly_sentiments = all_sentiments.groupby('semantic_key')['avg_sentiment'].mean().reset_index()
        monthly_sentiments.rename(columns={'avg_sentiment': 'monthly_avg_sentiment'}, inplace=True)
        
        # 파일 저장
        monthly_sentiments.to_csv(monthly_sent_path, index=False, encoding='utf-8-sig', float_format='%.4f')
        print(f"[INFO] Monthly sentiment aggregation (Positivity) saved to {monthly_sent_path} ({len(monthly_sentiments)} keys)")
    else:
        print("[WARN] No sentiment data found to aggregate.")
        # 3단계를 위해 빈 파일 생성
        pd.DataFrame(columns=['semantic_key', 'monthly_avg_sentiment']).to_csv(monthly_sent_path, index=False, encoding='utf-8-sig')
    # --- ▲▲▲ [개선된] 월간 평균 감성 점수 저장 완료 ▲▲▲ ---
        
    print(f"[SUCCESS] Monthly data aggregation complete.")

if __name__ == "__main__":
    aggregate_monthly_data()