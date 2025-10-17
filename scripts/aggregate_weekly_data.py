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
    """지난 7일간의 주요 데이터에 날짜 정보를 포함하여 집계합니다."""
    print(f"[INFO] Aggregating data from the last {DAYS_TO_AGGREGATE} days with date info...")

    all_keywords = []
    all_trends = pd.DataFrame()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_AGGREGATE - 1)

    for i in range(DAYS_TO_AGGREGATE):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        
        date_folders = sorted(glob.glob(os.path.join(DAILY_ARCHIVE_DIR, date_str, "*")))
        if not date_folders:
            continue
        
        latest_daily_folder = date_folders[-1]
        
        # 1. keywords.json 데이터 로드
        kw_path = os.path.join(latest_daily_folder, "keywords.json")
        if os.path.exists(kw_path):
            with open(kw_path, 'r', encoding='utf-8') as f:
                kw_data = json.load(f)
                # 각 키워드에 날짜 정보 추가 (내부 집계용)
                for kw in kw_data.get("keywords", []):
                    kw['date'] = date_str
                    all_keywords.append(kw)

        # 2. trend_strength.csv 데이터 로드
        trends_path = os.path.join(latest_daily_folder, "export", "trend_strength.csv")
        if os.path.exists(trends_path):
            trends_df = pd.read_csv(trends_path)
            # --- ▼▼▼ [수정] 데이터프레임에 'date' 컬럼을 추가합니다 ▼▼▼ ---
            trends_df['date'] = date_str
            all_trends = pd.concat([all_trends, trends_df], ignore_index=True)

    # --- 집계 후 저장 ---

    # 1. 주간 keywords.json 생성 (기존과 동일하게 주간 점수 합산)
    if all_keywords:
        weekly_keyword_scores = defaultdict(float)
        for k in all_keywords:
            weekly_keyword_scores[k['keyword']] += k.get('score', 0.0)
        
        sorted_keywords = sorted(weekly_keyword_scores.items(), key=lambda item: item[1], reverse=True)
        
        final_keywords_obj = { "keywords": [{"keyword": k, "score": v} for k, v in sorted_keywords] }
        with open(os.path.join(ROOT_OUTPUT_DIR, "keywords.json"), 'w', encoding='utf-8') as f:
            json.dump(final_keywords_obj, f, ensure_ascii=False, indent=2)
        print("[INFO] Weekly keywords.json created.")

    # 2. 주간 trend_strength.csv 생성 (이제 'date' 컬럼 포함)
    if not all_trends.empty:
        all_trends.to_csv(os.path.join(ROOT_OUTPUT_DIR, "export", "trend_strength.csv"), index=False, encoding='utf-8-sig')
        print("[INFO] Weekly trend_strength.csv with date column created.")

if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT_OUTPUT_DIR, "export"), exist_ok=True)
    aggregate_data()