import os
import json
import glob
from datetime import datetime, timedelta

# --- 설정 ---
WAREHOUSE_META_DIR = "data/warehouse/meta"
OUTPUT_DIR = "outputs/debug"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "monthly_meta_agg.json")
DAYS_TO_AGGREGATE = 30

def aggregate_monthly_data():
    """
    지난 30일간 웨어하우스에 저장된 news_meta 파일들을 찾아 하나의 파일로 집계합니다.
    하루에 여러 파일이 있을 경우 가장 최신 파일을 선택합니다.
    """
    print(f"[INFO] Aggregating news meta data from warehouse for the last {DAYS_TO_AGGREGATE} days...")
    
    all_articles = []
    seen_urls = set()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_AGGREGATE)
    
    target_files_by_day = {}

    # 1. 지난 30일간의 날짜별 폴더 순회
    for i in range(DAYS_TO_AGGREGATE + 1):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        daily_folder = os.path.join(WAREHOUSE_META_DIR, date_str)
        
        if os.path.isdir(daily_folder):
            # 2. 해당 날짜 폴더의 모든 메타 파일 검색
            daily_meta_files = glob.glob(os.path.join(daily_folder, "news_meta_*.json"))
            if daily_meta_files:
                # 3. 파일명 기준(시간순)으로 정렬하여 가장 최신 파일 선택
                latest_file_for_day = sorted(daily_meta_files)[-1]
                target_files_by_day[date_str] = latest_file_for_day

    if not target_files_by_day:
        print(f"[WARN] No news_meta files found in warehouse for the last 30 days.")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    print(f"[INFO] Found meta files from {len(target_files_by_day)} days to aggregate.")

    # 4. 선택된 최신 파일들을 읽어 기사 목록 합치기
    for date_str in sorted(target_files_by_day.keys()):
        file_path = target_files_by_day[date_str]
        try:
            with open(file_path, "r", encoding="utf-8") as fp:
                daily_articles = json.load(fp)
                for article in daily_articles:
                    url = article.get("url")
                    if url and url not in seen_urls:
                        all_articles.append(article)
                        seen_urls.add(url)
        except Exception as e:
            print(f"[WARN] Failed to process {file_path}: {e}")
            
    # 5. 최종 집계 파일 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
        
    print(f"[SUCCESS] Aggregated {len(all_articles)} unique articles into {OUTPUT_FILE}")

if __name__ == "__main__":
    aggregate_monthly_data()
