# 파일 경로: scripts/select_top_articles.py
import pandas as pd
from src.utils import load_json, latest
import os
from datetime import datetime, timedelta
import glob
from src.config import load_config
from src.timeutil import now_kst, to_date
from collections import Counter # <--- 1. Counter 임포트

# --- ▼▼▼ sklearn 임포트 추가 ▼▼▼ ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] sklearn 라이브러리를 찾을 수 없습니다. 코사인 유사도 중복 제거를 건너뜁니다.")
# --- ▲▲▲ 임포트 완료 ▲▲▲ ---

TOP_N = 3
OUTPUT_CSV = "outputs/export/today_article_list.csv"
CUMULATIVE_SIGNAL_COUNTS_CSV = "outputs/export/daily_signal_counts.csv" # 이름 변경
# --- ▼▼▼ [신규] 비율 CSV 경로 추가 ▼▼▼ ---
CUMULATIVE_RATIOS_CSV = "outputs/export/daily_article_ratios.csv"
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
SCORE_THRESHOLD = 10.0

CFG = load_config()
DOMAIN_HINTS = set(hint.lower() for hint in CFG.get("domain_hints", []))
TODAY_DT = now_kst().date()
TODAY_STR = TODAY_DT.strftime("%Y-%m-%d")
TWO_DAYS_AGO_STR = (TODAY_DT - timedelta(days=2)).strftime("%Y-%m-%d")

# --- ▼▼▼ 코사인 유사도 중복 제거 함수 (신규 추가) ▼▼▼ ---
def deduplicate_by_content(articles: list, threshold=0.9) -> list:
    """
    기사 본문(body)의 코사인 유사도를 기반으로 중복 기사를 제거합니다.
    (articles 리스트는 'score' 기준 내림차순으로 정렬되어 있어야 합니다.)
    """
    if not SKLEARN_AVAILABLE or len(articles) < 2:
        return articles # 라이브러리가 없거나 기사가 1개 이하면 통과

    # 1. 기사 본문만 추출
    docs = [item.get("body", "") for item in articles]

    # 2. 모든 본문이 비어있으면 건너뛰기
    if not any(docs):
        print("[WARN] 기사 본문이 모두 비어있어, 내용 기반 중복 제거를 건너뜁니다.")
        return articles

    try:
        # 3. TF-IDF 벡터화
        vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb', ngram_range=(4, 7))
        X = vectorizer.fit_transform(docs)

        # 4. 코사인 유사도 계산
        sim_matrix = cosine_similarity(X)

        keep_indices = []
        removed_indices = set()

        for i in range(len(articles)):
            if i in removed_indices:
                continue

            # (i)번째 기사는 보관 (가장 점수가 높은 원본)
            keep_indices.append(i) 

            # (i)와 유사한 (j)번째 기사들을 제거
            for j in range(i + 1, len(articles)):
                if j in removed_indices:
                    continue
                if sim_matrix[i, j] >= threshold:
                    removed_indices.add(j) # (i)와 유사하므로 제거

        print(f"[INFO] 내용 기반 중복 제거 완료. (유효 {len(articles)}개 -> {len(keep_indices)}개)")
        return [articles[i] for i in keep_indices]

    except Exception as e:
        print(f"[WARN] 코사인 유사도 중복 제거 중 오류 발생: {e}. 중복 제거를 건너뜁니다.")
        return articles
# --- ▲▲▲ 함수 추가 완료 ▲▲▲ ---

def select_articles():
    """
    기사 점수를 계산하고, 총 기사 수/관심 기사 수를 집계하여 daily_signal_counts.csv에 저장한 뒤,
    비율을 계산하여 daily_article_ratios.csv에 저장합니다.
    """
    is_monthly_run = os.getenv("MONTHLY_RUN", "false").lower() == "true"
    if is_monthly_run:
        print("[INFO] Monthly Run: Skipping daily top article selection.")
        # ... (월간 실행 시 빈 파일 생성 로직, CUMULATIVE_RATIOS_CSV도 포함) ...
        if not os.path.exists(CUMULATIVE_SIGNAL_COUNTS_CSV):
             os.makedirs(os.path.dirname(CUMULATIVE_SIGNAL_COUNTS_CSV), exist_ok=True)
             pd.DataFrame(columns=["date", "signal_article_count", "meta_article_count"]).to_csv(
                 CUMULATIVE_SIGNAL_COUNTS_CSV, index=False, encoding="utf-8-sig"
             )
        if not os.path.exists(CUMULATIVE_RATIOS_CSV): # [신규] 비율 파일도 생성
             os.makedirs(os.path.dirname(CUMULATIVE_RATIOS_CSV), exist_ok=True)
             pd.DataFrame(columns=['date', 'signal_articles', 'meta_articles', 'signal_ratio']).to_csv(
                 CUMULATIVE_RATIOS_CSV, index=False, encoding="utf-8-sig"
             )
        return

    meta_path = latest("data/news_meta_*.json")
    if not meta_path or not os.path.exists(meta_path):
        print(f"[ERROR] Meta file not found: {meta_path}. Skipping article selection.")
        # ... (메타 파일 없을 시 빈 파일 생성 로직, CUMULATIVE_RATIOS_CSV도 포함) ...
        if not os.path.exists(CUMULATIVE_SIGNAL_COUNTS_CSV):
             os.makedirs(os.path.dirname(CUMULATIVE_SIGNAL_COUNTS_CSV), exist_ok=True)
             pd.DataFrame(columns=["date", "signal_article_count", "meta_article_count"]).to_csv(
                 CUMULATIVE_SIGNAL_COUNTS_CSV, index=False, encoding="utf-8-sig"
             )
        if not os.path.exists(CUMULATIVE_RATIOS_CSV): # [신규] 비율 파일도 생성
             os.makedirs(os.path.dirname(CUMULATIVE_RATIOS_CSV), exist_ok=True)
             pd.DataFrame(columns=['date', 'signal_articles', 'meta_articles', 'signal_ratio']).to_csv(
                 CUMULATIVE_RATIOS_CSV, index=False, encoding="utf-8-sig"
             )
        return

    meta_items = load_json(meta_path, [])
    meta_article_count = len(meta_items)
    print(f"[INFO] Loaded {meta_article_count} articles from {meta_path} for scoring.")

    try:
        df_strength = pd.read_csv("outputs/export/trend_strength.csv")
        top_keywords = set(df_strength.head(5)['term'])
    except FileNotFoundError:
        print("[WARN] trend_strength.csv not found. Using empty set for top keywords.")
        top_keywords = set()

    try:
        df_events = pd.read_csv("outputs/export/events.csv")
        event_titles = set(df_events['title']) if 'title' in df_events.columns else set()
    except FileNotFoundError:
        print("[WARN] events.csv not found. Using empty set for event titles.")
        event_titles = set()

    # --- ▼▼▼ 2. Counter 초기화 ▼▼▼ ---
    score_distribution = Counter()
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    
    # 기사 스코어링 (기존 로직과 동일)
    scored_articles = []
    signal_article_count = 0
    print(f"[INFO] Scoring articles based on Today={TODAY_STR}, D-2={TWO_DAYS_AGO_STR}")
    for item in meta_items:
        if not isinstance(item, dict):
             print(f"[WARN] Skipping non-dictionary item in meta file: {item}")
             continue
        
        score = 0
        title = item.get("title", "")
        body = item.get("body") or item.get("description", "")
        content = f"{title} {body}".lower() if title or body else ""
        
        if content:
            # 1. Domain Hints Buff (+7)
            for hint in DOMAIN_HINTS:
                if hint and hint in content:
                    score += 20
                    break
            # 2. Top Keyword Buff (+3)
            for keyword in top_keywords:
                if keyword and keyword.lower() in content:
                     score += 5
            # 3. Event Buff (+2)
            if title and title in event_titles:
                 score += 2

        # 4. Date Buff/Debuff (+5 / -5)
        pub_date_raw = item.get("published_time") or item.get("pubDate_raw") or ""
        article_date_str = to_date(pub_date_raw)
        if article_date_str == TODAY_STR:
            score += 5
        elif article_date_str == TWO_DAYS_AGO_STR:
            score -= 5

        score_distribution[score] += 1

        if score >= SCORE_THRESHOLD:
            signal_article_count += 1
            scored_articles.append({ "title": title, "url": item.get("url"), "score": score, "body": body }) # <-- "body": body 추가
            
    # --- ▼▼▼ 3. 분포 결과 출력 ▼▼▼ ---
    print("\n[DEBUG] --- Article Score Distribution (All Articles) ---")
    if not score_distribution:
        print("  No articles were scored.")
    else:
        # 점수(key)가 높은 순으로 정렬하여 출력
        print("  Score | Count")
        print("  -----------------")
        for score_val, count in sorted(score_distribution.items(), key=lambda item: item[0], reverse=True):
            print(f"  {score_val:5d} | {count:5d} articles")
    print("  --------------------------------------------------\n")

    # Top N 기사 저장 (기존 로직)
    if not scored_articles:
        df_top_articles = pd.DataFrame(columns=["title", "url"])
    else:
        # 1. 점수(score) 기준으로 내림차순 정렬 (필수!)
        scored_articles.sort(key=lambda x: x.get('score', 0), reverse=True)

        # 2. 내용 기반(코사인 유사도)으로 중복 제거
        deduped_articles = deduplicate_by_content(scored_articles, threshold=0.2)

        # 3. 상위 N개 선택
        top_articles = deduped_articles[:TOP_N]

        # 4. CSV 저장을 위해 DataFrame으로 변환
        df_top_articles = pd.DataFrame(top_articles)[['title', 'url']]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_top_articles.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Top {len(df_top_articles)} articles for report saved to {OUTPUT_CSV}")

    # --- ▼▼▼ 수정: 'daily_signal_counts.csv' 저장 로직 ▼▼▼ ---
    new_data = {"date": TODAY_STR, "signal_article_count": signal_article_count, "meta_article_count": meta_article_count}

    df_existing = pd.DataFrame()
    archive_paths = sorted(glob.glob("outputs/daily/*/*"))
    if archive_paths:
        latest_archive_path = archive_paths[-1]
        latest_cumulative_file = os.path.join(latest_archive_path, "export", "daily_signal_counts.csv")
        
        print(f"[INFO] Loading existing signal counts from latest archive: {latest_cumulative_file}")
        if os.path.exists(latest_cumulative_file):
            try:
                 df_existing = pd.read_csv(latest_cumulative_file)
                 if 'signal_article_count' not in df_existing.columns: df_existing['signal_article_count'] = 0
                 if 'meta_article_count' not in df_existing.columns: df_existing['meta_article_count'] = 0
                 print(f"  -> Found {len(df_existing)} existing records.")
            except Exception as e:
                 print(f"[WARN] Failed to load or process existing CSV {latest_cumulative_file}: {e}. Starting fresh.")
                 df_existing = pd.DataFrame()
        else:
            print("  -> No signal count file found in the latest archive. Starting fresh.")
    else:
        print("[INFO] No daily archives found. Starting fresh.")

    # 데이터 병합
    df_new = pd.DataFrame([new_data])
    if not df_existing.empty:
        df_existing = df_existing[df_existing["date"] != TODAY_STR]
        if 'meta_article_count' not in df_existing.columns: df_existing['meta_article_count'] = 0
        df_final_counts = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final_counts = df_new

    # 'daily_signal_counts.csv' 저장
    df_final_counts = df_final_counts[["date", "signal_article_count", "meta_article_count"]]
    df_final_counts.sort_values(by="date", inplace=True)
    os.makedirs(os.path.dirname(CUMULATIVE_SIGNAL_COUNTS_CSV), exist_ok=True)
    df_final_counts.to_csv(CUMULATIVE_SIGNAL_COUNTS_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Daily signal counts ({signal_article_count} / {meta_article_count}) saved to {CUMULATIVE_SIGNAL_COUNTS_CSV}")
    
    # --- ▼▼▼ [신규] 'daily_article_ratios.csv' 계산 및 저장 로직 ▼▼▼ ---
    try:
        df_ratios = df_final_counts.copy()
        # 컬럼명 변경 (시각화 파일과 일치)
        df_ratios.rename(columns={
            'signal_article_count': 'signal_articles',
            'meta_article_count': 'meta_articles'
        }, inplace=True)
        
        # 비율 계산 (분모: meta_articles)
        df_ratios['signal_ratio'] = (df_ratios['signal_articles'] / df_ratios['meta_articles']).where(df_ratios['meta_articles'] > 0, 0)
        
        # 날짜 정렬 (이미 되어있지만 확인)
        df_ratios['date'] = pd.to_datetime(df_ratios['date'])
        df_ratios.sort_values(by='date', inplace=True)
        df_ratios['date'] = df_ratios['date'].dt.strftime('%Y-%m-%d')
        
        # 필요한 컬럼만 저장
        cols_to_save = ['date', 'signal_articles', 'meta_articles', 'signal_ratio']
        df_ratios[cols_to_save].to_csv(CUMULATIVE_RATIOS_CSV, index=False, encoding="utf-8-sig", float_format='%.4f')
        print(f"[INFO] Daily article ratios (using meta count) calculated and saved to: {CUMULATIVE_RATIOS_CSV}")

    except Exception as e:
        print(f"[ERROR] Failed to calculate or save daily_article_ratios.csv: {e}")
        # 오류 발생 시 빈 파일 생성
        if not os.path.exists(CUMULATIVE_RATIOS_CSV):
             os.makedirs(os.path.dirname(CUMULATIVE_RATIOS_CSV), exist_ok=True)
             pd.DataFrame(columns=['date', 'signal_articles', 'meta_articles', 'signal_ratio']).to_csv(
                 CUMULATIVE_RATIOS_CSV, index=False, encoding="utf-8-sig"
             )
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

if __name__ == "__main__":
    select_articles()
