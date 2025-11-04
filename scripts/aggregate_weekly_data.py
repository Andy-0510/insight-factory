# 파일 경로: scripts/aggregate_weekly_data.py

import os
import json
import glob
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
from pathlib import Path
from src.utils import load_json

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    import numpy as np
    # --- ▼▼▼ [신규 추가] 코사인 유사도 임포트 ▼▼▼ ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
except ImportError:
    SentenceTransformer = None
    KMeans = None
    np = None
    # --- ▼▼▼ [신규 추가] 코사인 유사도 임포트 (Fallback) ▼▼▼ ---
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_AVAILABLE = False
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


ROOT_OUTPUT_DIR = "outputs"
DAILY_ARCHIVE_DIR = os.path.join(ROOT_OUTPUT_DIR, "daily")
# --- ▼▼▼ [수정] EXPORT_DIR 전역 정의 ▼▼▼ ---
EXPORT_DIR = os.path.join(ROOT_OUTPUT_DIR, "export") # 전역 변수로 정의
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
DAYS_TO_AGGREGATE = 7


# --- ▼▼▼ [수정] 키워드 클러스터링 함수 (aggregate_data 함수 전에 위치해야 함) ▼▼▼ ---
def generate_keyword_clusters(final_keywords: list[dict[str, any]], model_name: str, n_clusters: int = 8, out_path: str = os.path.join(EXPORT_DIR, "keyword_clusters.csv")):
    """Extracts keyword embeddings, performs clustering, and saves the result."""
    # ... (함수 내용은 이전과 동일) ...
    if not final_keywords or SentenceTransformer is None or KMeans is None or pd is None or np is None: #
        print("[WARN] Skipping keyword clustering due to missing dependencies or data.") #
        # Create empty file if it doesn't exist #
        if not os.path.exists(out_path): #
             Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True) #
             pd.DataFrame(columns=['cluster_id', 'keywords']).to_csv(out_path, index=False, encoding='utf-8-sig') #
        return #

    print(f"[INFO] Starting weekly keyword clustering (n_clusters={n_clusters})...") #
    keywords = [item['keyword'] for item in final_keywords if item.get('keyword')] # 키워드 유효성 체크 #

    if not keywords: #
        print("[WARN] No valid keywords found for clustering.") #
        # Create empty file #
        if not os.path.exists(out_path): #
             Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True) #
             pd.DataFrame(columns=['cluster_id', 'keywords']).to_csv(out_path, index=False, encoding='utf-8-sig') #
        return #

    try: #
        # 1. Get Embeddings #
        # Consider making model_name configurable via config.json #
        embedder = SentenceTransformer(model_name) #
        embeddings = embedder.encode(keywords, show_progress_bar=False) #

        # 2. Perform Clustering (KMeans example) #
        actual_n_clusters = min(n_clusters, len(keywords)) #
        if actual_n_clusters < 2: #
             print("[WARN] Not enough keywords to cluster meaningfully.") #
             # Create empty file #
             Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True) #
             pd.DataFrame(columns=['cluster_id', 'keywords']).to_csv(out_path, index=False, encoding='utf-8-sig') #
             return #

        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10) #
        cluster_labels = kmeans.fit_predict(embeddings) #

        # 3. Group keywords by cluster and save #
        clusters = defaultdict(list) #
        for keyword, label in zip(keywords, cluster_labels): #
            clusters[label].append(keyword) #

        cluster_rows = [] #
        for cluster_id, words in clusters.items(): #
            cluster_rows.append({ #
                "cluster_id": cluster_id, #
                "keywords": ", ".join(words) #
            }) #

        df_clusters = pd.DataFrame(cluster_rows).sort_values("cluster_id") #
        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True) #
        df_clusters.to_csv(out_path, index=False, encoding='utf-8-sig') #
        print(f"[INFO] Weekly keyword clusters saved to {out_path} ({len(df_clusters)} clusters found)") #

    except Exception as e: #
        print(f"[ERROR] Failed to generate weekly keyword clusters: {e}") #
        # Create empty file on error #
        if not os.path.exists(out_path): #
             Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True) #
             pd.DataFrame(columns=['cluster_id', 'keywords']).to_csv(out_path, index=False, encoding='utf-8-sig') #
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- ▼▼▼ [신규 추가] 이벤트 중복제거 헬퍼 함수 ▼▼▼ ---
def deduplicate_events_by_content(df_events: pd.DataFrame, meta_articles: list, threshold=0.2) -> pd.DataFrame:
    """
    이벤트 목록(df_events)을 메타 기사(meta_articles)의 본문 내용과 병합한 후,
    코사인 유사도(threshold=0.2)를 기준으로 중복을 제거합니다.
    """
    if not SKLEARN_AVAILABLE or df_events.empty or not meta_articles:
        print("[INFO] Skipping event content deduplication (sklearn unavailable or data empty).")
        return df_events

    # 1. URL-본문 맵(dict) 생성 (빠른 조회를 위해)
    url_to_body = {}
    for item in meta_articles:
        if isinstance(item, dict) and item.get("url"):
            body = item.get("body") or item.get("description") or item.get("title", "")
            url_to_body[item["url"]] = body

    # 2. 이벤트 DataFrame에 'body' 컬럼 추가
    df_events['body'] = df_events['url'].map(url_to_body).fillna("")

    # 3. 본문이 있는 기사 / 없는 기사 분리
    df_with_body = df_events[df_events['body'] != ""].copy()
    df_no_body = df_events[df_events['body'] == ""]

    if df_with_body.empty:
        print("[INFO] No event bodies found for content deduplication.")
        return df_events # 원본 반환

    print(f"[INFO] Running content deduplication on {len(df_with_body)} events...")

    try:
        # 4. URL/Title 기준 1차 중복 제거 (다른 날짜에 동일 기사 집계 방지)
        df_with_body = df_with_body.drop_duplicates(subset=['url'])
        df_with_body = df_with_body.drop_duplicates(subset=['title'])
        df_with_body = df_with_body.reset_index(drop=True)

        if len(df_with_body) < 2:
            return df_events.drop(columns=['body'], errors='ignore')

        # 5. TF-IDF 벡터화 (n-gram 사용)
        vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb', ngram_range=(4, 7))
        X = vectorizer.fit_transform(df_with_body['body'])

        # 6. 코사인 유사도 계산
        sim_matrix = cosine_similarity(X)

        keep_indices = []
        removed_indices = set()

        for i in range(len(df_with_body)):
            if i in removed_indices:
                continue
            keep_indices.append(i) # (i)번째 기사는 보관
            for j in range(i + 1, len(df_with_body)):
                if j in removed_indices:
                    continue
                if sim_matrix[i, j] >= threshold:
                    removed_indices.add(j) # (i)와 유사하므로 제거

        df_deduped_body = df_with_body.iloc[keep_indices]
        print(f"[INFO] Content deduplication complete: {len(df_with_body)} -> {len(df_deduped_body)} events.")

        # 7. 본문 없는 기사와 다시 합치고, 'body' 컬럼 제거
        final_df = pd.concat([df_deduped_body, df_no_body], ignore_index=True)
        return final_df.drop(columns=['body'])

    except Exception as e:
        print(f"[WARN] Content deduplication failed: {e}. Returning original events.")
        return df_events.drop(columns=['body'], errors='ignore')
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

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
        # ... (기존 데이터 로드 및 집계 로직) ...
        current_date = start_date + timedelta(days=i) #
        date_str = current_date.strftime("%Y-%m-%d") #

        date_folders = sorted(glob.glob(os.path.join(DAILY_ARCHIVE_DIR, date_str, "*"))) #
        if not date_folders: #
            continue #

        latest_daily_folder = date_folders[-1] #

        # 각 데이터 파일 경로 정의 #
        kw_path = os.path.join(latest_daily_folder, "keywords.json") #
        trends_path = os.path.join(latest_daily_folder, "export", "trend_strength.csv") #
        events_path = os.path.join(latest_daily_folder, "export", "events.csv") #
        weak_signals_path = os.path.join(latest_daily_folder, "export", "weak_signals.csv") #
        meta_path = os.path.join(latest_daily_folder, "debug", "news_meta_latest.json") # debug 폴더의 메타 파일 #

        # 데이터 로드 및 집계 #
        if os.path.exists(kw_path): #
            try: # Add try-except for JSON loading #
                # --- ▼▼▼ [수정] load_json 사용 ▼▼▼ ---
                kw_data = load_json(kw_path, default={}) # load_json 함수 사용
                all_keywords.extend(kw_data.get("keywords", [])) #
                # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
            except Exception as e: #
                 print(f"[WARN] Error processing {kw_path}: {e}") #


        if os.path.exists(trends_path): #
            try: # Add try-except for CSV loading #
                df = pd.read_csv(trends_path); #
                df['date'] = date_str # Add date column #
                all_trends = pd.concat([all_trends, df], ignore_index=True) #
            except Exception as e: #
                 print(f"[WARN] Error processing {trends_path}: {e}") #


        if os.path.exists(events_path): #
             try: # Add try-except #
                 all_events = pd.concat([all_events, pd.read_csv(events_path)], ignore_index=True) #
             except Exception as e: #
                  print(f"[WARN] Error processing {events_path}: {e}") #

        if os.path.exists(weak_signals_path): #
             try: # Add try-except #
                 all_weak_signals = pd.concat([all_weak_signals, pd.read_csv(weak_signals_path)], ignore_index=True) #
             except Exception as e: #
                  print(f"[WARN] Error processing {weak_signals_path}: {e}") #


        if os.path.exists(meta_path): #
            try: # Add try-except #
                 # --- ▼▼▼ [수정] load_json 사용 ▼▼▼ ---
                 articles = load_json(meta_path, default=[]) # load_json 함수 사용
                 for article in articles: #
                    # Ensure article is a dict before using .get() #
                    if isinstance(article, dict): #
                        url = article.get("url") #
                        if url and url not in seen_urls: #
                            all_meta_articles.append(article) #
                            seen_urls.add(url) #
                 # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
            except Exception as e: #
                 print(f"[WARN] Error processing {meta_path}: {e}") #


    # --- 집계 후 최종 파일 저장 ---
    os.makedirs(os.path.join(ROOT_OUTPUT_DIR, "export"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_OUTPUT_DIR, "debug"), exist_ok=True)

    # 1. 주간 keywords.json 생성
    final_kws_list = [] # 클러스터링에 사용할 리스트
    if all_keywords:
        # ... (기존 keywords.json 생성 로직) ...
        weekly_scores = defaultdict(float) #
        for k in all_keywords: #
             # Ensure 'keyword' exists and score is valid #
             keyword = k.get('keyword') #
             score = k.get('score', 0.0) #
             if keyword and isinstance(score, (int, float)): #
                 weekly_scores[keyword] += score # #
        sorted_kws = sorted(weekly_scores.items(), key=lambda item: item[1], reverse=True) #
        final_kws_list = [{"keyword": k, "score": v} for k, v in sorted_kws] # 리스트 생성 #
        final_kws_json = { "keywords": final_kws_list } # JSON용 객체 #
        try: # Add try-except for JSON saving #
            with open(os.path.join(ROOT_OUTPUT_DIR, "keywords.json"), 'w', encoding='utf-8') as f: #
                json.dump(final_kws_json, f, ensure_ascii=False, indent=2) # #
            print("[INFO] Weekly keywords.json created.") #
        except Exception as e: #
            print(f"[ERROR] Failed to save weekly keywords.json: {e}") #


    # 2. 주간 events.csv, weak_signals.csv, trend_strength.csv 생성
    # ... (기존 CSV 저장 로직) ...
    if not all_events.empty: 
            try: 
                # --- ▼▼▼ [신규 추가] 저장 전, 본문 기반 중복 제거 ▼▼▼ ---
                print(f"[INFO] Aggregated {len(all_events)} events. Starting content deduplication...")
                # (all_meta_articles은 224줄에서 이미 집계 완료됨)
                all_events_deduped = deduplicate_events_by_content(all_events, all_meta_articles) 
                print(f"[INFO] Final events count after deduplication: {len(all_events_deduped)}")
                # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

                # [수정] 원본(all_events) 대신 중복 제거된(all_events_deduped) DataFrame 저장
                all_events_deduped.to_csv(os.path.join(EXPORT_DIR, "events.csv"), index=False, encoding='utf-8-sig')
                print("[INFO] Weekly events.csv created (deduplicated).") # 로그 메시지 수정
            except Exception as e: 
                print(f"[ERROR] Failed to save weekly events.csv: {e}")

    if not all_weak_signals.empty: #
        try: # Add try-except #
            all_weak_signals.to_csv(os.path.join(EXPORT_DIR, "weak_signals.csv"), index=False, encoding='utf-8-sig') # Use EXPORT_DIR #
            print("[INFO] Weekly weak_signals.csv created.") #
        except Exception as e: #
            print(f"[ERROR] Failed to save weekly weak_signals.csv: {e}") #

    if not all_trends.empty: #
        try: # Add try-except #
            all_trends.to_csv(os.path.join(EXPORT_DIR, "trend_strength.csv"), index=False, encoding='utf-8-sig') # Use EXPORT_DIR #
            print("[INFO] Weekly trend_strength.csv created.") #
        except Exception as e: #
             print(f"[ERROR] Failed to save weekly trend_strength.csv: {e}") #


    # 3. 주간 news_meta_agg.json 생성 (module_c 용)
    # ... (기존 JSON 저장 로직) ...
    try: # Add try-except #
        with open(os.path.join(ROOT_OUTPUT_DIR, "debug", "weekly_meta_agg.json"), 'w', encoding='utf-8') as f: #
            json.dump(all_meta_articles, f, ensure_ascii=False, indent=2) # #
        print("[INFO] Weekly weekly_meta_agg.json created.") #
    except Exception as e: #
        print(f"[ERROR] Failed to save weekly_meta_agg.json: {e}") #


    # --- ▼▼▼ [수정] 주간 키워드 클러스터링 실행 ▼▼▼ ---
    if SentenceTransformer and KMeans and pd and np:
        try:
            # config.json 로드하여 모델 이름 가져오기
            cfg = load_json("config.json", {}) # load_json 사용
            model_name_for_clusters = cfg.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") #
            # generate_keyword_clusters 함수 호출
            generate_keyword_clusters(final_kws_list, model_name=model_name_for_clusters) #
        except Exception as e:
             print(f"[ERROR] Error during weekly keyword clustering setup or execution: {e}") #
             # Ensure empty cluster file exists even if setup fails #
             cluster_out_path = os.path.join(EXPORT_DIR, "keyword_clusters.csv") #
             if not os.path.exists(cluster_out_path): #
                 Path(os.path.dirname(cluster_out_path)).mkdir(parents=True, exist_ok=True) #
                 pd.DataFrame(columns=['cluster_id', 'keywords']).to_csv(cluster_out_path, index=False, encoding='utf-8-sig') #

    else:
        print("[WARN] Skipping weekly keyword clustering step as dependencies are missing.") #
        # Create empty cluster file as placeholder #
        cluster_out_path = os.path.join(EXPORT_DIR, "keyword_clusters.csv") #
        if not os.path.exists(cluster_out_path): #
            Path(os.path.dirname(cluster_out_path)).mkdir(parents=True, exist_ok=True) #
            pd.DataFrame(columns=['cluster_id', 'keywords']).to_csv(cluster_out_path, index=False, encoding='utf-8-sig') #

    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


if __name__ == "__main__":
    # ... (기존 main 호출 로직) ...
    try: #
        aggregate_data() #
    except Exception as e: #
        print(f"[FATAL] An error occurred during weekly aggregation: {e}") #
        import traceback #
        traceback.print_exc() # Print detailed traceback #
        # Optionally exit with an error code #
        # sys.exit(1) #