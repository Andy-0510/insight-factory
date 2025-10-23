# 파일 경로: scripts/aggregate_weekly_data.py

import os
import json
import glob
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
from pathlib import Path
# --- ▼▼▼ [수정] load_json import 추가 ▼▼▼ ---
from src.utils import load_json # src.utils에서 load_json 함수 가져오기
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- ▼▼▼ [수정] 클러스터링 관련 라이브러리 import ▼▼▼ ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    import numpy as np # NumPy 추가
except ImportError:
    SentenceTransformer = None
    KMeans = None
    np = None
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

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
    if not all_events.empty: #
        try: # Add try-except for CSV saving #
             all_events.to_csv(os.path.join(EXPORT_DIR, "events.csv"), index=False, encoding='utf-8-sig') # Use EXPORT_DIR #
             print("[INFO] Weekly events.csv created.") #
        except Exception as e: #
            print(f"[ERROR] Failed to save weekly events.csv: {e}") #

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