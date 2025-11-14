import pandas as pd
from src.utils import load_json, latest
import os
import re
import glob
from transformers import pipeline
from datetime import datetime, timedelta
import json
import numpy as np
from src.config import load_config # [신규] config 로더 임포트
import math # <--- 이 줄을 추가합니다.

# --- ▼▼▼ [신규] 1. 임베딩 라이브러리 임포트 ▼▼▼ ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_LIBS_AVAILABLE = True
except ImportError:
    print("[ERROR] 'sentence_transformers' 또는 'sklearn' 라이브러리가 없습니다. pip install sentence-transformers sklearn")
    EMBEDDING_LIBS_AVAILABLE = False
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


# --- 모델 로드 ---
HF_MODEL_NAME = "beomi/KcELECTRA-base-v2022"
OUTPUT_CSV = "outputs/export/daily_topic_sentiment.csv"

# [신규] config에서 임베딩 모델명 로드 (module_b와 동일 모델 사용)
CFG = load_config()
EMBEDDING_MODEL_NAME = CFG.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# [신규] 전역 변수로 모델 및 마스터 임베딩 캐싱
EMBEDDING_MODEL = None
MASTER_TOPIC_EMBEDDINGS = None


def get_embedding_model():
    """임베딩 모델을 로드하고 전역 변수에 캐싱합니다."""
    global EMBEDDING_MODEL
    if not EMBEDDING_LIBS_AVAILABLE:
        return None
    if EMBEDDING_MODEL is None:
        try:
            EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"[INFO] Semantic similarity model '{EMBEDDING_MODEL_NAME}' loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
    return EMBEDDING_MODEL


# [신규] 마스터 토픽의 임베딩을 미리 계산하는 함수
def get_master_topic_embeddings(master_topics_map, model):
    """마스터 토픽 키워드들을 문서로 변환하고 임베딩을 미리 계산하여 캐싱합니다."""
    global MASTER_TOPIC_EMBEDDINGS
    if MASTER_TOPIC_EMBEDDINGS is None:
        if not model or not master_topics_map:
            return None
        try:
            # 마스터 토픽의 키워드 리스트를 하나의 '문서'로 결합
            master_topic_docs = [" ".join(keywords) for keywords in master_topics_map.values()]
            master_topic_keys = list(master_topics_map.keys())
            
            # 모든 마스터 토픽 문서를 한번에 임베딩
            embeddings = model.encode(master_topic_docs, show_progress_bar=False)
            
            MASTER_TOPIC_EMBEDDINGS = {"keys": master_topic_keys, "embeddings": embeddings}
            print(f"[INFO] Master topic embeddings ({len(master_topic_keys)} keys) calculated.")
        except Exception as e:
            print(f"[ERROR] Failed to calculate master topic embeddings: {e}")
            return None
    return MASTER_TOPIC_EMBEDDINGS


def get_sentiment_analyzer():
    """Hugging Face Hub에서 감성 분석 모델 파이프라인을 로드합니다."""
    # (기존 코드와 동일)
    from transformers import pipeline
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model=HF_MODEL_NAME, tokenizer=HF_MODEL_NAME, device=-1)
        print(f"[INFO] 감성 분석 모델을 Hugging Face Hub에서 로드했습니다: {HF_MODEL_NAME}")
        return sentiment_analyzer
    except Exception as e:
        print(f"[WARN] 감성 분석 모델 로드 실패: {e}. 감성 점수는 0으로 처리됩니다.")
        return None

# [기존] 마스터 토픽 로드
MASTER_TOPICS = load_json("data/dictionaries/master_topics.json", {})


# --- ▼▼▼ 2단계: 'find_semantic_key' 함수를 'calculate_semantic_similarities'로 대체 ▼▼▼ ---

# [삭제] def get_jaccard_similarity(...) - 더 이상 사용하지 않음
# [삭제] def find_semantic_key(...) - 더 이상 사용하지 않음

def calculate_semantic_similarities(daily_topic_keywords, model, master_embeddings_data):
    """
    한 토픽의 키워드 셋과 (미리 계산된) 모든 마스터 토픽 임베딩 간의
    '코사인 유사도'를 계산하여 (Key, Score) 리스트를 반환합니다.
    """
    if not model or not master_embeddings_data or not daily_topic_keywords:
        return [] # 계산 불가
    
    try:
        # 1. 일간 토픽의 키워드를 하나의 '문서'로 결합
        daily_topic_doc = [" ".join(daily_topic_keywords)]
        
        # 2. 이 문서의 임베딩(벡터) 계산
        daily_topic_embedding = model.encode(daily_topic_doc, show_progress_bar=False)
        
        # 3. 미리 계산된 마스터 토픽 임베딩과 코사인 유사도 비교
        master_embeddings = master_embeddings_data["embeddings"]
        master_keys = master_embeddings_data["keys"]
        
        # (1, 768) 벡터와 (N, 768) 벡터 비교 -> (1, N) 결과 매트릭스
        scores = cosine_similarity(daily_topic_embedding, master_embeddings)[0] # 첫 번째 (유일한) 행 선택
        
        similarities = []
        for key, score in zip(master_keys, scores):
            if score > 0: # [핵심] 0% 초과 유사도만 후보로 인정
                similarities.append((key, float(score))) # numpy float를 python float로 변환
        
        # 유사도 높은 순으로 정렬하여 반환
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    except Exception as e:
        print(f"[WARN] Semantic similarity calculation failed: {e}")
        return []
# --- ▲▲▲ 2단계 수정 완료 ▲▲▲ ---


def calculate_sentiments():
    """
    [수정됨] 1:1 매핑(경매) 로직 및 점수 분포 확장 로직 적용
    """
    analyzer = get_sentiment_analyzer()
    if not analyzer:
        return

    # --- ▼▼▼ 3단계: 1:1 매핑(경매) 로직 및 점수 분포 확장 적용 ▼▼▼ ---

    # [신규] 임베딩 모델 로드 및 마스터 임베딩 미리 계산
    model = get_embedding_model()
    master_embeddings_data = get_master_topic_embeddings(MASTER_TOPICS, model)
    
    if not model or not master_embeddings_data:
        print("[ERROR] Embedding model or master embeddings not available. Aborting sentiment calculation.")
        return

    meta_items = load_json(latest("data/news_meta_*.json"), [])
    topics_data = load_json("outputs/topics.json", {"topics": []})
    
    if not meta_items or not topics_data.get("topics"):
        print("[INFO] No data to process for sentiment calculation.")
        return

    today_date = pd.to_datetime("today").strftime("%Y-%m-%d")

    # 1. 모든 Topic ID -> 모든 Semantic Key 간의 유사도 매트릭스 생성
    topic_id_to_data = {} # 감성 분석 루프를 위해 토픽 데이터(키워드) 보관
    candidate_pool = [] # (topic_id, key, score) 경매 후보 풀
    all_topic_ids = set() # 전체 topic_id 추적
    debug_all_scores = {} # [디버깅] 토픽별 전체 점수표

    print("[INFO] 1:N 매핑 시작: (SEMANTIC) 모든 토픽 vs 모든 키 유사도 계산 중...")
    for topic in topics_data["topics"]:
        topic_id = topic["topic_id"]
        all_topic_ids.add(topic_id)
        topic_keywords = [w["word"] for w in topic.get("top_words", [])]
        
        # [수정] 2단계에서 만든 '시맨틱 유사도' 함수 호출
        key_score_list = calculate_semantic_similarities(topic_keywords, model, master_embeddings_data) 
        
        topic_id_to_data[topic_id] = {"keywords": topic_keywords}
        debug_all_scores[topic_id] = key_score_list # [디버깅]
        
        if not key_score_list:
            # 0% 유사도 토픽 (Uncategorized 후보)
            pass
        else:
            # 0% 초과 유사도 토픽 (경매 후보)
            for key, score in key_score_list:
                candidate_pool.append((topic_id, key, score))

    # 2. 점수(score)가 높은 순으로 풀(pool) 정렬 (경매 준비)
    candidate_pool.sort(key=lambda x: x[2], reverse=True)

    # 3. 1:1 매핑 (선점 로직 - 경매 시작)
    assigned_topics = set()
    assigned_keys = set()
    topic_semantic_keys = {} # 최종 매핑 결과
    debug_matches = [] # [디버깅] 매칭 과정 로그

    print("[INFO] 1:1 매핑 중: 최적의 짝 찾기 (Best Match First)...")
    for topic_id, semantic_key, score in candidate_pool:
        # 이미 Topic_id가 배정되었거나, Semantic_key가 선점되었다면 스킵
        if topic_id in assigned_topics:
            # debug_matches.append(f"[SKIP] Topic {topic_id}는 이미 다른 키에 배정됨.") # 로그가 너무 많아짐
            continue
        if semantic_key in assigned_keys:
            debug_matches.append(f"[SKIP] Key '{semantic_key}'는 이미 선점됨. (Topic {topic_id}가 {score:.4f}점으로 입찰 시도)")
            continue
        
        # [성공] 둘 다 사용 가능하므로 매칭
        topic_semantic_keys[topic_id] = semantic_key
        assigned_topics.add(topic_id)
        assigned_keys.add(semantic_key)
        debug_matches.append(f"[WIN] Topic {topic_id} -> '{semantic_key}' (Score: {score:.4f})")

    # 4. 배정되지 못한 토픽들(0% 유사도 포함)을 'Uncategorized'로 처리
    for topic_id in all_topic_ids:
        if topic_id not in assigned_topics:
            topic_semantic_keys[topic_id] = "Uncategorized"
            debug_matches.append(f"[ASSIGN] Topic {topic_id} -> 'Uncategorized' (0% 유사도 또는 모든 입찰 실패)")
    
    num_assigned = len(assigned_keys)
    num_uncategorized = len(all_topic_ids) - num_assigned
    print(f"[INFO] 1:1 매핑 완료. (할당됨: {num_assigned}개, Uncategorized로 처리: {num_uncategorized}개)")
    
    # 5. 감성 점수 계산 (기존 로직 재사용 + 점수 분포 확장)
    topic_sentiments = {}
    debug_sentiments = [] # [디버깅] 감성 점수 로그
    
    for topic_id in all_topic_ids:
        semantic_key = topic_semantic_keys.get(topic_id, "Uncategorized")
        
        # [핵심] Uncategorized로 분류된 토픽은 감성 분석/리스크 분석에서 제외
        if semantic_key == "Uncategorized":
            continue 

        topic_keywords = topic_id_to_data[topic_id]["keywords"]
        
        topic_sentiments[topic_id] = []
        keyword_pattern = re.compile('|'.join(re.escape(kw) for kw in topic_keywords), re.IGNORECASE)
        for item in meta_items:
            content = item.get("body") or item.get("description", "")
            if content and keyword_pattern.search(content):
                try:
                    result = analyzer(content, truncation=True, max_length=512)[0]
                    
                    # --- ▼▼▼ 3-A. 점수 분포 확장 로직 (포함) ▼▼▼ ---
                    # 1. 기존 로직으로 0.0 ~ 1.0 사이의 기본 점수 계산
                    score_raw = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']

                    # 2. [수정] 스트레칭 로직 제거
                    # GAIN = 100.0 
                    # score_scaled = (math.tanh(GAIN * (score_raw - 0.5)) + 1) / 2.0

                    # 3. [수정] 원본 점수를 그대로 사용 (0.0 ~ 1.0 보정은 유지)
                    score = max(0.0, min(1.0, score_raw))
                    # --- ▲▲▲ 점수 분포 확장 로직 종료 ▲▲▲ ---
                    
                    topic_sentiments[topic_id].append(score)
                    
                    # [디버깅]
                    debug_sentiments.append(f"Topic {topic_id} ({semantic_key}): score_raw={score_raw:.4f} -> score_final={score:.4f}")
                    
                except Exception:
                    continue
    
    # 6. 최종 CSV 생성 (Uncategorized 제외)
    results = []
    for topic_id, scores in topic_sentiments.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            semantic_key = topic_semantic_keys[topic_id] # 1:1 매핑된 키
            
            results.append({
                "date": today_date, 
                "semantic_key": semantic_key,
                "topic_id": topic_id, 
                "avg_sentiment": round(avg_score, 4), 
                "article_count": len(scores)
            })

    # [수정] Uncategorized는 이미 topic_sentiments 계산에서 제외되었으므로,
    # final_results 필터링은 필요 없음.
    df_new = pd.DataFrame(results)

    # 과거 데이터 로드 및 병합 로직 (기존과 동일)
    df_existing = pd.DataFrame()
    archive_paths = sorted(glob.glob("outputs/daily/*/*"))
    if archive_paths:
        latest_archive_path = archive_paths[-1]
        latest_sentiment_file = os.path.join(latest_archive_path, "export", "daily_topic_sentiment.csv")
        if os.path.exists(latest_sentiment_file):
            df_existing = pd.read_csv(latest_sentiment_file)

    if not df_existing.empty:
        df_existing = df_existing[df_existing["date"] != today_date]
    df_final = pd.concat([df_existing, df_new], ignore_index=True)

    # 90일 이전 데이터 삭제 및 저장 (기존과 동일)
    df_final['date'] = pd.to_datetime(df_final['date'])
    ninety_days_ago = datetime.now() - timedelta(days=90)
    df_final = df_final[df_final['date'] >= ninety_days_ago]
    df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')
    df_final.sort_values(by=["date", "topic_id"], inplace=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Daily topic sentiments calculated and saved to {OUTPUT_CSV}. ({len(df_new)} topics processed)")

    # --- ▼▼▼ 4단계: 디버그 로그 저장 (상세화) ▼▼▼ ---
    os.makedirs("outputs/debug", exist_ok=True)
    debug_log_path = "outputs/debug/daily_topic_matching_log.json"
    try:
        debug_output = {
            "metadata": {
                "date": today_date,
                "assigned_count": num_assigned,
                "uncategorized_count": num_uncategorized,
                "embedding_model": EMBEDDING_MODEL_NAME
            },
            "matching_process_log": debug_matches,
            "topic_similarity_scores": debug_all_scores, # 토픽별 전체 점수표
            "final_mapping": topic_semantic_keys # (Topic_ID -> Semantic_Key) 최종본
        }
        with open(debug_log_path, "w", encoding="utf-8") as f:
            json.dump(debug_output, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 1:1 매칭 디버그 로그 저장: {debug_log_path}")
    except Exception as e:
        print(f"[WARN] 디버그 로그(매칭) 저장 실패: {e}")

    sentiment_log_path = "outputs/debug/daily_sentiment_scaling_log.txt"
    try:
        with open(sentiment_log_path, "w", encoding="utf-8") as f:
            f.write(f"--- Sentiment Scaling Log ({today_date}) ---\n")
            f.write("--- (Topic_ID (Semantic_Key): raw_score -> final_score) ---\n")
            f.write("--------------------------------------------------\n")
            f.write("\n".join(debug_sentiments))
        print(f"[INFO] 감성 점수 디버그 로그 저장: {sentiment_log_path}")
    except Exception as e:
        print(f"[WARN] 디버그 로그(감성) 저장 실패: {e}")
    # --- ▲▲▲ 4단계 수정 완료 ▲▲▲ ---

if __name__ == "__main__":
    # 임베딩 모델 로드 시 예외 처리를 위해 라이브러리 가용성 먼저 체크
    if EMBEDDING_LIBS_AVAILABLE:
        calculate_sentiments()
    else:
        print("[ERROR] 'sentence_transformers' 또는 'sklearn'이 설치되지 않아 calculate_sentiments를 실행할 수 없습니다.")
