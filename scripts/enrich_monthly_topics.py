# 파일 경로: scripts/enrich_monthly_topics.py
# (신규 생성 파일)

import os
import json
import pandas as pd
from src.utils import load_json, save_json
from collections import defaultdict
import numpy as np
from src.config import load_config
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_LIBS_AVAILABLE = True
except ImportError:
    EMBEDDING_LIBS_AVAILABLE = False
    print("[ERROR] 'sentence_transformers' 또는 'sklearn' 라이브러리가 없습니다. pip install sentence-transformers sklearn")
    
# [신규] 전역 변수로 모델 및 마스터 임베딩 캐싱
EMBEDDING_MODEL = None
MASTER_TOPIC_EMBEDDINGS = None
CFG = load_config()
EMBEDDING_MODEL_NAME = CFG.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_embedding_model():
    """임베딩 모델을 로드하고 전역 변수에 캐싱합니다."""
    global EMBEDDING_MODEL
    if not EMBEDDING_LIBS_AVAILABLE: return None
    if EMBEDDING_MODEL is None:
        try:
            EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"[INFO] enrich_monthly_topics: Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
        except Exception as e:
            print(f"[ERROR] enrich_monthly_topics: Embedding model load failed: {e}")
    return EMBEDDING_MODEL

def get_master_topic_embeddings(master_topics_map, model):
    """마스터 토픽 키워드들을 문서로 변환하고 임베딩을 미리 계산하여 캐싱합니다."""
    global MASTER_TOPIC_EMBEDDINGS
    if MASTER_TOPIC_EMBEDDINGS is None:
        if not model or not master_topics_map: return None
        try:
            # 마스터 토픽의 키워드 리스트를 하나의 '문서'로 결합
            master_topic_docs = [" ".join(keywords) for keywords in master_topics_map.values()]
            master_topic_keys = list(master_topics_map.keys())
            
            # 모든 마스터 토픽 문서를 한번에 임베딩
            embeddings = model.encode(master_topic_docs, show_progress_bar=False)
            
            MASTER_TOPIC_EMBEDDINGS = {"keys": master_topic_keys, "embeddings": embeddings}
            print(f"[INFO] enrich_monthly_topics: Master topic embeddings ({len(master_topic_keys)} keys) calculated.")
        except Exception as e:
            print(f"[ERROR] enrich_monthly_topics: Master topic embedding failed: {e}")
            return None
    return MASTER_TOPIC_EMBEDDINGS


def find_best_semantic_key_embedding(topic_word_set, model, master_embeddings_data):
    """
    [신규] 시맨틱 유사도(코사인)를 기반으로 가장 일치하는 마스터 토픽 키를 찾습니다.
    """
    if not model or not master_embeddings_data or not topic_word_set:
        return "Uncategorized" # 필수 요소 없으면 Uncategorized

    try:
        # 1. 월간 토픽의 키워드 셋을 하나의 '문서'로 결합
        topic_doc = [" ".join(topic_word_set)]
        
        # 2. 이 문서의 임베딩(벡터) 계산
        topic_embedding = model.encode(topic_doc, show_progress_bar=False)
        
        # 3. 미리 계산된 마스터 토픽 임베딩과 코사인 유사도 비교
        master_embeddings = master_embeddings_data["embeddings"]
        master_keys = master_embeddings_data["keys"]
        
        scores = cosine_similarity(topic_embedding, master_embeddings)[0]
        
        best_score_index = np.argmax(scores)
        best_score = scores[best_score_index]
        best_key = master_keys[best_score_index]

        # 4. [핵심] 유사도가 0% 초과일 때만 유효한 키로 인정
        if best_score > 0.0:
            return best_key
        else:
            return "Uncategorized" # 0% 이하면 Uncategorized
            
    except Exception as e:
        print(f"[WARN] enrich_monthly_topics: Similarity calculation failed: {e}")
        return "Uncategorized"
        

# --- 파일 경로 정의 ---
ROOT_OUTPUT_DIR = "outputs"
EXPORT_DIR = os.path.join(ROOT_OUTPUT_DIR, "export")
DICT_DIR = "data/dictionaries"

TOPICS_FILE = os.path.join(ROOT_OUTPUT_DIR, "topics.json")
GROWTH_FILE = os.path.join(EXPORT_DIR, "topic_growth.csv")
SENTIMENT_FILE = os.path.join(EXPORT_DIR, "monthly_sentiment_agg.csv")
MASTER_TOPICS_FILE = os.path.join(DICT_DIR, "master_topics.json")

def enrich_topics():
    """
    1. topics.json (interest)
    2. topic_growth.csv (activity)
    3. monthly_sentiment_agg.csv (positivity)
    
    위 3개 파일을 병합하여 최종 topics.json을 덮어씁니다.
    """
    print("[INFO] [Step 3/3] Enriching topics with Positivity (Sentiment) and Activity (Momentum)...")

    # 1. 입력 파일 로드
    topics_data = load_json(TOPICS_FILE, {"topics": []})
    master_topics = load_json(MASTER_TOPICS_FILE, {})

    try:
        growth_df = pd.read_csv(GROWTH_FILE)
    except FileNotFoundError:
        print(f"[WARN] {GROWTH_FILE} not found. Activity (Y-axis) will be 0.")
        growth_df = pd.DataFrame(columns=['topic_id', 'momentum_score'])

    try:
        sentiment_df = pd.read_csv(SENTIMENT_FILE)
    except FileNotFoundError:
        print(f"[WARN] {SENTIMENT_FILE} not found. Positivity (Bubble Size) will be 0.5.")
        sentiment_df = pd.DataFrame(columns=['semantic_key', 'monthly_avg_sentiment'])

    if not topics_data.get("topics"):
        print("[ERROR] Base topics.json is empty. Cannot proceed with enrichment.")
        return

    # --- ▼▼▼ [신규] 임베딩 모델 및 마스터 임베딩 로드 ▼▼▼ ---
    model = get_embedding_model()
    master_embeddings_data = get_master_topic_embeddings(master_topics, model)

    if not model or not master_embeddings_data:
        print("[ERROR] Embedding model failed to load. Aborting enrichment.")
        # 임베딩 로드 실패 시, Jaccard로 폴백 (선택적)
        # print("[WARN] Falling back to Jaccard similarity...")
        # master_topic_sets = {key: set(keywords) for key, keywords in master_topics.items()}
        # model = None # 모델을 None으로 설정
        return # 또는 그냥 중단
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    # 2. 빠른 조회를 위한 룩업(Lookup) 테이블 생성
    # Activity (Y축) 룩업
    growth_lookup = growth_df.set_index('topic_id')['momentum_score'].to_dict()

    # Positivity (원 크기) 룩업
    sentiment_lookup = sentiment_df.set_index('semantic_key')['monthly_avg_sentiment'].to_dict()

    # 3. 토픽 데이터 순회 및 3개 변수 병합
    enriched_topics = []
    for topic in topics_data.get("topics", []):
        topic_id = topic.get("topic_id")
        
        # 1. Interest (X축): 1단계에서 이미 저장됨 (기본값 0)
        topic['interest'] = topic.get('interest', 0)
        
        # 2. Activity (Y축): topic_growth.csv (momentum_score)에서 매핑
        topic['activity'] = float(growth_lookup.get(topic_id, 0.0))
        
        # 3. Positivity (원 크기): [수정] Embedding 유사도를 통해 semantic_key를 찾고,
        #    monthly_sentiment_agg.csv에서 값을 매핑
        topic_word_set = set([w.get("word") for w in topic.get("top_words", [])])

        # [수정] Jaccard 대신 임베딩 기반 함수 호출
        best_semantic_key = find_best_semantic_key_embedding(topic_word_set, model, master_embeddings_data)

        # --- ▼▼▼ [신규] 이 줄을 추가합니다 ▼▼▼ ---
        topic['semantic_key'] = best_semantic_key # semantic_key를 토픽 객체에 저장
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        # 매핑된 키(Uncategorized 포함)를 사용하여 점수 조회
        topic['positivity'] = float(sentiment_lookup.get(best_semantic_key, 0.5)) 

        enriched_topics.append(topic)

    # 4. 최종 topics.json 덮어쓰기
    final_output = {"topics": enriched_topics}
    save_json(TOPICS_FILE, final_output)
    
    print(f"[SUCCESS] [Step 3/3] Enrichment complete. {TOPICS_FILE} has been overwritten with Interest, Positivity, and Activity data.")

if __name__ == "__main__":
    enrich_topics()