# 파일 경로: scripts/enrich_monthly_topics.py
# (신규 생성 파일)

import os
import json
import pandas as pd
from src.utils import load_json, save_json
from collections import defaultdict

# --- Jaccard 유사도 함수 (calculate_daily_sentiment.py에서 복사) ---
def get_jaccard_similarity(set1, set2):
    """두 세트 간의 Jaccard 유사도를 계산합니다."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def find_best_semantic_key(topic_word_set, master_topic_sets, threshold=0.1):
    """
    토픽의 단어 세트와 가장 일치하는 마스터 토픽(semantic_key)을 찾습니다.
    (calculate_daily_sentiment.py의 로직 [cite: 241-243] 기반)
    """
    best_match_key = "Uncategorized"
    max_similarity = 0

    for key, master_set in master_topic_sets.items():
        similarity = get_jaccard_similarity(topic_word_set, master_set)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_key = key
            
    return best_match_key if max_similarity >= threshold else "Uncategorized"

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

    # 2. 빠른 조회를 위한 룩업(Lookup) 테이블 생성
    # Activity (Y축) 룩업
    growth_lookup = growth_df.set_index('topic_id')['momentum_score'].to_dict()
    
    # Positivity (원 크기) 룩업
    sentiment_lookup = sentiment_df.set_index('semantic_key')['monthly_avg_sentiment'].to_dict()
    
    # Positivity 매핑을 위한 Jaccard 세트 준비
    master_topic_sets = {key: set(keywords) for key, keywords in master_topics.items()}

    # 3. 토픽 데이터 순회 및 3개 변수 병합
    enriched_topics = []
    for topic in topics_data.get("topics", []):
        topic_id = topic.get("topic_id")
        
        # 1. Interest (X축): 1단계에서 이미 저장됨 (기본값 0)
        topic['interest'] = topic.get('interest', 0)
        
        # 2. Activity (Y축): topic_growth.csv (momentum_score)에서 매핑
        topic['activity'] = float(growth_lookup.get(topic_id, 0.0))
        
        # 3. Positivity (원 크기): Jaccard 유사도를 통해 semantic_key를 찾고, 
        #    monthly_sentiment_agg.csv에서 값을 매핑
        topic_word_set = set([w.get("word") for w in topic.get("top_words", [])])
        best_semantic_key = find_best_semantic_key(topic_word_set, master_topic_sets)
        
        # 매핑된 키가 없으면 중립(0.5) 처리
        topic['positivity'] = float(sentiment_lookup.get(best_semantic_key, 0.5)) 
        
        enriched_topics.append(topic)

    # 4. 최종 topics.json 덮어쓰기
    final_output = {"topics": enriched_topics}
    save_json(TOPICS_FILE, final_output)
    
    print(f"[SUCCESS] [Step 3/3] Enrichment complete. {TOPICS_FILE} has been overwritten with Interest, Positivity, and Activity data.")

if __name__ == "__main__":
    enrich_topics()