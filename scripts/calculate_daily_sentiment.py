import pandas as pd
from src.utils import load_json, latest
import os
import re
import glob
from transformers import pipeline
from datetime import datetime, timedelta


# --- 모델 로드 ---
# Hugging Face Hub 모델 이름을 사용합니다.
HF_MODEL_NAME = "beomi/KcELECTRA-base-v2022"
OUTPUT_CSV = "outputs/export/daily_topic_sentiment.csv"

def get_sentiment_analyzer():
    """Hugging Face Hub에서 감성 분석 모델 파이프라인을 로드합니다."""
    # ✅ Hugging Face Hub에서 바로 모델 로드 (로컬/깃액션 동일)
    from transformers import pipeline
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model=HF_MODEL_NAME, tokenizer=HF_MODEL_NAME, device=-1) # CPU 사용
        print(f"[INFO] 감성 분석 모델을 Hugging Face Hub에서 로드했습니다: {HF_MODEL_NAME}")
        return sentiment_analyzer
    except Exception as e:
        print(f"[WARN] 감성 분석 모델 로드 실패: {e}. 감성 점수는 0으로 처리됩니다.")
        return None
# --- 💡 수정 완료 ---

# [수정] 마스터 토픽을 불러오고, 유사도를 계산하는 헬퍼 함수 추가
MASTER_TOPICS = load_json("data/dictionaries/master_topics.json", {})

def get_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def find_semantic_key(daily_topic_keywords, master_topics, threshold=0.1):
    best_match_key = "Uncategorized"
    max_similarity = 0
    daily_set = set(daily_topic_keywords)

    for key, master_keywords in master_topics.items():
        master_set = set(master_keywords)
        similarity = get_jaccard_similarity(daily_set, master_set)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_key = key
            
    return best_match_key if max_similarity >= threshold else "Uncategorized"

def calculate_sentiments():
    """
    최신 기사 데이터와 토픽 데이터를 기반으로 토픽별 일일 감성 점수를 계산하고 누적 저장합니다.
    """
    is_monthly_run = os.getenv("MONTHLY_RUN", "false").lower() == "true"
    if is_monthly_run:
        print("[INFO] Monthly Run: Skipping daily sentiment calculation.")
        return

    analyzer = get_sentiment_analyzer()
    if not analyzer:
        return

    meta_items = load_json(latest("data/news_meta_*.json"), [])
    topics_data = load_json("outputs/topics.json", {"topics": []})
    
    if not meta_items or not topics_data.get("topics"):
        print("[INFO] No data to process for sentiment calculation.")
        return

    today_date = pd.to_datetime("today").strftime("%Y-%m-%d")

    topic_sentiments = {}
    # [수정] semantic_key를 저장할 딕셔너리 추가
    topic_semantic_keys = {}

    for topic in topics_data["topics"]:
        topic_id = topic["topic_id"]
        topic_keywords = [w["word"] for w in topic.get("top_words", [])]

        # [수정] Semantic Key 매칭 로직 호출
        semantic_key = find_semantic_key(topic_keywords, MASTER_TOPICS)
        topic_semantic_keys[topic_id] = semantic_key
        
        topic_sentiments[topic_id] = []
        keyword_pattern = re.compile('|'.join(re.escape(kw) for kw in topic_keywords), re.IGNORECASE)
        for item in meta_items:
            content = item.get("body") or item.get("description", "")
            if content and keyword_pattern.search(content):
                try:
                    result = analyzer(content, truncation=True, max_length=512)[0]
                    score = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']
                    topic_sentiments[topic_id].append(score)
                except Exception:
                    continue
    
    results = []
    for topic_id, scores in topic_sentiments.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            # [수정] 결과에 semantic_key 추가
            results.append({
                "date": today_date, 
                "semantic_key": topic_semantic_keys.get(topic_id, "Uncategorized"),
                "topic_id": topic_id, 
                "avg_sentiment": round(avg_score, 4), 
                "article_count": len(scores)
            })

    df_new = pd.DataFrame(results)

    # 과거 데이터 로드 및 병합 로직 (기존과 동일하나, 새 컬럼이 자연스럽게 처리됨)
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
    else:
        df_final = df_new

    # 90일 이전 데이터 삭제 및 저장 (기존과 동일)
    df_final['date'] = pd.to_datetime(df_final['date'])
    ninety_days_ago = datetime.now() - timedelta(days=90)
    df_final = df_final[df_final['date'] >= ninety_days_ago]
    df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')
    df_final.sort_values(by=["date", "topic_id"], inplace=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Daily topic sentiments calculated and saved to {OUTPUT_CSV}. ({len(df_new)} topics processed)")

if __name__ == "__main__":
    calculate_sentiments()
