import pandas as pd
from src.utils import load_json, latest
import os
import re
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

def calculate_sentiments():
    """
    최신 기사 데이터와 토픽 데이터를 기반으로 토픽별 일일 감성 점수를 계산하고 누적 저장합니다.
    """
    analyzer = get_sentiment_analyzer()
    if not analyzer:
        return

    # 1. 데이터 로드
    meta_items = load_json(latest("data/news_meta_*.json"), [])
    topics_data = load_json("outputs/topics.json", {"topics": []})
    
    if not meta_items or not topics_data.get("topics"):
        print("[INFO] No data to process for sentiment calculation.")
        return

    today_date = pd.to_datetime("today").strftime("%Y-%m-%d")

    # 2. 토픽별로 기사 매핑 및 감성 분석
    topic_sentiments = {} # {topic_id: [score1, score2, ...]}
    for topic in topics_data["topics"]:
        topic_id = topic["topic_id"]
        topic_keywords = [w["word"] for w in topic.get("top_words", [])]
        topic_sentiments[topic_id] = []

        keyword_pattern = re.compile('|'.join(re.escape(kw) for kw in topic_keywords), re.IGNORECASE)

        for item in meta_items:
            content = item.get("body") or item.get("description", "")
            if content and keyword_pattern.search(content):
                try:
                    result = analyzer(content, truncation=True, max_length=512)[0]
                    # KcELECTRA 모델은 'LABEL_0'(부정), 'LABEL_1'(긍정)을 반환합니다.
                    # 'LABEL_1'일 경우 score, 'LABEL_0'일 경우 1 - score로 변환하여 긍정 점수로 통일합니다.
                    score = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']
                    topic_sentiments[topic_id].append(score)
                except Exception:
                    continue
    
    # 3. 토픽별 평균 감성 점수 계산
    results = []
    for topic_id, scores in topic_sentiments.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            results.append({
                "date": today_date,
                "topic_id": topic_id,
                "avg_sentiment": round(avg_score, 4),
                "article_count": len(scores)
            })

    # 4. CSV 파일에 누적 저장 (최신 90일 데이터만 유지)
    df_new = pd.DataFrame(results)
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        df_existing = df_existing[df_existing["date"] != today_date]
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
    
    # 90일 이전 데이터 삭제 로직
    df_final['date'] = pd.to_datetime(df_final['date'])
    ninety_days_ago = datetime.now() - timedelta(days=90)
    df_final = df_final[df_final['date'] >= ninety_days_ago]
    df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')
    
    df_final.sort_values(by=["date", "topic_id"], inplace=True)

    # CSV 저장 전 폴더가 존재하는지 확인하고 없으면 생성
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Daily topic sentiments calculated and saved to {OUTPUT_CSV}. ({len(df_new)} topics processed)")

if __name__ == "__main__":
    calculate_sentiments()
