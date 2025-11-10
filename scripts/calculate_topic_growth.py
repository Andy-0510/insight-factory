# 파일 경로: scripts/calculate_topic_growth.py

import os
import pandas as pd
from src.utils import load_json, save_json
from collections import defaultdict

ROOT_OUTPUT_DIR = "outputs"
EXPORT_DIR = os.path.join(ROOT_OUTPUT_DIR, "export")
OUTPUT_CSV = os.path.join(EXPORT_DIR, "topic_growth.csv")
TOP_N_KEYWORDS_PER_TOPIC = 20 # 각 토픽의 모멘텀 계산에 사용할 상위 키워드 수

def calculate_topic_momentum():
    """
    월간 토픽과 키워드 트렌드 데이터를 사용하여 토픽별 모멘텀 점수를 계산하고
    topic_growth.csv 파일로 저장합니다.
    """
    print("[INFO] Calculating monthly topic momentum...")

    # 1. 필요 데이터 로드
    topics_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "topics.json"), {"topics": []})
    trends_df = None
    try:
        trends_df = pd.read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv"))
    except FileNotFoundError:
        print("[WARN] trend_strength.csv not found. Cannot calculate topic momentum.")
        # 빈 파일 생성 후 종료
        os.makedirs(EXPORT_DIR, exist_ok=True)
        pd.DataFrame(columns=['topic_id', 'topic_name', 'momentum_score']).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        return
    except Exception as e:
        print(f"[ERROR] Failed to load trend_strength.csv: {e}")
        # 빈 파일 생성 후 종료
        os.makedirs(EXPORT_DIR, exist_ok=True)
        pd.DataFrame(columns=['topic_id', 'topic_name', 'momentum_score']).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        return


    topics = topics_data.get("topics", [])
    if not topics or trends_df is None or trends_df.empty:
        print("[WARN] No topics or trend data available to calculate momentum.")
        # 빈 파일 생성 후 종료
        os.makedirs(EXPORT_DIR, exist_ok=True)
        pd.DataFrame(columns=['topic_id', 'topic_name', 'momentum_score']).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        return

    # 키워드별 z_like 점수를 빠르게 조회하기 위한 딕셔너리 생성
    keyword_momentum = trends_df.set_index('term')['z_like'].to_dict()

    # 2. 토픽별 모멘텀 점수 계산
    topic_momentum_scores = []
    for topic in topics:
        topic_id = topic.get("topic_id")
        topic_name = topic.get("topic_name", f"Topic #{topic_id}") # 이름 없으면 ID 사용
        top_words = topic.get("top_words", [])

        # 상위 N개 키워드의 z_like 점수 합산 (가중 평균 대신 단순 합 사용)
        momentum_sum = 0.0
        keyword_count = 0
        for i, word_info in enumerate(top_words):
            if i >= TOP_N_KEYWORDS_PER_TOPIC: break # 상위 N개만 사용
            keyword = word_info.get("word")
            if keyword in keyword_momentum:
                momentum_sum += keyword_momentum[keyword]
                keyword_count += 1

        # 평균 모멘텀 계산 (키워드가 하나라도 있을 경우)
        avg_momentum = momentum_sum / keyword_count if keyword_count > 0 else 0.0

        topic_momentum_scores.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "momentum_score": round(avg_momentum, 3) # 소수점 3자리까지
        })

    # 3. DataFrame 생성 및 저장
    df_growth = pd.DataFrame(topic_momentum_scores)
    # 모멘텀 점수 기준으로 정렬 (상승 높은 순 -> 하락 낮은 순)
    df_growth = df_growth.sort_values(by="momentum_score", ascending=False)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    df_growth.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"[SUCCESS] Topic momentum calculated and saved to {OUTPUT_CSV} ({len(df_growth)} topics)")

if __name__ == "__main__":
    calculate_topic_momentum()