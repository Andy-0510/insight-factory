import pandas as pd
import numpy as np
from src.utils import load_json, latest
import os
import json
import re
import glob
from src.config import load_config


# --- 설정 ---
OUTPUT_CSV_PATH = "outputs/export/risk_issues.csv"
MOVING_AVG_WINDOW = 7
STD_DEV_THRESHOLD = 2.0 # 2.0 표준편차 이상 하락 시 리스크로 간주

def get_negative_sentences(topic_keywords, articles):
    """특정 토픽과 관련된 기사에서 부정적인 문장을 추출합니다."""
    keyword_pattern = re.compile('|'.join(re.escape(kw) for kw in topic_keywords), re.IGNORECASE)
    
    # 간단한 부정 키워드 목록
    neg_words = ["논란", "우려", "리스크", "규제", "지연", "하락", "부진", "문제", "비판", "악화", "경고"]
    neg_pattern = re.compile('|'.join(neg_words))
    
    evidence_sentences = []
    for item in articles:
        content = item.get("body") or item.get("description", "")
        if content and keyword_pattern.search(content):
            sentences = re.split(r'(?<=[.!?다])\s+', content)
            for sent in sentences:
                if neg_pattern.search(sent) and len(evidence_sentences) < 3:
                    evidence_sentences.append(sent.strip())
    return evidence_sentences

def call_gemini_for_risk_analysis(topic_name, sentiment_drop, evidence):
    """LLM을 호출하여 리스크를 분석합니다."""
    try:
        # ... (api 키 설정 등)
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)
        if evidence:
            evidence_str = "- " + "\n- ".join(evidence)
        else:
            evidence_str = "N/A"

        prompt = f"""
        당신은 디스플레이 산업 전문 리스크 분석가입니다. 아래 데이터를 바탕으로 리스크를 분석하고 지정된 JSON 형식으로만 답변해주세요.

        ### 분석 대상 데이터:
        - **토픽:** {topic_name}
        - **감성 점수 하락폭:** {sentiment_drop:.2f} (0-1 스케일, 클수록 부정적)
        - **관련 부정 기사 내용(발췌):**
        {evidence_str}

        ### 분석 요청:
        1. **impact_range**: 이 리스크의 예상 영향 범위를 "단기/재무", "중기/PR", "장기/운영" 중에서 하나만 선택하세요. 
        2. **summary**: 이 리스크의 핵심 내용을 한글 2문장으로 요약하세요. 
        
        --- ▼▼▼ 수정된 부분 ▼▼▼ ---
        3. **mitigation**: 이 리스크에 대한 1차적인 핵심 완화 액션 1가지를 **개조식(명사형)**으로 제안하고, 괄호 안에 예상 담당 조직을 명시해주세요. (예: 프리미엄 제품 차별화 전략 수립 (제품기획팀))
        --- ▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        ### 출력 형식 (JSON):
        ```json
        {{
          "impact_range": "...",
          "summary": "...",
          "mitigation": "..."
        }}
        ```
        """
        response = model.generate_content(prompt)
        # ... (JSON 파싱 로직)
        match = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            return json.loads(response.text)

    except Exception as e:
        # ... (예외 처리)
        print(f"[ERROR] Gemini 리스크 분석 실패: {e}")
        return {
            "impact_range": "분석 실패",
            "summary": "LLM 호출에 실패했습니다.",
            "mitigation": "API 키 및 네트워크 상태를 확인하세요."
        }

def analyze_risks(articles):
    """토픽별 감성 점수 시계열을 분석하여 리스크를 탐지합니다."""
    print("[INFO] [module_g_risk] 리스크 분석 시작")

    sentiment_file_path = None
    archive_paths = sorted(glob.glob("outputs/daily/*/*"))
    if archive_paths:
        latest_archive_path = archive_paths[-1]
        candidate_path = os.path.join(latest_archive_path, "export", "daily_topic_sentiment.csv")
        if os.path.exists(candidate_path):
            sentiment_file_path = candidate_path

    if not sentiment_file_path:
        fallback_path = "outputs/export/daily_topic_sentiment.csv"
        if os.path.exists(fallback_path):
             sentiment_file_path = fallback_path
        else:
             print(f"[WARN] 리스크 분석에 필요한 'daily_topic_sentiment.csv' 파일을 찾을 수 없습니다. 분석을 건너뜁니다.")
             return

    print(f"[INFO] [module_g_risk] 감성 점수 데이터 로드: {sentiment_file_path}")
    df = pd.read_csv(sentiment_file_path)
    
    # [수정] topics.json 대신 master_topics.json 파일을 직접 로드합니다.
    master_topics = load_json("data/dictionaries/master_topics.json", {})

    risk_issues = []
    
    print("[INFO] [module_g_risk] 토픽별 감성 점수 하락 패턴 분석 중...")
    for semantic_key, group in df.groupby('semantic_key'):
        if semantic_key == "Uncategorized":
            continue
        
        if len(group) < MOVING_AVG_WINDOW:
            continue
            
        group = group.sort_values('date').set_index('date')
        group['ma'] = group['avg_sentiment'].rolling(window=MOVING_AVG_WINDOW, min_periods=MOVING_AVG_WINDOW).mean()
        group['std'] = group['avg_sentiment'].rolling(window=MOVING_AVG_WINDOW, min_periods=MOVING_AVG_WINDOW).std()
        
        today_data = group.iloc[-1]
        
        threshold_value = today_data['ma'] * 0.8
        if pd.notna(threshold_value) and today_data['avg_sentiment'] < threshold_value:
            sentiment_drop = today_data['ma'] - today_data['avg_sentiment']
            
            # --- ▼▼▼ [신규] 일간 토픽 키워드 역추적 로직 ▼▼▼ ---
            topic_name = semantic_key
            risk_date = today_data.name # today_data의 인덱스(날짜 문자열)
            risk_topic_id = today_data['topic_id']

            topic_keywords = []
            source = "master_topics (fallback)" # 키워드 출처 (디버깅용)

            try:
                date_dir = os.path.join("outputs/daily", risk_date)
                # 해당 날짜의 가장 최신 시간 폴더 찾기
                time_folders = [d for d in os.listdir(date_dir) if os.path.isdir(os.path.join(date_dir, d))]
                if not time_folders:
                    raise FileNotFoundError(f"No time folders found in {date_dir}")

                latest_time_folder = sorted(time_folders)[-1]
                topics_json_path = os.path.join(date_dir, latest_time_folder, "topics.json")

                if not os.path.exists(topics_json_path):
                    raise FileNotFoundError(f"topics.json not found at {topics_json_path}")

                topics_data = load_json(topics_json_path, {"topics": []})

                for topic in topics_data.get("topics", []):
                    if topic.get("topic_id") == risk_topic_id:
                        topic_keywords = [w.get("word") for w in topic.get("top_words", [])]
                        source = f"daily_topics.json (T_ID: {risk_topic_id})"
                        break

                if not topic_keywords:
                    print(f"[WARN] Topic ID {risk_topic_id} not found in {topics_json_path}. Falling back to master_topics.")

            except Exception as e:
                print(f"[WARN] Failed to load daily topics.json for {risk_date}: {e}. Falling back to master_topics.")

            # [Fallback] 일간 topics.json 로드 실패 시 master_topics에서 키워드 가져오기
            if not topic_keywords:
                topic_keywords = master_topics.get(semantic_key, [])

            if not topic_keywords:
                print(f"[WARN] No keywords found for {semantic_key}. Skipping risk analysis.")
                continue # 키워드가 전혀 없으면 리스크 분석 불가
            # --- ▲▲▲ 키워드 역추적 로직 완료 ▲▲▲ ---

            print(f"[INFO] 리스크 탐지: Topic '{topic_name}', 하락폭: {sentiment_drop:.2f}, Keywords Source: {source}")

            evidence = get_negative_sentences(topic_keywords, articles)
            llm_analysis = call_gemini_for_risk_analysis(topic_name, sentiment_drop, evidence)

            risk_issues.append({
                "Date": risk_date,
                "Topic": topic_name,
                "Keywords": ", ".join(topic_keywords[:5]), # [신규] 키워드 컬럼 추가 (상위 5개)
                "sentiment_drop": round(sentiment_drop, 3),
                **llm_analysis
            })

    if risk_issues:
        df_risks = pd.DataFrame(risk_issues)
        df_risks.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"[SUCCESS] {len(risk_issues)}개의 리스크를 탐지하여 {OUTPUT_CSV_PATH}에 저장했습니다.")
    else:
        print("[INFO] 금일 탐지된 신규 리스크가 없습니다.")
        if not os.path.exists(OUTPUT_CSV_PATH):
            # [수정] "Keywords" 컬럼 추가
            pd.DataFrame(columns=["Date", "Topic", "Keywords", "sentiment_drop", "impact_range", "summary", "mitigation"]).to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

# --- ▼▼▼▼▼ [수정] main 함수가 analyze_risks를 호출하도록 변경 ▼▼▼▼▼ ---
def main():
    is_monthly_run = os.getenv("MONTHLY_RUN", "false").lower() == "true"
    
    if is_monthly_run:
        meta_path = "outputs/debug/monthly_meta_agg.json"
        print(f"[INFO] Monthly Run: Using aggregated meta file for {__name__}.")
    else:
        meta_path = "outputs/debug/news_meta_latest.json"
        if not os.path.exists(meta_path):
            meta_path = latest("data/news_meta_*.json")

    if not meta_path or not os.path.exists(meta_path):
        raise SystemExit("Input meta file not found.")
        
    print(f"[INFO] Loading meta data from: {meta_path}")
    meta_items = load_json(meta_path, [])

    # 로드한 데이터를 analyze_risks 함수에 전달하여 실행
    analyze_risks(articles=meta_items)

# --- ▼▼▼▼▼ [수정] main 함수를 호출하도록 변경 ▼▼▼▼▼ ---
if __name__ == "__main__":
    main()