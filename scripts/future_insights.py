import os
import json
import pandas as pd
import glob
import re
from collections import defaultdict
from src.utils import load_json, save_json, latest


def latest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getctime) if files else None


def call_gemini_for_maturity(tech_name, data):
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 환경 변수가 없습니다.")
        genai.configure(api_key=api_key)

        llm_config = load_json("config.json", {}).get("llm", {})
        model_name = llm_config.get("model", "gemini-1.5-flash-latest")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
너는 기술 시장 분석 전문가야. 아래 데이터를 바탕으로 '{tech_name}' 기술의 현재 시장 성숙도 단계를 'Emerging', 'Growth', 'Maturity' 중 하나로 추론하고, 그 이유를 한글 한 문장으로 설명해줘.

### 데이터:
- 뉴스 언급 빈도 (최근 30일): {data['frequency']}회
- 시장 감성 점수 (0~1, 높을수록 긍정적): {data['sentiment']:.2f}
- 주요 이벤트 발생 비율:
  - 투자(INVEST): {data['events'].get('INVEST', 0)}회
  - 출시(LAUNCH): {data['events'].get('LAUNCH', 0)}회
  - 수주(ORDER): {data['events'].get('ORDER', 0)}회

### 출력 형식 (반드시 아래 JSON 형식만 출력):
```json
{{
  "stage": "추론한 단계",
  "reason": "판단 이유 요약"
}}
"""
        response = model.generate_content(prompt)
        json_text_match = re.search(r'json\s*(\{.*?\})\s*', response.text, re.DOTALL)
        if json_text_match:
            return json.loads(json_text_match.group(1))
        else:
            return json.loads(response.text)
    except Exception as e:
        print(f"[ERROR] '{tech_name}' 기술 성숙도 분석 중 LLM 호출 오류: {e}")
        return {"stage": "Error", "reason": str(e)}


def call_gemini_for_weak_signal(signal, context_sentences):
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        llm_config = load_json("config.json", {}).get("llm", {})
        model_name = llm_config.get("model", "gemini-1.5-flash-latest")
        model = genai.GenerativeModel(model_name)

        context_str = "\n".join([f"- {s}" for s in context_sentences])
        prompt = f"""
너는 미래 기술 트렌드 분석가야. '{signal}'이라는 새로운 약한 신호가 포착되었어. 아래 참고 문장들을 바탕으로, 이 용어가 무엇이며 어떤 맥락에서 등장했고, 미래에 어떤 잠재력이 있는지 한글 한 문장으로 명확하게 요약해줘.

### 약한 신호:
{signal}

### 참고 문장:
{context_str}

### 출력 형식 (반드시 아래 JSON 형식만 출력):
```json
{{
  "signal": "{signal}",
  "interpretation": "한 문장 요약"
}}
"""
        response = model.generate_content(prompt)
        json_text_match = re.search(r'json\s*(\{.*?\})\s*', response.text, re.DOTALL)
        if json_text_match:
            return json.loads(json_text_match.group(1))
        else:
            return json.loads(response.text)
    except Exception as e:
        print(f"[ERROR] '{signal}' 약한 신호 분석 중 LLM 호출 오류: {e}")
        return {"signal": signal, "interpretation": f"Error: {e}"}


# --- 1. 기술 성숙도 분석 ---
def analyze_tech_maturity():
    """ 기술 성숙도 분석 (Hugging Face Hub 모델 사용 버전) """
    print("\n--- 1. 기술 성숙도 분석 시작 ---")

    config = load_json("config.json")
    keywords_data = load_json("outputs/keywords.json")

    top_keywords = {item['keyword'] for item in keywords_data.get('keywords', [])[:20]}
    tech_filter = set(config.get('domain_hints', []))
    target_techs = list(top_keywords.intersection(tech_filter))

    print(f"[INFO] 분석 대상 기술 ({len(target_techs)}개): {target_techs}")
    if not target_techs:
        save_json('outputs/tech_maturity.json', {"results": []})
        return

    try:
        trends_df = pd.read_csv("outputs/export/trend_strength.csv")
        events_df = pd.read_csv("outputs/export/events.csv")
    except FileNotFoundError as e:
        print(f"[ERROR] 분석에 필요한 CSV 파일 없음: {e}")
        return

    meta_data = load_json(latest("data/news_meta_*.json"), [])

    # ✅ Hugging Face Hub에서 바로 모델 로드 (로컬/깃액션 동일)
    from transformers import pipeline
    try:
        hf_model = "beomi/KcELECTRA-base-v2022"  # 공개 모델
        sentiment_analyzer = pipeline("sentiment-analysis", model=hf_model, tokenizer=hf_model)
        print(f"[INFO] 감성 분석 모델을 Hugging Face Hub에서 로드했습니다: {hf_model}")
    except Exception as e:
        print(f"[WARN] 감성 분석 모델 로드 실패: {e}. 감성 점수는 0으로 처리됩니다.")
        sentiment_analyzer = None

    maturity_results = []
    for tech in target_techs:
        print(f"\n[분석 중] 기술: {tech}")

        freq_series = trends_df[trends_df['term'] == tech]['total']
        freq = int(freq_series.sum())

        tech_events = events_df[events_df['title'].str.contains(tech, case=False, na=False)]
        event_counts = tech_events['types'].str.split(',').explode().value_counts().to_dict()

        sentiment_score = 0.0
        if sentiment_analyzer:
            tech_articles = [item['body'] for item in meta_data if item.get('body') and tech in item['body'] and len(item['body']) > 50]
            if tech_articles:
                sentiments = sentiment_analyzer(tech_articles[:10], truncation=True, max_length=512)
                positive_scores = [s['score'] for s in sentiments if s['label'].upper() in ['POSITIVE', '1', 'LABEL_1']]
                if positive_scores:
                    sentiment_score = sum(positive_scores) / len(positive_scores)

        analysis_data = {"frequency": freq, "sentiment": sentiment_score, "events": event_counts}
        print(f"[INFO] 수집된 데이터: {analysis_data}")

        llm_result = call_gemini_for_maturity(tech, analysis_data)

        maturity_results.append({
            "technology": tech, "metrics": analysis_data, "analysis": llm_result
        })

    save_json('outputs/tech_maturity.json', {"results": maturity_results})
    print("\n--- 기술 성숙도 분석 완료 ---")
    print("결과가 'outputs/tech_maturity.json'에 저장되었습니다.")


# --- 2. 약한 신호 심층 분석 (기존 유지) ---
def analyze_weak_signals():
    print("\n--- 2. 약한 신호 심층 분석 시작 ---")
    weak_signal_insights = []

    try:
        weak_signals_df = pd.read_csv("outputs/export/weak_signals.csv")
        top_weak_signals = weak_signals_df.head(3)['term'].tolist()
        print(f"[INFO] 분석 대상 약한 신호 ({len(top_weak_signals)}개): {top_weak_signals}")

        if not top_weak_signals:
            save_json('outputs/weak_signal_insights.json', {"results": []})
            return

        meta_file = latest("data/news_meta_*.json")
        if not meta_file:
            save_json('outputs/weak_signal_insights.json', {"results": []})
            return

        meta_data = load_json(meta_file, [])

        for signal in top_weak_signals:
            context_sentences = []
            for item in meta_data:
                body = item.get('body', '')
                if body and signal in body:
                    sentences = re.split(r'(?<=[.!?])\s+', body)
                    for sent in sentences:
                        if signal in sent and len(context_sentences) < 3:
                            context_sentences.append(sent.strip())
                    if len(context_sentences) >= 3:
                        break

            print(f"\n[분석 중] 약한 신호: {signal} (참고 문장 {len(context_sentences)}개)")
            llm_result = call_gemini_for_weak_signal(signal, context_sentences)
            weak_signal_insights.append(llm_result)

    except FileNotFoundError:
        print("[WARN] weak_signals.csv 파일을 찾을 수 없습니다.")

    save_json('outputs/weak_signal_insights.json', {"results": weak_signal_insights})
    print("\n--- 약한 신호 심층 분석 완료 ---")
    print("결과가 'outputs/weak_signal_insights.json'에 저장되었습니다.")


if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    analyze_tech_maturity()
    analyze_weak_signals()
