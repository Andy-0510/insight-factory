# 파일 경로: src/module_h_planning.py

import pandas as pd
from src.utils import load_json, save_json
import os
import json
import re
from datetime import datetime, timedelta
from src.config import load_config, llm_config  

# --- 설정 ---
OUTPUT_CSV_PATH = "outputs/export/two_week_plan.csv"

def call_gemini_for_planning(context):
    """LLM을 호출하여 실행 계획을 생성합니다."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        
        # config.json에서 모델명을 동적으로 불러옵니다.
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001") # fallback 모델명도 수정

        print(f"[INFO] Using Gemini model for planning: {model_name}")
        
        model = genai.GenerativeModel(model_name)


        # 기한(Due)을 오늘 날짜 기준으로 계산
        due_date_14 = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

        prompt = f"""
        당신은 최고의 비즈니스 전략가입니다. 아래 제공된 '핵심 인사이트'와 '최우선 신사업 아이디어'를 바탕으로, 다음 2주간 실행해야 할 구체적인 실행 계획(Action Plan) 3가지를 제안해주세요.

        ### 입력 데이터:
        {json.dumps(context, ensure_ascii=False, indent=2)}

        ### 요청사항:
        1. 제안된 아이디어를 검증하고 구체화하기 위한 실행 계획을 세워주세요.
        2. 각 계획은 아래 5가지 항목을 반드시 포함해야 합니다.
           - **Hypothesis**: 검증하려는 가설. (예: "고객들은 OOO 기능에 비용을 지불할 의사가 있다.")
           - **Target**: 가설 검증을 위한 구체적인 실행 대상. (예: "주요 고객사 3곳의 구매 담당자 인터뷰")
           - **KPI**: 성공을 측정할 핵심 성과 지표. (예: "인터뷰 대상 중 2곳 이상에서 긍정적 구매의사(LOI) 확보")
           - **Owner**: 책임 담당 조직. (예: "사업개발팀", "기술기획팀")
           - **Due**: 완료 기한. (반드시 "{due_date_14}"로 고정)
        3. 모든 결과는 지정된 JSON 배열 형식으로만 답변해주세요. 설명은 필요 없습니다.

        ### 출력 형식 (JSON 배열):
        ```json
        [
          {{
            "Hypothesis": "...",
            "Target": "...",
            "KPI": "...",
            "Owner": "...",
            "Due": "{due_date_14}"
          }},
          {{
            "Hypothesis": "...",
            "Target": "...",
            "KPI": "...",
            "Owner": "...",
            "Due": "{due_date_14}"
          }}
        ]
        ```
        """
        response = model.generate_content(prompt)
        # 마크다운 JSON 코드 블록을 안전하게 파싱
        match = re.search(r'```json\s*(\[.*?\])\s*```', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else: # 마크다운이 없는 순수 JSON 배열 응답 처리
            return json.loads(response.text)

    except Exception as e:
        print(f"[ERROR] Gemini 실행 계획 생성 실패: {e}")
        return []


def generate_plan():
    """최종 분석 결과를 바탕으로 실행 계획을 생성하고 저장합니다."""
    
    # 1. 컨텍스트 데이터 로드
    try:
        insights_data = load_json("outputs/trend_insights.json")
        opps_data = load_json("outputs/biz_opportunities.json")
    except FileNotFoundError:
        print("[WARN] 분석 파일이 없어 실행 계획 생성을 건너뜁니다.")
        return

    # LLM에 전달할 핵심 정보만 추출
    context = {
        "핵심 인사이트 요약": insights_data.get("summary", "N/A"),
        "최우선 신사업 아이디어": (opps_data.get("ideas") or [{}])[0]
    }
    
    # 2. LLM 호출하여 실행 계획 생성
    plan_items = call_gemini_for_planning(context)

    # 3. 결과 저장
    if plan_items:
        df_plan = pd.DataFrame(plan_items)
        # 컬럼 순서 고정
        df_plan = df_plan[["Hypothesis", "Target", "KPI", "Owner", "Due"]]
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        df_plan.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"[SUCCESS] {len(df_plan)}개의 실행 계획을 생성하여 {OUTPUT_CSV_PATH}에 저장했습니다.")
    else:
        print("[INFO] 생성된 실행 계획이 없습니다.")
        # 빈 파일 생성
        if not os.path.exists(OUTPUT_CSV_PATH):
            pd.DataFrame(columns=["Hypothesis", "Target", "KPI", "Owner", "Due"]).to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    generate_plan()