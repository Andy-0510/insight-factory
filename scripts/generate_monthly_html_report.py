import os
import json
import pandas as pd
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader, select_autoescape
import re
import markdown # 테이블 HTML 변환용
import time # LLM 호출 지연용
from collections import Counter, defaultdict # 섹션 3에서 사용

# --- 설정 (경로는 실제 프로젝트 구조에 맞게 조정 필요) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
TEMPLATE_NAME = 'monthly_report_template.html'
OUTPUT_BASE_DIR = os.path.join(ROOT_DIR, 'outputs')
EXPORT_DIR = os.path.join(OUTPUT_BASE_DIR, 'export')
FIG_DIR = os.path.join(OUTPUT_BASE_DIR, 'fig')
DEBUG_DIR = os.path.join(OUTPUT_BASE_DIR, 'debug')

# --- 필요한 헬퍼 함수 (utils, timeutil 등에서 import) ---
from src.utils import load_json, latest
from src.timeutil import now_kst
from src.config import load_config

# --- 헬퍼 함수 ---
def format_int_filter(value):
    try: return f"{int(value):,}"
    except (ValueError, TypeError): return value

def load_json_safe(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception: return default

def safe_read_csv(path, **kwargs):
    try:
        if os.path.exists(path): return pd.read_csv(path, **kwargs)
        else: return pd.DataFrame()
    except Exception: return pd.DataFrame()

def get_relative_image_path(image_name):
    # outputs/monthly_report.html 기준 상대 경로
    return f"fig/{image_name}"

def dataframe_to_html_table(df, max_rows=50):
    if df is None or df.empty:
        return "<p>(데이터 없음)</p>"
    # 테이블 스타일 클래스 추가
    return df.head(max_rows).to_html(index=False, escape=False, border=0, classes=["dataframe-table"])

# --- ▼▼▼ 2. 헬퍼 함수 추가 ▼▼▼ ---
def convert_md_to_html(md_text):
    """LLM이 반환한 Markdown 텍스트를 HTML로 변환합니다."""
    if not md_text:
        return ""
    try:
        # 'extra' 확장 기능은 테이블, 각주 등 다양한 마크다운을 지원합니다.
        html = markdown.markdown(md_text, extensions=['extra'])
        return html
    except Exception as e:
        print(f"[WARN] Markdown 변환 실패: {e}")
        return md_text # 변환 실패 시 원본 텍스트 반환
# --- ▲▲▲ 헬퍼 함수 추가 완료 ▲▲▲ ---

def dataframe_to_html_table(df, max_rows=50, classes="dataframe-table"): # <-- 1. classes 인자 추가 (기본값 설정)
    if df is None or df.empty:
        return "<p>(데이터 없음)</p>"
    return df.head(max_rows).to_html(index=False, escape=False, border=0, classes=classes) # <-- 2. 인자로 받은 classes 사용

# --- ▼▼▼ LLM 호출 함수들 (월간용) ▼▼▼ ---
def _call_gemini_safe(prompt: str, default_resp: str = "AI 분석 실패") -> str:
    """Gemini 호출을 안전하게 감싸는 내부 함수 (지연 시간 추가)"""
    try:
        # 분당 호출 제한을 피하기 위해 지연 (월간은 호출 횟수가 많으므로 필수)
        print("[INFO] Waiting before Gemini API call (Rate Limit)...")
        time.sleep(4.1) # 분당 15회 제한 (4초+)
        
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "LLM API 키 없음."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)
        request_options = {"timeout": 120} # 월간 분석은 타임아웃 120초
        response = model.generate_content(prompt, request_options=request_options)

        if response and hasattr(response, 'text') and response.text:
             text = response.text.strip()
             text = re.sub(r"^```[\w]*\n", "", text)
             text = re.sub(r"\n```$", "", text)
             return text.strip()
        # ... (기타 오류 처리) ...
        elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             print(f"[WARN] Gemini request blocked: {response.prompt_feedback.block_reason}")
             return f"AI 분석 실패 (콘텐츠 차단: {response.prompt_feedback.block_reason})"
        else:
             print(f"[WARN] Gemini returned empty or unexpected response.")
             return default_resp
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e.__class__.__name__}: {e}")
        if "Timeout" in str(e): return "AI 분석 실패 (응답 시간 초과)"
        elif "API key not valid" in str(e): return "AI 분석 실패 (API 키 오류)"
        elif "429" in str(e): return "AI 분석 실패 (API 요청 한도 초과)"
        return f"AI 분석 실패 ({e.__class__.__name__})"

# --- ▼▼▼ [추가] 섹션 요약 함수 ▼▼▼ ---
def call_gemini_for_section_summary(section_title, context_summary):
    """LLM 호출: 각 섹션별 Executive Summary 생성 (간결하게)"""
    if isinstance(context_summary, (list, dict)):
        # 컨텍스트가 너무 길어지는 것을 방지하기 위해 요약 (예: 상위 5개)
        if isinstance(context_summary, list) and len(context_summary) > 5:
             context_str = json.dumps(context_summary[:5], ensure_ascii=False, indent=2) + "\n... (이하 생략)"
        else:
             context_str = json.dumps(context_summary, ensure_ascii=False, indent=2)
    else:
        context_str = str(context_summary)

    prompt = f"""
    당신은 디스플레이 산업 분야의 수석 애널리스트입니다. '{section_title}' 섹션의 핵심 데이터를 요약했습니다.
    이 데이터를 바탕으로 해당 섹션의 **핵심 결론(Executive Summary)**을 **1~2 문장**으로 작성해주세요. (Markdown 형식)
    
    ### '{section_title}' 섹션 핵심 데이터:
    {context_str}

    ### Section Summary (1~2 문장, Markdown):
    """
    md_response = _call_gemini_safe(prompt, default_resp="섹션 요약 생성 실패.")
    return markdown.markdown(md_response) # HTML로 변환하여 반환
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def call_gemini_for_monthly_summary(context):
    """LLM 호출: 월간 Executive Summary 생성"""
    prompt = f"""
    당신은 디스플레이 제조 기업의 최고 전략 책임자 입니다.
    아래는 지난 한 달간의 시장 데이터 분석 결과 요약입니다. 
    이 데이터를 종합하여 CEO 및 경영진을 위한 '월간 전략 보고서 Executive Summary'를 Markdown 형식으로 작성해주세요.
    지난달 시장의 핵심적인 변화, 우리에게 가장 큰 기회와 위협 요인, 그리고 다음 분기에 집중해야 할 최우선 전략 방향을 중심으로 서술해주세요.
    답변 이외 군더더기는 절대로 포함하지 않으며, **본론부터 바로 답변**해주세요. **일반론적인 논지와 전개는 지양**하며, **주어진 데이터에 충실히 근거**하여 **깊이 있는 통찰**을 제공해주세요.
    ### 월간 데이터 요약:
    {json.dumps(context, ensure_ascii=False, indent=2)}

    ### Executive Summary (Markdown 형식):
    """
    return _call_gemini_safe(prompt, default_resp="월간 AI 요약 생성 실패.")


def call_gemini_for_positioning_analysis(topics_context):
    """LLM 호출: 토픽 데이터를 기반으로 사분면 분석, 인사이트, 시사점을 JSON으로 생성"""
    prompt = f"""
    당신은 **디스플레이 제조 기업**의 최고 전략 책임자 입니다. 아래는 이번 달 시장의 핵심 토픽 데이터입니다.
    ### 핵심 토픽 데이터:
    {json.dumps(topics_context, ensure_ascii=False, indent=2)}

    ### 분석 요청:
    아래 10가지 항목에 대해 **디스플레이 제조 기업의 관점**으로 **반드시 JSON 형식으로만** 답변해주세요. (다른 설명 금지)
    1.  `insight` (Markdown): 데이터를 종합한 시장의 거시적 흐름 (1~2 문단)
    2.  `q1_topics` (String): '고관심/고긍정' 영역의 **핵심 토픽 1~2개**.
    3.  `q1_strategy` (String): 해당 영역의 권고 전략 (1문장).
    4.  `q2_topics` (String): '고관심/저긍정' 영역의 **핵심 토픽 1~2개**.
    5.  `q2_strategy` (String): 해당 영역의 권고 전략 (1문장).
    6.  `q3_topics` (String): '저관심/고긍정' 영역의 **핵심 토픽 1~2개**.
    7.  `q3_strategy` (String): 해당 영역의 권고 전략 (1문장).
    8.  `q4_topics` (String): '저관심/저긍정' 영역의 **핵심 토픽 1~2개**.
    9.  `q4_strategy` (String): 해당 영역의 권고 전략 (1문장).
    10. `implications` (List[String]): 전략적 시사점 2가지 (문자열 리스트)

    ### 출력 형식 (JSON):
    ```json
    {{
      "insight": "#### 전략적 인사이트 (AI)\\n...",
      "q1_topics": "핵심 토픽 (예: IT OLED)",
      "q1_strategy": "...",
      "q2_topics": "...",
      "q2_strategy": "...",
      "q3_topics": "...",
      "q3_strategy": "...",
      "q4_topics": "...",
      "q4_strategy": "...",
      "implications": ["시사점 1", "시사점 2"]
    }}
    ```
    """
    raw_response = _call_gemini_safe(prompt, default_resp="{}")

    # LLM 응답에서 JSON만 파싱
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
        else:
            parsed_json = json.loads(raw_response)
    except json.JSONDecodeError:
        parsed_json = {}

    # 템플릿에 필요한 기본값 설정
    insights = {
        'positioning_insight': convert_md_to_html(parsed_json.get('insight', "AI 분석 실패")),
        'quadrant1_topics': parsed_json.get('q1_topics', "AI 분석 실패"),
        'quadrant1_strategy': parsed_json.get('q1_strategy', "AI 권고 생성 실패"),
        'quadrant2_topics': parsed_json.get('q2_topics', "AI 분석 실패"),
        'quadrant2_strategy': parsed_json.get('q2_strategy', "AI 권고 생성 실패"),
        'quadrant3_topics': parsed_json.get('q3_topics', "AI 분석 실패"),
        'quadrant3_strategy': parsed_json.get('q3_strategy', "AI 권고 생성 실패"),
        'quadrant4_topics': parsed_json.get('q4_topics', "AI 분석 실패"),
        'quadrant4_strategy': parsed_json.get('q4_strategy', "AI 권고 생성 실패"),
        'strategic_implications': parsed_json.get('implications', ["AI 시사점 분석 실패"])
    }
    return insights

'''
def call_gemini_for_tech_recommendation(tech_name, stage, reason):
    """LLM 호출: 개별 기술 성숙도 기반 투자 권고 생성 (간결화)"""
    prompt = f"""
    디스플레이 전략가로서, '{tech_name}' 기술이 현재 '{stage}' 단계로 분석되었습니다.
    (이유: {reason})
    이 기술에 대한 **투자 권고** (예: 공격적 Capa 확대, 원천 기술 확보, 모니터링 유지 등)를 **한 문장**으로 간결하게 제안해주세요. (명사형 어구 선호)
    
    ### 투자 권고 (한 문장, 명사형 어구):
    """
    # _call_gemini_safe 함수는 이미 정의되어 있다고 가정
    return _call_gemini_safe(prompt, default_resp="AI 권고 생성 실패.")
'''

'''
def call_gemini_for_rd_recommendation(tech_details_context):
    """LLM 호출: 기술 성숙도 데이터 전체 기반 R&D 자원 배분 권고 생성"""
    prompt = f"""
    당신은 디스플레이 기업의 CTO입니다. 아래는 이번 달 주요 기술들의 성숙도 분석 결과입니다.
    
    ### 기술 성숙도 요약:
    {json.dumps(tech_details_context, ensure_ascii=False, indent=2)}

    ### 분석 요청:
    이 데이터를 바탕으로, 다음 분기 **R&D 자원 배분 전략** (예: Growth 기술 비중 확대, Emerging 기술 탐색 강화 등)에 대한 **핵심 권고 사항**을 2~3 문장으로 요약해주세요. (레퍼런스의 3개 항목 비율처럼)

    ### R&D 자원 배분 권고 (2~3 문장):
    """
    # _call_gemini_safe 함수는 이미 정의되어 있다고 가정
    return _call_gemini_safe(prompt, default_resp="R&D AI 권고 생성 실패.")
'''

def call_gemini_for_strategy_insight(company_name, topics_str):
    """LLM 호출: 기업의 토픽 집중도 기반 전략 방향성 분석 (월간용)"""
    prompt = f"""
    당신은 **디스플레이 제조 기업** 전문 애널리스트입니다.
    '{company_name}'라는 기업이 최근 한 달간 아래 토픽들에 집중하고 있습니다.
    이를 바탕으로 이 기업의 현재 사업 방향성과 단기 전략을 **한 문장**으로 간결하게 해석해주세요.
    ### 집중 토픽:
    {topics_str}
    ### 분석 결과 (한 문장 요약):
    """
    return _call_gemini_safe(prompt, default_resp=f"'{company_name}' 전략 분석 실패.")

def call_gemini_for_competition_alerts(matrix_summary, network_summary):
    """LLM 호출: 경쟁 강도 변화 경보 생성"""
    prompt = f"""
    당신은 **디스플레이 시장** 경쟁 분석 전문가입니다. 아래는 이번 달 산업 내 주요 기업들의 분석 요약 데이터입니다.
    ### 기업-토픽 집중도 요약:
    {matrix_summary}
    ### 기업 관계망 요약:
    {network_summary}

    ### 분석 요청:
    이 데이터를 바탕으로, **디스플레이 제조 기업의 관점**에서 이번 달 감지된 **경쟁 강도 변화에 대한 핵심 경보** (예: 특정 영역 경쟁 심화, 신규 파트너십으로 인한 구도 변화 등)를 2~3개의 불릿포인트(-)로 요약해주세요.
    
    ### 경쟁 강도 변화 경보 (Markdown):
    - (경보 1)
    - (경보 2)
    """
    response = _call_gemini_safe(prompt, default_resp="경쟁 강도 분석 실패.")
    # Markdown 리스트 파싱
    alerts = re.findall(r"-\s+(.*)", response)
    return [a.strip() for a in alerts] if alerts else ["AI 분석 실패 또는 특이사항 없음."]

def call_gemini_for_network_action_item(pair_info):
    """LLM 호출: 기업 관계망 기반 액션 아이템 제안"""
    prompt = f"""
    당신은 **디스플레이 제조 기업**의 전략기획팀 리더입니다. 시장 분석 결과, 아래 기업 간의 주요 관계가 포착되었습니다.
    ### 관계 정보:
    {pair_info}
    ### 액션 아이템 제안 (1개, 명사형 어구, 20자 내외):
    """
    recommendation_raw = _call_gemini_safe(prompt, default_resp="액션 아이템 생성 실패.")
    # 후처리 (줄바꿈, 마크다운 제거)
    return recommendation_raw.replace("**", "").split('\n')[0].strip()

''' ### 아래와 동일 함수 ### 
def call_gemini_for_risk_analysis(risk_context):
    """LLM 호출: 리스크 데이터를 기반으로 매트릭스, 즉시 대응, 종합 평가를 생성"""
    prompt = f"""
    당신은 기업 리스크 관리 최고 책임자(CRO)입니다. 아래는 이번 달 탐지된 주요 리스크 목록입니다.
    
    ### 주요 리스크 목록:
    {json.dumps(risk_context, ensure_ascii=False, indent=2)}

    ### 분석 요청:
    아래 3가지 항목에 대해 Markdown 형식으로 답변해주세요.

    1.  **리스크 대응 매트릭스**:
        - 각 리스크를 '회피(Avoid)', '완화(Mitigate)', '전가(Transfer)', '수용(Accept)' 4가지 전략으로 분류하고, 각 전략별 대상 리스크와 핵심 전략을 요약해주세요.
        - `회피(Avoid)`: (대상 리스크), (핵심 전략)
        - `완화(Mitigate)`: (대상 리스크), (핵심 전략)
        - `전가(Transfer)`: (대상 리스크), (핵심 전략)
        - `수용(Accept)`: (대상 리스크), (핵심 전략)
    
    2.  **즉시 대응 필요 항목 (AI Priority)**:
        - 위 리스크 중 가장 시급하게 대응해야 할 Top 3 리스크를 선정하고, 30일 이내 실행할 구체적인 액션 아이템을 제안해주세요. (담당 조직, 기한 포함)
        - `[Risk ID 또는 토픽명]`: (액션 아이템) (담당: [조직], 기한: [YYYY-MM-DD])
        - `[Risk ID 또는 토픽명]`: (액션 아이템) (담당: [조직], 기한: [YYYY-MM-DD])

    3.  **리스크 종합 평가 및 대응 제안 (AI)**:
        - 이번 달 리스크 현황을 1~2 문단으로 종합 평가하고, 차월 대응 방향을 제안해주세요.

    ### 출력 형식 (Markdown):
    #### 리스크 대응 매트릭스
    - **회피(Avoid)**: (1번 항목 답변)
    - **완화(Mitigate)**: (1번 항목 답변)
    - **전가(Transfer)**: (1번 항목 답변)
    - **수용(Accept)**: (1번 항목 답변)

    #### 즉시 대응 필요 항목 (AI Priority)
    - (2번 항목 답변 1)
    - (2번 항목 답변 2)
    - (2번 항목 답변 3)

    #### 리스크 종합 평가 및 대응 제안 (AI)
    (3번 항목 답변)
    """
    response = _call_gemini_safe(prompt, default_resp="리스크 분석 실패")
    
    # LLM 응답 파싱
    insights = {
        'risk_matrix': {
            "avoid": {"targets": "AI 분석 중...", "strategy": "AI 분석 중..."},
            "mitigate": {"targets": "AI 분석 중...", "strategy": "AI 분석 중..."},
            "transfer": {"targets": "AI 분석 중...", "strategy": "AI 분석 중..."},
            "accept": {"targets": "AI 분석 중...", "strategy": "AI 분석 중..."}
        },
        'immediate_actions': [],
        'risk_assessment': "리스크 AI 종합 평가 생성 실패."
    }

    try:
        # 1. 리스크 매트릭스 파싱
        matrix_match = re.search(r"#### 리스크 대응 매트릭스\s*(.*?)\s*#### 즉시 대응 필요 항목", response, re.DOTALL)
        if matrix_match:
            matrix_text = matrix_match.group(1)
            avoid_match = re.search(r"회피\(Avoid\)\s*:\s*(.*?),\s*(.*)", matrix_text)
            if avoid_match: insights['risk_matrix']['avoid'] = {"targets": avoid_match.group(1).strip(), "strategy": avoid_match.group(2).strip()}
            mitigate_match = re.search(r"완화\(Mitigate\)\s*:\s*(.*?),\s*(.*)", matrix_text)
            if mitigate_match: insights['risk_matrix']['mitigate'] = {"targets": mitigate_match.group(1).strip(), "strategy": mitigate_match.group(2).strip()}
            transfer_match = re.search(r"전가\(Transfer\)\s*:\s*(.*?),\s*(.*)", matrix_text)
            if transfer_match: insights['risk_matrix']['transfer'] = {"targets": transfer_match.group(1).strip(), "strategy": transfer_match.group(2).strip()}
            accept_match = re.search(r"수용\(Accept\)\s*:\s*(.*?),\s*(.*)", matrix_text)
            if accept_match: insights['risk_matrix']['accept'] = {"targets": accept_match.group(1).strip(), "strategy": accept_match.group(2).strip()}

        # 2. 즉시 대응 항목 파싱
        actions_match = re.search(r"#### 즉시 대응 필요 항목 \(AI Priority\)\s*(.*?)\s*#### 리스크 종합 평가", response, re.DOTALL)
        if actions_match:
            actions_text = actions_match.group(1)
            actions = re.findall(r"-\s+(.*)", actions_text)
            if actions:
                # LLM 응답 형식(Risk ID, 액션, 담당, 기한)을 파싱하여 딕셔너리로 만듦
                actions_list = []
                for action_str in actions:
                    # 정규표현식으로 각 부분 추출 (더 견고하게 수정)
                    action_match = re.search(r"\*\*(.*?)\*\*:\s*(.*)\s+\(담당:\s*(.*),\s+기한:\s*(.*)\)", action_str)
                    if action_match:
                        actions_list.append({
                            "risk_id": action_match.group(1).strip(),
                            "action": action_match.group(2).strip(),
                            "owner": action_match.group(3).strip(),
                            "due_date": action_match.group(4).strip().rstrip(')')
                        })
                    else:
                         # 단순 텍스트일 경우
                         actions_list.append({"risk_id": "N/A", "action": action_str, "owner": "N/A", "due_date": "N/A"})
                insights['immediate_actions'] = actions_list

        # 3. 종합 평가 파싱
        assessment_match = re.search(r"#### 리스크 종합 평가 및 대응 제안 \(AI\)\s*(.*)", response, re.DOTALL)
        if assessment_match:
            insights['risk_assessment'] = markdown.markdown(assessment_match.group(1).strip())
            
    except Exception as e:
        print(f"[WARN] Failed to parse risk LLM response: {e}")
        
    return insights
'''

def call_gemini_for_risk_analysis(risk_context):
    """LLM 호출: 리스크 데이터를 기반으로 매트릭스, 즉시 대응, 종합 평가를 JSON으로 생성"""
    prompt = f"""
    당신은 **디스플레이 제조 기업**의 리스크 관리 최고 책임자(CRO)입니다. 아래는 이번 달 탐지된 주요 리스크 목록입니다.
    ### 주요 리스크 목록:
    {json.dumps(risk_context, ensure_ascii=False, indent=2)}

    ### 분석 요청:
    아래 4가지 항목에 대해 **반드시 JSON 형식으로만** 답변해주세요. (다른 설명 금지)
    1.  `matrix` (Object): 4가지 전략('avoid', 'mitigate', 'transfer', 'accept')별 대상 리스크와 핵심 전략.
        - **[중요]** `targets` 필드에는 '주요 리스크 목록' 컨텍스트에 포함된 **'Keywords'만** 요약하여 기재하세요. (토픽 이름 제외) (예: "oled, tv, 패널")
    2.  `immediate_actions` (List[Object]): 가장 시급한 Top 3 리스크의 액션 아이템. (예: [{{"risk_id": "...", "action": "...", "owner": "...", "due_date": "..."}}, ...])
    3.  `assessment` (Markdown): 리스크 종합 평가 및 차월 대응 방향 (1~2 문단)

    ### 출력 형식 (JSON):
    ```json
    {{
      "matrix": {{
        "avoid": {{"targets": "...", "strategy": "..."}},
        "mitigate": {{"targets": "...", "strategy": "..."}},
        "transfer": {{"targets": "...", "strategy": "..."}},
        "accept": {{"targets": "...", "strategy": "..."}}
      }},
      "immediate_actions": [
        {{"risk_id": "Risk ID", "action": "액션 아이템", "owner": "담당조직", "due_date": "YYYY-MM-DD"}}
      ],
      "assessment": "#### 리스크 종합 평가 (AI)\\n..."
    }}
    ```
    """
    raw_response = _call_gemini_safe(prompt, default_resp="{}")

    # LLM 응답에서 JSON만 파싱
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
        else:
            parsed_json = json.loads(raw_response)
    except json.JSONDecodeError:
        parsed_json = {}

    # 기본값 설정
    default_matrix = {"targets": "LLM 호출 한도 초과", "strategy": "LLM 호출 한도 초과"}
    insights = {
        'risk_matrix': parsed_json.get('matrix', {
            "avoid": default_matrix,
            "mitigate": default_matrix,
            "transfer": default_matrix,
            "accept": default_matrix
        }),
        'immediate_actions': parsed_json.get('immediate_actions', []),
        'risk_assessment': convert_md_to_html(parsed_json.get('assessment', "리스크 AI 종합 평가 생성 실패."))
    }
    return insights


def call_gemini_for_final_recommendation(summary_context):
    """LLM 호출: 모든 섹션의 요약을 바탕으로 최종 종합 전략 권고 생성"""
    prompt = f"""
    당신은 **디스플레이** 산업 분야의 전략 컨설턴트입니다. 아래는 이번 달 시장 분석의 모든 핵심 요약본입니다.
    
    ### 월간 핵심 데이터 요약:
    {json.dumps(summary_context, ensure_ascii=False, indent=2)}

    ### 분석 요청 (Markdown 형식):
        1. **종합 해석**: 토픽 포지셔닝 맵(버블 차트), 미니 트렌드 차트, 모멘텀 점수 데이터를 종합적으로 해석하여, 현재 시장의 **주요 동력(Driving Force)**과 **새롭게 부상하는 기회(Emerging Opportunity)** 영역이 무엇인지 설명해주세요. # 이름 변경
        2. **전략적 제언**: 분석 결과를 바탕으로, 우리 회사(**디스플레이 제조 기업**)가 다음 분기에 **자원을 집중해야 할 토픽 영역**과 **주의 깊게 모니터링해야 할 토픽 영역**을 각각 제안해주세요.

    ### 분석 결과 (Markdown):
        #### 종합 시장 동향 해석
            - **주요 동력**: (설명)
            - **부상하는 기회**: (설명)

        #### 전략적 제언
            - **집중 영역**: (제안)
            - **모니터링 영역**: (제안)
    """
    response = _call_gemini_safe(prompt, default_resp="최종 AI 종합 권고 생성 실패.")
    # Markdown을 HTML로 변환하여 반환 (템플릿에서 |safe 필터 사용)
    return markdown.markdown(response.strip())


# --- LLM 호출 함수들 (필요한 함수 정의 또는 import) ---
# 예: call_gemini_for_monthly_summary, call_gemini_for_positioning_insight 등
# ... (LLM 함수 정의 영역) ...

# --- 데이터 로딩 및 가공 함수 ---
def prepare_monthly_report_data():
    """월간 HTML 템플릿에 필요한 데이터를 로드하고 가공하는 함수"""
    print("[INFO] Loading and preparing data for Monthly HTML report...")
    data = {}
    end_dt = now_kst()
    start_dt = end_dt - timedelta(days=29)
    data['report_period'] = f"{start_dt.strftime('%Y.%m.%d')} - {end_dt.strftime('%Y.%m.%d')}"
    data['report_month'] = end_dt.strftime('%Y년 %m월')

    # 1. 월간 집계 데이터 로드
    topics_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, "topics.json"), {"topics": []})
    # tech_maturity_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, "tech_maturity.json"), {"results": []})
    company_network_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, "company_network.json"), {})
    biz_opps_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, "biz_opportunities.json"), {"ideas": []})
    growth_df = safe_read_csv(os.path.join(EXPORT_DIR, "topic_growth.csv"))
    matrix_df_long = safe_read_csv(os.path.join(EXPORT_DIR, "company_topic_matrix_long.csv"))
    risk_issues_df = safe_read_csv(os.path.join(EXPORT_DIR, "risk_issues.csv"))
    action_plan_df = safe_read_csv(os.path.join(EXPORT_DIR, "two_week_plan.csv"))
    monthly_meta = load_json_safe(os.path.join(DEBUG_DIR, "monthly_meta_agg.json"), [])
    weak_signals_df = safe_read_csv(os.path.join(EXPORT_DIR, "weak_signals.csv"))

    # 2. 목차 데이터 생성 (부제목 추가)
    data['toc_items'] = [
        {'number': '📄', 'id': 'summary', 'title': 'Executive Summary', 'subtitle': '월간 전략 요약 및 핵심 KPI'},
        {'number': '1', 'id': 'positioning', 'title': '전략적 시장 포지셔닝 맵', 'subtitle': '시장 거시 환경 분석 · Topic Bubble Map'},
        # {'number': '2', 'id': 'lifecycle', 'title': '기술 수명 주기 분석', 'subtitle': 'R&D 투자 타이밍 · Technology Maturity Map'},
        {'number': '2', 'id': 'competitors', 'title': '주요 기업의 전략적 의도 분석', 'subtitle': '경쟁 구도 분석 · Company Network'},
        {'number': '3', 'id': 'risk', 'title': 'Market Risk Management', 'subtitle': '리스크/이슈 탐지 · Mitigation Plan'},
        {'number': '4', 'id': 'actionplan', 'title': '전략 방향성 및 실행 방안 제안', 'subtitle': '중장기 전략 제안 · Resource Allocation'},
        {'number': '🔍', 'id': 'opportunities', 'title': '잠재 사업 기회', 'subtitle': '데이터 기반 아이디어 제안 · Top 5 Opportunities'},
    ]
    
    # --- ▼▼▼ Section 0: Executive Summary & KPI ▼▼▼ ---
    summary_context = {
        "period": data['report_period'],
        "top_biz_idea": (biz_opps_data.get("ideas", [{}]))[0].get("idea", "N/A"),
        "top_risk": risk_issues_df.iloc[0]['Topic'] if not risk_issues_df.empty and 'Topic' in risk_issues_df.columns else "N/A",
        # "emerging_tech": next((t['technology'] for t in tech_maturity_data.get("results", []) if t.get("analysis", {}).get("stage") == "Emerging"), "N/A"),
        "top_rising_topic": growth_df.iloc[0]['topic_name'] if not growth_df.empty and 'topic_name' in growth_df.columns else "N/A"
    }
    data['executive_summary'] = convert_md_to_html(call_gemini_for_monthly_summary(summary_context))
    data['kpi_total_articles'] = len(monthly_meta)
    data['kpi_article_change_mom'] = "+0%"; data['kpi_article_change_class'] = "change-neutral" # Placeholder
    data['kpi_key_topics_count'] = len(topics_data.get("topics", []))
    data['kpi_topic_change_mom'] = "+0"; data['kpi_topic_change_class'] = "change-neutral" # Placeholder
    source_diversity_count = len(set(item.get('site_name') for item in monthly_meta if item.get('site_name')))
    data['kpi_source_diversity'] = source_diversity_count
    data['kpi_source_change_mom'] = "+0"; data['kpi_source_change_class'] = "change-neutral" # Placeholder
    weak_signals_count = len(weak_signals_df) if not weak_signals_df.empty else 0
    data['kpi_weak_signals_count'] = weak_signals_count
    data['kpi_weak_signal_change_mom'] = "+0"; data['kpi_weak_signal_change_class'] = "change-neutral" # Placeholder

    # --- ▼▼▼ Section 1: Market Positioning ▼▼▼ ---
    data['topic_bubble_map_path'] = get_relative_image_path("topics_bubble.png")
    data['topic_mini_trends_path'] = get_relative_image_path("topics_mini_trends.png")
    df_topic_details = pd.DataFrame(topics_data.get("topics", []))
    topics_context = []
    topic_id_name_map = {}
    if not df_topic_details.empty and 'topic_id' in df_topic_details.columns:
        if not growth_df.empty and 'topic_id' in growth_df.columns:
             df_topic_details = pd.merge(df_topic_details, growth_df[['topic_id', 'momentum_score']], on='topic_id', how='left')
             df_topic_details['momentum_score'] = df_topic_details['momentum_score'].fillna(0.0)
        else: df_topic_details['momentum_score'] = 0.0
        df_topic_details['Topic Identifier'] = df_topic_details.apply(lambda row: row.get('topic_name', f"Topic #{row.get('topic_id')}"), axis=1)
        if 'topic_summary' not in df_topic_details.columns: df_topic_details['topic_summary'] = ""
        
        topic_id_name_map = df_topic_details.set_index('topic_id')['Topic Identifier'].to_dict()

        topics_context = df_topic_details.head(10).apply(
            lambda row: { "토픽명": row['Topic Identifier'], "요약": row['topic_summary'], "모멘텀 (z-like)": row.get('momentum_score', 0.0) }, axis=1
        ).tolist()
    positioning_insights = call_gemini_for_positioning_analysis(topics_context)
    data.update(positioning_insights)
    data['positioning_exec_summary'] = call_gemini_for_section_summary("전략적 시장 포지셔닝 맵", topics_context)
    if not df_topic_details.empty:
        df_topic_details['top_words_str'] = df_topic_details.get('top_words', pd.Series(dtype='object')).apply(lambda words: ", ".join([w.get('word', '') for w in words[:3]]) if isinstance(words, list) else "")
        df_topic_details_final = df_topic_details[['Topic Identifier', 'topic_summary', 'top_words_str']].rename(columns={'Topic Identifier': '토픽', 'topic_summary': '요약', 'top_words_str': '핵심 키워드'})
        data['topic_details_table'] = dataframe_to_html_table(df_topic_details_final, max_rows=20, classes="dataframe-table topics-table")
    else: data['topic_details_table'] = "<p>(토픽 데이터 없음)</p>"
    if not growth_df.empty and 'topic_id' in growth_df.columns and topic_id_name_map:
         growth_df['토픽'] = growth_df['topic_id'].map(topic_id_name_map).fillna(growth_df['topic_id'].apply(lambda x: f"Topic #{x}"))
         if 'topic_name' not in growth_df.columns: growth_df['topic_name'] = ""
         growth_table_df = growth_df[['토픽', 'topic_name', 'momentum_score']].rename(columns={'topic_name': 'LLM 이름 (참고)', 'momentum_score': '모멘텀 점수'})
         data['topic_growth_table'] = dataframe_to_html_table(growth_table_df, max_rows=15, classes="dataframe-table growth-table")
    else: data['topic_growth_table'] = "<p>(토픽 성장률 데이터 없음)</p>"

    # --- ▼▼▼ Section 2: Tech Maturity ▼▼▼ ---
    '''
    data['tech_maturity_map_path'] = get_relative_image_path("tech_maturity_map.png")
    tech_details_list = []
    llm_context_for_rd_rec = []
    tech_results = tech_maturity_data.get("results", [])
    print(f"[INFO] Processing {len(tech_results)} tech maturity items...")
    for tech in tech_results:
        tech_name = tech.get("technology", "N/A"); stage = tech.get("analysis", {}).get("stage", "N/A"); reason = tech.get("analysis", {}).get("reason", "분석 데이터 없음")
        stage_class_map = {"Emerging": "stage-emerging", "Growth": "stage-growth", "Maturity": "stage-maturity"}
        recommendation = call_gemini_for_tech_recommendation(tech_name, stage, reason)
        tech_details_list.append({ "stage_class": stage_class_map.get(stage, ""), "stage_text": stage, "name": tech_name, "description": reason, "recommendation": recommendation })
        llm_context_for_rd_rec.append(f"{tech_name}: {stage} 단계 (권고: {recommendation})")
    data['tech_maturity_details'] = tech_details_list
    data['rd_recommendation'] = call_gemini_for_rd_recommendation(llm_context_for_rd_rec)
    data['lifecycle_exec_summary'] = call_gemini_for_section_summary("기술 수명 주기 분석", llm_context_for_rd_rec)
    df_tech_maturity = pd.DataFrame(tech_details_list)
    data['tech_maturity_table'] = dataframe_to_html_table(df_tech_maturity[['name', 'stage_text', 'description', 'recommendation']].head(10), classes="dataframe-table tech-maturity-table")
    '''
    
    # --- ▼▼▼ Section 3: Competitor Analysis ▼▼▼ ---
    data['matrix_heatmap_path'] = get_relative_image_path("matrix_heatmap.png")
    data['company_network_path'] = get_relative_image_path("company_network.png")
    data['keyword_network_path'] = get_relative_image_path("keyword_network.png")
    comp_pos_list = []
    matrix_summary_for_llm = []
    if not matrix_df_long.empty and topic_id_name_map:
        top_orgs = matrix_df_long.groupby('org')['hybrid_score'].sum().nlargest(4).index
        try:
             focus_quantiles = matrix_df_long.groupby('org')['hybrid_score'].sum().quantile([0.25, 0.5, 0.75]).to_dict()
             def get_focus_level(score):
                 if score > focus_quantiles[0.75]: return "Very High"
                 elif score > focus_quantiles[0.5]: return "High"
                 elif score > focus_quantiles[0.25]: return "Medium"
                 return "Low"
        except Exception:
             print("[WARN] Failed to calculate focus quantiles. Using default 'N/A'.")
             def get_focus_level(score):
                 return "N/A"
        total_market_score = matrix_df_long['hybrid_score'].sum()
        print(f"[INFO] Processing {len(top_orgs)} top competitors for positioning map...")
        for comp in top_orgs:
            comp_data = matrix_df_long[matrix_df_long['org'] == comp]
            total_score = comp_data['hybrid_score'].sum()
            focus_level = get_focus_level(total_score)
            topic_share_pct = (total_score / total_market_score * 100) if total_market_score > 0 else 0.0
            top_topics = comp_data.nlargest(3, 'hybrid_score')
            top_topics_str = ", ".join([topic_id_name_map.get(int(t_id), f"Topic {t_id}") for t_id in top_topics['topic'] if str(t_id).isdigit()])
            strategy_insight = call_gemini_for_strategy_insight(comp, top_topics_str)
            comp_pos_list.append({ "name": comp, "badge_text": f"{focus_level} Focus", "topic_share": f"{topic_share_pct:.1f}%", "focus_level": focus_level, "strategy_insight": strategy_insight })
            matrix_summary_for_llm.append(f"- {comp}: {focus_level} Focus, Top Topic: {top_topics_str.split(',')[0]}")
    data['competitor_positioning'] = comp_pos_list
    network_summary_for_llm = [f"- {p['source']} <-> {p['target']} ({p['rel_type']})" for p in company_network_data.get("top_pairs", [])[:3]]
    # 1. LLM 호출 (변수 이름을 alerts_list로 바로 받음)
    alerts_list = call_gemini_for_competition_alerts("\n".join(matrix_summary_for_llm), "\n".join(network_summary_for_llm))
    # 2. HTML 변환 로직은 그대로 유지
    if alerts_list:
        data['competition_alerts'] = "<ul>" + "".join([f"<li>{convert_md_to_html(item)}</li>" for item in alerts_list]) + "</ul>"
    else:
        data['competition_alerts'] = "<p>AI 분석 실패 또는 특이사항 없음.</p>"
    df_centrality = pd.DataFrame(company_network_data.get("centrality", []))
    if not df_centrality.empty:
         data['competitor_strategy_table'] = dataframe_to_html_table(df_centrality[['org', 'degree_centrality']].rename(columns={'org': '기업 (허브)', 'degree_centrality': '연결 중심성'}), max_rows=5, classes="dataframe-table centrality-table")
    else: data['competitor_strategy_table'] = "<p>(네트워크 중심성 데이터 없음)</p>"
    top_pairs = company_network_data.get("top_pairs", [])
    pair_actions_list = []
    if top_pairs:
        print(f"[INFO] Formatting {len(top_pairs[:5])} top company pairs with evidence...")
        for pair in top_pairs[:5]:
            # (LLM 호출 삭제)
            key_evidence = pair.get('evidence', '근거 문장 없음') # 1단계에서 저장한 evidence 가져오기

            pair_actions_list.append({ 
                "관계": f"{pair['source']} ↔ {pair['target']}", 
                "유형": pair['rel_type'], 
                "강도(언급빈도)": pair['weight'], 
                "주요 연관 근거": key_evidence  # <-- 컬럼명 및 값 변경
            })
    data['competitor_actions_table'] = dataframe_to_html_table(pd.DataFrame(pair_actions_list), classes="dataframe-table actions-table")
    data['competitors_exec_summary'] = call_gemini_for_section_summary("경쟁사 전략적 의도 분석", matrix_summary_for_llm + network_summary_for_llm)

    # --- ▼▼▼ Section 4: Risk Management (테이블 가공 로직 수정) ▼▼▼ ---
    # --- ▼▼▼ [신규] risk_negative_spikes.png 파일 존재 여부 확인 ▼▼▼ ---
    # FIG_DIR 경로는 파일 상단에 정의되어 있어야 합니다. (os.path.join(ROOT_OUTPUT_DIR, 'fig'))
    risk_spikes_path_full = os.path.join(FIG_DIR, "risk_negative_spikes.png")

    if os.path.exists(risk_spikes_path_full):
        data['risk_spikes_path'] = get_relative_image_path("risk_negative_spikes.png")
    else:
        data['risk_spikes_path'] = None # 파일이 없으면 None으로 설정

    data['risk_network_path'] = get_relative_image_path("risk_keyword_network.png")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    
    if not risk_issues_df.empty:
        # --- ▼▼▼ [수정] get_risk_level_display 함수 ▼▼▼ ---
        def get_risk_level_display(score_str):
            """sentiment_drop 점수에 따라 레벨 배지와 괄호 안 점수를 반환"""
            try:
                score = float(score_str)
                level = "Low"
                css_class = "risk-low" 
                if score > 0.15: # (임계값은 예시, 필요시 조정)
                    level = "High"
                    css_class = "risk-high"
                elif score > 0.05: # (임계값은 예시, 필요시 조정)
                    level = "Medium"
                    css_class = "risk-medium"
                
                # HTML 문자열 반환 (CSS 클래스 + 레벨 + 괄호 안 점수)
                return f'<span class="risk-level {css_class}">{level}</span><br>({score:.3f})'
            except (ValueError, TypeError):
                return "N/A"
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        # 2. '완화 액션' 내용에 볼드 처리
        def format_mitigation(text):
             if not text or pd.isna(text): return "N/A"
             text_html = str(text)
             text_html = re.sub(r'^(30일|60일|90일):', r'<strong>\1:</strong>', text_html, flags=re.MULTILINE)
             return text_html

        # 3. DataFrame 가공
        df_risk_processed = risk_issues_df.head(10).copy()
        
        # 4. '심각도' (리스크 수준) 컬럼 생성
        df_risk_processed['심각도'] = df_risk_processed['sentiment_drop'].apply(get_risk_level_display)
        
        # 5. '완화 액션' 포맷팅
        df_risk_processed['완화 액션'] = df_risk_processed['mitigation'].apply(format_mitigation)

        # 6. 컬럼 선택 및 이름 변경 (내용 유지: '요약' 컬럼 포함)
        df_risk_processed = df_risk_processed[
            ['Topic', 'summary', '심각도', 'impact_range', '완화 액션'] 
        ].rename(columns={
            'Topic': '토픽/분야',
            'summary': '요약',
            'impact_range': '영향 범위'
        })
        
        # 7. HTML 테이블로 변환 (순서 재배치)
        data['risk_list_table'] = dataframe_to_html_table(
            df_risk_processed[['토픽/분야', '요약', '심각도', '영향 범위', '완화 액션']], # 순서 유지
            max_rows=10,
            classes="dataframe-table risk-table" # 'risk-table' 클래스 전달
        )
    else:
        data['risk_list_table'] = "<p> </p>"
    
    risk_context = risk_issues_df.head(5).to_dict('records') if not risk_issues_df.empty else []
    risk_analysis_results = call_gemini_for_risk_analysis(risk_context)
    data.update(risk_analysis_results)
    data['risk_exec_summary'] = call_gemini_for_section_summary("전략적 리스크 관리", risk_context)
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    # --- ▼▼▼ Section 5: Biz Opps & Action Plan ▼▼▼ ---
    data['idea_score_dist_path'] = get_relative_image_path("idea_score_distribution.png")
    df_opps = pd.DataFrame(biz_opps_data.get("ideas", []))
    if not df_opps.empty:
         df_opps_sorted = df_opps.sort_values(by="score", ascending=False)
         data['opportunity_list_table'] = dataframe_to_html_table(df_opps_sorted[['idea', 'value_prop', 'score', 'target_customer']].head(5), max_rows=5, classes="dataframe-table opps-table")
    else: data['opportunity_list_table'] = "<p>(신사업 아이디어 데이터 없음)</p>"
    data['action_plan_table'] = dataframe_to_html_table(action_plan_df, max_rows=10, classes="dataframe-table action-plan-table")
    data['opportunities_exec_summary'] = call_gemini_for_section_summary("신사업 기회 발굴", (biz_opps_data.get("ideas", [{}]))[0])

    # --- ▼▼▼ Section 6: Final Recommendation ▼▼▼ ---
    final_context = {
        "Positioning": data.get('positioning_insight', 'N/A'),
        # "R&D": data.get('rd_recommendation', 'N/A'),
        "Competition": data.get('competition_alerts', []),
        "Risk": data.get('risk_assessment', 'N/A'),
        "Opportunity": (biz_opps_data.get("ideas", [{}]))[0].get("idea", "N/A"),
    }
    data['final_recommendation'] = call_gemini_for_final_recommendation(final_context)

    # Footer 정보
    data['dashboard_link'] = '#'; data['data_source_link'] = '#'; data['contact_link'] = '#'

    print("[INFO] Monthly data preparation complete.")
    return data

# --- Jinja2 템플릿 렌더링 함수 ---
def render_html_report(template_dir, template_name, data):
    """Jinja2를 사용하여 HTML 리포트를 렌더링"""
    print(f"[INFO] Rendering Monthly HTML template: {template_name}")
    try:
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        env.filters['format_int'] = format_int_filter

        template = env.get_template(template_name)
        html_content = template.render(data)
        print("[INFO] Monthly HTML rendering successful.")
        return html_content
    except Exception as e:
        print(f"[ERROR] Monthly HTML template rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return f"<html><body><h1>Monthly Report Generation Failed</h1><pre>{e}</pre></body></html>"


# --- 메인 실행 로직 ---
def main():
    start_time = now_kst()
    print(f"[INFO] Starting monthly HTML report generation at {start_time.strftime('%Y-%m-%d %H:%M:%S KST')}")
    report_data = prepare_monthly_report_data()
    html_output = render_html_report(TEMPLATE_DIR, TEMPLATE_NAME, report_data)
    output_html_path = os.path.join(OUTPUT_BASE_DIR, 'monthly_report.html')
    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        print(f"[SUCCESS] Monthly HTML report saved to: {output_html_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save monthly HTML report: {e}")
    end_time = now_kst()
    print(f"[INFO] Monthly report generation finished at {end_time.strftime('%Y-%m-%d %H:%M:%S KST')}. Duration: {end_time - start_time}")

if __name__ == '__main__':
    main()