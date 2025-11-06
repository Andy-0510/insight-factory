import os
import json
import pandas as pd
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
import re # 정규 표현식 사용 위해 추가
from src.utils import latest # <--- 이 줄을 추가하세요
from transformers import pipeline
from src.config import load_config # config 로더 추가
from src.timeutil import now_kst, to_date

TARGET_COMPETITORS_LIST = [
    "LG디스플레이", "삼성디스플레이", "BOE", "CSOT", "AUO", "Innolux", 
    "Visionox", "Tianma", "JDI", "Sharp", "LG Display", "Samsung Display"
]
# (검색용 - 모두 소문자)
TARGET_COMPETITORS_SET_LOWER = {
    comp.lower() for comp in TARGET_COMPETITORS_LIST
}

# --- 설정 ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 프로젝트 루트 경로 추정
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
TEMPLATE_NAME = 'daily_report_template.html'
OUTPUT_BASE_DIR = os.path.join(ROOT_DIR, 'outputs')
EXPORT_DIR = os.path.join(OUTPUT_BASE_DIR, 'export')
FIG_DIR = os.path.join(OUTPUT_BASE_DIR, 'fig')

# --- 헬퍼 함수 ---
def format_int_filter(value):
    try: return f"{int(value):,}"
    except (ValueError, TypeError): return value

def load_json_safe(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # print(f"[WARN] JSON file not found: {path}") # 로깅 최소화
        return default
    except json.JSONDecodeError:
        print(f"[WARN] Error decoding JSON from: {path}")
        return default
    except Exception as e:
        print(f"[WARN] Error loading JSON {path}: {e}")
        return default

def safe_read_csv(path, **kwargs):
    try:
        if os.path.exists(path):
            return pd.read_csv(path, **kwargs)
        else:
            # print(f"[WARN] CSV file not found: {path}") # 로깅 최소화
            return pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Error reading CSV {path}: {e}")
        return pd.DataFrame()

def get_relative_image_path(image_name):
    # HTML 파일은 outputs/daily/YYYY-MM-DD/HHMM-KST/ 에 생성되므로,
    # fig 폴더까지의 상대 경로는 ../../../fig/ 가 됩니다.
    # 하지만 워크플로우에서 복사 후에는 같은 시간 폴더 내 fig에 있을 것이므로 fig/ 로 유지합니다.
    return f"fig/{image_name}"

# --- ▼▼▼ summarize_text_with_hf 함수 추가 ▼▼▼ ---
summarizer = None # 전역 변수로 선언

def summarize_text_with_hf(text: str) -> str:
    """허깅페이스 모델을 사용해 기사 본문을 자연스러운 요약문으로 생성합니다 (num_beams 추가)."""
    global summarizer

    if summarizer is None:
        print("[INFO] Initializing Hugging Face summarization model (may take a moment)...")
        try:
            # 모델 로드 시 device=-1 로 CPU 사용 명시 (GPU 없을 경우 대비)
            summarizer = pipeline('summarization', model='gogamza/kobart-summarization', device=-1)
            print("[INFO] Summarization model initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize summarization model: {e}")
            return "> 요약 모델 초기화 실패."

    if not text or len(text.split()) < 30: # 요약 최소 길이 조정 (예: 30단어)
        return "> 기사 본문이 너무 짧아 요약할 수 없습니다."

    try:
        # 요약 최대/최소 길이 조정
        summary_result = summarizer(text, max_length=130, min_length=30, do_sample=False, num_beams=10)

        if summary_result and isinstance(summary_result, list) and 'summary_text' in summary_result[0]:
            summary = summary_result[0]['summary_text']
            if summary:
                return f"{summary.strip()}" # ">" 제거하고 반환
            else:
                return "요약문 생성에 실패했습니다. 제목을 클릭하여 기사 전문을 확인할 수 있습니다."
        else:
            return "요약 생성 실패 (모델 결과 형식 오류). 제목을 클릭하여 기사 전문을 확인할 수 있습니다."
    except Exception as e:
        print(f"[WARN] Hugging Face summarization failed for a text: {e}")
        error_detail = str(e)
        # 메모리 부족 오류 감지 (예시, 실제 오류 메시지 확인 필요)
        if "out of memory" in error_detail.lower():
             return "요약 생성 실패 (메모리 부족). 제목을 클릭하여 기사 전문을 확인할 수 있습니다."
        return f"로컬 요약 모델 실행 중 오류 발생: 제목을 클릭하여 기사 전문을 확인할 수 있습니다."
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    
# --- ▼▼▼ LLM 호출 함수들 추가 ▼▼▼ ---

def _call_gemini_safe(prompt: str, default_resp: str = "AI 분석 실패") -> str:
    """Gemini 호출을 안전하게 감싸는 내부 함수"""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "LLM API 키 없음."

        genai.configure(api_key=api_key)
        cfg = load_config() # config 로드
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        # print(f"[DEBUG] Using Gemini model: {model_name}") # 필요시 디버깅 로그
        model = genai.GenerativeModel(model_name)

        # 타임아웃 설정 추가 (예: 60초)
        request_options = {"timeout": 60}
        response = model.generate_content(prompt, request_options=request_options)

        # 응답 유효성 검사 강화
        if response and hasattr(response, 'text') and response.text:
             return response.text.strip().replace("\n", " ")
        elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             # 안전 설정 등으로 차단된 경우
             print(f"[WARN] Gemini request blocked: {response.prompt_feedback.block_reason}")
             return f"AI 분석 실패 (콘텐츠 안전 문제로 차단됨: {response.prompt_feedback.block_reason})"
        else:
             # 빈 응답 또는 예기치 않은 응답 형식
             print(f"[WARN] Gemini returned empty or unexpected response.")
             return default_resp

    except Exception as e:
        # 오류 로깅 개선
        print(f"[ERROR] Gemini API call failed: {e.__class__.__name__}: {e}")
        # traceback.print_exc() # 상세 오류 필요시 주석 해제
        # 네트워크 오류, 타임아웃 등 특정 오류 처리
        if "Timeout" in str(e):
             return "AI 분석 실패 (응답 시간 초과)"
        elif "API key not valid" in str(e):
             return "AI 분석 실패 (API 키 오류)"
        return f"AI 분석 실패 ({e.__class__.__name__})"


def call_gemini_for_spike_analysis(spike_info: str, article_titles: list) -> str:
    """LLM 호출: 스파이크 발생 원인 분석"""
    if not article_titles:
        return "분석 근거가 되는 당일 기사가 없음."

    titles_str = "- " + "\n- ".join(article_titles)
    prompt = f"""
    당신은 데이터 분석가입니다. 아래 포착된 시장 활동량 이상 급등 현상(Spike)에 대해,
    함께 제공된 관련일 뉴스 헤드라인들을 근거로 가장 유력한 **핵심 원인**을 **한 문장**으로 명확하게 분석해주세요.
    ### 포착된 이상 징후:
    {spike_info}
    ### 관련일 주요 뉴스 헤드라인:
    {titles_str}
    ### 분석 결과 (핵심 원인 한 문장):
    """
    return _call_gemini_safe(prompt, default_resp="스파이크 원인 분석 실패")

def call_gemini_for_signal_commentary(signal_term: str, article_titles: list, z_like_score: float, diff: int) -> dict:
    """LLM 호출: 급등 신호의 (1)원인과 (2)전략적 의미를 JSON으로 반환"""
    if not article_titles:
        return {"commentary": "관련 기사 없음.", "interpretation": "분석 불가"}

    titles_str = "- " + "\n- ".join(article_titles)

    prompt = f"""
    당신은 디스플레이 패널 및 인접 산업의 애널리스트입니다. 오늘 '{signal_term}' 키워드가 급증했습니다.
    (관련 지표: z_like={z_like_score:.2f}, 전일대비={diff})

    ### 관련 기사 헤드라인:
    {titles_str}

    ### 요청:
    1.  **commentary**: 급증한 핵심 이유를 **50자 내외 한 문장**으로 요약해주세요.
    2.  **interpretation**: 이 급등이 의미하는 바를 15자 내외로 간결하게 분석해주세요.

    ### 출력 (반드시 JSON 형식만, 다른 설명 절대 금지):
    {{
      "commentary": "...",
      "interpretation": "..."
    }}
    """

    raw_response = _call_gemini_safe(prompt, default_resp="{}")
    try:
        # LLM 응답에서 JSON만 파싱
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
        else:
            parsed_json = json.loads(raw_response) # JSON만 반환했을 경우

        # 두 키가 모두 존재하도록 보장
        parsed_json.setdefault("commentary", "AI 요약 실패")
        parsed_json.setdefault("interpretation", "AI 해석 실패")
        return parsed_json
    except json.JSONDecodeError:
        return {"commentary": "JSON 디코딩 실패", "interpretation": "JSON 디코딩 실패"}

def call_gemini_for_event_analysis(event_title: str, event_type: str) -> dict:
    """LLM 호출: 경쟁사 이벤트의 5가지 핵심 요소(회사명, 요약, 사실, 영향, 대응)를 JSON으로 반환"""

    prompt = f"""
    당신은 경쟁 전략 분석가입니다. 경쟁사가 '{event_title}'({event_type}) 이벤트를 실행했습니다.
    이 이벤트에 대해 아래 5가지 항목을 추출하고 분석하여 **반드시 JSON 형식으로만** 응답해주세요.

    1.  `company_name`: 이벤트의 주체(회사 이름).
    2.  `summary_title`: 이벤트를 대표하는 10단어 이내의 핵심 요약 제목.
    3.  `fact_summary`: 이벤트의 핵심 사실(Fact) 2문장 요약.
    4.  `impact`: 이 이벤트가 우리 회사에 미칠 잠재적 영향 (1문장).
    5.  `next_step`: 우리가 고려해야 할 초기 대응 방향 (1문장).

    ### 분석 결과 (JSON 형식):
    ```json
    {{
      "company_name": "...",
      "summary_title": "...",
      "fact_summary": "...",
      "impact": "...",
      "next_step": "..."
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
    parsed_json.setdefault("company_name", "분석 실패")
    parsed_json.setdefault("summary_title", event_title) # 실패 시 원본 제목 사용
    parsed_json.setdefault("fact_summary", "AI 사실 요약 실패")
    parsed_json.setdefault("impact", "AI 영향 분석 실패")
    parsed_json.setdefault("next_step", "AI 대응 분석 실패")

    return parsed_json


def call_gemini_for_risk_commentary(topic_name, sentiment_drop, related_titles):
    """LLM 호출: 감성 급락 원인 및 영향 분석 (하나의 문단으로 통합)"""
    if not related_titles: related_titles = ["관련 기사 제목 없음"]
    titles_str = "- " + "\n- ".join(related_titles)

    prompt = f"""
    디스플레이 및 연관 산업의 리스크 분석가로서, '{topic_name}' 토픽의 감성 점수가 오늘 {sentiment_drop:.2f} 만큼 급락했습니다.
    아래 참고 기사 제목을 바탕으로, **급락 원인과 잠재적 영향을 2~3문장으로 간결하게** 요약 분석해주세요
    근거 기반의 추정은 가능하지만, 절대 데이터가 부족하다거나 혹은 분석이 어렵다는 말은 하지 마세요.

    ### 관련 기사 제목:
    {titles_str}

    ### 종합 분석 (2-3 문장):
    """

    # LLM으로부터 직접 텍스트를 받습니다.
    result_text = _call_gemini_safe(prompt, default_resp="AI 리스크 분석에 실패했습니다.")

    # LLM 응답에 포함될 수 있는 불필요한 마크다운을 제거하고 공백을 정리합니다.
    commentary = result_text.replace("\n", " ").replace("*", "").replace("-", "").strip()

    return commentary
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


# --- 데이터 로딩 및 가공 함수 ---
def prepare_report_data():
    """HTML 템플릿에 필요한 데이터를 로드하고 가공하는 함수"""
    print("[INFO] Loading data for HTML report...")
    data = {}
    today_dt = now_kst() # KST 사용
    today_str = today_dt.strftime("%Y-%m-%d")
    weekday_ko = ['월', '화', '수', '목', '금', '토', '일'][today_dt.weekday()]

    # --- ▼▼▼ 모든 데이터 로딩을 이 함수 시작 부분으로 이동 ▼▼▼ ---
    df_ratios = safe_read_csv(os.path.join(EXPORT_DIR, 'daily_article_ratios.csv'))
    ts_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, 'trend_timeseries.json'), {"daily": []}) # <-- 1. 추가
    df_ts = pd.DataFrame(ts_data.get("daily", [])) # <-- 2. 추가
    df_spikes = safe_read_csv(os.path.join(EXPORT_DIR, 'timeseries_spikes_enhanced.csv'))
    df_hot = safe_read_csv(os.path.join(EXPORT_DIR, 'daily_hot_signals.csv'))
    df_events = safe_read_csv(os.path.join(EXPORT_DIR, 'events.csv'))
    df_top_articles = safe_read_csv(os.path.join(EXPORT_DIR, 'today_article_list.csv'))
    df_sentiment = safe_read_csv(os.path.join(EXPORT_DIR, "daily_topic_sentiment.csv")) # 감성 데이터

    # 메타 파일 로드 (최신 파일)
    latest_meta_file = latest(os.path.join(ROOT_DIR, "data", "news_meta_*.json"))
    meta_items = load_json_safe(latest_meta_file, [])
    url_to_meta = {item.get("url"): item for item in meta_items if isinstance(item, dict)}

    # 오늘 기사 제목 리스트 (LLM 컨텍스트용)
    today_article_titles = [
         item.get("title", "") for item in meta_items
         if isinstance(item, dict) and item.get("title") and to_date(item.get("pubDate_raw", "")) == today_str
    ][:10] # 최대 10개
    master_topics = load_json_safe(os.path.join(ROOT_DIR, "data/dictionaries/master_topics.json"), {})


    # 1. 기본 정보
    data['report_date'] = today_dt.strftime(f"%Y.%m.%d ({weekday_ko})")
    data['report_date_short'] = today_str

    # 2. 시장 활동량 데이터 (trend_timeseries.json 기준으로 수정)
    article_count = 0
    article_wow_change = "N/A"
    wow_change_class = "change-neutral"
    article_z_score = "< ±2σ"# <-- 이렇게 변경
    latest_data_date_str = today_str

    if not df_ts.empty:
        # 날짜 기준으로 정렬
        df_ts['date_dt'] = pd.to_datetime(df_ts['date'])
        df_ts = df_ts.sort_values(by='date_dt')

        # trend_timeseries.json의 *가장 마지막* 데이터(D-1)를 가져옴
        latest_data = df_ts.iloc[-1]
        article_count = latest_data['count']
        latest_data_dt = latest_data['date_dt']
        latest_data_date_str = latest_data['date'] # 실제 데이터 날짜

        # WoW 계산 (D-1일 기준 7일 전)
        seven_days_ago_dt = latest_data_dt - pd.Timedelta(days=7)
        seven_days_ago_str = seven_days_ago_dt.strftime("%Y-%m-%d")
        past_ts_data = df_ts[df_ts['date'] == seven_days_ago_str]

        if not past_ts_data.empty:
            past_count = past_ts_data.iloc[0]['count']
            if past_count > 0:
                change_pct = ((article_count - past_count) / past_count) * 100
                article_wow_change = f"{'+' if change_pct >= 0 else ''}{change_pct:.0f}%"
                if change_pct > 5: wow_change_class = "change-up"
                elif change_pct < -5: wow_change_class = "change-down"

    # Z-Score (스파이크) 로직: "오늘"이 아닌 "최신 데이터 날짜" 기준으로 스파이크를 찾음
    if not df_spikes.empty:
        today_spike = df_spikes[(df_spikes['date'] == latest_data_date_str) & (df_spikes['metric'] == '전체 기사량')]
        if not today_spike.empty:
            z_val = today_spike.iloc[0]['z_score']
            article_z_score = f"{z_val:+.1f}σ"

    data['latest_data_date'] = latest_data_date_str # 템플릿에 날짜 전달
    data['article_count'] = article_count
    data['article_wow_change'] = article_wow_change
    data['wow_change_class'] = wow_change_class
    data['article_z_score'] = article_z_score
    data['timeseries_image_path'] = get_relative_image_path("timeseries.png")

    # --- 스파이크 및 액션 아이템 로직 (수정본) ---
    data['spike_detected'] = False
    data['spike_message'] = ""
    data['spike_action_items'] = [] # 1. data 딕셔너리에 미리 빈 리스트로 초기화

    if not df_spikes.empty:
        today_spikes_list = df_spikes[df_spikes['date'] == latest_data_date_str].to_dict('records') 

        if today_spikes_list:
            data['spike_detected'] = True
            messages = []

            for spike in today_spikes_list:
                msg = f"{spike['metric']}에서 7일 평균 대비 {spike['z_score']:.1f} 표준편차 급등/급락({spike['value']})"
                messages.append(msg)

                # 2. data['spike_action_items'] 리스트에 직접 추가
                if spike['metric'] == '전체 기사량':
                    data['spike_action_items'].append("시장 전반의 관심이 급증했습니다. 급등 원인이 된 주요 토픽과 이벤트를 상세 분석하세요.")
                elif spike['metric'] == '신호 기사 비율':
                    data['spike_action_items'].append("핵심 도메인 관련 기사가 집중 발생했습니다. '오늘의 급등 신호' 섹션을 즉시 확인하세요.")

            data['spike_message'] = " ".join(messages)

    # 4. 오늘의 급등 신호 (Hot Signals) 및 LLM 해설
    hot_signals_list = []
    if not df_hot.empty:
        print(f"[INFO] Analyzing {len(df_hot.head(4))} hot signals...")
        for index, row in df_hot.head(4).iterrows(): # 상위 4개로 수정
            term = row.get('term', '')
            if not term: continue

            # 관련 기사 제목 찾기
            related_titles = [
                 meta.get("title", "") for meta in meta_items
                 if isinstance(meta, dict) and meta.get("title") and term.lower() in meta.get("title", "").lower()
            ][:5] # 최대 5개

            # --- ▼▼▼ 수정된 LLM 호출 ▼▼▼ ---
            z_score = row.get('z_like', 0.0)
            diff = row.get('diff', 0)

            # LLM에 z_score와 diff를 추가로 전달
            llm_result = call_gemini_for_signal_commentary(term, related_titles, z_score, diff)

            commentary = llm_result.get("commentary", "AI 요약 실패")
            interpretation = llm_result.get("interpretation", "AI 해석 실패") # <-- 새 변수
            # --- ▲▲▲ 수정 완료 ▲▲▲ ---

            # (이전 답변에서 수정한 'next_step_message' 로직...)
            next_step_message = f"'{term}' 관련 상세 기사 및 경쟁사 반응 모니터링" # 1. 기본값
            if not df_events.empty and 'org' in df_events.columns and term in df_events['org'].values:
                 next_step_message = f"주요 기업 '{term}'의 이벤트가 감지되었습니다. 4번 섹션(주요 기업 EVENTS)에서 상세 내용을 확인하세요."
            else: 
                found_topic = None
                for topic_key, keywords in master_topics.items():
                     if term in keywords:
                         found_topic = topic_key
                         break
                if found_topic:
                     next_step_message = f"핵심 토픽 '{found_topic}' 관련 신호입니다. 3번 섹션(키워드 감성분석)에서 해당 토픽의 감성 변동을 교차 확인하세요."

            hot_signals_list.append({
                "term": term,
                "z_like_display": f"{z_score:+.1f}σ",
                "commentary": commentary,
                "interpretation": interpretation, # <-- 새 필드 추가
                "cur": row.get('cur', 0),
                "diff_display": f"{'+' if diff >= 0 else ''}{diff}",
                "next_step": next_step_message 
            })
            # print(f"  Processed signal: {term}") # 진행 로깅
    data['hot_signals'] = hot_signals_list
    data['hot_signals_image_path'] = get_relative_image_path("weekly_strong_signals_barchart.png") # 임시

    # 5. 경쟁사 주요 활동 (Events) 및 LLM 분석
    event_list = []
    event_colors = { "LAUNCH": "#3b82f6", "INVEST": "#10b981", "PARTNERSHIP": "#f59e0b", "ORDER": "#8b5cf6", "CERT": "#6366f1", "REGUL": "#ef4444" }
    if not df_events.empty:
        # 1. 오늘 날짜의 이벤트만 필터링
        today_events_raw = df_events[df_events['date'] == today_str]

        # --- ▼▼▼ 2. 경쟁사 필터링 및 이름 찾기 로직 (수정) ▼▼▼ ---

        # (Helper function: 제목을 검색하여 '표시할 이름'을 반환)
        def find_target_competitor_proper(title_str):
            title_lower = str(title_str).lower()

            found_lower = None
            for comp_lower in TARGET_COMPETITORS_SET_LOWER: # 소문자 셋으로 검색
                if comp_lower in title_lower:
                    found_lower = comp_lower
                    break # 일단 하나 찾으면 중지

            if not found_lower:
                return None

            # 소문자로 찾은 이름을 원본(대소문자 포함) 리스트에서 다시 찾아서 반환
            for comp_proper in TARGET_COMPETITORS_LIST:
                if comp_proper.lower() == found_lower:
                    return comp_proper # (예: "LG디스플레이")
            return None

        print(f"[INFO] Filtering {len(today_events_raw)} total events for target competitors...")

        processed_count = 0 # 5개 제한 카운터

        # 3. 필터링된 최종 목록으로 LLM 호출 루프 실행
        for index in today_events_raw.index: # 순서 유지를 위해 index 사용
            if processed_count >= 5: # 5개 제한
                break

            row = today_events_raw.loc[index]
            event_title = row.get('title', '')

            # 필터링: 제목에서 경쟁사 이름 찾기
            found_company_name = find_target_competitor_proper(event_title)

            if not found_company_name:
                continue # 타겟 경쟁사가 아니면 스킵

            # --- 타겟 경쟁사 이벤트 감지됨 ---
            processed_count += 1
            print(f"  -> Processing event {processed_count}/5: {found_company_name}")

            event_type = str(row.get('types', 'OTHER')).split(',')[0].strip().upper()
            event_url = row.get('url', '#')

            # LLM 호출 (나머지 분석 항목 요청)
            analysis = call_gemini_for_event_analysis(event_title, event_type)

            event_list.append({
                "type": event_type,
                "date": row.get('date', ''),
                "color_code": event_colors.get(event_type, "#6b7280"),
                "url": event_url,

                # --- ▼▼▼ 4. LLM 응답 대신 우리가 찾은 이름으로 강제 적용 ▼▼▼ ---
                "company_name": found_company_name, 

                "summary_title": analysis.get("summary_title", event_title),
                "fact_summary": analysis.get("fact_summary", "N/A"),
                "impact": analysis.get("impact", "N/A"),
                "next_step": analysis.get("next_step", "N/A")
            })
    data['competitor_events'] = event_list

    # 6. 주요 기사 (Top Articles) 및 요약 (이전 단계에서 완료됨)
    article_list = []
    if not df_top_articles.empty:
        print(f"[INFO] Processing {len(df_top_articles)} top articles for summarization...")
        for index, row in df_top_articles.iterrows():
            url = row.get('url')
            meta = url_to_meta.get(url, {})
            body = meta.get("body") or meta.get("description", "")
            summary = summarize_text_with_hf(body) # 이미 요약 함수 호출
            article_list.append({
                "url": url,
                "title": row.get('title', ''),
                "source": meta.get('site_name', ''),
                "date": (meta.get('published_time') or '')[:10],
                "summary": summary
            })
            # print(f"  Summarized article {index + 1}/{len(df_top_articles)}") # 진행 로깅
    data['top_articles'] = article_list


    # 7. 리스크 신호 (Risk Signals) 및 LLM 해설
    risk_signals_list = []
    if not df_sentiment.empty and 'semantic_key' in df_sentiment.columns:
        print("[INFO] Analyzing risk signals based on sentiment drop...")
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
        
        df_sentiment_today = df_sentiment[df_sentiment['date'].dt.date == today_dt.date()] # 오늘 데이터만

        if not df_sentiment_today.empty:
            # 8일 전 날짜를 계산
            eight_days_ago = today_dt.date() - pd.Timedelta(days=7) 
            # CSV의 날짜에서도 시간(.dt.date)을 버리고 8일 전과 비교
            df_sentiment_recent = df_sentiment[df_sentiment['date'].dt.date >= eight_days_ago]

            analyzed_keys = 0
            for key, group in df_sentiment_recent.groupby('semantic_key'):
                
                if key == "Uncategorized" or len(group) < 2: continue # 최소 2일 데이터 필요

                group = group.sort_values('date')
                today_row = group[group['date'].dt.date == today_dt.date()]
                
                if today_row.empty: continue # 오늘 데이터 없으면 스킵

                past_days = group[group['date'].dt.date < today_dt.date()]

                if past_days.empty: continue # 과거 데이터 없으면 스킵

                today_score = today_row.iloc[0]['avg_sentiment']
                avg_past = past_days['avg_sentiment'].mean()
                std_past = past_days['avg_sentiment'].std()

                threshold_std = avg_past - 1.5 * std_past if pd.notna(std_past) and std_past > 0 else avg_past * 0.9
                threshold_pct = avg_past * 0.8
                
                is_risky = pd.notna(today_score) and pd.notna(avg_past) and \
                           (today_score < threshold_std or today_score < threshold_pct)

                if is_risky:
                    analyzed_keys += 1
                    sentiment_drop = avg_past - today_score
                    topic_keywords = master_topics.get(key, [])
                    
                    related_titles = []
                    if topic_keywords:
                         kw_pattern = re.compile('|'.join(re.escape(kw) for kw in topic_keywords), re.IGNORECASE)
                         related_titles = [
                             item.get("title", "") for item in meta_items
                             if isinstance(item, dict) and item.get("title") and kw_pattern.search(item.get("title", ""))
                         ][:5] 

                    llm_comment = call_gemini_for_risk_commentary(key, sentiment_drop, related_titles)

                    risk_signals_list.append({
                        "topic": key,
                        "drop_display": f"{sentiment_drop:.2f}",
                        "commentary": llm_comment, # LLM 결과 사용
                        "score_today": f"{today_score:.2f}",
                        "score_avg_7d": f"{avg_past:.2f}",
                        "next_step": f"'{key}' 토픽 관련 부정 이슈 및 고객사 영향 점검" # 예시
                    })
            print(f"  Analyzed {analyzed_keys} risk signals.")
    data['risk_signals'] = sorted(risk_signals_list, key=lambda x: float(x.get('drop_display', 0)), reverse=True) # 하락폭 큰 순 정렬

    # 8. Footer 정보
    data['contact_email'] = 'intelligence@company.com'
    data['dashboard_link'] = '#'
    data['settings_link'] = '#'
    data['unsubscribe_link'] = '#'

    print("[INFO] Data preparation complete.")
    return data


# --- Jinja2 템플릿 렌더링 함수 ---
# ... (render_html_report 함수는 동일) ...
def render_html_report(template_dir, template_name, data):
    print(f"[INFO] Rendering HTML template: {template_name}")
    try:
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape(['html', 'xml']))
        env.filters['format_int'] = format_int_filter
        template = env.get_template(template_name)
        html_content = template.render(data)
        print("[INFO] HTML rendering successful.")
        return html_content
    except Exception as e:
        print(f"[ERROR] HTML template rendering failed: {e}")
        return f"<html><body><h1>Report Generation Failed</h1><p>{e}</p></body></html>"
    

# --- 메인 실행 로직 ---
def main():
    start_time = now_kst() # KST 사용
    print(f"[INFO] Starting daily HTML report generation at {start_time.strftime('%Y-%m-%d %H:%M:%S KST')}")

    # 1. 데이터 준비
    report_data = prepare_report_data()

    # 2. HTML 렌더링
    html_output = render_html_report(TEMPLATE_DIR, TEMPLATE_NAME, report_data)

    output_html_path = os.path.join(OUTPUT_BASE_DIR, 'daily_report.html') # outputs/ 바로 아래에 저장
    # 3. HTML 파일 저장
    try:
        # outputs 폴더가 없으면 생성
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        print(f"[SUCCESS] Daily HTML report saved to: {output_html_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save HTML report: {e}")

    end_time = now_kst() # KST 사용
    print(f"[INFO] Report generation finished at {end_time.strftime('%Y-%m-%d %H:%M:%S KST')}. Duration: {end_time - start_time}")

if __name__ == '__main__':
    main()