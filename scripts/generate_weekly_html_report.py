# 파일 경로: scripts/generate_weekly_html_report.py

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader, select_autoescape
import re
from collections import defaultdict, Counter # Counter 추가
import time # time 모듈 추가

# --- 설정 (경로는 실제 프로젝트 구조에 맞게 조정 필요) ---
# ... (기존 설정 유지) ...
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
TEMPLATE_NAME = 'weekly_report_template.html'
OUTPUT_BASE_DIR = os.path.join(ROOT_DIR, 'outputs')
EXPORT_DIR = os.path.join(OUTPUT_BASE_DIR, 'export')
FIG_DIR = os.path.join(OUTPUT_BASE_DIR, 'fig')
DEBUG_DIR = os.path.join(OUTPUT_BASE_DIR, 'debug')
TARGET_COMPETITORS = ["삼성디스플레이", "LG디스플레이", "BOE", "CSOT", "Visionox", "Tianma"] # 경쟁사 목록 예시

# --- 필요한 헬퍼 함수 ---
from src.utils import load_json, latest
from src.timeutil import now_kst
from src.config import load_config

# Jinja2 필터 등
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
    return f"fig/{image_name}"

# --- ▼▼▼ LLM 호출 함수 (주간 요약) ▼▼▼ ---
def _call_gemini_safe(prompt: str, default_resp: str = "AI 분석 실패") -> str:
    """Gemini 호출을 안전하게 감싸는 내부 함수 (지연 시간 추가)"""
    try:
        # --- ▼▼▼ 지연 시간 추가 (예: 4.1초) ▼▼▼ ---
        # 분당 15회 제한이므로, 호출당 60/15 = 4초 이상 간격 필요
        print("[INFO] Waiting before Gemini API call to respect rate limits...")
        time.sleep(4.1) # 4초보다 약간 길게 설정
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        import google.generativeai as genai
        # ... (기존 API 키 확인 및 모델 설정 로직) ...
        api_key = os.getenv("GEMINI_API_KEY") # ...
        if not api_key: return "LLM API 키 없음." # ...
        genai.configure(api_key=api_key) # ...
        cfg = load_config() # ...
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001") # ...
        model = genai.GenerativeModel(model_name) # ...

        request_options = {"timeout": 90}
        response = model.generate_content(prompt, request_options=request_options)
        # ... (기존 응답 처리 및 오류 처리 로직) ...
        if response and hasattr(response, 'text') and response.text: # ...
             text = response.text.strip() # ...
             text = re.sub(r"^```[\w]*\n", "", text) # ...
             text = re.sub(r"\n```$", "", text) # ...
             return text.strip() # ...
        # ... (콘텐츠 차단 등 다른 오류 처리) ...
        elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: # ...
             print(f"[WARN] Gemini request blocked: {response.prompt_feedback.block_reason}") # ...
             return f"AI 분석 실패 (콘텐츠 차단: {response.prompt_feedback.block_reason})" # ...
        else: # ...
             print(f"[WARN] Gemini returned empty or unexpected response.") # ...
             return default_resp # ...

    except Exception as e:
        # ... (기존 오류 처리 로직) ...
        print(f"[ERROR] Gemini API call failed: {e.__class__.__name__}: {e}") # ...
        if "Timeout" in str(e): return "AI 분석 실패 (응답 시간 초과)" # ...
        elif "API key not valid" in str(e): return "AI 분석 실패 (API 키 오류)" # ...
        # --- ▼▼▼ 429 오류 메시지 추가 ▼▼▼ ---
        elif "429" in str(e) and "ResourceExhausted" in str(e):
             print("[WARN] Gemini API rate limit likely exceeded.")
             return "AI 분석 실패 (API 요청 한도 초과)"
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
        return f"AI 분석 실패 ({e.__class__.__name__})"

def call_gemini_for_weekly_summary(context):
    """LLM을 호출하여 주간 경영 요약을 생성합니다."""
    prompt = f"""
    당신은 디스플레이 산업 전문 수석 비즈니스 분석가입니다.
    아래는 지난 한 주간의 시장 데이터 요약입니다. 이 데이터를 종합하여 경영진 및 팀 리더를 위한 '주간 인텔리전스 요약'을 작성해주세요.
    ### 주간 데이터 요약:
    {json.dumps(context, ensure_ascii=False, indent=2)}

    ### 작성 가이드 (Markdown 형식):
    1. **핵심 맥락**: 데이터를 관통하는 가장 중요한 시장의 흐름 1~2가지를 설명해주세요.
    2. **전략적 인사이트**: 이 흐름이 우리 비즈니스에 주는 기회 또는 위협 요소를 분석해주세요.
    3. **추천 Action Items**: 다음 주에 팀이 우선적으로 실행해야 할 구체적인 액션 아이템 2가지를 제안해주세요.
    4. 각 항목을 명확하게 구분하고, 전문가의 시각에서 간결하고 명확한 톤으로 작성해주세요.
    ### 출력 형식 (Markdown):
    #### 핵심 맥락
    - (분석 내용)

    #### 전략적 인사이트
    - (분석 내용)

    #### 추천 Action Items
    - (실행 제안 1)
    - (실행 제안 2)
    """
    return _call_gemini_safe(prompt, default_resp="주간 AI 요약 생성 실패")
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- ▼▼▼ LLM 호출 함수 (주간 테마 설명 - 신규 추가) ▼▼▼ ---
def call_gemini_for_theme_description(theme_name, keywords):
    """LLM을 호출하여 시장 테마에 대한 설명을 생성합니다."""
    keywords_str = ", ".join(keywords)
    prompt = f"""
    시장 분석가로서, '{theme_name}'라는 주간 시장 테마가 포착되었습니다.
    이 테마와 관련된 주요 키워드는 [{keywords_str}] 입니다.
    이 테마가 현재 시장에서 어떤 의미를 가지는지 **1~2 문장**으로 간결하게 설명해주세요.
    ### 테마 설명 (1~2 문장):
    """
    # _call_gemini_safe 함수는 이미 정의되어 있다고 가정
    return _call_gemini_safe(prompt, default_resp=f"'{theme_name}' 테마에 대한 AI 설명 생성 실패.")
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- ▼▼▼ LLM 호출 함수 (경쟁사 분석 - 신규 추가) ▼▼▼ ---
def call_gemini_for_competitor_insight(competitor_name, mentions, momentum):
    """LLM을 호출하여 경쟁사의 주간 활동에 대한 해설을 생성합니다."""
    prompt = f"""
    시장 분석가로서, 경쟁사 '{competitor_name}'의 지난 주 활동 데이터는 다음과 같습니다:
    - 주간 총 언급량: {mentions}
    - 주간 평균 모멘텀 (z-like): {momentum:.2f}

    이 데이터를 바탕으로 '{competitor_name}'의 **이번 주 핵심 활동**과 **주목해야 할 전략적 움직임**을 **한 문장**으로 간결하게 요약해주세요.
    ### 분석 요약 (한 문장):
    """
    # _call_gemini_safe 함수는 이미 정의되어 있다고 가정
    return _call_gemini_safe(prompt, default_resp=f"'{competitor_name}' 관련 AI 분석 실패.")
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- ▼▼▼ LLM 호출 함수 (주간 약한 신호 분석 - 신규 추가) ▼▼▼ ---
def call_gemini_for_weekly_insight(weak_signals):
    """LLM을 호출하여 주간 약한 신호의 의미를 분석합니다."""
    if not weak_signals:
        return "금주에 주목할 만한 신규 약한 신호가 포착되지 않았습니다."

    # LLM 프롬프트에 전달할 데이터 형식 변경 (dict 리스트)
    signals_context = [{"signal": s.get('term'), "momentum": s.get('z_like'), "mentions": s.get('total')} for s in weak_signals]

    prompt = f"""
    당신은 미래 기술 트렌드 분석가입니다. 아래는 지난 한 주간 포착된 초기 신호(Weak Signals) 목록입니다.
    ### 주간 초기 신호 목록 (상위):
    {json.dumps(signals_context, ensure_ascii=False, indent=2)}

    ### 분석 요청:
    1. 목록에서 가장 중요하고 잠재력 있는 신호 **2~3개**를 선별해주세요.
    2. 각 신호가 **무엇을 의미**하는지, 그리고 **왜 지금 주목**해야 하는지에 대한 **해석**을 한 문장으로 요약해주세요.
    3. 각 신호의 **잠재적 영향**(기회 또는 위협)을 한 문장으로 설명해주세요.
    4. 각 신호에 대한 **초기 검증 액션**을 한 문장으로 제안해주세요.
    5. 분석 결과를 아래 **마크다운 형식**으로만 답변해주세요. 다른 설명은 필요 없습니다.
    ### 출력 형식 (Markdown):
    - **[신호명 1]:**
        - **해석:** (분석 및 해석 요약)
        - **잠재 영향:** (기회/위협 설명)
        - **검증 액션:** (실행 제안)
    - **[신호명 2]:**
        - **해석:** (분석 및 해석 요약)
        - **잠재 영향:** (기회/위협 설명)
        - **검증 액션:** (실행 제안)
    """
    # _call_gemini_safe 함수는 이미 정의되어 있다고 가정
    llm_response = _call_gemini_safe(prompt, default_resp="약한 신호 AI 분석 실패.")

    # LLM 응답 파싱 (Markdown 리스트 형식 가정)
    parsed_insights = {}
    try:
        # 각 신호별 블록 분리 (신호명을 키로 사용)
        signal_blocks = re.split(r'-\s+\*\*(.*?):\*\*', llm_response)
        if len(signal_blocks) > 1:
            for i in range(1, len(signal_blocks), 2):
                signal_name = signal_blocks[i].strip()
                details_text = signal_blocks[i+1].strip()
                details = {}
                # 각 항목 (해석, 잠재 영향, 검증 액션) 추출
                desc_match = re.search(r'-\s+\*\*해석:\*\*\s+(.*)', details_text)
                imp_match = re.search(r'-\s+\*\*잠재 영향:\*\*\s+(.*)', details_text)
                act_match = re.search(r'-\s+\*\*검증 액션:\*\*\s+(.*)', details_text)
                if desc_match: details['description'] = desc_match.group(1).strip()
                if imp_match: details['implication'] = imp_match.group(1).strip()
                if act_match: details['next_step'] = act_match.group(1).strip()
                parsed_insights[signal_name] = details
    except Exception as e:
        print(f"[WARN] Failed to parse LLM response for weak signals: {e}")

    return parsed_insights # 파싱된 결과를 딕셔너리로 반환

# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


# --- ▼▼▼ LLM 호출 함수 (모멘텀 분석 권고 - 프롬프트 수정) ▼▼▼ ---
def call_gemini_for_momentum_recommendation(term, z_like_score, change_display):
    """LLM을 호출하여 모멘텀 변화에 대한 권고사항을 생성합니다."""
    prompt = f"""
    디스플레이 제조 기업의 시장 분석가로서, '{term}' 키워드의 주간 모멘텀 변화에 대한 **실행 중심의 권고사항**을 작성해주세요.
    - 주간 평균 모멘텀 (z-like): {z_like_score:.2f}
    - 주간 변화량: {change_display}

    **권고사항은 다음 주에 실행할 구체적인 행동 1가지를 명사형 어구로 간결하게, **절대로 다른 설명 없이** 제안**해주세요. (예: '경쟁사 가격 정책 변화 심층 분석', '신규 라인업 적용 가능성 검토 착수', '관련 기술 특허 동향 재확인') **Markdown(`**` 등) 서식은 사용하지 마세요.**

    ### 권고사항 (핵심 액션 1가지, 명사형 어구, 설명/서식 절대 금지):
    """
    recommendation_raw = _call_gemini_safe(prompt, default_resp=f"'{term}' 관련 AI 권고 생성 실패.")

    # --- ▼▼▼ 후처리: Markdown 볼드 제거 및 첫 줄만 사용 ▼▼▼ ---
    # 1. 양 끝 공백 제거
    recommendation_clean = recommendation_raw.strip()
    # 2. Markdown 볼드(**) 제거
    recommendation_clean = recommendation_clean.replace("**", "")
    # 3. 여러 줄 응답일 경우 첫 줄만 사용 (혹은 첫 문장)
    recommendation_clean = recommendation_clean.split('\n')[0].strip()
    # 4. 혹시 모를 리스트 마커(-) 제거
    if recommendation_clean.startswith('- '):
        recommendation_clean = recommendation_clean[2:]

    return recommendation_clean
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def call_gemini_for_portfolio_actions(momentum_summary):
    """LLM을 호출하여 모멘텀 분석 기반 포트폴리오/전략 조정 권고를 생성합니다."""
    prompt = f"""
    당신은 전략 기획 담당자입니다. 아래는 이번 주 주요 키워드들의 모멘텀 변화 요약입니다.
    ### 주간 모멘텀 요약:
    {momentum_summary}

    이 정보를 바탕으로, 다음 분기 **기술/제품 포트폴리오** 관점에서 고려해야 할 **전략적 시사점** 또는 **조정 방향**에 대한 권고 사항 **2가지**를 제안해주세요. (각 제안은 한 문장)
    ### 포트폴리오 권고 (2가지):
    - (제안 1)
    - (제안 2)
    """
    llm_response = _call_gemini_safe(prompt, default_resp="포트폴리오 권고 생성 실패.")
    # LLM 응답 파싱 (Markdown 리스트 형식 가정)
    actions = []
    try:
        lines = llm_response.split('\n')
        for line in lines:
            if line.strip().startswith('-'):
                actions.append(line.strip()[1:].strip())
    except Exception:
        pass
    return actions if actions else ["AI 권고 생성 실패"]

# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


# --- 데이터 로딩 및 가공 함수 ---
def prepare_weekly_report_data():
    """주간 HTML 템플릿에 필요한 데이터를 로드하고 가공하는 함수"""
    # ... (기존 데이터 로딩 및 Section 1-4, 5, 6 처리 로직 유지) ...
    print("[INFO] Loading and preparing data for Weekly HTML report...")
    data = {}
    end_dt = now_kst(); start_dt = end_dt - timedelta(days=6)
    data['date_range_display'] = f"{start_dt.strftime('%Y.%m.%d')} - {end_dt.strftime('%Y.%m.%d')}"

    # 1. 주간 집계 데이터 로드
    keywords_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, "keywords.json"), {"keywords": []})
    events_df = safe_read_csv(os.path.join(EXPORT_DIR, "events.csv"))
    weak_signals_df = safe_read_csv(os.path.join(EXPORT_DIR, "weak_signals.csv"))
    trends_df = safe_read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv")) # 주간 통합본
    weekly_meta = load_json_safe(os.path.join(DEBUG_DIR, "weekly_meta_agg.json"), [])
    clusters_df = safe_read_csv(os.path.join(EXPORT_DIR, "keyword_clusters.csv"))
    topics_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, "topics.json"), {"topics": []})
    growth_df = safe_read_csv(os.path.join(EXPORT_DIR, "topic_growth.csv"))

    # 2. Executive Summary & Key Insights (기존 처리)
    # ... (LLM 요약, Key Insight 계산 등) ...
    top_keywords_list = [k.get('keyword') for k in keywords_data.get('keywords', [])[:5]]
    competitor_mentions_counter = Counter() # ... (경쟁사 언급량 계산) ...
    if not trends_df.empty:
        for competitor in TARGET_COMPETITORS:
            total_mentions = trends_df[trends_df['term'] == competitor]['cur'].sum()
            if total_mentions > 0: competitor_mentions_counter[competitor] = total_mentions
    top_competitors_list = [c for c, _ in competitor_mentions_counter.most_common(3)]
    top_weak_signals_list = weak_signals_df.sort_values(by='z_like', ascending=False)['term'].head(3).tolist() if not weak_signals_df.empty else []
    summary_context = { # ... 컨텍스트 구성 ...
        "분석 기간": data['date_range_display'], "주간 Top 키워드": top_keywords_list,
        "주간 활동량 Top 경쟁사": top_competitors_list, "주목할 만한 약한 신호": top_weak_signals_list
    }
    llm_summary_markdown = call_gemini_for_weekly_summary(summary_context)
    data['executive_summary'] = llm_summary_markdown
    data['total_articles'] = len(weekly_meta); data['article_change_info'] = "..."
    data['top_keywords_count'] = len(keywords_data.get('keywords', [])); data['top_keywords_preview'] = ", ".join(top_keywords_list) + " 등" if top_keywords_list else "..."
    unique_events = events_df.drop_duplicates(subset=['title']) if not events_df.empty else pd.DataFrame()
    data['total_events'] = len(unique_events)
    event_type_counts = unique_events['types'].str.split(',').explode().str.strip().value_counts() if not unique_events.empty else pd.Series()
    data['event_type_summary'] = ", ".join([f"{idx}: {val}건" for idx, val in event_type_counts.items()]) if not event_type_counts.empty else "..."
    # ... (Action Items 추출 로직) ...
    recommended_actions_list = [] # Placeholder 수정 필요
    data['recommended_actions'] = recommended_actions_list

    # 4. 시장 테마 데이터 (기존 처리)
    market_themes_list = []
    # ... (토픽 데이터 가공 및 LLM 설명 호출 로직) ...
    processed_topics = topics_data.get("topics", [])
    growth_map = {} # ... (growth_map 생성) ...
    if not growth_df.empty and 'topic_id' in growth_df.columns and 'momentum_score' in growth_df.columns:
        growth_map = growth_df.set_index('topic_id')['momentum_score'].to_dict()
    for topic in processed_topics[:4]: # ... (테마 정보 생성 및 LLM 설명 호출) ...
        topic_id = topic.get("topic_id"); theme_name = topic.get("topic_name", f"Topic #{topic_id}")
        top_words = [w.get('word') for w in topic.get('top_words', [])[:4]]
        momentum = growth_map.get(topic_id, 0.0); badge_text = "안정"; badge_color = "#718096"
        if momentum > 0.5: badge_text = "급상승"; badge_color = "#38a169"
        elif momentum > 0.1: badge_text = "상승"; badge_color = "#3b82f6"
        elif momentum < -0.3: badge_text = "하락"; badge_color = "#e53e3e"
        llm_description = call_gemini_for_theme_description(theme_name, top_words)
        market_themes_list.append({ "name": theme_name, "badge_text": badge_text, "badge_color": badge_color, "tags": top_words, "description": llm_description })
    data['market_themes'] = market_themes_list
    data['wordcloud_image_path'] = get_relative_image_path("weekly_wordcloud.png")
    data['weekly_topic_barchart_path'] = get_relative_image_path("weekly_topics_barchart.png") # <-- 이 줄 추가
    data['keyword_network_image_path'] = get_relative_image_path("keyword_network.png")
    data['strong_signals_bar_image_path'] = get_relative_image_path("weekly_strong_signals_barchart.png")

    # --- ▼▼▼ 5. 경쟁사 분석 데이터 (Placeholder 채우기) ▼▼▼ ---
    competitor_analysis_list = []
    trend_classes = {"up": "trend-up", "down": "trend-down", "stable": "trend-stable"}
    trend_icons = {"up": "↑", "down": "↓", "stable": "→"}

    print(f"[INFO] Analyzing {len(TARGET_COMPETITORS)} target competitors...")
    if not trends_df.empty:
        weekly_momentum = trends_df.groupby('term')['z_like'].mean()
        # --- ▼▼▼ 추가 계산 ▼▼▼ ---
        weekly_peak_momentum = trends_df.groupby('term')['z_like'].max() # 주간 최고 z_like
        # 30일 누적 언급량 계산 (weekly_trend_details.csv 사용 가정, 없으면 trends_df 사용)
        trends_details_df = safe_read_csv(os.path.join(DEBUG_DIR, "weekly_trend_details.csv"))
        if not trends_details_df.empty:
             cumulative_mentions_30d = trends_details_df.groupby('term')['cur'].sum()
        else:
             # Fallback: Use weekly data if details not available (less accurate for 30d)
             cumulative_mentions_30d = trends_df.groupby('term')['cur'].sum()
        # --- ▲▲▲▲▲▲▲▲▲▲▲ ---

        for competitor in TARGET_COMPETITORS:
            mentions = competitor_mentions_counter.get(competitor, 0)
            if mentions == 0: continue

            momentum = weekly_momentum.get(competitor, 0.0)
            # --- ▼▼▼ 추가 데이터 가져오기 ▼▼▼ ---
            peak_momentum = weekly_peak_momentum.get(competitor, 0.0)
            total_30d_mentions = cumulative_mentions_30d.get(competitor, 0)
            # --- ▲▲▲▲▲▲▲▲▲▲▲ ---

            trend_key = "stable"
            if momentum > 0.3: trend_key = "up"
            elif momentum < -0.3: trend_key = "down"

            llm_insight = call_gemini_for_competitor_insight(competitor, mentions, momentum)

            competitor_analysis_list.append({
                "name": competitor,
                "trend_class": trend_classes[trend_key],
                "trend_icon": trend_icons[trend_key],
                "trend_percentage": f"{momentum:+.1f}σ",
                "mentions": mentions, # 주간 언급량
                "momentum_score": f"{momentum:+.2f}", # 주간 평균 모멘텀
                # --- ▼▼▼ 추가 데이터 추가 ▼▼▼ ---
                "peak_momentum_score": f"{peak_momentum:+.2f}", # 주간 최대 모멘텀
                "total_30d_mentions": total_30d_mentions, # 30일 누적 언급량
                # --- ▲▲▲▲▲▲▲▲▲▲▲ ---
                "insight": llm_insight
            })

        competitor_analysis_list.sort(key=lambda x: x.get('mentions', 0), reverse=True)

    data['competitor_analysis'] = competitor_analysis_list
    data['competitor_actions'] = []
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    # --- ▼▼▼ 6. 미래 신호 (Weak Signals) 데이터 (Placeholder 채우기) ▼▼▼ ---
    weak_signals_list = []
    if not weak_signals_df.empty:
        top_weak = weak_signals_df.sort_values(by="z_like", ascending=False).drop_duplicates(subset=['term']).head(5)
        weak_signals_for_llm = top_weak[['term', 'z_like', 'total']].to_dict('records')
        llm_insights_map = call_gemini_for_weekly_insight(weak_signals_for_llm)

        print(f"[INFO] Processing {len(top_weak)} weak signals...")
        for index, row in top_weak.iterrows():
            term = row.get('term')
            if not term: continue
            llm_analysis = llm_insights_map.get(term, {})
            # ... (description, implication, next_step 가져오기) ...
            description = llm_analysis.get('description', "...")
            implication = llm_analysis.get('implication', "...")
            next_step = llm_analysis.get('next_step', "...")
            z_like = row.get('z_like', 0.0)
            total_mentions = row.get('total', 0) # total 값 가져오기
            cur_mentions = row.get('cur', 0) # cur 값 가져오기 (주간 빈도 계산용)

            # ... (배지 로직) ...
            type_text = "Emerging"; type_badge_class="badge-emerging"
            confidence_text = "Low Freq"; confidence_badge_class="badge-confidence"
            if total_mentions > 20: confidence_text = "Medium Freq"
            elif total_mentions > 50: confidence_text = "High Freq"


            weak_signals_list.append({
                "term": term,
                "icon": "💡",
                "type_badge_class": type_badge_class,
                "type_text": type_text,
                "confidence_badge_class": confidence_badge_class,
                "confidence_text": confidence_text,
                "description": description,
                "implication": implication,
                # "frequency": f"주 {total_mentions}회", # total 을 사용하므로 주간 빈도는 cur 로 변경
                "frequency": f"최근 {cur_mentions}회",
                "z_like_score": f"{z_like:.1f}σ",
                "total_mentions": total_mentions, # 누적 언급량 추가
                "next_step": next_step
            })
    data['weak_signals'] = weak_signals_list
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


    # --- ▼▼▼ 7. 모멘텀 변화 데이터 (Placeholder 채우기) ▼▼▼ ---
    momentum_items_list = []
    momentum_summary_for_llm = []
    if not trends_df.empty:
        # ... (weekly_momentum, weekly_mentions, selected_terms 계산) ...
        weekly_momentum = trends_df.groupby('term')['z_like'].mean()
        weekly_mentions = trends_details_df.groupby('term')['cur'].sum() if not trends_details_df.empty else pd.Series()
        top_rising = weekly_momentum[weekly_momentum > 0].nlargest(5)
        top_falling = weekly_momentum[weekly_momentum < 0].nsmallest(5)
        selected_terms = list(top_rising.index) + list(top_falling.index)


        print(f"[INFO] Analyzing momentum for {len(selected_terms)} terms using detailed data...")
        for term in selected_terms:
            momentum = weekly_momentum.get(term, 0.0)
            change_display = "변화 없음" # ... (change_display 계산) ...
            if not trends_details_df.empty:
                 term_daily_history = trends_details_df[trends_details_df['term'] == term].sort_values('date')
                 if len(term_daily_history) >= 2:
                      # ... (change 계산) ...
                      try:
                          z_start = float(term_daily_history.iloc[0].get('z_like', 0.0))
                          z_end = float(term_daily_history.iloc[-1].get('z_like', 0.0))
                          change = z_end - z_start
                          change_display = f"{change:+.1f}σ"
                      except (ValueError, TypeError): change_display = "계산 오류"
                 elif len(term_daily_history) == 1: change_display = "신규 진입"

            # --- ▼▼▼ 추세 결정 로직 확인 ▼▼▼ ---
            trend_key = "stable"; trend_text="안정"
            # 기준값을 좀 더 명확하게 조정 (예: 상승/하락 기준 강화)
            if momentum > 0.5: # 예: 0.5 초과 시 '상승'
                trend_key = "up"; trend_text="상승"
            elif momentum < -0.3: # 예: -0.3 미만 시 '하락'
                trend_key = "down"; trend_text="하락"
            # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

            # LLM 권고 호출 (수정된 함수 사용)
            recommendation = call_gemini_for_momentum_recommendation(term, momentum, change_display)

            momentum_items_list.append({
                "term": term,
                "z_like_score": f"{momentum:+.2f}",
                "change_display": change_display,
                "trend_class": trend_classes[trend_key], # 올바른 클래스 할당 확인
                "trend_icon": trend_icons[trend_key],
                "trend_text": trend_text,
                "recommendation": recommendation # 후처리된 결과
            })
            momentum_summary_for_llm.append(f"- {term}: 모멘텀 {momentum:.2f} ({trend_text})")

        momentum_items_list.sort(key=lambda x: float(x.get('z_like_score', 0.0)), reverse=True)

    data['momentum_items'] = momentum_items_list
    data['momentum_bar_image_path'] = get_relative_image_path("topics_mini_trends.png")

    # 포트폴리오 권고 LLM 호출
    print(f"DEBUG: Portfolio context:\n{momentum_summary_for_llm}")
    portfolio_actions_list = call_gemini_for_portfolio_actions("\n".join(momentum_summary_for_llm))
    data['portfolio_actions'] = portfolio_actions_list
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


    # 8. Footer 정보 (기존 유지)
    data['dashboard_link'] = '#'; data['data_source_link'] = '#'; data['methodology_link'] = '#'; data['contact_link'] = '#'

    print("[INFO] Weekly data preparation complete.")
    return data

# --- Jinja2 템플릿 렌더링 함수 ---
# ... (render_html_report 함수는 동일) ...
def render_html_report(template_dir, template_name, data):
    # ... (내용 동일) ...
    print(f"[INFO] Rendering Weekly HTML template: {template_name}")
    try:
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape(['html', 'xml']))
        env.filters['format_int'] = format_int_filter
        env.add_extension('jinja2.ext.do')
        template = env.get_template(template_name)
        html_content = template.render(data)
        print("[INFO] Weekly HTML rendering successful.")
        return html_content
    except Exception as e:
        print(f"[ERROR] Weekly HTML template rendering failed: {e}")
        import traceback; traceback.print_exc()
        return f"<html><body><h1>Weekly Report Generation Failed</h1><pre>{e}</pre></body></html>"

# --- 메인 실행 로직 ---
# ... (main 함수는 동일) ...
def main():
    start_time = now_kst()
    print(f"[INFO] Starting weekly HTML report generation at {start_time.strftime('%Y-%m-%d %H:%M:%S KST')}")
    report_data = prepare_weekly_report_data()
    html_output = render_html_report(TEMPLATE_DIR, TEMPLATE_NAME, report_data)
    output_html_path = os.path.join(OUTPUT_BASE_DIR, 'weekly_report.html')
    try:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        with open(output_html_path, 'w', encoding='utf-8') as f: f.write(html_output)
        print(f"[SUCCESS] Weekly HTML report saved to: {output_html_path}")
    except Exception as e: print(f"[ERROR] Failed to save weekly HTML report: {e}")
    end_time = now_kst()
    print(f"[INFO] Weekly report generation finished at {end_time.strftime('%Y-%m-%d %H:%M:%S KST')}. Duration: {end_time - start_time}")

if __name__ == '__main__':
    main()