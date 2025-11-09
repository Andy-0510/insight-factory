import os
import json
import pandas as pd
from datetime import datetime, timedelta
from src.utils import load_json
from src.config import load_config
# [수정] 필요한 헬퍼 함수들을 직접 가져오거나 정의
from .daily_commentary_report import (_safe_read_csv, _to_markdown_table, _section_header, _insert_image, build_html_from_md)

# --- 설정 ---
ROOT_OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(ROOT_OUTPUT_DIR, "fig")
EXPORT_DIR = os.path.join(ROOT_OUTPUT_DIR, "export")
OUT_MD = os.path.join(ROOT_OUTPUT_DIR, "monthly_commentary_report.md")
OUT_HTML = os.path.join(ROOT_OUTPUT_DIR, "monthly_commentary_report.html")

# --- LLM 호출 함수 ---
# --- ▼▼▼ [추가] call_gemini_for_strategy_insight 함수 정의 ▼▼▼ ---
def call_gemini_for_strategy_insight(company_name, topics_str):
    """LLM을 호출하여 기업의 토픽 집중도를 기반으로 전략 방향성을 분석합니다."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "LLM API 키가 없습니다."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        당신은 B2B 기술 기업 전문 애널리스트입니다.
        '{company_name}'라는 기업이 최근 아래 토픽들에 집중하고 있습니다.
        이를 바탕으로 이 기업의 현재 사업 방향성과 단기 전략을 1~2 문장으로 간결하게 해석해주세요.
        ### 집중 토픽:
        {topics_str}

        ### 분석 결과 (1~2 문장 요약):
        """
        response = model.generate_content(prompt)
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        return f"LLM 분석 실패: {e}"
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def call_gemini_for_monthly_exec_summary(context: dict) -> str:
    """LLM을 호출하여 월간 리포트의 Executive Summary를 생성합니다."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "> LLM API 키가 없어 Executive Summary를 생성할 수 없습니다."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        당신은 최고 전략 책임자(CSO)입니다. 아래는 지난 한 달간의 시장 분석 결과 최종 요약본입니다.
        이를 바탕으로 CEO 및 이사회 보고를 위한 '월간 전략 보고서 Executive Summary'를 작성해주세요.
        지난달 시장의 핵심적인 변화, 우리에게 가장 큰 기회와 위협 요인, 그리고 다음 분기에 집중해야 할 최우선 전략 방향을 중심으로 작성해주세요.

        ### 월간 핵심 데이터 요약:
        {json.dumps(context, ensure_ascii=False, indent=2)}

        ### Executive Summary:
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"> LLM 요약 생성 실패: {e}"
    
# --- ▼▼▼ [수정] 포지셔닝 맵 해설 LLM 함수 (토픽 트렌드/성장률 컨텍스트 추가) ▼▼▼ ---
def call_gemini_for_positioning_commentary(topics_data, growth_df):
    """LLM을 호출하여 토픽 포지셔닝 맵, 트렌드, 성장률(모멘텀)을 종합적으로 분석합니다.""" # 주석 수정
    simplified_topics = [
         {"topic_name": t.get("topic_name", f"Topic {t.get('topic_id')}"), "summary": t.get("topic_summary")}
         for t in topics_data.get("topics", [])
    ]
    growth_info = "토픽 모멘텀 데이터 없음." # 이름 변경
    if growth_df is not None and not growth_df.empty:
        growth_info_list = []
        topic_map = {t.get("topic_id"): t.get("topic_name", f"Topic {t.get('topic_id')}") for t in topics_data.get("topics", [])}

        # --- ▼▼▼ [수정] 'momentum_score' 컬럼 사용 ▼▼▼ ---
        if 'topic_id' in growth_df.columns and 'momentum_score' in growth_df.columns: # 컬럼명 변경
            # topic_name 컬럼이 없을 경우 생성
            if 'topic_name' not in growth_df.columns:
                 growth_df['topic_name'] = growth_df['topic_id'].apply(lambda tid: topic_map.get(tid, f"Topic {tid}"))

            # Sort by 'momentum_score' instead of 'growth_rate'
            rising = growth_df[growth_df['momentum_score'] > 0].nlargest(3, 'momentum_score') # 컬럼명 변경
            falling = growth_df[growth_df['momentum_score'] < 0].nsmallest(3, 'momentum_score') # 컬럼명 변경
            if not rising.empty:
                # Use 'momentum_score' for display
                growth_info_list.append("- **주요 상승 토픽**: " + ", ".join([f"{row.get('topic_name', '?')} ({row.get('momentum_score', 0):.2f})" for _, row in rising.iterrows()])) # 컬럼명 변경 및 포맷 조정 (.1% -> .2f)
            if not falling.empty:
                 # Use 'momentum_score' for display
                 growth_info_list.append("- **주요 하락 토픽**: " + ", ".join([f"{row.get('topic_name', '?')} ({row.get('momentum_score', 0):.2f})" for _, row in falling.iterrows()])) # 컬럼명 변경 및 포맷 조정
            if growth_info_list: growth_info = "\n".join(growth_info_list)
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
        else:
            growth_info = "모멘텀 데이터 컬럼('topic_id', 'momentum_score') 오류." # 이름 변경


    try:
        import google.generativeai as genai
        # ... (LLM 호출 부분은 동일) ...
        api_key = os.getenv("GEMINI_API_KEY") #
        if not api_key: return "> LLM API 키 없음." #

        genai.configure(api_key=api_key) #
        cfg = load_config() #
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001") #
        model = genai.GenerativeModel(model_name) #

        prompt = f"""
        당신은 시장 전략 컨설턴트입니다.
        아래는 시장 토픽 포지셔닝(관심도/긍정성), 주요 토픽 목록, 그리고 토픽별 모멘텀 점수 데이터입니다. # 이름 변경

        ### 주요 토픽 목록 (버블맵 참고):
        {json.dumps(simplified_topics, ensure_ascii=False, indent=2)}

        ### 토픽 모멘텀 요약 (미니 트렌드 차트 참고): # 이름 변경
        {growth_info}

        ### 분석 요청:
        1. **종합 해석**: 토픽 포지셔닝 맵(버블 차트), 미니 트렌드 차트, 모멘텀 점수 데이터를 종합적으로 해석하여, 현재 시장의 **주요 동력(Driving Force)**과 **새롭게 부상하는 기회(Emerging Opportunity)** 영역이 무엇인지 설명해주세요. # 이름 변경
        2. **전략적 제언**: 분석 결과를 바탕으로, 우리 회사가 다음 분기에 **자원을 집중해야 할 토픽 영역**과 **주의 깊게 모니터링해야 할 토픽 영역**을 각각 제안해주세요.

        ### 분석 결과 (Markdown):
        #### 종합 시장 동향 해석
        - **주요 동력**: (설명)
        - **부상하는 기회**: (설명)

        #### 전략적 제언
        - **집중 영역**: (제안)
        - **모니터링 영역**: (제안)
        """ #
        response = model.generate_content(prompt) #
        return response.text.strip() #
    except Exception as e:
        return f"> LLM 해설 생성 실패: {e}"
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- ▼▼▼ [수정] 리스크 해설 LLM 함수 (시각화 컨텍스트 추가) ▼▼▼ ---
def call_gemini_for_risk_commentary_monthly(risks_df):
    """LLM을 호출하여 월간 리스크 데이터와 시각화를 종합적으로 분석합니다."""
    if risks_df is None or risks_df.empty:
        return "> 분석할 리스크 데이터가 없습니다."

    risk_summary_list = []
    for _, row in risks_df.head(5).iterrows(): # 상위 5개 리스크 정보 사용
        risk_summary_list.append(f"- **{row.get('Topic', '?')}**: {row.get('summary', '')} (영향: {row.get('impact_range', '?')})")
    risk_summary = "\n".join(risk_summary_list)

    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "> LLM API 키 없음."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        당신은 기업 리스크 관리 최고 책임자(CRO)입니다.
        지난 한 달간 탐지된 주요 리스크 목록과 관련 시각화 자료(토픽별 부정 감성 추이, 리스크 키워드 네트워크)를 검토했습니다.

        ### 주요 리스크 요약:
        {risk_summary}

        ### 분석 요청:
        위 정보와 시각화 자료(부정 감성 급락 추이, 리스크 키워드 연관성)를 종합적으로 고려하여,
        1. **가장 시급하게 대응해야 할 Top 1 리스크**는 무엇이며, 그 이유는 무엇입니까?
        2. 이 리스크에 대해 다음 달에 **즉시 실행해야 할 완화 조치 1~2가지**를 구체적으로 제안해주세요.

        ### 분석 결과 (Markdown):
        #### 월간 리스크 종합 평가
        - **최우선 관리 대상 리스크**: (선정된 리스크 및 이유)
        - **차월 실행 조치 제안**:
            - (제안 1)
            - (제안 2 - 선택적)
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"> LLM 해설 생성 실패: {e}"
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- 섹션별 컨텐츠 생성 함수 ---

# --- ▼▼▼ [수정] 월간 포지셔닝 맵 섹션 (이미지, 테이블, LLM 해설 추가) ▼▼▼ ---
def _section_monthly_positioning_map():
    lines = [_section_header("1. 전략적 시장 포지셔닝 맵")]
    lines.append("> 시장의 핵심 토픽 지형, 트렌드, 성장률을 통해 거시적인 동향과 기회 영역을 분석합니다.\n") # 문구 수정

    # 이미지 추가 (버블 + 미니 트렌드)
    lines.append(_insert_image(os.path.join(FIG_DIR, "topics_bubble.png"), "시장 토픽 포지셔닝 맵"))
    lines.append(_insert_image(os.path.join(FIG_DIR, "topics_mini_trends.png"), "주요 토픽별 주간 트렌드"))

    # 토픽 상세 테이블 (기존 로직 유지)
    topics_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "topics.json"), {})
    df_topics = pd.DataFrame(topics_data.get("topics", []))
    if not df_topics.empty:
        df_topics['top_words_str'] = df_topics['top_words'].apply(
            lambda words: ", ".join([w.get('word', '') for w in words[:3]]) if isinstance(words, list) else ""
        )
        df_topics['Topic Identifier'] = df_topics.apply(lambda row: row.get('topic_name', f"Topic #{row.get('topic_id')}"), axis=1)
        lines.append("\n### 토픽 상세 정보\n")
        lines.append(_to_markdown_table(df_topics[['Topic Identifier', 'topic_summary', 'top_words_str']].rename(columns={
            'Topic Identifier': '토픽', 'topic_summary': '요약', 'top_words_str': '핵심 키워드'
        })))
    else:
        lines.append("\n> - 토픽 상세 정보 없음\n")

    # 토픽 성장/하락 테이블 추가
    df_growth = _safe_read_csv(os.path.join(EXPORT_DIR, "topic_growth.csv"))
    if not df_growth.empty:
        lines.append("\n### 토픽 성장/하락 추세\n")
        lines.append(_to_markdown_table(df_growth))
    else:
        lines.append("\n> - 토픽 성장/하락 데이터 없음\n")

    # LLM 종합 해설 추가
    llm_commentary = call_gemini_for_positioning_commentary(topics_data, df_growth)
    lines.append("\n### 포지셔닝 및 트렌드 종합 해설 (AI)\n")
    lines.append(llm_commentary)

    return "\n".join(lines)
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

'''
def _section_monthly_tech_lifecycle():
    lines = [_section_header("2. 기술 수명 주기 및 R&D 투자 타이밍 분석")]
    lines.append("> 주요 기술의 시장 성숙도 단계를 진단하고, R&D 및 사업화 투자 시점을 판단합니다.\n")
    lines.append(_insert_image(os.path.join(FIG_DIR, "tech_maturity_map.png"), "기술 성숙도 맵"))

    tech_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "tech_maturity.json"), {})
    df_tech = pd.DataFrame(tech_data.get("results", []))
    if not df_tech.empty:
        rows = [{"기술": item.get("technology"), "단계": item.get("analysis", {}).get("stage"), "판단 근거": item.get("analysis", {}).get("reason")} for item in df_tech.to_dict('records')]
        lines.append(_to_markdown_table(pd.DataFrame(rows)))
    return "\n".join(lines)
'''
    
# --- ▼▼▼ [수정] 월간 경쟁사 전략 섹션 (키워드 네트워크 이미지 추가) ▼▼▼ ---
def _section_monthly_competitors():
    lines = [_section_header("3. 경쟁사 전략적 의도 및 파트너 관계망 분석")]
    lines.append("> 경쟁사의 토픽 집중도와 키워드 연관성, 기업 간 관계망을 통해 경쟁 구도와 전략을 분석합니다.\n") # 문구 수정

    lines.append("### 기업-토픽 집중도\n")
    lines.append(_insert_image(os.path.join(FIG_DIR, "matrix_heatmap.png"), "기업-토픽 집중도 히트맵"))
    df_matrix = _safe_read_csv(os.path.join(EXPORT_DIR, "company_topic_matrix_wide.csv"))
    lines.append("\n#### 집중도 매트릭스 (상위 5개사)\n") # 서브헤더 추가
    lines.append(_to_markdown_table(df_matrix.head(5)))

    lines.append("\n### 키워드 및 기업 관계망\n") # 서브헤더 추가
    lines.append(_insert_image(os.path.join(FIG_DIR, "keyword_network.png"), "월간 키워드 네트워크")) # 키워드 네트워크 추가
    lines.append(_insert_image(os.path.join(FIG_DIR, "company_network.png"), "기업 경쟁/협력 관계망"))

    # 기존 기업 네트워크 분석 테이블 (관계쌍) 추가
    network_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "company_network.json"), {})
    top_pairs = network_data.get("top_pairs", [])
    if top_pairs:
         lines.append("\n#### 주요 기업 관계 Top 5\n")
         df_pairs = pd.DataFrame(top_pairs).head(5)
         lines.append(_to_markdown_table(df_pairs[['source', 'target', 'weight', 'rel_type']].rename(columns={
             'source':'기업1', 'target':'기업2', 'weight':'관계강도(언급빈도)', 'rel_type':'관계유형(추정)'
         })))

    # LLM 기반 기업 전략 해석 (기존 로직 유지 - monthly_report.py 에서 가져옴)
    matrix_df_long = _safe_read_csv(os.path.join(EXPORT_DIR, "company_topic_matrix_long.csv"))
    topics_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "topics.json"), {})
    topic_map = {t.get("topic_id"): t.get("topic_name", f"Topic #{t.get('topic_id')}") for t in topics_data.get("topics", [])}
    if not matrix_df_long.empty:
        top_companies = matrix_df_long.groupby('org')['hybrid_score'].sum().nlargest(3).index
        insight_rows = []
        lines.append("\n#### Top 3 경쟁사 전략 방향성 해석 (AI)\n")
        for company in top_companies:
            top_topics = matrix_df_long[matrix_df_long['org'] == company].nlargest(3, 'hybrid_score')
            top_topics_str = ", ".join([topic_map.get(int(t_id), f"Topic {t_id}") for t_id in top_topics['topic'] if str(t_id).isdigit()])
            insight = call_gemini_for_strategy_insight(company, top_topics_str)
            insight_rows.append({ "기업": company, "핵심 집중 토픽": top_topics_str, "전략 방향성 해석": insight })
        lines.append(_to_markdown_table(pd.DataFrame(insight_rows)))


    return "\n".join(lines)
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- ▼▼▼ [수정] 월간 리스크 관리 섹션 (이미지 및 LLM 해설 추가) ▼▼▼ ---
def _section_monthly_risks():
    lines = [_section_header("4. 전략적 리스크 관리 및 완화 액션 제안")]
    lines.append("> 데이터 기반으로 탐지된 주요 리스크와 관련 시그널을 분석하고, 대응 방안을 제시합니다.\n") # 문구 수정

    # 리스크 시각화 이미지 추가
    lines.append("\n### 리스크 관련 시그널 시각화\n")
    lines.append(_insert_image(os.path.join(FIG_DIR, "risk_negative_spikes.png"), "주요 토픽 부정 감성 추이"))
    lines.append(_insert_image(os.path.join(FIG_DIR, "risk_keyword_network.png"), "리스크 연관 키워드 네트워크"))

    # 리스크 목록 테이블
    df_risks = _safe_read_csv(os.path.join(EXPORT_DIR, "risk_issues.csv"))
    lines.append("\n### 탐지된 주요 리스크 목록\n")
    lines.append(_to_markdown_table(df_risks))

    # LLM 종합 해설 추가
    llm_commentary = call_gemini_for_risk_commentary_monthly(df_risks)
    lines.append("\n### 리스크 종합 평가 및 대응 제안 (AI)\n")
    lines.append(llm_commentary)

    return "\n".join(lines)
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def _section_monthly_biz_opps():
    lines = [_section_header("5. 데이터 기반 신사업 아이디어 제안")]
    lines.append("> 시장 데이터 분석을 통해 도출된 Top 5 신사업 아이디어와 각 아이디어의 사업성을 평가합니다.\n")
    lines.append(_insert_image(os.path.join(FIG_DIR, "idea_score_distribution.png"), "신사업 아이디어 점수 분포"))

    opps_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "biz_opportunities.json"), {})
    df_opps = pd.DataFrame(opps_data.get("ideas", []))
    if not df_opps.empty:
        lines.append(_to_markdown_table(df_opps[['idea', 'value_prop', 'score']].rename(columns={'idea': '아이디어', 'value_prop': '가치 제안', 'score': '총점'})))
    return "\n".join(lines)

def _section_monthly_action_plan():
    lines = [_section_header("6. Top 1 신사업 아이디어 검증 (향후 2주 실행 계획)")]
    lines.append("> 가장 우선순위가 높은 신사업 아이디어를 검증하기 위한 구체적인 실행 계획을 제시합니다.\n")
    df_plan = _safe_read_csv(os.path.join(EXPORT_DIR, "two_week_plan.csv"))
    lines.append(_to_markdown_table(df_plan))
    return "\n".join(lines)

def main():
    """논문/기사 형식의 월간 상세 해설 리포트 생성 메인 함수"""
    print("[INFO] Generating Professional Monthly Commentary Report...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=29) # 월간 기준 (약 30일)
    period = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"

    # 1. Executive Summary 컨텍스트 준비 (기존 로직 유지)
    opps_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "biz_opportunities.json"), {}) #
    risks_df = _safe_read_csv(os.path.join(EXPORT_DIR, "risk_issues.csv")) #
    tech_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "tech_maturity.json"), {}) #

    summary_context = {
        "period": period,
        "top_biz_idea": (opps_data.get("ideas", [{}]))[0].get("idea", "N/A"),
        "top_risk": risks_df.iloc[0]['Topic'] if not risks_df.empty else "N/A",
        # "emerging_tech": next((t['technology'] for t in tech_data.get("results", []) if t.get("analysis", {}).get("stage") == "Emerging"), "N/A")
    }

    # 2. 리포트 컨텐츠 조립
    lines = [f"# 월간 상세 해설 리포트\n<div class='subtitle'>Period: {period} | Generated by Market Intelligence Team</div>\n"] #

    lines.append(_section_header("Executive Summary", level=2))
    summary_text = call_gemini_for_monthly_exec_summary(summary_context) #
    lines.append(f"<div class='executive-summary'>{summary_text}</div>\n")

    lines.append(_section_monthly_positioning_map()) # 수정된 함수 호출
    lines.append(_section_monthly_tech_lifecycle()) # 기존 함수 호출
    lines.append(_section_monthly_competitors()) # 수정된 함수 호출
    lines.append(_section_monthly_risks()) # 수정된 함수 호출
    lines.append(_section_monthly_biz_opps()) # 기존 함수 호출
    lines.append(_section_monthly_action_plan()) # 기존 함수 호출

    # 3. 파일 저장 및 변환
    report_content = "\n".join(lines)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(report_content)

    build_html_from_md(OUT_MD, OUT_HTML) # HTML 생성 함수 호출
    print(f"[SUCCESS] Professional monthly commentary report generated: {OUT_MD}, {OUT_HTML}")

if __name__ == "__main__": #
    main()