# 파일 경로: src/module_f/monthly_commentary_report.py

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from src.utils import load_json
from src.config import load_config
from .daily_commentary_report import _safe_read_csv, _to_markdown_table, _section_header, _insert_image, build_html_from_md

# --- 설정 ---
ROOT_OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(ROOT_OUTPUT_DIR, "fig")
EXPORT_DIR = os.path.join(ROOT_OUTPUT_DIR, "export")
OUT_MD = os.path.join(ROOT_OUTPUT_DIR, "monthly_commentary_report.md")
OUT_HTML = os.path.join(ROOT_OUTPUT_DIR, "monthly_commentary_report.html")

# --- LLM 호출 함수 ---
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

# --- 섹션별 컨텐츠 생성 함수 ---

def _section_monthly_positioning_map():
    lines = [_section_header("1. 전략적 시장 포지셔닝 맵")]
    lines.append("> 시장의 핵심 토픽 지형을 통해 거시적인 트렌드와 기회 영역을 분석합니다.\n")
    lines.append(_insert_image(os.path.join(FIG_DIR, "topics_bubble.png"), "시장 토픽 포지셔닝 맵"))
    
    topics_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "topics.json"), {})
    df_topics = pd.DataFrame(topics_data.get("topics", []))
    
    if not df_topics.empty:
        # --- ▼▼▼ [수정] 'topic_name', 'topic_summary'가 없을 경우에 대한 방어 코드 추가 ▼▼▼ ---

        # 'top_words_str' 컬럼은 항상 생성합니다.
        df_topics['top_words_str'] = df_topics['top_words'].apply(
            lambda words: ", ".join([w['word'] for w in words[:3]]) if isinstance(words, list) else ""
        )

        # 표에 표시할 컬럼과 컬럼명을 동적으로 결정합니다.
        columns_to_display = []
        rename_map = {}

        if 'topic_name' in df_topics.columns:
            columns_to_display.append('topic_name')
            rename_map['topic_name'] = '토픽명'
        else:
            # topic_name이 없으면 topic_id를 대신 사용합니다.
            columns_to_display.append('topic_id')
            rename_map['topic_id'] = '토픽 ID'

        if 'topic_summary' in df_topics.columns:
            columns_to_display.append('topic_summary')
            rename_map['topic_summary'] = '요약'

        columns_to_display.append('top_words_str')
        rename_map['top_words_str'] = '핵심 키워드'
        
        # 준비된 컬럼만으로 테이블을 생성합니다.
        lines.append(_to_markdown_table(df_topics[columns_to_display].rename(columns=rename_map)))
        
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    else:
        lines.append("> - 데이터 없음\n")
        
    return "\n".join(lines)

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
    
def _section_monthly_competitors():
    lines = [_section_header("3. 경쟁사 전략적 의도 및 파트너 관계망 분석")]
    lines.append("> 경쟁사의 토픽 집중도를 통해 전략 방향을 유추하고, 시장 내 기업 간 관계망을 분석합니다.\n")
    lines.append(_insert_image(os.path.join(FIG_DIR, "matrix_heatmap.png"), "기업-토픽 집중도 히트맵"))
    lines.append(_insert_image(os.path.join(FIG_DIR, "company_network.png"), "기업 경쟁/협력 관계망"))
    
    df_matrix = _safe_read_csv(os.path.join(EXPORT_DIR, "company_topic_matrix_wide.csv"))
    lines.append("### 기업별 토픽 집중도 (상위 5개사)\n")
    lines.append(_to_markdown_table(df_matrix.head(5)))
    return "\n".join(lines)

def _section_monthly_risks():
    lines = [_section_header("4. 전략적 리스크 관리 및 완화 액션 제안")]
    lines.append("> 데이터 기반으로 탐지된 주요 리스크와 그에 대한 구체적인 대응 방안을 제시합니다.\n")
    df_risks = _safe_read_csv(os.path.join(EXPORT_DIR, "risk_issues.csv"))
    lines.append(_to_markdown_table(df_risks))
    return "\n".join(lines)

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
    start_date = end_date - timedelta(days=29)
    period = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"

    # 1. Executive Summary 컨텍스트 준비
    opps_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "biz_opportunities.json"), {})
    risks_df = _safe_read_csv(os.path.join(EXPORT_DIR, "risk_issues.csv"))
    tech_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "tech_maturity.json"), {})
    
    summary_context = {
        "period": period,
        "top_biz_idea": (opps_data.get("ideas", [{}]))[0].get("idea", "N/A"),
        "top_risk": risks_df.iloc[0]['Topic'] if not risks_df.empty else "N/A",
        "emerging_tech": next((t['technology'] for t in tech_data.get("results", []) if t.get("analysis", {}).get("stage") == "Emerging"), "N/A")
    }

    # 2. 리포트 컨텐츠 조립
    lines = [f"# 월간 상세 해설 리포트\n<div class='subtitle'>Period: {period} | Generated by Market Intelligence Team</div>\n"]
    
    lines.append(_section_header("Executive Summary", level=2))
    summary_text = call_gemini_for_monthly_exec_summary(summary_context)
    lines.append(f"<div class='executive-summary'>{summary_text}</div>\n")
    
    lines.append(_section_monthly_positioning_map())
    lines.append(_section_monthly_tech_lifecycle())
    lines.append(_section_monthly_competitors())
    lines.append(_section_monthly_risks())
    lines.append(_section_monthly_biz_opps())
    lines.append(_section_monthly_action_plan())

    # 3. 파일 저장 및 변환
    report_content = "\n".join(lines)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    build_html_from_md(OUT_MD, OUT_HTML)
    print(f"[SUCCESS] Professional monthly commentary report generated: {OUT_MD}, {OUT_HTML}")

if __name__ == "__main__":
    main()