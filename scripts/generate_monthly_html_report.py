# 파일 경로: scripts/generate_monthly_html_report.py

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader, select_autoescape
import re
import markdown # 테이블 HTML 변환용

# --- 설정 (경로는 실제 프로젝트 구조에 맞게 조정 필요) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
TEMPLATE_NAME = 'monthly_report_template.html' # 월간 템플릿 사용
OUTPUT_BASE_DIR = os.path.join(ROOT_DIR, 'outputs')
EXPORT_DIR = os.path.join(OUTPUT_BASE_DIR, 'export')
FIG_DIR = os.path.join(OUTPUT_BASE_DIR, 'fig')
DEBUG_DIR = os.path.join(OUTPUT_BASE_DIR, 'debug')

# --- 필요한 헬퍼 함수 (utils, timeutil 등에서 import) ---
from src.utils import load_json, latest # load_json, latest 필요
from src.timeutil import now_kst # now_kst 필요
from src.config import load_config # LLM 호출 등에 필요

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
    # 월간 HTML은 outputs/monthly_report.html 에 생성될 예정 (워크플로우에서 복사 전)
    # fig 폴더까지의 상대 경로는 fig/
    return f"fig/{image_name}"

def dataframe_to_html_table(df, max_rows=50):
    """Pandas DataFrame을 HTML 테이블 문자열로 변환 (Markdown 대신)"""
    if df is None or df.empty:
        return "<p>(데이터 없음)</p>"
    # 테이블 스타일링을 위해 CSS 클래스 추가 가능
    return df.head(max_rows).to_html(index=False, escape=False, border=0, classes=["dataframe-table"]) # CSS 클래스 추가

# --- LLM 호출 함수들 (필요한 함수 정의 또는 import) ---
# 예: call_gemini_for_monthly_summary, call_gemini_for_positioning_insight 등
# ... (LLM 함수 정의 영역) ...

# --- 데이터 로딩 및 가공 함수 ---
def prepare_monthly_report_data():
    """월간 HTML 템플릿에 필요한 데이터를 로드하고 가공하는 함수"""
    print("[INFO] Loading and preparing data for Monthly HTML report...")
    data = {}
    end_dt = now_kst()
    start_dt = end_dt - timedelta(days=29) # 약 30일
    data['report_period'] = f"{start_dt.strftime('%Y.%m.%d')} - {end_dt.strftime('%Y.%m.%d')}"
    data['report_month'] = end_dt.strftime('%Y년 %m월') # 제목용

    # 1. 월간 집계 데이터 로드
    topics_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, "topics.json"), {"topics": []})
    tech_maturity_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, "tech_maturity.json"), {"results": []})
    company_network_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, "company_network.json"), {})
    biz_opps_data = load_json_safe(os.path.join(OUTPUT_BASE_DIR, "biz_opportunities.json"), {"ideas": []})
    # CSV 데이터 로드
    growth_df = safe_read_csv(os.path.join(EXPORT_DIR, "topic_growth.csv"))
    matrix_df_long = safe_read_csv(os.path.join(EXPORT_DIR, "company_topic_matrix_long.csv")) # 분석용 long format
    risk_issues_df = safe_read_csv(os.path.join(EXPORT_DIR, "risk_issues.csv"))
    action_plan_df = safe_read_csv(os.path.join(EXPORT_DIR, "two_week_plan.csv"))
    # 월간 집계 메타 (기사 수 등 계산용)
    monthly_meta = load_json_safe(os.path.join(DEBUG_DIR, "monthly_meta_agg.json"), [])

    # 2. 목차 데이터 생성
    data['toc_items'] = [
        {'number': '📄', 'id': 'summary', 'title': 'Executive Summary'},
        {'number': '1', 'id': 'positioning', 'title': '전략적 시장 포지셔닝 맵'},
        {'number': '2', 'id': 'lifecycle', 'title': '기술 수명 주기 분석'},
        {'number': '3', 'id': 'competitors', 'title': '경쟁사 전략적 의도 분석'},
        {'number': '4', 'id': 'risk', 'title': '전략적 리스크 관리'},
        {'number': '5', 'id': 'opportunities', 'title': '신사업 기회 발굴'},
        {'number': '6', 'id': 'actionplan', 'title': '종합 전략 방향 및 실행 방안'},
    ]

    # --- ▼▼▼ 각 섹션별 데이터 준비 (Placeholder 채우기 시작) ▼▼▼ ---

    # Section 0: Executive Summary & KPI
    summary_context = { # LLM 요약용 컨텍스트 준비
        "period": data['report_period'],
        # ... (monthly_report.py 에서 정의한 컨텍스트 구성) ...
    }
    # data['executive_summary'] = call_gemini_for_monthly_summary(summary_context)
    data['executive_summary'] = "월간 AI 요약 생성 예정..." # Placeholder
    # KPI 계산 (Placeholder, 실제 계산 로직 필요)
    data['kpi_total_articles'] = len(monthly_meta)
    data['kpi_article_change_mom'] = "+0%" # 계산 필요
    data['kpi_article_change_class'] = "change-neutral"
    data['kpi_key_topics_count'] = len(topics_data.get("topics", []))
    data['kpi_topic_change_mom'] = "+0" # 계산 필요
    data['kpi_topic_change_class'] = "change-neutral"
    data['kpi_source_diversity'] = "계산 필요" # Placeholder
    data['kpi_source_change_mom'] = "+0" # 계산 필요
    data['kpi_source_change_class'] = "change-neutral"
    data['kpi_weak_signals_count'] = "계산 필요" # Placeholder
    data['kpi_weak_signal_change_mom'] = "+0" # 계산 필요
    data['kpi_weak_signal_change_class'] = "change-neutral"

    # Section 1: Market Positioning
    data['topic_bubble_map_path'] = get_relative_image_path("topics_bubble.png")
    data['topic_mini_trends_path'] = get_relative_image_path("topics_mini_trends.png")
    # data['positioning_insight'] = call_gemini_for_positioning_insight(topics_data, growth_df) # LLM 호출
    data['positioning_insight'] = "포지셔닝 AI 분석 예정..." # Placeholder
    # 사분면 데이터 (Placeholder, 실제 분류 로직 필요)
    data['quadrant1_topics'] = "차량용, QD-OLED (예시)"; data['quadrant1_strategy'] = "AI 권고 생성 예정"
    data['quadrant2_topics'] = "IT OLED, 폴더블 (예시)"; data['quadrant2_strategy'] = "AI 권고 생성 예정"
    data['quadrant3_topics'] = "마이크로LED, 투명 (예시)"; data['quadrant3_strategy'] = "AI 권고 생성 예정"
    data['quadrant4_topics'] = "E-paper (예시)"; data['quadrant4_strategy'] = "AI 권고 생성 예정"
    # 시사점 (Placeholder, LLM 호출 또는 로직 필요)
    data['strategic_implications'] = ["AI 시사점 1 분석 예정", "AI 시사점 2 분석 예정"]
    # 테이블 데이터 준비 (DataFrame -> HTML)
    df_topic_details = pd.DataFrame(topics_data.get("topics", [])) # ... 가공 ...
    data['topic_details_table'] = dataframe_to_html_table(df_topic_details[['topic_id', 'topic_name', 'topic_summary']].head()) # 예시
    data['topic_growth_table'] = dataframe_to_html_table(growth_df)

    # Section 2: Tech Maturity
    data['tech_maturity_map_path'] = get_relative_image_path("tech_maturity_map.png")
    tech_details_list = [] # Placeholder, tech_maturity_data 가공
    data['tech_maturity_details'] = tech_details_list
    df_tech_maturity = pd.DataFrame(tech_maturity_data.get("results", [])) # ... 가공 ...
    data['tech_maturity_table'] = dataframe_to_html_table(df_tech_maturity[['technology', 'analysis']].head()) # 예시
    # data['rd_recommendation'] = call_gemini_for_rd_recommendation(...) # LLM 호출
    data['rd_recommendation'] = "R&D AI 권고 생성 예정..." # Placeholder

    # Section 3: Competitor Analysis
    data['matrix_heatmap_path'] = get_relative_image_path("matrix_heatmap.png")
    data['company_network_path'] = get_relative_image_path("company_network.png")
    data['keyword_network_path'] = get_relative_image_path("keyword_network.png")
    comp_pos_list = [] # Placeholder, matrix_df_long 등 가공
    data['competitor_positioning'] = comp_pos_list
    # data['competition_alerts'] = call_gemini_for_competition_alerts(...) # LLM 호출
    data['competition_alerts'] = ["경쟁 강도 AI 분석 예정..."] # Placeholder
    df_comp_strategy = pd.DataFrame() # Placeholder, LLM 결과 가공
    data['competitor_strategy_table'] = dataframe_to_html_table(df_comp_strategy)
    df_comp_actions = pd.DataFrame() # Placeholder, LLM 결과 가공
    data['competitor_actions_table'] = dataframe_to_html_table(df_comp_actions)

    # Section 4: Risk Management
    data['risk_spikes_path'] = get_relative_image_path("risk_negative_spikes.png")
    data['risk_network_path'] = get_relative_image_path("risk_keyword_network.png")
    data['risk_list_table'] = dataframe_to_html_table(risk_issues_df)
    risk_matrix_data = {"avoid": {}, "mitigate": {}, "transfer": {}, "accept": {}} # Placeholder, LLM 결과 가공
    data['risk_matrix'] = risk_matrix_data
    immediate_actions_list = [] # Placeholder, LLM 결과 가공
    data['immediate_actions'] = immediate_actions_list
    # data['risk_assessment'] = call_gemini_for_risk_assessment(...) # LLM 호출
    data['risk_assessment'] = "리스크 AI 종합 평가 생성 예정..." # Placeholder

    # Section 5: Business Opportunities & Action Plan
    data['idea_score_dist_path'] = get_relative_image_path("idea_score_distribution.png")
    df_opps = pd.DataFrame(biz_opps_data.get("ideas", [])) # ... 가공 ...
    data['opportunity_list_table'] = dataframe_to_html_table(df_opps[['idea', 'value_prop', 'score']].head()) # 예시
    data['action_plan_table'] = dataframe_to_html_table(action_plan_df)

    # Section 6: Final Recommendation
    # data['final_recommendation'] = call_gemini_for_final_recommendation(...) # LLM 호출
    data['final_recommendation'] = "최종 AI 종합 권고 생성 예정..." # Placeholder

    # Footer 정보
    data['dashboard_link'] = '#'
    data['data_source_link'] = '#'
    data['contact_link'] = '#'

    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

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
        # 필터 등록
        env.filters['format_int'] = format_int_filter
        # 테이블 HTML 렌더링을 위해 safe 필터 사용 예정이므로 추가 필터 불필요

        template = env.get_template(template_name)
        html_content = template.render(data)
        print("[INFO] Monthly HTML rendering successful.")
        return html_content
    except Exception as e:
        print(f"[ERROR] Monthly HTML template rendering failed: {e}")
        import traceback
        traceback.print_exc() # 상세 오류 출력
        return f"<html><body><h1>Monthly Report Generation Failed</h1><pre>{e}</pre></body></html>"


# --- 메인 실행 로직 ---
def main():
    start_time = now_kst()
    print(f"[INFO] Starting monthly HTML report generation at {start_time.strftime('%Y-%m-%d %H:%M:%S KST')}")

    # 1. 데이터 준비
    report_data = prepare_monthly_report_data()

    # 2. HTML 렌더링
    html_output = render_html_report(TEMPLATE_DIR, TEMPLATE_NAME, report_data)

    # 3. 출력 파일 경로 설정 ( outputs/monthly_report.html )
    output_html_path = os.path.join(OUTPUT_BASE_DIR, 'monthly_report.html')

    # 4. HTML 파일 저장
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
    # 월간 스크립트는 필요한 월간 집계 파일이 생성된 후 실행되어야 함
    # 예: outputs/topics.json, outputs/export/topic_growth.csv 등
    main()