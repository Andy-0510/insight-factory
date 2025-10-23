import os
import json
import pandas as pd
from datetime import datetime, timedelta
from src.utils import load_json
from src.config import load_config
# [수정] weekly_report에 필요한 헬퍼 함수들을 직접 가져오거나 정의
from .daily_commentary_report import (_safe_read_csv, _to_markdown_table, _section_header, _insert_image, build_html_from_md)

# --- 설정 ---
ROOT_OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(ROOT_OUTPUT_DIR, "fig")
EXPORT_DIR = os.path.join(ROOT_OUTPUT_DIR, "export")
OUT_MD = os.path.join(ROOT_OUTPUT_DIR, "weekly_commentary_report.md")
OUT_HTML = os.path.join(ROOT_OUTPUT_DIR, "weekly_commentary_report.html")
TARGET_COMPETITORS = ["삼성디스플레이", "LG디스플레이", "BOE", "CSOT", "Visionox", "Tianma"] #

# --- LLM 호출 함수 ---
def call_gemini_for_weekly_exec_summary(context: dict) -> str:
    """LLM을 호출하여 주간 리포트의 Executive Summary를 생성합니다."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "> LLM API 키가 없어 Executive Summary를 생성할 수 없습니다."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        당신은 시장 분석팀의 수석 애널리스트입니다. 아래는 지난 한 주간의 시장 데이터 요약입니다.
        이를 바탕으로 경영진을 위한 'Executive Summary'를 4~5 문장의 상세한 문단으로 작성해주세요.
        지난주 시장의 가장 중요한 흐름, 핵심적인 기회 및 위협 요인, 그리고 다음 주에 반드시 주목해야 할 관전 포인트를 중심으로 논리적으로 서술해주세요.

        ### 주간 핵심 데이터:
        - **분석 기간**: {context.get('period', 'N/A')}
        - **주간 Top5 키워드**: {', '.join(context.get('top_keywords', []))}
        - **가장 활동적인 경쟁사 Top3**: {', '.join(context.get('top_competitors', []))}
        - **주목할 만한 약한 신호**: {', '.join(context.get('top_weak_signals', []))}
        - **상승 모멘텀 Top 신호**: {context.get('top_rising_signal', 'N/A')}

        ### Executive Summary (4~5 문장):
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"> LLM 요약 생성 실패: {e}"
    
# --- ▼▼▼ [추가] 약한 신호 심층 분석을 위한 LLM 호출 함수 ▼▼▼ ---
def call_gemini_for_weak_signal_analysis(weak_signals: list) -> str:
    """LLM을 호출하여 약한 신호의 의미와 잠재력을 심층 분석합니다."""
    if not weak_signals:
        return "> 분석할 약한 신호가 없습니다."
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "> LLM API 키가 없습니다."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        당신은 미래 기술 트렌드를 분석하는 전문 애널리스트입니다. 아래는 이번 주에 포착된 주목할 만한 초기 신호(Weak Signals) 목록입니다.

        ### 분석 대상 약한 신호:
        {json.dumps(weak_signals, ensure_ascii=False, indent=2)}

        ### 요청:
        목록에 있는 각 '약한 신호'에 대해, 그것이 무엇을 의미하는지, 왜 지금 주목해야 하는지, 그리고 미래에 어떤 잠재력(기회 또는 위협)을 가질 수 있는지 2~3문장으로 심층 분석해주세요. 
        각 신호는 `### [신호명]` 형식의 헤더로 구분하여 명확하게 설명해주세요.
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"> LLM 해설 생성 실패: {e}"

# --- ▼▼▼ [신규 추가] 키워드 네트워크/클러스터 해설 생성을 위한 LLM 함수 ▼▼▼ ---
def call_gemini_for_keyword_network_commentary(top_keywords, clusters_df):
    """LLM을 호출하여 키워드 네트워크 및 클러스터의 의미를 분석합니다."""
    if clusters_df is None or clusters_df.empty:
        cluster_info = "키워드 클러스터 정보 없음."
    else:
        cluster_info_list = []
        for _, row in clusters_df.head(5).iterrows(): # 상위 5개 클러스터 정보 사용
             cluster_info_list.append(f"- Cluster {row.get('cluster_id', '?')}: {row.get('keywords', '')}")
        cluster_info = "\n".join(cluster_info_list)

    top_keywords_str = ", ".join(top_keywords)

    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "> LLM API 키 없음."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        당신은 시장 동향 분석 전문가입니다.
        지난 주 시장의 주요 키워드와 이들의 클러스터링 결과는 다음과 같습니다.

        ### 주간 Top 키워드 (참고용):
        {top_keywords_str}

        ### 주요 키워드 클러스터:
        {cluster_info}

        ### 분석 요청:
        위 정보를 바탕으로, 지난 주 시장을 관통하는 **핵심 테마 2~3가지**를 도출하고, 각 테마가 **어떤 키워드 클러스터와 연관**되는지 간략하게 설명해주세요. 키워드 네트워크 시각화 자료를 함께 본다고 가정하고 해설해주세요.

        ### 분석 결과 (Markdown 형식):
        #### 주간 핵심 테마 분석
        - **테마 1**: (테마 설명 및 관련 클러스터 언급)
        - **테마 2**: (테마 설명 및 관련 클러스터 언급)
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"> LLM 해설 생성 실패: {e}"
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


# --- 섹션별 컨텐츠 생성 함수 ---
# --- ▼▼▼ [수정] 주간 시장 테마 섹션 수정 (네트워크/클러스터 및 LLM 해설 추가) ▼▼▼ ---
def _section_weekly_market_themes(df_keywords):
    lines = [_section_header("1. 주간 시장 테마 및 거시적 흐름 분석")]
    lines.append("> 지난 한 주간 누적된 키워드 점수를 통해 시장의 핵심 테마와 그 연관성을 분석합니다.\n") # 문구 수정
    lines.append(_insert_image(os.path.join(FIG_DIR, "weekly_wordcloud.png"), "주간 키워드 워드클라우드"))
    lines.append(_to_markdown_table(df_keywords.rename(columns={'keyword': 'Top 10 키워드', 'score': '누적 점수'}))) #

    # --- ▼▼▼ [추가] 키워드 네트워크 및 클러스터, LLM 해설 ▼▼▼ ---
    lines.append("\n### 키워드 연관성 분석\n")
    lines.append(_insert_image(os.path.join(FIG_DIR, "keyword_network.png"), "주간 키워드 네트워크")) # 이미지 추가
    df_clusters = _safe_read_csv(os.path.join(EXPORT_DIR, "keyword_clusters.csv")) # 클러스터 데이터 로드
    if not df_clusters.empty:
        lines.append("\n#### 주요 키워드 클러스터\n")
        lines.append(_to_markdown_table(df_clusters)) # 클러스터 테이블 추가
        # LLM 해설 생성 및 추가
        top_keywords_list = df_keywords['keyword'].tolist() if not df_keywords.empty else []
        llm_commentary = call_gemini_for_keyword_network_commentary(top_keywords_list, df_clusters)
        lines.append("\n#### 네트워크 및 클러스터 해설 (AI)\n")
        lines.append(llm_commentary)
    else:
        lines.append("\n> 키워드 네트워크/클러스터 데이터가 없습니다.\n")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    return "\n".join(lines)
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


def _section_weekly_competitors(trends_df):
    lines = [_section_header("2. 경쟁사 활동 강도 및 전략 전환 경보")]
    lines.append("> 주요 경쟁사의 주간 언급량과 모멘텀 변화를 통해 경쟁 구도를 분석합니다.\n")
    
    if not trends_df.empty:
        competitor_stats = []
        for competitor in TARGET_COMPETITORS:
            comp_df = trends_df[trends_df['term'] == competitor]
            if not comp_df.empty:
                weekly_mentions = comp_df['cur'].sum()
                avg_z_like = comp_df['z_like'].mean()
                if weekly_mentions > 0:
                    competitor_stats.append({
                        "경쟁사": competitor,
                        "주간 총 언급량": int(weekly_mentions),
                        "주간 평균 모멘텀": round(avg_z_like, 2)
                    })
        
        if competitor_stats:
            df_competitors = pd.DataFrame(competitor_stats).sort_values(by="주간 총 언급량", ascending=False)
            lines.append(_to_markdown_table(df_competitors))
        else:
            lines.append("> 금주 감지된 주요 경쟁사 활동이 없습니다.\n")
    else:
        lines.append("> 트렌드 데이터가 없습니다.\n")
    return "\n".join(lines)

def _section_weekly_future_signals(df_weak):
    lines = [_section_header("3. 잠재적 미래 성장 동력 및 초기 신호 발견")]
    lines.append("> 시장의 초기 단계이거나 새롭게 부상하는 약한 신호(Weak Signals)를 분석합니다.\n")
    
    if not df_weak.empty:
        # 분석할 상위 약한 신호 선정 (최대 3개)
        top_weak = df_weak.sort_values(by="z_like", ascending=False).head(3).copy()
        
        lines.append("### 금주 포착된 주요 약한 신호\n")
        lines.append(_to_markdown_table(top_weak[['term', 'cur', 'z_like']].rename(columns={'term': '약한 신호', 'cur': '최근 언급량', 'z_like': '모멘텀'})))
        
        # --- ▼▼▼ [수정] LLM 심층 분석 로직 추가 ▼▼▼ ---
        weak_signals_for_llm = top_weak.to_dict('records')
        llm_commentary = call_gemini_for_weak_signal_analysis(weak_signals_for_llm)
        
        lines.append("\n### 신호별 심층 분석\n")
        lines.append(llm_commentary)
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    else:
        lines.append("> 금주 포착된 약한 신호가 없습니다.\n")
    return "\n".join(lines)
    
def _section_weekly_momentum(trends_df):
    lines = [_section_header("4. 핵심 트렌드 모멘텀 변화 및 우선순위 검토")]
    lines.append("> 주간 상승/하강 모멘텀이 가장 강했던 신호들을 통해 시장의 동적인 변화를 파악합니다.\n")
    lines.append(_insert_image(os.path.join(FIG_DIR, "weekly_strong_signals_barchart.png"), "주간 상승/하강 신호 Top 5"))
    
    if not trends_df.empty:
        weekly_momentum_df = trends_df.groupby('term')['z_like'].mean().reset_index()
        top_rising = weekly_momentum_df.nlargest(5, 'z_like')
        lines.append("### 상승 모멘텀 Top 5\n")
        lines.append(_to_markdown_table(top_rising.rename(columns={'term': '신호', 'z_like': '주간 평균 모멘텀'})))
    
    return "\n".join(lines)

def main():
    """논문/기사 형식의 주간 상세 해설 리포트 생성 메인 함수"""
    print("[INFO] Generating Professional Weekly Commentary Report...")
    today_str = datetime.now().strftime("%Y-%m-%d")
    start_date_str = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
    period = f"{start_date_str} ~ {today_str}"

    # 1. 데이터 로드
    keywords_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "keywords.json"), {"keywords": []}) #
    df_keywords = pd.DataFrame(keywords_data.get("keywords", [])).head(10) #

    trends_df = _safe_read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv")) #
    df_weak = _safe_read_csv(os.path.join(EXPORT_DIR, "weak_signals.csv")) #

    # 2. Executive Summary 컨텍스트 준비 (기존 로직 유지)
    top_competitors = []
    if not trends_df.empty:
        competitor_mentions = {}
        for competitor in TARGET_COMPETITORS:
            total_mentions = trends_df[trends_df['term'] == competitor]['cur'].sum()
            if total_mentions > 0:
                competitor_mentions[competitor] = total_mentions
        top_competitors = [comp for comp, _ in sorted(competitor_mentions.items(), key=lambda item: item[1], reverse=True)][:3] #

    rising_signals = trends_df[trends_df['z_like'] > 0].sort_values(by='z_like', ascending=False) if not trends_df.empty else pd.DataFrame() #

    summary_context = {
        "period": period,
        "top_keywords": df_keywords['keyword'].tolist()[:5] if not df_keywords.empty else [],
        "top_competitors": top_competitors,
        "top_weak_signals": df_weak['term'].tolist()[:3] if not df_weak.empty else [],
        "top_rising_signal": rising_signals.iloc[0]['term'] if not rising_signals.empty else "N/A"
    }

    # 3. 리포트 컨텐츠 조립
    lines = [f"# 주간 상세 해설 리포트\n<div class='subtitle'>Period: {period} | Generated by Market Intelligence Team</div>\n"] #

    lines.append(_section_header("Executive Summary", level=2))
    summary_text = call_gemini_for_weekly_exec_summary(summary_context) #
    lines.append(f"<div class='executive-summary'>{summary_text}</div>\n")

    lines.append(_section_weekly_market_themes(df_keywords)) # 수정된 함수 호출
    lines.append(_section_weekly_competitors(trends_df)) #
    lines.append(_section_weekly_future_signals(df_weak)) #
    lines.append(_section_weekly_momentum(trends_df)) #

    # 4. 파일 저장 및 변환
    report_content = "\n".join(lines)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(report_content)

    build_html_from_md(OUT_MD, OUT_HTML) # HTML 생성 함수 호출
    print(f"[SUCCESS] Professional weekly commentary report generated: {OUT_MD}, {OUT_HTML}")

if __name__ == "__main__": #
    main()