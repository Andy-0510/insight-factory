# 파일 경로: src/module_f/monthly_report.py

import os
import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from src.utils import load_json
from src.config import load_config
import google.generativeai as genai


# 헬퍼 함수
def _safe_read_csv(path, **kwargs):
    try:
        if os.path.exists(path): return pd.read_csv(path, **kwargs)
    except Exception: pass
    return pd.DataFrame()

def _to_markdown_table(df: pd.DataFrame, max_rows=50):
    if df is None or df.empty: return "- (데이터 없음)\n"
    return df.head(max_rows).copy().to_markdown(index=False) + "\n"

def _section_header(title):
    return f"\n## {title}\n"

def _exists(path):
    return path and os.path.exists(path)

def _insert_images(image_paths, md_out_path, captions=None):
    lines = []
    if not isinstance(image_paths, (list, tuple)): image_paths = [image_paths]
    captions = captions or []
    md_dir = os.path.dirname(md_out_path)
    for i, p in enumerate(image_paths):
        if _exists(p):
            relative_path = os.path.relpath(p, start=md_dir).replace("\\", "/")
            cap = captions[i] if i < len(captions) else ""
            lines.append(f"![{cap or 'Figure'}]({relative_path})")
    return ("\n".join(lines) + "\n") if lines else ""

def build_html_from_md_new(md_path, out_html):
    try:
        import markdown
        with open(md_path, "r", encoding="utf-8") as f: md = f.read()
        html = markdown.markdown(md, extensions=["extra", "tables", "toc"])
        # Use a more professional HTML template
        html_tpl = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Monthly Strategic Review</title><style>body{{font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif;line-height:1.6;padding:30px;max-width:960px;margin:30px auto;color:#333}}img{{max-width:100%;height:auto;border:1px solid #e0e0e0;margin:1.5em 0;display:block}}table{{border-collapse:collapse;width:100%;margin:1.5em 0;font-size:0.9em}}th,td{{border:1px solid #ddd;padding:12px;text-align:left;vertical-align:top}}th{{background:#f9f9f9;font-weight:600}}h1,h2,h3{{font-weight:600;margin-top:2.2em;margin-bottom:1em}}h1{{font-size:2em;text-align:center;border-bottom:none;margin-bottom:1.2em}}h2{{font-size:1.6em;border-bottom:2px solid #eee;padding-bottom:0.4em}}h3{{font-size:1.25em;border-bottom:1px solid #eee;padding-bottom:0.3em}}blockquote{{border-left:4px solid #eee;padding-left:1em;margin-left:0;color:#555}}code{{background:#f0f0f0;padding:2px 4px;border-radius:3px;font-size:0.9em}}a{{color:#007bff;text-decoration:none}}a:hover{{text-decoration:underline}}</style></head><body>{html}</body></html>"""
        with open(out_html, "w", encoding="utf-8") as f: f.write(html_tpl)
    except Exception as e: print(f"[WARN] HTML 변환 실패: {e}")
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- 설정 ---
ROOT_OUTPUT_DIR = "outputs"
EXPORT_DIR = os.path.join(ROOT_OUTPUT_DIR, "export")
FIG_DIR = os.path.join(ROOT_OUTPUT_DIR, "fig")
OUT_MD = os.path.join(ROOT_OUTPUT_DIR, "monthly_report.md")
OUT_HTML = os.path.join(ROOT_OUTPUT_DIR, "monthly_report.html")

def load_monthly_data():
    """월간 리포트에 필요한 모든 최종 데이터 산출물을 로드합니다."""
    print(f"[INFO] Loading data for monthly report...")
    monthly_data = {
        "topics": load_json(os.path.join(ROOT_OUTPUT_DIR, "topics.json"), {}),
        "tech_maturity": load_json(os.path.join(ROOT_OUTPUT_DIR, "tech_maturity.json"), {}),
        "company_matrix": _safe_read_csv(os.path.join(EXPORT_DIR, "company_topic_matrix_wide.csv")),
        "company_network": load_json(os.path.join(ROOT_OUTPUT_DIR, "company_network.json"), {}),
        "risk_issues": _safe_read_csv(os.path.join(EXPORT_DIR, "risk_issues.csv")),
        "biz_opps": load_json(os.path.join(ROOT_OUTPUT_DIR, "biz_opportunities.json"), {}),
        "action_plan": _safe_read_csv(os.path.join(EXPORT_DIR, "two_week_plan.csv")),
    }
    return monthly_data

# --- LLM 호출 ---
def call_gemini_for_monthly_summary(context):
    """
    LLM을 호출하여 월간 경영 요약을 생성합니다.
    
    Args:
        context (dict): 월간 데이터 분석 결과를 포함한 입력 데이터
        
    Returns:
        str: 생성된 월간 전략 브리핑 텍스트 (실패 시 오류 메시지 반환)
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_config = cfg.get("llm", {})
        model_name = model_config.get("model", "gemini-1.5-flash-001")
        
        model = genai.GenerativeModel(model_name)
        print(f"[INFO] Using Gemini model for monthly executive summary: {model_name}")

        prompt = f"""
        당신은 디스플레이 산업 최고 전략 책임자(CSO)입니다. 아래는 지난 한 달간의 시장 데이터 분석 결과 요약입니다. 
        이 데이터를 종합하여 CEO 및 경영진을 위한 '월간 전략 브리핑'을 작성해주세요.

        ### 월간 데이터 요약:
        {json.dumps(context, ensure_ascii=False, indent=2)}

        ### 작성 가이드 (Markdown 형식):
        1. **월간 핵심 동향 (Key Trends)**: 데이터를 관통하는 가장 중요한 시장의 변화와 거시적 흐름 2-3가지를 짚어주세요.
        2. **전략적 시사점 (Strategic Implications)**: 이 동향이 우리 비즈니스에 주는 기회와 위협 요소를 명확히 분석해주세요.
        3. **최우선 실행 과제 (Top Priority Action Items)**: 분석 결과를 바탕으로 다음 달에 가장 먼저 집중해야 할 
           구체적인 액션 아이템 2-3가지를 제안해주세요.
        """
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"[ERROR] Gemini 월간 요약 생성 실패: {e}", exc_info=True)
        return "LLM 요약 생성 중 오류가 발생했습니다."
        

def call_gemini_for_positioning_insight(topics_data):
    """LLM을 호출하여 토픽 포지셔닝 맵 분석 및 액션 아이템을 제안받습니다."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        # 프롬프트에 전달할 토픽 데이터 간소화 (이름, 요약 정보만 전달)
        simplified_topics = [
            {"topic_name": t.get("topic_name"), "summary": t.get("topic_summary")}
            for t in topics_data.get("topics", [])
        ]

        prompt = f"""
        당신은 시장 전략 컨설턴트입니다. 아래는 시장의 주요 토픽들을 '시장 관심도(X축)'와 '시장 긍정성(Y축)'으로 분석한 포지셔닝 데이터입니다.

        ### 토픽 데이터:
        {json.dumps(simplified_topics, ensure_ascii=False, indent=2)}

        ### 분석 요청:
        1. **4분면 분석**: 위 토픽들을 아래 4가지 영역으로 분류하고, 각 영역의 전략적 의미를 1~2 문장으로 해석해주세요.
            - **주력 영역 (관심도 높음, 긍정성 높음)**: 현재 시장의 주류이자 핵심 동력인 토픽.
            - **기회 영역 (관심도 낮음, 긍정성 높음)**: 미래 성장 잠재력이 있는 유망 토픽.
            - **경쟁/위험 영역 (관심도 높음, 긍정성 낮음)**: 경쟁이 치열하거나 부정적 이슈가 많은 토픽.
            - **틈새/장기 영역 (관심도 낮음, 긍정성 낮음)**: 장기적 관찰이 필요한 틈새 토픽.
        2. **핵심 토픽 선정 및 액션 아이템 제안**: 위 분석을 바탕으로, 지금 가장 주목해야 할 '기회 영역'과 '경쟁/위험 영역'의 토픽을 각각 하나씩 선정하고, 그에 대한 초기 액션 아이템을 구체적으로 제안해주세요.

        ### 출력 형식 (Markdown):
        #### 📈 4분면 분석
        - **주력 영역**: (분석 내용)
        - **기회 영역**: (분석 내용)
        - **경쟁/위험 영역**: (분석 내용)
        - **틈새/장기 영역**: (분석 내용)

        ####  actionable insights
        - **[선정된 '기회' 토픽명]**: (초기 액션 아이템 제안. 예: 해당 기술 보유 스타트업 3곳 리스트업 및 기술 검토 착수)
        - **[선정된 '위험' 토픽명]**: (초기 액션 아이템 제안. 예: 관련 부정 이슈에 대한 언론 반응 및 고객사 문의 현황 전수 조사)
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[ERROR] Gemini 포지셔닝 분석 실패: {e}")
        return "LLM 기반 포지셔닝 분석 중 오류가 발생했습니다."
    

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
        당신은 B2B 기술 기업 전문 애널리스트입니다. '{company_name}'라는 기업이 최근 아래 토픽들에 집중하고 있습니다. 
        이를 바탕으로 이 기업의 현재 사업 방향성과 단기 전략을 1~2 문장으로 간결하게 해석해주세요.

        ### 집중 토픽:
        {topics_str}

        ### 분석 결과 (1~2 문장 요약):
        """
        response = model.generate_content(prompt)
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        return f"LLM 분석 실패: {e}"

    
def call_gemini_for_network_action_item(pair_info):
    """LLM을 호출하여 경쟁/협력 관계에 대한 액션 아이템을 제안합니다."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "LLM API 키가 없습니다."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        당신은 전략기획팀 리더입니다. 시장 분석 결과, 아래와 같은 기업 간의 주요 관계가 포착되었습니다. 
        이 관계를 기반으로 우리 팀이 다음 주에 실행해야 할 현실적인 액션 아이템을 1개만 제안해주세요. (20자 내외)

        ### 관계 정보:
        {pair_info}

        ### 액션 아이템 제안 (1개, 20자 내외):
        """
        response = model.generate_content(prompt)
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        return f"LLM 분석 실패: {e}"


# --- ▼▼▼▼▼▼ [수정] 월간 리포트 섹션 상세 구현 ▼▼▼▼▼▼ ---

def _section_monthly_executive_summary(data):
    """
    섹션 0: EXECUTIVES SUMMARY
    
    Args:
        data (dict): 월간 분석 결과를 포함한 전체 데이터
        
    Returns:
        str: LLM이 생성한 경영진 요약 텍스트
    """
    # LLM에 전달할 핵심 월간 데이터 요약 생성
    context = {
        "가장 중요한 토픽 Top3": [t.get("topic_name") for t in data.get("topics", {}).get("topics", [])[:3]],
        "주요 탐지 리스크": [r.get("Topic") for r in data.get("risk_issues", pd.DataFrame()).to_dict('records')[:2]],
        "주요 기술 성숙도 변화": [
            f"{t.get('technology')}: {t.get('analysis', {}).get('stage')}" 
            for t in data.get("tech_maturity", {}).get("results", [])[:3]
        ],
        "가장 강한 관계를 맺은 경쟁사/파트너": data.get("company_network", {}).get("top_pairs", [{}])[0],
        "최우선 신사업 아이디어": (data.get("biz_opps", {}).get("ideas", [{}]))[0].get("idea")
    }

    # LLM 호출하여 월간 경영 요약 생성
    llm_summary = call_gemini_for_monthly_summary(context)
    
    return llm_summary

def _section_monthly_positioning_map(data):
    """섹션 1: 전략적 시장 포지셔닝 맵"""
    topics_data = data.get("topics", {})
    topic_list = topics_data.get("topics", [])

    # 1. 토픽 버블 차트 이미지 추가
    image_bubble = _insert_images(os.path.join(FIG_DIR, "topics_bubble.png"), OUT_MD, captions=["시장 토픽 포지셔닝 맵"])

    # 2. [추가] 토픽 미니 트렌드 이미지 (파일 존재 시)
    mini_trends_img = os.path.join(FIG_DIR, "topics_mini_trends.png")
    image_trends = _insert_images(mini_trends_img, OUT_MD, captions=["주요 토픽별 주간 트렌드"])

    # 3. LLM 기반 인사이트 및 액션 아이템 생성
    llm_insight = call_gemini_for_positioning_insight(topics_data)
    insight_section = f"\n### 전략적 인사이트 및 실행과제 제안\n{llm_insight}"

    # 4. 상세 데이터 테이블 생성
    if not topic_list:
        table_detail = "- (토픽 데이터 없음)\n"
    else:
        df_topics = pd.DataFrame(topic_list)
        df_topics['top_words_str'] = df_topics['top_words'].apply(
            lambda words: ", ".join([w.get('word', '') for w in words[:3]]) if isinstance(words, list) else "" # word 키 확인
        )
        # Use topic_name if available, otherwise topic_id
        df_topics['Topic Identifier'] = df_topics.apply(lambda row: row.get('topic_name', f"Topic #{row.get('topic_id')}"), axis=1)

        table_detail = "\n### 토픽 상세 정보\n" + _to_markdown_table(df_topics[['Topic Identifier', 'topic_summary', 'top_words_str']].rename(columns={
            'Topic Identifier': '토픽', 'topic_summary': '요약', 'top_words_str': '핵심 키워드'
        }))

    # 5. [추가] 토픽 성장/하락 테이블 (파일 존재 시)
    table_growth = ""
    growth_csv = os.path.join(EXPORT_DIR, "topic_growth.csv")
    df_growth = _safe_read_csv(growth_csv)
    if not df_growth.empty:
        table_growth = "\n### 토픽 성장/하락 추세\n" + _to_markdown_table(df_growth)
    else:
        table_growth = "\n- (토픽 성장/하락 데이터 없음)\n" # 데이터 없을 경우 메시지

    # 최종적으로 이미지, 인사이트, 테이블 순서로 조합하여 반환
    return image_bubble + image_trends + insight_section + table_detail + table_growth
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def _section_monthly_tech_lifecycle(data):
    """섹션 2: 기술 수명 주기 및 R&D 투자 타이밍 분석"""
    tech_maturity_data = data.get("tech_maturity", {})
    results = tech_maturity_data.get("results", [])

    rows = []
    for item in results:
        rows.append({
            "기술": item.get("technology"),
            "단계": item.get("analysis", {}).get("stage"),
            "판단 근거": item.get("analysis", {}).get("reason")
        })
    table = _to_markdown_table(pd.DataFrame(rows))
    image = _insert_images(os.path.join(FIG_DIR, "tech_maturity_map.png"), OUT_MD, captions=["기술 성숙도 맵"])
    return image + table

def _section_monthly_competitor_strategy(data):
    """섹션 3: 경쟁사 전략적 의도 및 파트너 관계망 분석"""
    matrix_df_wide = data.get("company_matrix")
    network_data = data.get("company_network", {})
    
    lines = []

    # --- 3.1: 기업의 사업 방향성 및 전략 해석 ---
    lines.append("### 3.1 기업별 전략 방향성 분석")
    
    # Long format 데이터가 인사이트 도출에 더 유용하므로, wide 대신 long을 로드
    matrix_df_long = _safe_read_csv(os.path.join(EXPORT_DIR, "company_topic_matrix_long.csv"))
    topics_data = data.get("topics", {})
    topic_map = {t.get("topic_id"): t.get("topic_name", f"Topic #{t['topic_id']}") for t in topics_data.get("topics", [])}

    if not matrix_df_long.empty:
        # 총점이 가장 높은 상위 3개 기업 선정
        top_companies = matrix_df_long.groupby('org')['hybrid_score'].sum().nlargest(3).index

        insight_rows = []
        for company in top_companies:
            # 해당 기업의 상위 토픽 3개 추출
            top_topics = matrix_df_long[matrix_df_long['org'] == company].nlargest(3, 'hybrid_score')
            top_topics_str = ", ".join([topic_map.get(int(t_id), f"Topic {t_id}") for t_id in top_topics['topic']])
            
            # LLM 호출하여 전략 방향성 해석
            insight = call_gemini_for_strategy_insight(company, top_topics_str)
            insight_rows.append({
                "기업": company,
                "핵심 집중 토픽": top_topics_str,
                "전략 방향성 해석": insight
            })
        lines.append(_to_markdown_table(pd.DataFrame(insight_rows)))
    else:
        lines.append("- (기업별 토픽 집중도 데이터가 없습니다.)\n")

    # --- 3.2: 경쟁/협력 관계망 분석 및 액션 아이템 ---
    lines.append("\n### 3.2 경쟁/협력 관계망 분석 및 실행과제")
    # --- ▼▼▼ [추가] 월간 키워드 네트워크 이미지 삽입 ▼▼▼ ---
    lines.append(_insert_images(os.path.join(FIG_DIR, "keyword_network.png"), OUT_MD, captions=["월간 키워드 네트워크"]))
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    lines.append(_insert_images(os.path.join(FIG_DIR, "company_network.png"), OUT_MD, captions=["기업 경쟁/협력 관계망"]))
    
    top_pairs = network_data.get("top_pairs", [])
    if top_pairs:
        actionable_pairs = []
        for pair in top_pairs[:3]: # 상위 3개 관계에 대해서만 분석
            pair_info_str = f"기업1: {pair['source']}, 기업2: {pair['target']}, 관계 유형: {pair['rel_type']}"

            # LLM 호출하여 액션 아이템 제안
            action_item = call_gemini_for_network_action_item(pair_info_str)
            
            actionable_pairs.append({
                "주요 관계": f"{pair['source']} ↔ {pair['target']}",
                "유형": pair['rel_type'],
                "액션 아이템 제안": action_item
            })
        lines.append(_to_markdown_table(pd.DataFrame(actionable_pairs)))
        lines.append("\n >두 기업이 함께 언급된 문맥 안에서, 키워드의 빈도가 **경쟁 > 협력**인 경우 'rivalry', 반대의 경우 'partnership'으로 분류")        
    else:
        lines.append("- (주요 관계 데이터가 없습니다.)\n")
        
    return "\n".join(lines)

def _section_monthly_risk_management(data):
    """섹션 4: 전략적 리스크 관리 및 완화 액션 제안"""
    lines = []
    # [추가] 리스크 관련 시각화 이미지 (파일 존재 시)
    risk_images = [
        os.path.join(FIG_DIR, "risk_negative_spikes.png"),
        os.path.join(FIG_DIR, "risk_keyword_network.png")
    ]
    lines.append(_insert_images(risk_images, OUT_MD, captions=["주요 토픽 부정 감성 추이", "리스크 연관 키워드 네트워크"]))

    df_risks = data.get("risk_issues")
    # 테이블 추가 전에 데이터 유효성 검사
    if df_risks is not None and not df_risks.empty:
        lines.append(_to_markdown_table(df_risks))
    else:
        lines.append("- (리스크/이슈 데이터 없음)\n") # 데이터 없을 경우 메시지

    return "\n".join(lines)
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def _section_monthly_new_biz_ideas(data):
    """섹션 5: 데이터 기반 신사업 아이디어 제안"""
    biz_opps_data = data.get("biz_opps", {})
    ideas = biz_opps_data.get("ideas", [])

    # 5.1 데이터 기반 신사업 아이디어 TOP 5
    # 아이디어 표 생성        
    rows = []
    for idea in ideas:
        rows.append({
            "아이디어": idea.get("idea"),
            "가치 제안": idea.get("value_prop"),
            "총점": idea.get("score")
        })
    table = _to_markdown_table(pd.DataFrame(rows))

    # 이미지 삽입
    image = _insert_images(os.path.join(FIG_DIR, "idea_score_distribution.png"), OUT_MD, captions=["신사업 아이디어 점수 분포"])

    # 5.2 TOP 1 아이디어 2주 Action Items
    df_plan = data.get("action_plan")
    if df_plan is not None:
        action_plan_table = _to_markdown_table(df_plan)
        plan_section = f"\n\n #### [아이디어 검증: 2주 Action Plan]\n{action_plan_table}"
    else:
        plan_section = "\n\n"

    return table + "\n\n" + image + plan_section

def build_monthly_markdown():
    monthly_data = load_monthly_data()
    
    days_to_analyze = 30
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_analyze - 1)
    date_range_str = f"({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})"
    
    lines = [f"# Monthly Strategic Review {date_range_str}"]
    
    lines.append(_section_header("Executive Summary"))
    lines.append(_section_monthly_executive_summary(monthly_data))

    lines.append(_section_header("1. 전략적 시장 포지셔닝 맵")); lines.append(_section_monthly_positioning_map(monthly_data))
    lines.append(_section_header("2. 기술 수명 주기 및 R&D 투자 타이밍 분석")); lines.append(_section_monthly_tech_lifecycle(monthly_data))
    lines.append(_section_header("3. 경쟁사 전략적 의도 및 파트너 관계망 분석")); lines.append(_section_monthly_competitor_strategy(monthly_data))
    lines.append(_section_header("4. 전략적 리스크 관리 및 완화 액션 제안")); lines.append(_section_monthly_risk_management(monthly_data))
    lines.append(_section_header("5. 데이터 기반 신사업 아이디어 제안")); lines.append(_section_monthly_new_biz_ideas(monthly_data))

    with open(OUT_MD, "w", encoding="utf-8") as f: f.write("\n".join(lines))
    return OUT_MD

def main():
    try:
        md_path = build_monthly_markdown()
        build_html_from_md_new(md_path, OUT_HTML) # HTML generation call
        print(f"[INFO] Monthly report generated: {md_path}, {OUT_HTML}")
    except Exception as e:
        import traceback
        traceback.print_exc() #
        print(f"[ERROR] Monthly report generation failed: {e}")

if __name__ == "__main__":
    main()

