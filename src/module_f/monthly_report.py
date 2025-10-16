# 파일 경로: src/module_f/monthly_report.py

import os
import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from src.utils import load_json

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
        html_tpl = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Monthly Strategic Review</title><style>body{{font-family:sans-serif;line-height:1.6;padding:24px;max-width:900px;margin:20px auto}}img{{max-width:100%;border:1px solid #ddd}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}th{{background:#f7f7f7}}h2{{margin-top:32px;border-bottom:2px solid #eee}}</style></head><body>{html}</body></html>"""
        with open(out_html, "w", encoding="utf-8") as f: f.write(html_tpl)
    except Exception as e: print(f"[WARN] HTML 변환 실패: {e}")

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

# --- ▼▼▼▼▼▼ [수정] 월간 리포트 섹션 상세 구현 ▼▼▼▼▼▼ ---

def _section_monthly_positioning_map(data):
    """섹션 1: 전략적 시장 포지셔닝 맵"""
    topics_data = data.get("topics", {})
    topic_list = topics_data.get("topics", [])
    
    df_topics = pd.DataFrame(topic_list)
    if not df_topics.empty:
        df_topics['top_words_str'] = df_topics['top_words'].apply(lambda words: ", ".join([w['word'] for w in words[:3]]))
        table = _to_markdown_table(df_topics[['topic_name', 'topic_summary', 'top_words_str']].rename(columns={
            'topic_name': '토픽명', 'topic_summary': '요약', 'top_words_str': '핵심 키워드'
        }))
    else:
        table = "- (토픽 데이터 없음)\n"
        
    image = _insert_images(os.path.join(FIG_DIR, "topics_bubble.png"), OUT_MD, captions=["시장 토픽 포지셔닝 맵"])
    return image + table

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
    matrix_df = data.get("company_matrix")
    network_data = data.get("company_network", {})
    
    lines = [_insert_images(os.path.join(FIG_DIR, "company_network.png"), OUT_MD, captions=["기업 경쟁/협력 관계망"])]
    lines.append("### 기업별 토픽 집중도 (상위 5개사)")
    lines.append(_to_markdown_table(matrix_df.head(5)))
    
    top_pairs = network_data.get("top_pairs", [])
    if top_pairs:
        lines.append("### 가장 강한 관계 Top 5")
        lines.append(_to_markdown_table(pd.DataFrame(top_pairs).head(5)))
        
    return "\n".join(lines)

def _section_monthly_risk_management(data):
    """섹션 4: 전략적 리스크 관리 및 완화 액션 제안"""
    df_risks = data.get("risk_issues")
    return _to_markdown_table(df_risks)

def _section_monthly_new_biz_ideas(data):
    """섹션 5: 데이터 기반 신사업 아이디어 및 초기 검증"""
    biz_opps_data = data.get("biz_opps", {})
    ideas = biz_opps_data.get("ideas", [])
    
    image = _insert_images(os.path.join(FIG_DIR, "idea_score_distribution.png"), OUT_MD, captions=["신사업 아이디어 점수 분포"])
    
    rows = []
    for idea in ideas:
        rows.append({
            "아이디어": idea.get("idea"),
            "가치 제안": idea.get("value_prop"),
            "총점": idea.get("score")
        })
    table = _to_markdown_table(pd.DataFrame(rows))
    return image + table

def _section_monthly_conclusion(data):
    """섹션 6: 종합 전략 방향 및 자원 배분 계획"""
    df_plan = data.get("action_plan")
    return _to_markdown_table(df_plan)

# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def build_monthly_markdown():
    monthly_data = load_monthly_data()
    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [f"# Monthly Strategic Review ({today_str})"]

    lines.append(_section_header("1. 전략적 시장 포지셔닝 맵")); lines.append(_section_monthly_positioning_map(monthly_data))
    lines.append(_section_header("2. 기술 수명 주기 및 R&D 투자 타이밍 분석")); lines.append(_section_monthly_tech_lifecycle(monthly_data))
    lines.append(_section_header("3. 경쟁사 전략적 의도 및 파트너 관계망 분석")); lines.append(_section_monthly_competitor_strategy(monthly_data))
    lines.append(_section_header("4. 전략적 리스크 관리 및 완화 액션 제안")); lines.append(_section_monthly_risk_management(monthly_data))
    lines.append(_section_header("5. 데이터 기반 신사업 아이디어 및 초기 검증")); lines.append(_section_monthly_new_biz_ideas(monthly_data))
    lines.append(_section_header("6. 향후 2주 실행 계획 (Action Plan)")); lines.append(_section_monthly_conclusion(monthly_data))

    with open(OUT_MD, "w", encoding="utf-8") as f: f.write("\n".join(lines))
    return OUT_MD

def main():
    try:
        md_path = build_monthly_markdown()
        build_html_from_md_new(md_path, OUT_HTML)
        print(f"[INFO] Monthly report generated: {md_path}, {OUT_HTML}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Monthly report generation failed: {e}")

if __name__ == "__main__":
    main()
