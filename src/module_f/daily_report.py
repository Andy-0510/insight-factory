# 파일 경로: src/module_f/daily_report.py

import os
import re
import glob
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from src.utils import load_json, save_json, latest

# --- 설정 ---
ROOT_OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(ROOT_OUTPUT_DIR, "fig")
EXPORT_DIR = os.path.join(ROOT_OUTPUT_DIR, "export")
OUT_MD = os.path.join(ROOT_OUTPUT_DIR, "report.md")
OUT_HTML = os.path.join(ROOT_OUTPUT_DIR, "report.html")

# --- 헬퍼 함수 ---
def _fmt_int(x):
    # ... (기존과 동일)
    try: return f"{int(x):,}"
    except Exception:
        try: return f"{float(x):.0f}"
        except Exception: return str(x) if x is not None else "-"
def _fmt_float(x, nd=2):
    # ... (기존과 동일)
    try: return f"{float(x):.{nd}f}"
    except Exception: return "-"
def _truncate(s, n=80):
    # ... (기존과 동일)
    s = (s or "").strip().replace("\n", " "); return s if len(s) <= n else s[:n-1] + "…"
def _exists(path):
    # ... (기존과 동일)
    return path and os.path.exists(path)
def _safe_read_csv(path, **kwargs):
    # ... (기존과 동일)
    try:
        if _exists(path): return pd.read_csv(path, **kwargs)
    except Exception: pass
    return pd.DataFrame()
def _to_markdown_table(df: pd.DataFrame, max_rows=50):
    # ... (기존과 동일)
    if df is None or df.empty: return "- (데이터 없음)\n"
    return df.head(max_rows).copy().to_markdown(index=False) + "\n"
def _load_data():
    # ... (기존과 동일)
    return {
        "keywords": load_json("outputs/keywords.json", {"keywords": [], "stats": {}}),
        "topics": load_json("outputs/topics.json", {"topics": []}),
        "ts": load_json("outputs/trend_timeseries.json", {"daily": []}),
        "insights": load_json("outputs/trend_insights.json", {"summary": "", "top_topics": [], "evidence": {}}),
        "opps": load_json("outputs/biz_opportunities.json", {"ideas": []}),
        "tech_maturity": load_json("outputs/tech_maturity.json", {"results": []}),
        "weak_insights": load_json("outputs/weak_signal_insights.json", {"results": []}),
        "meta_items": load_json(latest("data/news_meta_*.json"), [])
    }
def _section_header(title):
    # ... (기존과 동일)
    return f"\n## {title}\n"

# --- ▼▼▼▼▼▼ [수정] _insert_images 함수를 독립적으로 만듭니다 ▼▼▼▼▼▼ ---
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
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def _section_time_series(data):
    # ... (기존과 동일, 단 _insert_images 호출 방식 변경)
    ts = data.get("ts", {}); daily = ts.get("daily", [])
    total_cnt = sum(int(x.get("count", 0)) for x in daily)
    date_range = f"{daily[0].get('date', '?')} ~ {daily[-1].get('date', '?')}" if daily else "-"
    lines = [f"- **기간:** {date_range}", f"- **총 기사 수:** {_fmt_int(total_cnt)}"]
    lines.append(_insert_images(os.path.join(FIG_DIR, "timeseries.png"), OUT_MD, captions=["일별 기사 수 추이"]))
    lines.append(_insert_images(os.path.join(FIG_DIR, "timeseries_spikes.png"), OUT_MD, captions=["이상치/스파이크 마커"]))
    df_spikes = _safe_read_csv(os.path.join(EXPORT_DIR, "timeseries_spikes.csv"))
    if not df_spikes.empty:
        lines.append("### 스파이크 상세"); lines.append(_to_markdown_table(df_spikes, max_rows=10))
    return "\n".join(lines)

def _section_signals_board(data):
    # ... (기존과 동일)
    df_strong = _safe_read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv"))
    if not df_strong.empty:
        rows = [{"모멘텀 토픽": row.get("term"), "z_like 점수": _fmt_float(row.get("z_like"), 2), "금일 언급량": _fmt_int(row.get("cur"))} for _, row in df_strong.head(3).iterrows()]
        return _to_markdown_table(pd.DataFrame(rows))
    return "- (강한 신호 데이터 없음)\n"

def _section_competitor_events(data):
    # ... (기존과 동일)
    df_events = _safe_read_csv(os.path.join(EXPORT_DIR, "events.csv"))
    if not df_events.empty:
        rows = [{"날짜": row.get("date", ""), "유형": row.get("types", ""), "제목": _truncate(row.get("title", ""), 100)} for _, row in df_events.head(5).iterrows()]
        return _to_markdown_table(pd.DataFrame(rows))
    return "- (주요 이벤트 데이터 없음)\n"

def _section_top_articles(data):
    # ... (기존과 동일)
    df_articles = _safe_read_csv(os.path.join(EXPORT_DIR, "today_article_list.csv"))
    if not df_articles.empty:
        df_articles['제목'] = df_articles.apply(lambda row: f"[{_truncate(row['title'], 100)}]({row['url']})", axis=1)
        return _to_markdown_table(df_articles[['제목']])
    return "- (선정된 주요 기사 없음)\n"

def build_html_from_md_new(md_path=OUT_MD, out_html=OUT_HTML):
    # ... (기존과 동일)
    try:
        import markdown
        with open(md_path, "r", encoding="utf-8") as f: md = f.read()
        html = markdown.markdown(md, extensions=["extra", "tables", "toc"])
        html_tpl = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Daily Briefing</title><style>body{{font-family:sans-serif;line-height:1.6;padding:24px;max-width:900px;margin:20px auto}}img{{max-width:100%;border:1px solid #ddd}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}th{{background:#f7f7f7}}h2{{margin-top:32px;border-bottom:2px solid #eee}}</style></head><body>{html}</body></html>"""
        with open(out_html, "w", encoding="utf-8") as f: f.write(html_tpl)
    except Exception as e: print(f"[WARN] HTML 변환 실패: {e}")

# --- main 함수 ---
def build_daily_markdown():
    data = _load_data(); today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [f"# Daily Briefing ({today_str})"]
    lines.append(_section_header("1. 시장 활동량 및 이상 징후")); lines.append(_section_time_series(data))
    lines.append(_section_header("2. 핵심 모멘텀 토픽 Top 3")); lines.append(_section_signals_board(data))
    lines.append(_section_header("3. 경쟁사 주요 활동")); lines.append(_section_competitor_events(data))
    lines.append(_section_header("4. 주요 기사")); lines.append(_section_top_articles(data))
    with open(OUT_MD, "w", encoding="utf-8") as f: f.write("\n".join(lines))
    return OUT_MD

def main():
    try:
        md_path = build_daily_markdown()
        build_html_from_md_new(md_path, OUT_HTML)
        print(f"[INFO] Daily report generated: {md_path}, {OUT_HTML}")
    except Exception as e: print(f"[ERROR] Daily report generation failed: {e}")

if __name__ == "__main__":
    main()