# -*- coding: utf-8 -*-
import os
import re
import glob
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# 외부 의존 유틸 (기존 코드와 호환)
from src.utils import load_json, save_json, latest

# -----------------------------
# 상수/경로 (기존과 동일)
# -----------------------------
FIG_DIR = "outputs/fig"
EXPORT_DIR = "outputs/export"
OUT_MD = "outputs/report.md"
OUT_HTML = "outputs/report.html"

# -----------------------------
# 헬퍼 함수들 (기존 코드 전체를 그대로 복사)
# -----------------------------
def _fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        try:
            return f"{float(x):.0f}"
        except Exception:
            return str(x) if x is not None else "-"

def _fmt_float(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"

def _truncate(s, n=80):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n-1] + "…"

def _exists(path):
    return path and os.path.exists(path)

def _safe_read_csv(path, **kwargs):
    try:
        if _exists(path):
            return pd.read_csv(path, **kwargs)
    except Exception:
        pass
    return pd.DataFrame()

def _to_markdown_table(df: pd.DataFrame, max_rows=50):
    if df is None or df.empty:
        return "- (데이터 없음)\n"
    use = df.head(max_rows).copy()
    return use.to_markdown(index=False) + ("\n" if len(use) else "\n")

def _load_data():
    keywords = load_json("outputs/keywords.json", {"keywords": [], "stats": {}})
    topics = load_json("outputs/topics.json", {"topics": []})
    ts = load_json("outputs/trend_timeseries.json", {"daily": []})
    insights = load_json("outputs/trend_insights.json", {"summary": "", "top_topics": [], "evidence": {}})
    opps = load_json("outputs/biz_opportunities.json", {"ideas": []})
    tech_maturity = load_json("outputs/tech_maturity.json", {"results": []})
    weak_insights = load_json("outputs/weak_signal_insights.json", {"results": []})
    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, []) if meta_path else []
    return {
        "keywords": keywords, "topics": topics, "ts": ts,
        "insights": insights, "opps": opps, "tech_maturity": tech_maturity,
        "weak_insights": weak_insights, "meta_items": meta_items
    }

def _section_header(title):
    return f"\n## {title}\n"

# ▼▼▼▼▼▼ 일간 리포트에 필요한 섹션 함수들만 남겨두거나 그대로 둡니다 ▼▼▼▼▼▼
# 이 함수들은 주간/월간 리포트에서도 재사용될 수 있으므로 삭제하지 않아도 됩니다.

def _section_time_series(data):
    ts = data.get("ts", {})
    daily = ts.get("daily", [])
    n_days = len(daily)
    total_cnt = sum(int(x.get("count", 0)) for x in daily)
    date_range = f"{daily[0].get('date', '?')} ~ {daily[-1].get('date', '?')}" if n_days > 0 else "-"

    lines = [
        f"- **기간:** {date_range}",
        f"- **총 기사 수:** {_fmt_int(total_cnt)}",
        "> 일별 기사 수와 7일 이동평균, 통계적 이상치(Spike)를 통해 시장의 양적 변화를 확인합니다."
    ]
    lines.append(_insert_images(os.path.join(FIG_DIR, "timeseries.png"), captions=["일별 기사 수 추이"]))
    lines.append(_insert_images(os.path.join(FIG_DIR, "timeseries_spikes.png"), captions=["이상치/스파이크 마커"]))
    
    spikes_csv = os.path.join(EXPORT_DIR, "timeseries_spikes.csv")
    df_spikes = _safe_read_csv(spikes_csv)
    if not df_spikes.empty:
        lines.append("### 스파이크 상세")
        lines.append(_to_markdown_table(df_spikes, max_rows=10))
        
    return "\n".join(lines)

def _section_signals_board(data):
    lines = []
    strong_csv = os.path.join(EXPORT_DIR, "trend_strength.csv")
    df_strong = _safe_read_csv(strong_csv)

    if not df_strong.empty:
        df_strong_top3 = df_strong.head(3)
        rows = []
        for _, row in df_strong_top3.iterrows():
            rows.append({
                "모멘텀 토픽": row.get("term"),
                "z_like 점수": _fmt_float(row.get("z_like"), 2),
                "금일 언급량": _fmt_int(row.get("cur"))
            })
        lines.append(_to_markdown_table(pd.DataFrame(rows)))
    else:
        lines.append("- (강한 신호 데이터 없음)\n")
    return "\n".join(lines)

def _section_competitor_events(data):
    events_csv = os.path.join(EXPORT_DIR, "events.csv")
    df_events = _safe_read_csv(events_csv)
    
    if not df_events.empty:
        rows = []
        for _, row in df_events.head(5).iterrows():
            rows.append({
                "날짜": row.get("date", ""),
                "유형": row.get("types", ""),
                "제목": _truncate(row.get("title", ""), 100)
            })
        return _to_markdown_table(pd.DataFrame(rows))
    return "- (주요 이벤트 데이터 없음)\n"
    
def _insert_images(image_paths, captions=None):
    lines = []
    if not isinstance(image_paths, (list, tuple)):
        image_paths = [image_paths]
    captions = captions or []
    for i, p in enumerate(image_paths):
        rel = p
        if _exists(p):
            if p.startswith("outputs/"):
                rel = p.replace("outputs/", "")
            else:
                rel = p
            rel = rel.replace("\\", "/")
            cap = captions[i] if i < len(captions) else ""
            lines.append(f"![{cap or 'Figure'}]({rel})")
    return ("\n".join(lines) + "\n") if lines else ""

def build_html_from_md_new(md_path=OUT_MD, out_html=OUT_HTML):
    try:
        import markdown
        with open(md_path, "r", encoding="utf-8") as f:
            md = f.read()
        html = markdown.markdown(md, extensions=["extra", "tables", "toc"])
        html_tpl = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Daily Briefing</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif; line-height: 1.6; padding: 24px; color: #222; max-width: 900px; margin: 20px auto; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
  th {{ background: #f7f7f7; }}
  h2 {{ margin-top: 32px; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
</style>
</head>
<body>
{html}
</body>
</html>"""
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html_tpl)
    except Exception as e:
        print(f"[WARN] HTML 변환 실패: {e}")

# ▲▲▲▲▲▲ 헬퍼 함수들은 그대로 유지 ▲▲▲▲▲▲


# -----------------------------
# ▼▼▼▼▼▼ main 함수 로직 수정 ▼▼▼▼▼▼
# -----------------------------

def build_daily_markdown():
    """기획안에 맞춰 일간 브리핑 마크다운을 생성합니다."""
    data = _load_data()
    today_str = datetime.now().strftime("%Y-%m-%d")

    lines = [f"# Daily Briefing ({today_str})"]

    # 섹션 1: 일일 시장 활동량 및 이상 징후 포착
    lines.append(_section_header("1. 시장 활동량 및 이상 징후"))
    lines.append(_section_time_series(data))

    # 섹션 2: 핵심 모멘텀 토픽 Top 3 분석
    lines.append(_section_header("2. 핵심 모멘텀 토픽 Top 3"))
    lines.append(_section_signals_board(data))
    
    # 섹션 3: 경쟁사 주요 활동
    lines.append(_section_header("3. 경쟁사 주요 활동"))
    lines.append(_section_competitor_events(data))

    # 섹션 4: 주요 기사 (다음 단계에서 구현)
    lines.append(_section_header("4. 주요 기사"))
    lines.append("- (구현 예정)\n")

    # 마크다운 파일 저장
    Path(OUT_MD).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return OUT_MD

def main():
    try:
        md_path = build_daily_markdown()
        build_html_from_md_new(md_path, OUT_HTML)
        print(f"[INFO] Daily report generated: {md_path}, {OUT_HTML}")
    except Exception as e:
        print(f"[ERROR] Daily report generation failed: {e}")

if __name__ == "__main__":
    main()
