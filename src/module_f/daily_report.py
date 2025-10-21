# 파일 경로: src/module_f/daily_report.py

import os
import re
import glob
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from src.utils import load_json, save_json, latest
from src.config import load_config
from transformers import pipeline # [수정] transformers의 pipeline 추가


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

# --- ▼▼▼▼▼ [신규 추가] 일일 급등 신호 섹션 함수 ▼▼▼▼▼ ---
def _section_daily_hot_signals(data):
    """오늘 가장 뜨거운 '급등 신호'를 보여줍니다."""
    df_hot = _safe_read_csv(os.path.join(EXPORT_DIR, "daily_hot_signals.csv"))
    return _to_markdown_table(df_hot.rename(columns={
        'term': '급등 신호', 'cur': '오늘 언급량', 'diff': '어제 대비 증가', 'z_like': '모멘텀(z_like)'
    }))
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# 기사 요약을 위한 허깅페이스 모델 호출
# 전역 변수로 요약 파이프라인을 저장하여 매번 로드하지 않도록 최적화
summarizer = None

def summarize_text_with_hf(text: str) -> str:
    """허깅페이스 모델을 사용해 기사 본문을 자연스러운 요약문으로 생성합니다."""
    global summarizer
    
    if summarizer is None:
        print("[INFO] Initializing Hugging Face summarization model...")
        # You passed along ... 경고는 무시하셔도 괜찮습니다.
        summarizer = pipeline('summarization', model='gogamza/kobart-summarization')
        print("[INFO] Model initialized.")

    if not text or len(text.split()) < 50:
        return "> 기사 본문이 짧아 요약할 수 없습니다."
    
    try:
        summary_result = summarizer(text, max_length=150, min_length=40, do_sample=False)
        
        if summary_result and isinstance(summary_result, list) and 'summary_text' in summary_result[0]:
            summary = summary_result[0]['summary_text']
            
            # --- ▼▼▼ [수정] 불릿포인트 변환 로직 삭제 및 인용문 형식으로 변경 ▼▼▼ ---
            if summary:
                # Markdown 인용문 형식(>)을 사용하여 단락 전체를 깔끔하게 보여줍니다.
                return f"> {summary.strip()}"
            else:
                return "> 요약문 생성에 실패했습니다."
            # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
        else:
            return "> 요약 생성에 실패했습니다 (모델이 결과를 반환하지 않음)."

    except Exception as e:
        print(f"[WARN] Hugging Face summarization failed: {e}")
        return "> 로컬 요약 모델 실행 중 오류가 발생했습니다."

# 리포트 섹션
def _section_time_series(data):
    """일일 시장 활동량 및 이상 징후 (최근 30일 기준)"""
    ts = data.get("ts", {})
    daily = ts.get("daily", [])
    df_ts_full = pd.DataFrame(daily)
    
    # --- ▼▼▼ [추가] 신호 기사 비율 계산 ▼▼▼ ---
    df_signal = _safe_read_csv(os.path.join(EXPORT_DIR, "daily_signal_counts.csv"))
    if not df_ts_full.empty and not df_signal.empty:
        df_merged = pd.merge(df_ts_full, df_signal, on="date", how="left").fillna(0)
        df_merged['signal_ratio'] = (df_merged['signal_article_count'] / df_merged['count']).where(df_merged['count'] > 0, 0)
        avg_ratio_30days = df_merged.tail(30)['signal_ratio'].mean()
    else:
        avg_ratio_30days = 0
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    df_ts_30days = df_ts_full.tail(30)
    if df_ts_30days.empty: return "- (시계열 데이터 부족)\n"
    
    date_range = f"{df_ts_30days.iloc[0]['date']} ~ {df_ts_30days.iloc[-1]['date']}"
    
    lines = [
        f"- **분석 기간:** {date_range} (최근 30일)",
        f"- **최근 30일 평균 신호 기사 비율:** {avg_ratio_30days:.2%}" # <-- 비율 텍스트 추가
    ]
    
    # 이미지는 이제 강화된 버전으로 자동 교체됨
    lines.append(_insert_images(os.path.join(FIG_DIR, "timeseries.png"), OUT_MD, captions=["일일 기사량, 신호 기사 비율 및 스파이크 추이"]))
    
    # --- ▼▼▼▼▼ [수정] 스파이크 테이블을 두 개로 분리하여 표시 ▼▼▼▼▼ ---
    df_spikes = _safe_read_csv(os.path.join(EXPORT_DIR, "timeseries_spikes_enhanced.csv"))
    if not df_spikes.empty:
        start_date_30days = pd.to_datetime(df_ts_30days.iloc[0]['date'])
        df_spikes['date'] = pd.to_datetime(df_spikes['date'])
        df_spikes_recent = df_spikes[df_spikes['date'] >= start_date_30days].copy()
        
        if not df_spikes_recent.empty:
            df_spikes_recent['date'] = df_spikes_recent['date'].dt.strftime('%Y-%m-%d')
            
            # 1. 전체 기사량 스파이크 테이블
            df_count_spikes = df_spikes_recent[df_spikes_recent['metric'] == '전체 기사량'].copy()
            if not df_count_spikes.empty:
                lines.append("### 📈 전체 기사량 스파이크")
                lines.append(_to_markdown_table(df_count_spikes[['date', 'value', 'z_score']].rename(columns={
                    'date': '날짜', 'value': '기사량', 'z_score': 'Z-Score'
                })))

            # 2. 신호 기사 비율 스파이크 테이블
            df_ratio_spikes = df_spikes_recent[df_spikes_recent['metric'] == '신호 기사 비율'].copy()
            if not df_ratio_spikes.empty:
                lines.append("### 신호 기사 비율 스파이크")
                lines.append(_to_markdown_table(df_ratio_spikes[['date', 'value', 'z_score']].rename(columns={
                    'date': '날짜', 'value': '비율', 'z_score': 'Z-Score'
                })))
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
        
    return "\n".join(lines)

def _section_historical_strong_signals(data):
    """지난 30일 기준의 누적 '강한 신호'를 보여줍니다."""
    df_strong = _safe_read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv"))
    if not df_strong.empty:
        # 기존 로직과 동일
        rows = [{"모멘텀 토픽": row.get("term"), "z_like 점수": _fmt_float(row.get("z_like"), 2), "누적 언급량": _fmt_int(row.get("total"))} for _, row in df_strong.head(5).iterrows()]
        return _to_markdown_table(pd.DataFrame(rows))
    return "- (누적 강한 신호 데이터 없음)\n"

def _section_competitor_events(data):
    # ... (기존과 동일)
    df_events = _safe_read_csv(os.path.join(EXPORT_DIR, "events.csv"))
    if not df_events.empty:
        rows = [{"날짜": row.get("date", ""), "유형": row.get("types", ""), "제목": _truncate(row.get("title", ""), 100)} for _, row in df_events.head(5).iterrows()]
        return _to_markdown_table(pd.DataFrame(rows))
    return "- (주요 이벤트 데이터 없음)\n"

def _section_top_articles(data):
    """주요 기사 제목과 함께 허깅페이스 기반 요약을 제공합니다."""
    df_articles = _safe_read_csv(os.path.join(EXPORT_DIR, "today_article_list.csv"))
    if df_articles.empty:
        return "- (선정된 주요 기사 없음)\n"

    meta_items = data.get("meta_items", [])
    # [수정] 사용자가 명시한 'body' 필드를 우선적으로 사용하도록 보강
    url_to_body = {item.get("url"): (item.get("body") or item.get("description", "")) for item in meta_items if item.get("url")}
    
    md_lines = []
    for _, row in df_articles.iterrows():
        title = row.get("title", "제목 없음")
        url = row.get("url")
        
        md_lines.append(f"### [{_truncate(title, 100)}]({url})")
        
        body = url_to_body.get(url, "")
        
        if body:
            # LLM 대신 허깅페이스 요약 함수를 호출합니다.
            summary = summarize_text_with_hf(body)
            md_lines.append(summary)
        else:
            md_lines.append("- (기사 본문을 찾을 수 없어 요약을 생성하지 못했습니다.)")
        
        md_lines.append("---")

    return "\n".join(md_lines)

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
    
    # --- ▼▼▼▼▼ [수정] 리포트 섹션 순서 및 내용 변경 ▼▼▼▼▼ ---
    lines.append(_section_header("1. 시장 활동량 및 이상 징후"))
    lines.append(_section_time_series(data))

    lines.append(_section_header("2. 오늘의 급등 신호 (Today's Hot Signals)"))
    lines.append(_section_daily_hot_signals(data))
    lines.append(_section_header("참고: 최근 30일 누적 강한 신호 Top 5"))
    lines.append(_section_historical_strong_signals(data))

    lines.append(_section_header("3. 경쟁사 주요 활동"))
    lines.append(_section_competitor_events(data))
    
    lines.append(_section_header("4. 주요 기사"))
    lines.append(_section_top_articles(data))
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return OUT_MD

def main():
    try:
        md_path = build_daily_markdown()
        build_html_from_md_new(md_path, OUT_HTML)
        print(f"[INFO] Daily report generated: {md_path}, {OUT_HTML}")
    except Exception as e: print(f"[ERROR] Daily report generation failed: {e}")

if __name__ == "__main__":
    main()

