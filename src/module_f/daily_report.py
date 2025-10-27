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

# --- ▼▼▼ _load_data 함수 수정 (감성 데이터 로드 추가) ▼▼▼ ---
def _load_data():
    """리포트 생성에 필요한 모든 데이터 소스를 로드합니다."""
    data = {
        "keywords": load_json(os.path.join(ROOT_OUTPUT_DIR, "keywords.json"), {"keywords": [], "stats": {}}),
        "topics": load_json(os.path.join(ROOT_OUTPUT_DIR, "topics.json"), {"topics": []}),
        "ts": load_json(os.path.join(ROOT_OUTPUT_DIR, "trend_timeseries.json"), {"daily": []}),
        "insights": load_json(os.path.join(ROOT_OUTPUT_DIR, "trend_insights.json"), {"summary": "", "top_topics": [], "evidence": {}}),
        "opps": load_json(os.path.join(ROOT_OUTPUT_DIR, "biz_opportunities.json"), {"ideas": []}),
        "tech_maturity": load_json(os.path.join(ROOT_OUTPUT_DIR, "tech_maturity.json"), {"results": []}),
        "weak_insights": load_json(os.path.join(ROOT_OUTPUT_DIR, "weak_signal_insights.json"), {"results": []}),
        "meta_items": load_json(latest(os.path.join("data", "news_meta_*.json")), []) # 경로 수정
    }
    # --- ▼▼▼ [추가] 일간 감성 데이터 로드 ▼▼▼ ---
    sentiment_path = os.path.join(EXPORT_DIR, "daily_topic_sentiment.csv")
    data["sentiment"] = _safe_read_csv(sentiment_path)
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    return data
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

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
    """허깅페이스 모델을 사용해 기사 본문을 자연스러운 요약문으로 생성합니다 (num_beams 추가)."""
    global summarizer

    if summarizer is None:
        print("[INFO] Initializing Hugging Face summarization model...")
        try: # Add try-except for pipeline initialization
            summarizer = pipeline('summarization', model='gogamza/kobart-summarization')
            print("[INFO] Model initialized.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize summarization model: {e}")
            return "> 요약 모델 초기화 실패." # Return error message


    if not text or len(text.split()) < 50:
        return "> 기사 본문이 짧아 요약할 수 없습니다."

    try:
        # --- ▼▼▼ [수정] num_beams=4 파라미터 추가 ▼▼▼ ---
        summary_result = summarizer(text, max_length=150, min_length=40, do_sample=False, num_beams=4) # num_beams 추가
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        if summary_result and isinstance(summary_result, list) and 'summary_text' in summary_result[0]:
            summary = summary_result[0]['summary_text']
            if summary:
                return f"> {summary.strip()}"
            else:
                return "> 요약문 생성에 실패했습니다."
        else:
            return "> 요약 생성에 실패했습니다 (모델이 결과를 반환하지 않음)."
    except Exception as e:
        print(f"[WARN] Hugging Face summarization failed: {e}")
        # Add more detail to the error message if possible
        error_detail = str(e)
        return f"> 로컬 요약 모델 실행 중 오류 발생: {error_detail[:100]}" # Show first 100 chars of error

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

# --- ▼▼▼ [신규 추가] 일간 리스크 신호 섹션 함수 ▼▼▼ ---
def _section_daily_risk_signals(data):
    """토픽별 감성 점수 급락 현황을 분석하여 리스크 신호를 감지합니다."""
    df_sentiment = data.get("sentiment")
    if df_sentiment is None or df_sentiment.empty:
        return "- (감성 데이터 없음)\n"

    # semantic_key (또는 topic_id) 별로 그룹화하여 분석
    if 'semantic_key' not in df_sentiment.columns:
        return "- (감성 데이터에 'semantic_key' 컬럼 없음)\n"

    risky_topics = []
    today_date_dt = datetime.now() # 오늘 날짜 datetime 객체
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
    df_sentiment = df_sentiment[df_sentiment['semantic_key'] != "Uncategorized"] # 제외

    for key, group in df_sentiment.groupby('semantic_key'):
        # 최근 8일 데이터 필터링 (오늘 포함)
        group = group[group['date'] <= today_date_dt].sort_values('date').tail(8)
        if len(group) < 8: continue # 데이터 부족 시 건너뛰기

        # 오늘 데이터와 7일 평균 계산 (오늘 제외)
        today_row = group.iloc[-1]
        if today_row['date'].strftime('%Y-%m-%d') != today_date_dt.strftime('%Y-%m-%d'): continue # 오늘 데이터 없으면 스킵

        past_7_days = group.iloc[:-1]
        if past_7_days.empty: continue # 과거 데이터 없으면 스킵

        today_score = today_row['avg_sentiment']
        avg_7_days = past_7_days['avg_sentiment'].mean()
        std_7_days = past_7_days['avg_sentiment'].std()

        # 급락 조건: 오늘 점수가 (7일 평균 - 1.5 * 표준편차) 보다 낮거나, 평균의 80% 미만인 경우
        threshold_std = avg_7_days - 1.5 * std_7_days if pd.notna(std_7_days) and std_7_days > 0 else avg_7_days * 0.9 # 표준편차 유효성 체크
        threshold_pct = avg_7_days * 0.8

        is_risky = pd.notna(today_score) and pd.notna(avg_7_days) and \
                   (today_score < threshold_std or today_score < threshold_pct)

        if is_risky:
            drop_amount = avg_7_days - today_score
            risky_topics.append({
                "토픽 (Semantic Key)": key,
                "오늘 감성 점수": _fmt_float(today_score, 3),
                "최근 7일 평균": _fmt_float(avg_7_days, 3),
                "하락 폭": _fmt_float(drop_amount, 3)
            })

    if not risky_topics:
        return "- (감성 급락 토픽 없음)\n"

    df_risky = pd.DataFrame(risky_topics).sort_values(by="하락 폭", ascending=False)
    return _to_markdown_table(df_risky)
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def build_daily_markdown():
    data = _load_data()
    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [f"# Daily Briefing ({today_str})"]

    lines.append(_section_header("1. 시장 활동량 및 이상 징후"))
    lines.append(_section_time_series(data))

    lines.append(_section_header("2. 오늘의 리스크 신호 (Risk Signals)")) # <-- 새 섹션 추가
    lines.append(_section_daily_risk_signals(data)) # <-- 새 함수 호출

    lines.append(_section_header("3. 오늘의 급등 신호 (Today's Hot Signals)"))
    lines.append(_section_daily_hot_signals(data))
    lines.append(_section_header("참고: 최근 30일 누적 강한 신호 Top 5"))
    lines.append(_section_historical_strong_signals(data))

    lines.append(_section_header("4. 경쟁사 주요 활동"))
    lines.append(_section_competitor_events(data))

    lines.append(_section_header("5. 주요 기사"))
    lines.append(_section_top_articles(data))

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return OUT_MD
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

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
def main():
    try:
        md_path = build_daily_markdown()
        build_html_from_md_new(md_path, OUT_HTML)
        print(f"[INFO] Daily report generated: {md_path}, {OUT_HTML}") #
    except Exception as e: print(f"[ERROR] Daily report generation failed: {e}") #

if __name__ == "__main__":
    main()

