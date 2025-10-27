import os
import pandas as pd
import re
from datetime import datetime
from src.utils import load_json, latest
from src.config import load_config
from transformers import pipeline

# --- 설정 ---
ROOT_OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(ROOT_OUTPUT_DIR, "fig")
EXPORT_DIR = os.path.join(ROOT_OUTPUT_DIR, "export")
OUT_MD = os.path.join(ROOT_OUTPUT_DIR, "daily_commentary_report.md")
OUT_HTML = os.path.join(ROOT_OUTPUT_DIR, "daily_commentary_report.html")

# --- ▼▼▼ 필요한 헬퍼 함수 가져오기 (daily_report와 중복되는 부분) ▼▼▼ ---
def _fmt_float(x, nd=2):
    try: return f"{float(x):.{nd}f}"
    except Exception: return "-"

def _safe_read_csv(path, **kwargs):
    try:
        if os.path.exists(path): return pd.read_csv(path, **kwargs)
    except Exception: pass
    return pd.DataFrame()

def _to_markdown_table(df: pd.DataFrame, max_rows=50):
    if df is None or df.empty: return "> - 데이터 없음\n"
    return df.head(max_rows).copy().to_markdown(index=False) + "\n"

def _section_header(title, level=2):
    return f"\n{'#' * level} {title}\n"

def _insert_image(path, caption=""):
    # Use relative path logic from daily_report
    if os.path.exists(path):
        try:
            # Try making relative to ROOT_OUTPUT_DIR for consistency
            relative_path = os.path.relpath(path, start=ROOT_OUTPUT_DIR).replace("\\", "/")
            return f"![{caption}]({relative_path})\n"
        except ValueError: # Handle cases where path is outside ROOT_OUTPUT_DIR (e.g., during testing)
             return f"![{caption}]({os.path.basename(path)})\n" # Fallback to basename
    return ""
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


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
        

# --- ▼▼▼ [수정] HTML 디자인 개선 ▼▼▼ ---
def build_html_from_md(md_path, out_html):
    try:
        import markdown
        with open(md_path, "r", encoding="utf-8") as f: md = f.read()
        html = markdown.markdown(md, extensions=["extra", "tables", "toc"])
        
        # 전문적인 문서 스타일의 CSS 추가
        html_tpl = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Daily Detailed Commentary</title>
        <style>
            body {{ font-family: 'Nanum Myeongjo', serif; line-height: 1.8; padding: 2.5cm 2cm; color: #333; }}
            .report-container {{ max-width: 21cm; margin: 0 auto; }}
            h1, h2, h3 {{ font-family: 'Nanum Gothic', sans-serif; font-weight: 700; border-bottom: none; padding-bottom: 0; }}
            h1 {{ font-size: 24pt; text-align: center; margin-bottom: 0.5cm; }}
            .subtitle {{ text-align: center; font-size: 11pt; color: #666; margin-top: 0; margin-bottom: 1cm; }}
            .executive-summary {{ background-color: #f7f7ff; border-left: 5px solid #4a69bd; padding: 15px 20px; margin-bottom: 1.5cm; }}
            h2 {{ font-size: 16pt; border-bottom: 2px solid #4a69bd; margin-top: 1.5cm; padding-bottom: 5px; }}
            h3 {{ font-size: 13pt; border-bottom: 1px solid #ccc; margin-top: 1cm; padding-bottom: 3px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; font-size: 9pt; text-align: left; }}
            th {{ background-color: #f7f7ff; font-weight: bold; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 1em; }}
            blockquote {{ border-left: 3px solid #ccc; padding-left: 15px; color: #555; margin-left: 0; }}
            @media print {{ body {{ padding: 1cm; }} }}
        </style>
        </head><body><div class="report-container">{html}</div></body></html>"""
        with open(out_html, "w", encoding="utf-8") as f: f.write(html_tpl)
    except Exception as e: print(f"[WARN] HTML 변환 실패: {e}")

# --- ▼▼▼ [추가] 일간 Executive Summary 생성을 위한 LLM 호출 함수 ▼▼▼ ---
def call_gemini_for_daily_exec_summary(context: dict) -> str:
    """LLM을 호출하여 일간 리포트의 Executive Summary를 생성합니다."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "LLM API 키가 없습니다."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        당신은 시장 분석팀의 수석 애널리스트입니다. 아래는 오늘 시장에서 포착된 핵심 데이터입니다. 
        이 데이터를 바탕으로 경영진을 위한 'Executive Summary'를 3~4 문장의 간결한 문단으로 작성해주세요. 
        오늘의 가장 중요한 현상과 그 원인, 그리고 주목해야 할 포인트를 중심으로 논문 초록처럼 작성해주세요.

        ### 오늘의 핵심 데이터:
        - **시장 이상 징후**: {context.get('spike_info', '특이사항 없음')}
        - **가장 주목받은 급등 신호**: {context.get('top_hot_signal', '해당 없음')}
        - **주요 경쟁사 이벤트**: {context.get('top_event', '해당 없음')}

        ### Executive Summary (3~4 문장):
        """
        response = model.generate_content(prompt)
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        return f"LLM 요약 생성 실패: {e}"
    

def call_gemini_for_signal_commentary(signal_term: str, article_titles: list) -> str:
    """LLM을 호출하여 급등 신호의 발생 원인을 한 문장으로 요약합니다."""
    if not article_titles:
        return "관련 기사가 없어 해설을 생성할 수 없습니다."
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "LLM API 키가 없습니다."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        # --- ▼▼▼ [수정] f-string 오류를 해결하기 위해 문자열을 미리 생성합니다. ▼▼▼ ---
        formatted_titles = "\n- ".join(article_titles)

        prompt = f"""
        당신은 시장 애널리스트입니다. 오늘 '{signal_term}' 키워드가 시장에서 급증했습니다.

        ### 관련 기사 헤드라인:
        - {formatted_titles}

        ### 요청:
        위 헤드라인들을 근거로, '{signal_term}' 키워드가 급증한 핵심 이유를 **30자 내외의 한 문장**으로 간결하게 요약해주세요.
        """
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        response = model.generate_content(prompt)
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        return f"LLM 해설 생성 실패: {e}"
    
# --- ▼▼▼ [추가] 스파이크 분석용 LLM 호출 함수 ▼▼▼ ---
def call_gemini_for_spike_analysis(spike_info: str, article_titles: list) -> str:
    """LLM을 호출하여 스파이크 발생 원인을 분석합니다."""
    if not article_titles:
        return "분석의 근거가 되는 당일 기사가 없습니다."
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "LLM API 키가 없습니다."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        당신은 데이터 분석가이자 시장 분석 전문가입니다.
        
        ### 분석 대상 데이터:
        1. **포착된 이상 징후 (Spike)**: {spike_info}
        2. **관련일 주요 뉴스 헤드라인 (상위 10개)**: {', '.join(article_titles)}

        ### 요청:
        위 뉴스 헤드라인들을 근거로, 포착된 이상 징후(스파이크)의 가장 유력한 원인을 1~2문장으로 명확하게 분석해주세요.
        """
        response = model.generate_content(prompt)
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        return f"LLM 분석 실패: {e}"

# --- ▼▼▼ [신규 추가] 리스크 코멘트 생성을 위한 LLM 호출 함수 ▼▼▼ ---
def call_gemini_for_risk_commentary(topic_name, sentiment_drop, related_titles):
    """LLM을 호출하여 감성 급락 원인에 대한 해설을 생성합니다."""
    if not related_titles: related_titles = ["관련 기사 제목 없음"]
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "LLM API 키 없음."

        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)

        titles_str = "- " + "\n- ".join(related_titles[:5]) # 최대 5개 제목 사용

        prompt = f"""
        당신은 시장 리스크 분석가입니다.
        오늘 '{topic_name}' 토픽에서 평균 대비 {sentiment_drop:.2f} 만큼의 감성 점수 급락이 관측되었습니다.

        ### 관련 기사 제목 (참고용):
        {titles_str}

        ### 요청:
        위 정보를 바탕으로, '{topic_name}' 토픽의 감성 점수가 급락한 가장 유력한 **원인**과 이로 인해 발생할 수 있는 잠재적 **영향**을 각각 한 문장으로 간결하게 분석해주세요.

        ### 분석 결과:
        - **원인**: (원인 분석)
        - **잠재 영향**: (잠재 영향 분석)
        """
        response = model.generate_content(prompt)
        # 결과에서 "원인:", "잠재 영향:" 부분을 추출하여 결합
        lines = response.text.strip().split('\n')
        commentary = " ".join([line.split(':', 1)[1].strip() for line in lines if ':' in line])
        return commentary if commentary else "LLM 분석 실패"

    except Exception as e:
        return f"LLM 해설 생성 실패: {e}"
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---



# --- 섹션별 컨텐츠 생성 함수 ---
# --- ▼▼▼ [신규 추가] 일간 리스크 신호 섹션 함수 (daily_report.py 로직 기반) ▼▼▼ ---
def _section_daily_risk_signals_commentary(data):
    """토픽별 감성 점수 급락 현황 분석 및 LLM 해설 추가"""
    lines = [_section_header("1.5 오늘의 리스크 신호 (감성 급락)", level=3)] # 레벨 조정
    lines.append("> 토픽별 일일 감성 점수의 급락 패턴을 탐지하고 그 원인을 분석합니다.\n")

    df_sentiment = data.get("sentiment") # _load_data 함수 필요
    meta_items = data.get("meta_items", []) # _load_data 함수 필요

    if df_sentiment is None or df_sentiment.empty:
        lines.append("> - 감성 데이터 없음\n")
        return "\n".join(lines)
    if 'semantic_key' not in df_sentiment.columns:
        lines.append("> - 감성 데이터에 'semantic_key' 컬럼 없음\n")
        return "\n".join(lines)

    risky_topics_data = [] # 테이블용 데이터
    today_date_dt = datetime.now()
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
    df_sentiment = df_sentiment[df_sentiment['semantic_key'] != "Uncategorized"]

    # master_topics 로드 (semantic_key -> keywords 매핑용)
    master_topics = load_json("data/dictionaries/master_topics.json", {})

    for key, group in df_sentiment.groupby('semantic_key'):
        group = group[group['date'] <= today_date_dt].sort_values('date').tail(8)
        if len(group) < 8: continue

        today_row = group.iloc[-1]
        if today_row['date'].strftime('%Y-%m-%d') != today_date_dt.strftime('%Y-%m-%d'): continue

        past_7_days = group.iloc[:-1]
        if past_7_days.empty: continue

        today_score = today_row['avg_sentiment']
        avg_7_days = past_7_days['avg_sentiment'].mean()
        std_7_days = past_7_days['avg_sentiment'].std()

        threshold_std = avg_7_days - 1.5 * std_7_days if pd.notna(std_7_days) and std_7_days > 0 else avg_7_days * 0.9
        threshold_pct = avg_7_days * 0.8
        is_risky = pd.notna(today_score) and pd.notna(avg_7_days) and \
                   (today_score < threshold_std or today_score < threshold_pct)

        if is_risky:
            sentiment_drop = avg_7_days - today_score
            topic_keywords = master_topics.get(key, []) # semantic_key에 해당하는 키워드 찾기

            # 관련 기사 제목 추출
            related_titles = []
            if topic_keywords:
                 kw_pattern = re.compile('|'.join(re.escape(kw) for kw in topic_keywords), re.IGNORECASE)
                 related_titles = [
                     item.get("title", "") for item in meta_items
                     if item.get("title") and kw_pattern.search(item.get("title", ""))
                 ][:5] # 최대 5개

            # LLM 해설 생성
            llm_comment = call_gemini_for_risk_commentary(key, sentiment_drop, related_titles)

            risky_topics_data.append({
                "토픽": key,
                "하락 폭": _fmt_float(sentiment_drop, 3),
                "오늘 점수": _fmt_float(today_score, 3),
                "7일 평균": _fmt_float(avg_7_days, 3),
                "LLM 해설": llm_comment
            })

    if not risky_topics_data:
        lines.append("> 금일 감성 급락으로 인한 리스크 신호는 포착되지 않았습니다.\n")
    else:
        df_risky = pd.DataFrame(risky_topics_data).sort_values(by="하락 폭", ascending=False)
        lines.append(_to_markdown_table(df_risky))

    return "\n".join(lines)
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def _section_activity_spikes(meta_items: list):
    lines = [_section_header("1. 시장 활동량 및 이상 징후")]
    lines.append("> 최근 30일간의 기사량 변화와 금일 발생한 이상 급등(Spike) 현황 및 원인을 분석합니다.\n")
    lines.append(_insert_image(os.path.join(FIG_DIR, "timeseries.png"), "최근 30일 기사량 추이"))
    
    df_spikes = _safe_read_csv(os.path.join(EXPORT_DIR, "timeseries_spikes_enhanced.csv"))
    if not df_spikes.empty:
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_spikes = df_spikes[df_spikes['date'] == today_str]
        if not today_spikes.empty:
            lines.append("### 금일 발생 스파이크 분석\n")
            lines.append("> [!NOTE] 금일 시장의 관심이 집중된 이상 급등 현상이 포착되었습니다.\n")
            lines.append(_to_markdown_table(today_spikes))
            
            # --- ▼▼▼ [수정] LLM 해설 생성 로직 추가 ▼▼▼ ---
            today_articles = [
                item.get("title", "") for item in meta_items if item.get("title")
            ][:10] # LLM 컨텍스트로 사용할 당일 기사 제목 (최대 10개)
            
            analysis_results = []
            for _, spike_row in today_spikes.iterrows():
                spike_info = f"지표: '{spike_row['metric']}', 값: {spike_row['value']}, Z-Score: {spike_row['z_score']:.2f}"
                llm_commentary = call_gemini_for_spike_analysis(spike_info, today_articles)
                analysis_results.append(llm_commentary)

            lines.append("**해설**:")
            for result in analysis_results:
                lines.append(f"- {result}")
            lines.append("")
            # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
        else:
            lines.append("> 금일 유의미한 스파이크는 포착되지 않았습니다.\n")
    return "\n".join(lines)

def _section_hot_signals(data): # data 딕셔너리를 인자로 받음
    lines = [_section_header("2. 오늘의 급등 신호", level=3)] # 레벨 조정
    lines.append("> 전일 및 주간 평균 대비 언급량이 급증하고 모멘텀이 높은 신호를 분석합니다.\n")

    # Load hot signals (daily_hot_signals.csv preferred)
    df_hot = _safe_read_csv(os.path.join(EXPORT_DIR, "daily_hot_signals.csv"))
    if df_hot.empty:
        # Fallback to trend_strength if hot signals file not found
        df_trends = _safe_read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv"))
        if not df_trends.empty:
            # Calculate hot signals from trends_df if needed
            df_hot = df_trends[(df_trends['z_like'] > 1.5) & (df_trends['diff'] > 0)].head(10).copy()
        else:
            lines.append("> 금일 유의미한 급등 신호는 포착되지 않았습니다.\n")
            return "\n".join(lines)

    # Check again if df_hot is still empty after fallback
    if df_hot.empty:
        lines.append("> 금일 유의미한 급등 신호는 포착되지 않았습니다.\n")
        return "\n".join(lines)

    meta_items = data.get('meta_items', []) # data 딕셔너리에서 meta_items 가져오기

    commentaries = []
    for _, row in df_hot.iterrows():
        term = row.get('term', '') # Use .get for safety
        if not term: continue # Skip if term is empty

        # --- ▼▼▼ 수정된 부분 ▼▼▼ ---
        # Find related titles from meta_items, ensuring item is a dict
        related_titles = [
            item.get("title", "") # item이 dict일 때만 .get() 호출
            for item in meta_items
            # Check if item is a dictionary before accessing .get()
            if isinstance(item, dict) and item.get("title") and term.lower() in item.get("title", "").lower()
        ][:5] # Limit to 5 titles
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        # Generate commentary using LLM
        if related_titles:
            commentary = call_gemini_for_signal_commentary(term, related_titles)
            commentaries.append(commentary)
        else:
            commentaries.append("당일 연관 기사 부족으로 분석 불가") # No related articles found

    # Add commentaries to the DataFrame
    # Ensure the length matches, handle potential errors if lengths differ
    if len(commentaries) == len(df_hot):
         df_hot['해설'] = commentaries
    else:
         # Fallback if lengths don't match (e.g., add placeholder)
         df_hot['해설'] = ["해설 생성 오류"] * len(df_hot)
         print(f"[WARN] Length mismatch between hot signals ({len(df_hot)}) and commentaries ({len(commentaries)}).")


    # Prepare table for Markdown
    lines.append(_to_markdown_table(df_hot[['term', 'cur', 'diff', 'z_like', '해설']].rename(columns={
        'term': '급등 신호', 'cur': '금일 언급량', 'diff': '전일 대비 증가', 'z_like': '모멘텀'
    })))

    return "\n".join(lines)

def _section_competitor_events():
    lines = [_section_header("3. 경쟁사 주요 활동")]
    lines.append("> 금일 발생한 경쟁사의 주요 이벤트(신제품, 파트너십, 투자 등)를 추적합니다.\n")
    df_events = _safe_read_csv(os.path.join(EXPORT_DIR, "events.csv"))
    lines.append(_to_markdown_table(df_events))
    return "\n".join(lines)

def _section_top_articles(data): # data 딕셔너리를 인자로 받음
    lines = [_section_header("4. 주요 기사", level=3)] # 레벨 조정
    lines.append("> 오늘의 핵심 이슈와 가장 관련성이 높은 기사 목록과 요약입니다.\n")

    df_articles = _safe_read_csv(os.path.join(EXPORT_DIR, "today_article_list.csv"))
    if df_articles.empty:
        lines.append("> - 선정된 주요 기사 없음\n")
        return "\n".join(lines)

    meta_items = data.get('meta_items', []) # data 딕셔너리에서 meta_items 가져오기

    # --- ▼▼▼ 수정된 부분 ▼▼▼ ---
    # Create url_to_body mapping, ensuring item is a dict
    url_to_body = {
        item.get("url"): (item.get("body") or item.get("description", ""))
        for item in meta_items
        # Check if item is a dictionary before accessing .get()
        if isinstance(item, dict) and item.get("url")
    }
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    # --- ▼▼▼ 기사 요약 로직 추가 ▼▼▼ ---
    for _, row in df_articles.iterrows():
        title = row.get("title", "제목 없음")
        url = row.get("url")

        lines.append(f"### [{title}]({url})") # Add link to title

        body = url_to_body.get(url, "") # Use the created mapping

        if body:
            summary = summarize_text_with_hf(body) # Use HF summarizer
            lines.append(summary)
        else:
            lines.append("> (기사 본문을 찾을 수 없어 요약을 생성하지 못했습니다.)")

        lines.append("\n---\n") # 기사 간 구분선 추가
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    return "\n".join(lines)

def _section_appendix_strong_signals():
    lines = [_section_header("참고: 최근 30일 누적 강한 신호 Top 5", level=3)]
    lines.append("> 장기적 관점의 시장 핵심 키워드입니다.\n")
    df_trends = _safe_read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv"))
    if not df_trends.empty:
        top5 = df_trends.sort_values(by="total", ascending=False).head(5)
        lines.append(_to_markdown_table(top5[['term', 'total', 'z_like']].rename(columns={
            'term': '누적 신호', 'total': '30일 누적 언급량', 'z_like': '현재 모멘텀'
        })))
    else:
        lines.append("> - 데이터 없음\n")
    return "\n".join(lines)


# --- ▼▼▼ [수정] main 함수 수정 (데이터 로드 및 새 섹션 호출 추가) ▼▼▼ ---
def main():
    """논문/기사 형식의 일간 상세 해설 리포트 생성 메인 함수"""
    print("[INFO] Generating Professional Daily Commentary Report...")
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. 분석에 필요한 모든 데이터 로드 (sentiment 포함)
    #    _load_data 함수를 daily_report.py 에서 가져오거나 여기에 정의해야 함
    #    여기서는 필요한 데이터만 직접 로드하는 것으로 가정
    meta_items = load_json(latest(os.path.join("data", "news_meta_*.json")), [])
    df_spikes = _safe_read_csv(os.path.join(EXPORT_DIR, "timeseries_spikes_enhanced.csv"))
    df_trends = _safe_read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv"))
    df_events = _safe_read_csv(os.path.join(EXPORT_DIR, "events.csv"))
    df_sentiment = _safe_read_csv(os.path.join(EXPORT_DIR, "daily_topic_sentiment.csv")) # 감성 데이터 로드

    report_data = { # 섹션 함수에 전달할 데이터 딕셔너리
        "meta_items": meta_items,
        "sentiment": df_sentiment,
        # 필요한 다른 데이터도 추가 가능
    }


    # 2. Executive Summary 생성을 위한 컨텍스트 준비 (기존 로직 유지)
    today_spikes = df_spikes[df_spikes['date'] == today_str] if not df_spikes.empty else pd.DataFrame()
    hot_signals = df_trends[(df_trends['z_like'] > 1.5) & (df_trends['diff'] > 0)] if not df_trends.empty else pd.DataFrame()
    summary_context = {
        "spike_info": f"{len(today_spikes)}건의 스파이크 발생 ({', '.join(today_spikes['metric'])})" if not today_spikes.empty else "특이사항 없음",
        "top_hot_signal": hot_signals.iloc[0]['term'] if not hot_signals.empty else "해당 없음",
        "top_event": f"{df_events.iloc[0]['title']} ({df_events.iloc[0]['types']})" if not df_events.empty else "해당 없음"
    }

    # 3. 리포트 컨텐츠 조립
    lines = [f"# 일간 상세 해설 리포트\n<div class='subtitle'>Date: {today_str} | Generated by Market Intelligence Team</div>\n"]
    lines.append(_section_header("Executive Summary", level=2))
    summary_text = call_gemini_for_daily_exec_summary(summary_context)
    lines.append(f"<div class='executive-summary'>{summary_text}</div>\n")

    lines.append(_section_activity_spikes(report_data)) # 데이터 전달
    lines.append(_section_daily_risk_signals_commentary(report_data)) # <-- 새 섹션 호출
    lines.append(_section_hot_signals(report_data)) # 데이터 전달
    lines.append(_section_competitor_events()) # 기존 함수 (필요시 데이터 전달)
    lines.append(_section_top_articles(report_data)) # 데이터 전달
    lines.append(_section_appendix_strong_signals()) # 기존 함수 (필요시 데이터 전달)

    # 4. 파일 저장 및 변환
    report_content = "\n".join(lines)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(report_content)

    build_html_from_md(OUT_MD, OUT_HTML) # HTML 생성 함수 호출
    print(f"[SUCCESS] Professional daily commentary report generated: {OUT_MD}, {OUT_HTML}")

if __name__ == "__main__":
    main()
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---