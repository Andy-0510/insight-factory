import os
import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict, Counter

from src.utils import load_json
from src.config import load_config
from .daily_report import (_fmt_int, _safe_read_csv, _to_markdown_table, _section_header, build_html_from_md_new, _exists, _insert_images)

# --- 설정 ---
ROOT_OUTPUT_DIR = "outputs"
DAILY_ARCHIVE_DIR = os.path.join(ROOT_OUTPUT_DIR, "daily")
FIG_DIR = os.path.join(ROOT_OUTPUT_DIR, "fig")
OUT_MD = os.path.join(ROOT_OUTPUT_DIR, "weekly_report.md")
OUT_HTML = os.path.join(ROOT_OUTPUT_DIR, "weekly_report.html")
TARGET_COMPETITORS = ["삼성디스플레이", "LG디스플레이", "BOE", "CSOT", "Visionox", "Tianma"]

# --- ▼▼▼▼▼▼ 주간 경영 요약을 위한 LLM 호출 함수 ▼▼▼▼▼▼ ---
def call_gemini_for_weekly_summary(context):
    """LLM을 호출하여 주간 경영 요약을 생성합니다."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)
        print(f"[INFO] Using Gemini model for weekly summary: {model_name}")

        prompt = f"""
        당신은 디스플레이 산업 전문 수석 비즈니스 분석가입니다. 아래는 지난 한 주간의 시장 데이터 요약입니다. 이 데이터를 종합하여 경영진 및 팀 리더를 위한 '주간 인텔리전스 요약'을 작성해주세요.

        ### 주간 데이터 요약:
        {json.dumps(context, ensure_ascii=False, indent=2)}

        ### 작성 가이드:
        1. **핵심 맥락**: 데이터를 관통하는 가장 중요한 시장의 흐름 1~2가지를 설명해주세요.
        2. **전략적 인사이트**: 이 흐름이 우리 비즈니스에 주는 기회 또는 위협 요소를 분석해주세요.
        3. **추천 Action Items**: 다음 주에 팀이 우선적으로 실행해야 할 구체적인 액션 아이템 2가지를 제안해주세요.
        4. 각 항목을 명확하게 구분하고, 전문가의 시각에서 간결하고 명확한 톤으로 작성해주세요.

        ### 출력 형식 (Markdown):
        #### 핵심 맥락
        - (분석 내용)

        #### 전략적 인사이트
        - (분석 내용)

        #### 추천 Action Items
        - (실행 제안 1)
        - (실행 제안 2)
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[ERROR] Gemini 주간 요약 생성 실패: {e}")
        return "LLM 요약 생성 중 오류가 발생했습니다."

# --- ▼▼▼▼▼▼ 주간 약한 신호 분석을 위한 LLM 호출 함수 ▼▼▼▼▼▼ ---
def call_gemini_for_weekly_insight(weak_signals):
    """LLM을 호출하여 주간 약한 신호의 의미를 분석합니다."""
    if not weak_signals:
        return "금주에 주목할 만한 신규 약한 신호가 포착되지 않았습니다."

    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        cfg = load_config()
        model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash-001")
        model = genai.GenerativeModel(model_name)
        print(f"[INFO] Using Gemini model for weekly weak signal insight: {model_name}")

        prompt = f"""
        당신은 미래 기술 트렌드 분석가입니다. 아래는 지난 한 주간 포착된 초기 신호(Weak Signals) 목록입니다.

        ### 주간 초기 신호 목록:
        {json.dumps(weak_signals, ensure_ascii=False, indent=2)}

        ### 분석 요청:
        1. 목록에서 가장 중요하고 잠재력 있는 신호 2~3개를 선별해주세요.
        2. 각 신호가 무엇을 의미하는지, 그리고 왜 지금 주목해야 하는지에 대한 해석을 한 문장으로 요약해주세요.
        3. 분석 결과를 아래 마크다운 형식으로만 답변해주세요. 다른 설명은 필요 없습니다.

        ### 출력 형식 (Markdown):
        - **[신호명 1]:** (분석 및 해석 요약)
        - **[신호명 2]:** (분석 및 해석 요약)
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[ERROR] Gemini 주간 약한 신호 분석 실패: {e}")
        return "LLM 분석 중 오류가 발생했습니다."

def load_weekly_data(days=7):
    print(f"[INFO] Loading data from the last {days} days...")
    aggregated_data = {
        "start_date": "", "end_date": "", "total_articles": 0, "all_keywords": [],
        "all_events": pd.DataFrame(),
        "trend_strength_history": defaultdict(list),
        "all_weak_signals": pd.DataFrame() # 약한 신호 데이터프레임 추가
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)
    aggregated_data["start_date"] = start_date.strftime("%Y-%m-%d")
    aggregated_data["end_date"] = end_date.strftime("%Y-%m-%d")

    for i in range(days):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        date_folders = sorted(glob.glob(os.path.join(DAILY_ARCHIVE_DIR, date_str, "*")))
        if not date_folders: continue
        
        latest_daily_folder = date_folders[-1]
        print(f"[DEBUG] Loading from: {latest_daily_folder}")

        ts_path = os.path.join(latest_daily_folder, "trend_timeseries.json")
        kw_path = os.path.join(latest_daily_folder, "keywords.json")
        events_path = os.path.join(latest_daily_folder, "export", "events.csv")
        trends_path = os.path.join(latest_daily_folder, "export", "trend_strength.csv")
        weak_signals_path = os.path.join(latest_daily_folder, "export", "weak_signals.csv")
        
        ts_data = load_json(ts_path, {"daily": []}) if os.path.exists(ts_path) else {"daily": []}
        keywords_data = load_json(kw_path, {"keywords": []}) if os.path.exists(kw_path) else {"keywords": []}
        events_df = _safe_read_csv(events_path)
        trends_df = _safe_read_csv(trends_path)
        trends_df = _safe_read_csv(os.path.join(latest_daily_folder, "export", "trend_strength.csv"))
        weak_signals_df = _safe_read_csv(weak_signals_path)

        if ts_data and ts_data.get("daily"):
            today_count = next((d['count'] for d in reversed(ts_data['daily']) if d['date'] == date_str), 0)
            aggregated_data["total_articles"] += today_count
        if keywords_data and keywords_data.get("keywords"):
            aggregated_data["all_keywords"].extend(keywords_data["keywords"])
        if not events_df.empty:
            aggregated_data["all_events"] = pd.concat([aggregated_data["all_events"], events_df], ignore_index=True)
        if not trends_df.empty:
            for _, row in trends_df.iterrows():
                aggregated_data["trend_strength_history"][row['term']].append({
                    "date": date_str,
                    "cur": row.get('cur', 0),
                    "z_like": row.get('z_like', 0.0)
                })
        if not weak_signals_df.empty:
            aggregated_data["all_weak_signals"] = pd.concat([aggregated_data["all_weak_signals"], weak_signals_df], ignore_index=True)
        return aggregated_data
    
# --- ▼▼▼▼▼▼ 주간 경영 요약 섹션 최종 구현 ▼▼▼▼▼▼ ---
def _section_weekly_summary(data):
    """주간 경영 요약 섹션"""
    
    # LLM에 전달할 컨텍스트 데이터 준비
    keyword_scores = defaultdict(float)
    for k in data['all_keywords']: keyword_scores[k['keyword']] += k.get('score', 0.0)
    top_keywords = [k for k, v in sorted(keyword_scores.items(), key=lambda item: item[1], reverse=True)[:5]]

    competitor_mentions = Counter()
    for term in TARGET_COMPETITORS:
        history = data["trend_strength_history"].get(term, [])
        competitor_mentions[term] = sum(d.get('cur', 0) for d in history)
    top_competitors = [c for c, v in competitor_mentions.most_common(3) if v > 0]
    
    # --- ▼▼▼▼▼ [수정] 약한 신호 데이터가 비어있는 경우를 처리합니다 ▼▼▼▼▼ ---
    df_weak_signals = data.get("all_weak_signals")
    if df_weak_signals is not None and not df_weak_signals.empty:
        top_weak_signals = df_weak_signals.sort_values(by="z_like", ascending=False).drop_duplicates(subset=['term']).head(3)['term'].tolist()
    else:
        top_weak_signals = [] # 데이터가 없으면 빈 리스트로 처리
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    context = {
        "분석 기간": f"{data['start_date']} ~ {data['end_date']}",
        "주간 Top 키워드": top_keywords,
        "주간 활동량 Top 경쟁사": top_competitors,
        "주목할 만한 약한 신호": top_weak_signals
    }
    
    llm_summary = call_gemini_for_weekly_summary(context)
    
    basic_stats = f"""
- **분석 기간:** {data['start_date']} ~ {data['end_date']}
- **총 분석 기사 수:** {_fmt_int(data['total_articles'])}
- **주요 이벤트 발생 건수:** {_fmt_int(len(data['all_events'].drop_duplicates(subset=['title'])))}
"""
    
    return basic_stats + "\n" + llm_summary

# --- ▼▼▼▼▼▼ 시장 테마 및 거시적 흐름 분석 섹션 구현 ▼▼▼▼▼▼ ---
def _section_weekly_market_themes(data):
    """주간 시장 테마 분석 섹션"""
    keyword_scores = defaultdict(float)
    for k in data['all_keywords']:
        keyword_scores[k['keyword']] += k.get('score', 0.0)
    
    sorted_keywords = sorted(keyword_scores.items(), key=lambda item: item[1], reverse=True)[:10]
    df_top_keywords = pd.DataFrame(sorted_keywords, columns=["키워드", "주간 누적 점수"])
    
    lines = [_to_markdown_table(df_top_keywords)]
    # 생성된 워드클라우드 이미지를 리포트에 포함
    lines.append(_insert_images(os.path.join(FIG_DIR, "weekly_wordcloud.png"), OUT_MD, captions=["주간 키워드 워드클라우드"]))
    
    return "\n".join(lines)

# --- ▼▼▼▼▼▼ 경쟁사 동향 분석 섹션 구현 ▼▼▼▼▼▼ ---
def _section_weekly_competitor_trends(data):
    """주간 경쟁 동향 분석 섹션"""
    competitor_stats = []
    
    for competitor in TARGET_COMPETITORS:
        history = data["trend_strength_history"].get(competitor, [])
        if history:
            weekly_mentions = sum(day.get('cur', 0) for day in history)
            avg_z_like = sum(day.get('z_like', 0.0) for day in history) / len(history)
            competitor_stats.append({
                "경쟁사": competitor,
                "주간 총 언급량": weekly_mentions,
                "주간 평균 모멘텀": round(avg_z_like, 2)
            })

    if not competitor_stats:
        return "> 금주에 감지된 주요 경쟁사 활동이 없습니다."

    df_competitors = pd.DataFrame(competitor_stats)
    df_competitors = df_competitors.sort_values(by="주간 총 언급량", ascending=False)
    
    return _to_markdown_table(df_competitors)

# --- ▼▼▼▼▼▼ 잠재적 미래 성장 동력 섹션 구현 ▼▼▼▼▼▼ ---
def _section_weekly_future_signals(data):
    """주간 미래 신호 발견 섹션"""
    df_weak = data["all_weak_signals"]
    if df_weak.empty:
        return "> 금주에 포착된 신규 약한 신호가 없습니다."
        
    # 주간 동안 나타난 약한 신호를 z_like 점수 기준으로 정렬하여 상위 5개 선정
    top_weak_signals = df_weak.sort_values(by="z_like", ascending=False).drop_duplicates(subset=['term']).head(5)
    
    # LLM 분석을 위해 dict 리스트로 변환
    weak_signals_for_llm = top_weak_signals[['term', 'z_like', 'total']].to_dict('records')
    
    # LLM 호출하여 분석 결과 받기
    llm_interpretation = call_gemini_for_weekly_insight(weak_signals_for_llm)
    
    lines = [_to_markdown_table(top_weak_signals[['term', 'cur', 'z_like', 'total']])]
    lines.append("\n**AI 기반 주요 신호 해석:**\n")
    lines.append(llm_interpretation)
    
    return "\n".join(lines)

# --- ▼▼▼▼▼▼ 모멘텀 변화 섹션 구현 ▼▼▼▼▼▼ ---
def _section_weekly_momentum_change(data):
    """주요 신호 변화 추이 섹션"""
    
    # 주간 총 언급량 기준 상위 5개 신호 선정
    mention_counts = Counter()
    for term, history in data["trend_strength_history"].items():
        mention_counts[term] += sum(d['cur'] for d in history)
        
    top_5_terms = [term for term, count in mention_counts.most_common(5)]
    
    if not top_5_terms:
        return "> 금주에 추적할 만한 핵심 신호가 없습니다."

    momentum_data = []
    for term in top_5_terms:
        history = data["trend_strength_history"].get(term, [])
        # 일별 z_like 점수를 문자열로 표현
        trend_str = " → ".join([f"{d['z_like']:.1f}" for d in sorted(history, key=lambda x: x['date'])])
        momentum_data.append({
            "핵심 신호": term,
            "주간 총 언급량": mention_counts[term],
            "일별 모멘텀(z_like) 추이": trend_str
        })
        
    df_momentum = pd.DataFrame(momentum_data)

    lines = [_to_markdown_table(df_momentum)]
    lines.append(_insert_images(os.path.join(FIG_DIR, "weekly_strong_signals_barchart.png"), OUT_MD, captions=["주간 상승/하강 신호 Top 5"]))
    
    return "\n".join(lines)

def build_weekly_markdown():
    weekly_data = load_weekly_data(days=7)
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [f"# Weekly Intelligence ({today_str})"]

    lines.append(_section_header("1. 주간 경영 요약 및 전략적 시사점")); lines.append(_section_weekly_summary(weekly_data))
    lines.append(_section_header("2. 주간 시장 테마 및 거시적 흐름 분석")); lines.append(_section_weekly_market_themes(weekly_data))
    lines.append(_section_header("3. 경쟁사 활동 강도 및 전략 전환 경보")); lines.append(_section_weekly_competitor_trends(weekly_data))
    lines.append(_section_header("4. 잠재적 미래 성장 동력 및 초기 신호 발견")); lines.append(_section_weekly_future_signals(weekly_data))
    lines.append(_section_header("5. 핵심 트렌드 모멘텀 변화 및 우선순위 검토")); lines.append(_section_weekly_momentum_change(weekly_data))
    with open(OUT_MD, "w", encoding="utf-8") as f: f.write("\n".join(lines))
    return OUT_MD

def main():
    try:
        md_path = build_weekly_markdown()
        build_html_from_md_new(md_path, OUT_HTML)
        print(f"[INFO] Weekly report generated: {md_path}, {OUT_HTML}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[ERROR] Weekly report generation failed: {e}")

if __name__ == "__main__":
    main()