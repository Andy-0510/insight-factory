import pandas as pd
from src.utils import load_json, latest
import os
from datetime import datetime

TOP_N = 3 # 최종 선정할 기사 수
OUTPUT_CSV = "outputs/export/today_article_list.csv"
CUMULATIVE_OUTPUT_CSV = "outputs/export/daily_signal_counts.csv"
SCORE_THRESHOLD = 0.0 # 점수가 0.0 이상인 기사를 '관심 기사'로 간주하고 저장 (토픽 ∩ Event)


def select_articles():
    """
    그날의 핵심 토픽 및 이벤트와 가장 관련성 높은 기사를 선정합니다.
    월간 실행 시에는 이 작업을 건너뜁니다.
    """
    # --- ▼▼▼▼▼ [추가] 월간 실행 시 함수를 즉시 종료 ▼▼▼▼▼ ---
    is_monthly_run = os.getenv("MONTHLY_RUN", "false").lower() == "true"
    if is_monthly_run:
        print("[INFO] Monthly Run: Skipping daily top article selection.")
        return
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, [])
    
    try:
        df_strength = pd.read_csv("outputs/export/trend_strength.csv")
        top_keywords = set(df_strength.head(5)['term'])
    except FileNotFoundError:
        top_keywords = set()

    try:
        df_events = pd.read_csv("outputs/export/events.csv")
        event_titles = set(df_events['title'])
    except FileNotFoundError:
        event_titles = set()

    # 2. 각 기사별로 점수 계산
    scored_articles = []
    for item in meta_items:
        score = 0
        title = item.get("title", "")
        body = item.get("body", "")
        content = f"{title} {body}".lower()

        # 점수 로직: 핵심 토픽이 포함되면 +2점
        for keyword in top_keywords:
            if keyword.lower() in content:
                score += 2
        
        # 점수 로직: 주요 이벤트 기사면 +3점
        if title in event_titles:
            score += 3
        
        # --- ▼▼▼ 점수가 임계값을 넘는 모든 기사를 리스트에 추가 ▼▼▼ ---
        if score >= SCORE_THRESHOLD:
            scored_articles.append({
                "title": title,
                "url": item.get("url"),
                "score": score
            })
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---    

    if not scored_articles:
        df_top_articles = pd.DataFrame(columns=["title", "url"])
    else:
        # 점수 높은 순으로 정렬하여 리포트에는 TOP_N 개까지만 표시
        df_top_articles = pd.DataFrame(scored_articles).sort_values(by="score", ascending=False).drop_duplicates(subset=['title']).head(TOP_N)
        df_top_articles = df_top_articles[['title', 'url']]
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_top_articles.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Top articles for report (max 5) saved to {OUTPUT_CSV}")

    # --- ▼▼▼ '관심 기사 수'를 scored_articles의 전체 개수로 변경 ▼▼▼ ---
    today_date = datetime.now().strftime("%Y-%m-%d")
    signal_count = len(scored_articles) # TOP_N이 아닌, 점수를 넘긴 모든 기사의 수
    
    new_data = {"date": today_date, "signal_article_count": signal_count}
    
    if os.path.exists(CUMULATIVE_OUTPUT_CSV):
        df_existing = pd.read_csv(CUMULATIVE_OUTPUT_CSV)
        df_existing = df_existing[df_existing["date"] != today_date]
        df_final = pd.concat([df_existing, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df_final = pd.DataFrame([new_data])
        
    df_final.sort_values(by="date", inplace=True)
    df_final.to_csv(CUMULATIVE_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Daily signal count ({signal_count}) saved to {CUMULATIVE_OUTPUT_CSV}")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

if __name__ == "__main__":
    select_articles()
