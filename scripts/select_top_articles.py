import pandas as pd
from src.utils import load_json, latest
import os

TOP_N = 3 # 최종 선정할 기사 수

def select_articles():
    """
    그날의 핵심 토픽 및 이벤트와 가장 관련성 높은 기사 TOP 3를 선정합니다.
    """
    # 1. 분석에 필요한 데이터 로드
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        print("[WARN] No news_meta file found. Skipping article selection.")
        return

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
            
        if score > 0:
            scored_articles.append({
                "title": title,
                "url": item.get("url"),
                "score": score
            })

    # 3. 점수가 높은 순으로 정렬하여 상위 N개 선택 및 저장
    if not scored_articles:
        print("[INFO] No relevant articles to select.")
        # 빈 파일이라도 생성하여 워크플로우 에러 방지
        df_top_articles = pd.DataFrame(columns=["title", "url"])
    else:
        df_top_articles = pd.DataFrame(scored_articles)
        df_top_articles = df_top_articles.sort_values(by="score", ascending=False).head(TOP_N)
        df_top_articles = df_top_articles.drop_duplicates(subset=['title'])
        df_top_articles = df_top_articles[['title', 'url']]

    output_path = "outputs/export/today_article_list.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_top_articles.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Top {len(df_top_articles)} articles saved to {output_path}")

if __name__ == "__main__":
    select_articles()
