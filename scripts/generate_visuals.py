import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Matplotlib 한글 폰트 설정 ---
def ensure_fonts():
    import matplotlib.font_manager as fm
    
    # 시스템에 설치된 Nanum 폰트 또는 Noto Sans CJK 폰트 경로 탐색
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_gothic = next((path for path in font_paths if 'NanumGothic' in path), None)
    noto_sans_cjk = next((path for path in font_paths if 'NotoSansKR' in path or 'NotoSansCJK' in path), None)

    font_path = nanum_gothic or noto_sans_cjk
    
    if font_path:
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_name
    else:
        # 적절한 폰트가 없는 경우 기본 폰트로 설정 (경고 메시지 출력)
        print("[WARN] NanumGothic or NotoSansKR font not found. Please install it for proper Korean display.")
        plt.rcParams['font.family'] = 'sans-serif'
        
    plt.rcParams['axes.unicode_minus'] = False
    print(f"[INFO] Matplotlib font set to: {plt.rcParams['font.family']}")


def plot_heatmap(df, topics_map):
    """ 1. 기업x토픽 집중도 히트맵 생성 """
    try:
        heatmap_data = df.pivot_table(index='org', columns='topic', values='hybrid_score', aggfunc='sum').fillna(0)
        
        # 데이터가 너무 많으면 상위 20개 기업만 선택
        if len(heatmap_data) > 20:
            top_orgs = heatmap_data.sum(axis=1).nlargest(20).index
            heatmap_data = heatmap_data.loc[top_orgs]

        if heatmap_data.empty: return

        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, cmap="viridis", linewidths=.5)
        
        # 토픽 ID를 키워드로 변경
        plt.xticks(ticks=range(len(heatmap_data.columns)), labels=[topics_map.get(f"topic_{col}", col) for col in heatmap_data.columns], rotation=45, ha='right')
        plt.title('기업별 토픽 집중도 (Hybrid Score)', fontsize=16)
        plt.xlabel('토픽', fontsize=12)
        plt.ylabel('기업', fontsize=12)
        plt.tight_layout()
        plt.savefig('outputs/fig/matrix_heatmap.png', dpi=150)
        plt.close()
        print("[INFO] Saved matrix_heatmap.png")
    except Exception as e:
        print(f"[ERROR] Failed to generate heatmap: {e}")


def plot_topic_share(df, topics_map, top_n_topics=3):
    """ 2. 상위 토픽별 점유율 파이 차트 생성 """
    try:
        top_topics = df.groupby('topic')['hybrid_score'].sum().nlargest(top_n_topics).index
        
        for topic in top_topics:
            topic_df = df[df['topic'] == topic].copy()
            # 점유율이 낮은 기업은 'Others'로 묶기
            top_orgs = topic_df.nlargest(5, 'topic_share')
            if len(topic_df) > 5:
                others_share = topic_df[~topic_df['org'].isin(top_orgs['org'])]['topic_share'].sum()
                others_row = pd.DataFrame([{'org': 'Others', 'topic_share': others_share}])
                top_orgs = pd.concat([top_orgs, others_row], ignore_index=True)

            plt.figure(figsize=(10, 8))
            plt.pie(top_orgs['topic_share'], labels=top_orgs['org'], autopct='%1.1f%%', startangle=140, pctdistance=0.85)
            plt.title(f'토픽 점유율: {topics_map.get(f"topic_{topic}", topic)}', fontsize=16)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f'outputs/fig/topic_share_{topic}.png', dpi=150)
            plt.close()
            print(f"[INFO] Saved topic_share_{topic}.png")
    except Exception as e:
        print(f"[ERROR] Failed to generate pie charts: {e}")


def plot_company_focus(df, top_n_orgs=3):
    """ 3. 상위 기업별 집중도 바 차트 생성 """
    try:
        top_orgs = df.groupby('org')['hybrid_score'].sum().nlargest(top_n_orgs).index

        for org in top_orgs:
            org_df = df[df['org'] == org].nlargest(8, 'company_focus')
            if org_df.empty: continue

            plt.figure(figsize=(12, 7))
            sns.barplot(data=org_df, x='topic', y='company_focus', palette='coolwarm')
            plt.title(f'\'{org}\'의 토픽별 집중도', fontsize=16)
            plt.xlabel('토픽 ID', fontsize=12)
            plt.ylabel('집중도 점수', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'outputs/fig/company_focus_{org}.png', dpi=150)
            plt.close()
            print(f"[INFO] Saved company_focus_{org}.png")
    except Exception as e:
        print(f"[ERROR] Failed to generate bar charts: {e}")


def main():
    """ 메인 실행 함수 """
    # 폰트 설정
    ensure_fonts()
    
    # 데이터 로드
    try:
        df = pd.read_csv('outputs/export/company_topic_matrix_long.csv')
    except FileNotFoundError:
        print("[ERROR] company_topic_matrix_long.csv not found. Please run module_d.py first.")
        return

    # 토픽 키워드 맵 로드
    try:
        with open('outputs/topics.json', 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
        topics_map = {f"topic_{t['topic_id']}": ", ".join(w['word'] for w in t['top_words'][:2]) for t in topics_data['topics']}
    except Exception:
        topics_map = {}
        print("[WARN] topics.json not found or failed to parse. Topic IDs will be used as labels.")

    # 시각화 함수 호출
    os.makedirs('outputs/fig', exist_ok=True)
    plot_heatmap(df, topics_map)
    plot_topic_share(df, topics_map)
    plot_company_focus(df)
    
    print("\n[SUCCESS] All visualizations have been generated in 'outputs/fig/'")

if __name__ == '__main__':
    import json
    main()