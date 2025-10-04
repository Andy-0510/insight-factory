import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import networkx as nx
import itertools


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


def plot_idea_score_distribution(ideas: list, output_path: str = 'outputs/fig/idea_score_distribution.png'):
    """ 아이디어별 점수 분포 바 차트 생성 (Market, Urgency, Feasibility, Risk) """
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    if not ideas:
        print("[WARN] No ideas provided for score chart.")
        return

    # 아이디어 이름은 최대 15자까지만 표시
    labels = [idea.get("idea", "")[:15] + "…" if len(idea.get("idea", "")) > 15 else idea.get("idea", "") for idea in ideas]
    market = [idea["score_breakdown"]["market"] for idea in ideas]
    urgency = [idea["score_breakdown"]["urgency"] for idea in ideas]
    feasibility = [idea["score_breakdown"]["feasibility"] for idea in ideas]
    risk = [idea["score_breakdown"]["risk"] for idea in ideas]

    x = np.arange(len(ideas))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - 1.5*width, market, width, label='Market')
    bars2 = ax.bar(x - 0.5*width, urgency, width, label='Urgency')
    bars3 = ax.bar(x + 0.5*width, feasibility, width, label='Feasibility')
    bars4 = ax.bar(x + 1.5*width, risk, width, label='Risk')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel("Score (0.0 ~ 1.0)")
    ax.set_title("아이디어별 점수 분포", fontsize=16)
    ax.legend()

    # ✅ 막대 위에 값 표시
    for bars in [bars1, bars2, bars3, bars4]:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved idea_score_distribution.png")


def plot_company_network_from_json(json_path="outputs/company_network.json",
                                   output_path="outputs/fig/company_network.png",
                                   top_edges=30, top_nodes=10):
    """기업 네트워크를 단순화하여 시각화
       - 상위 top_edges개의 관계만 표시
       - 중심성 상위 top_nodes 기업만 강조
    """
    import json, os
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import matplotlib.font_manager as fm

    if not os.path.exists(json_path):
        print("[WARN] company_network.json not found")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges = data.get("edges", [])
    if not edges:
        print("[WARN] No edges in company_network.json")
        return

    # --- 1. 상위 top_edges 엣지만 선택 ---
    edges_sorted = sorted(edges, key=lambda e: e["weight"], reverse=True)[:top_edges]

    G = nx.Graph()
    for e in edges_sorted:
        a, b, w = e["source"], e["target"], float(e["weight"])
        G.add_edge(a, b, weight=w)

    if G.number_of_nodes() == 0:
        print("[WARN] Graph empty")
        return

    # --- 2. 중심성 계산 후 상위 top_nodes만 강조 ---
    deg = nx.degree_centrality(G)
    top_nodes_sorted = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_nodes]
    top_nodes = {n for n, _ in top_nodes_sorted}

    # --- 3. 폰트 지정 (NanumGothic) ---
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
    else:
        font_name = "sans-serif"

    # --- 4. 레이아웃 & 시각화 ---
    pos = nx.kamada_kawai_layout(G)  # 더 균형 잡힌 배치
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    w_arr = np.array(weights, dtype=float)
    q95 = np.quantile(w_arr, 0.95)
    w_arr = np.minimum(w_arr, q95)
    w_norm = (0.6 + 1.8 * (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-6)).tolist()

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_edges(G, pos, width=w_norm, edge_color="#7a7a7a", alpha=0.35)

    # 중심 기업은 빨간색, 나머지는 파란색
    node_colors = ["#e74c3c" if n in top_nodes else "#86b6f6" for n in G.nodes()]
    node_sizes = [1200 if n in top_nodes else 600 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           edgecolors="#333", linewidths=0.6, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="#222", font_family=plt.rcParams['font.family'][0])

    plt.title("기업 경쟁/협력 네트워크 (핵심 관계망)", fontsize=14, fontname=plt.rcParams['font.family'][0])
    plt.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved simplified company_network.png with {len(G.nodes())} nodes and {len(G.edges())} edges")




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

    #plot_idea_score_distribution
    try:
        with open('outputs/biz_opportunities.json', 'r', encoding='utf-8') as f:
            ideas_data = json.load(f)
        top_ideas = sorted(ideas_data["ideas"], key=lambda it: it.get("score", 0), reverse=True)[:5]
        plot_idea_score_distribution(top_ideas)
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to generate idea score chart: {e}")
        traceback.print_exc()
    
    # 기업 네트워크 시각화(신규)
    try:
        plot_company_network_from_json("outputs/company_network.json", "outputs/fig/company_network.png")
    except Exception as e:
        print(f"[WARN] company network visualization failed: {repr(e)}")


    # 시각화 함수 호출
    os.makedirs('outputs/fig', exist_ok=True)
    plot_heatmap(df, topics_map)
    plot_topic_share(df, topics_map)
    plot_company_focus(df)
    
    print("\n[SUCCESS] All visualizations have been generated in 'outputs/fig/'")

if __name__ == '__main__':
    import json
    main()