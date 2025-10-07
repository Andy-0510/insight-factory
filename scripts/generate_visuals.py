import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import networkx as nx
import itertools
import numpy as np
import matplotlib.font_manager as fm


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


def plot_keyword_network(keywords, docs, out_path="outputs/fig/keyword_network.png", topn=50, min_cooccur=2, max_edges=100, label_top=25):
    import matplotlib.pyplot as plt
    import networkx as nx
    import os
    ensure_fonts()
    apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    freq = {}
    for it in (keywords.get("keywords", [])[:topn] or []):
        w = (it.get("keyword") or "").strip()
        s = float(it.get("score", 0) or 0)
        if w:
            freq[w] = max(s, 0.0)
    if not freq or not docs:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "네트워크 데이터 없음", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return {"nodes": 0, "edges": 0}
    G = nx.Graph()
    for w in freq:
        G.add_node(w, weight=freq[w])
    cooccur = {}
    for doc in docs:
        words = set(simple_tokenize_ko(doc)).intersection(set(freq.keys()))
        words = sorted(words)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                pair = (words[i], words[j])
                cooccur[pair] = cooccur.get(pair, 0) + 1
    for (u, v), w in cooccur.items():
        if w >= min_cooccur:
            G.add_edge(u, v, weight=w)
    edges = sorted(G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)[:max_edges]
    G = nx.Graph()
    for w in freq:
        G.add_node(w, weight=freq[w])
    G.add_edges_from((u, v, d) for u, v, d in edges)
    if not G.number_of_nodes():
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "네트워크 데이터 없음", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return {"nodes": 0, "edges": 0}
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    fig, ax = plt.subplots(figsize=(10, 7))
    node_sizes = [max(100, 1000 * G.nodes[n]["weight"]) for n in G.nodes()]
    node_colors = ["#3b82f6" if G.degree(n) > sum(G.degree(n) for n in G.nodes()) / G.number_of_nodes() else "#93c5fd" for n in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    w_max = max(edge_weights, default=1)
    w_min = min(edge_weights, default=1)
    w_norm = [0.5 + 2.5 * (w - w_min) / (w_max - w_min + 1e-9) for w in edge_weights]
    font_name = ensure_fonts()
    nx.draw_networkx_edges(G, pos, ax=ax, width=w_norm, edge_color="#666", alpha=0.25)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, linewidths=0.5, edgecolors="#333")
    if label_top is None:
        label_nodes = list(G.nodes())
    else:
        label_nodes = [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:label_top]]
        label_nodes = [n for n in label_nodes if n in G.nodes()]
    for n in label_nodes:
        txt = (n or "").strip()
        if not txt:
            continue
        x, y = pos[n]
        ax.text(
            x, y, txt,
            ha="center", va="center",
            fontsize=8, color="#111111",
            zorder=5, clip_on=False,
            fontname=font_name,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.80)
        )
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        pad_x = (max(xs) - min(xs)) * 0.08 + 0.05
        pad_y = (max(ys) - min(ys)) * 0.08 + 0.05
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    plt.title("Keyword Co-occurrence Network")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

def plot_company_network_from_json(json_path="outputs/company_network.json",
                                   output_path="outputs/fig/company_network.png",
                                   top_edges=30, top_nodes=10):

    if not os.path.exists(json_path):
        print("[WARN] company_network.json not found")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges_all = data.get("edges", [])
    central = data.get("centrality", []) or []
    if not edges_all:
        print("[WARN] No edges in company_network.json")
        return

    # 1) 상위 엣지 선별
    edges_sorted = sorted(edges_all, key=lambda e: e.get("weight", 0), reverse=True)[:top_edges]

    # 2) 그래프 구성 (rel_type 유지)
    G = nx.Graph()
    for e in edges_sorted:
        u, v = e.get("source"), e.get("target")
        w = float(e.get("weight", 1.0))
        r = e.get("rel_type", "neutral")
        if not u or not v:
            continue
        G.add_edge(u, v, weight=w, rel_type=r)

    if G.number_of_nodes() == 0:
        print("[WARN] Graph empty")
        return

    # 3) 강조 노드 기준: JSON 중심성 상위 우선, 없으면 현재 그래프 기준
    if central:
        top_nodes = {c.get("org") for c in central[:top_nodes] if c.get("org")}
    else:
        deg = nx.degree_centrality(G)
        top_nodes = {n for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_nodes]}

    # 4) 폰트 안전 설정
    try:
        # 프로젝트 공통 한글 폰트 설정을 재사용
        font_name = plt.rcParams['font.family'][0]
    except Exception:
        font_name = "sans-serif"

    # 5) 레이아웃: 가중치 반영(Spring)
    pos = nx.spring_layout(G, weight="weight", seed=42)

    # 6) 엣지 스타일: rel_type별 색상
    edge_colors = []
    weights = []
    for u, v, d in G.edges(data=True):
        weights.append(float(d.get("weight", 1.0)))
        rt = d.get("rel_type", "neutral")
        if rt == "rivalry":
            edge_colors.append("#e74c3c")   # red
        elif rt == "partnership":
            edge_colors.append("#27ae60")   # green
        else:
            edge_colors.append("#7a7a7a")   # gray

    w_arr = np.array(weights, dtype=float)
    if w_arr.size == 0:
        print("[WARN] No edge weights")
        return
    q95 = np.quantile(w_arr, 0.95)
    w_arr = np.minimum(w_arr, q95)
    w_norm = (0.6 + 1.8 * (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-6)).tolist()

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_edges(G, pos, width=w_norm, edge_color=edge_colors, alpha=0.35)

    # 7) 노드 스타일
    node_colors = ["#e74c3c" if n in top_nodes else "#86b6f6" for n in G.nodes()]
    node_sizes = [1200 if n in top_nodes else 600 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           edgecolors="#333", linewidths=0.6, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="#222", font_family=font_name)

    # 8) 범례(간단 표기)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", lw=2, label="경쟁"),
        Line2D([0], [0], color="#27ae60", lw=2, label="협력"),
        Line2D([0], [0], color="#7a7a7a", lw=2, label="중립"),
        Line2D([0], [0], marker='o', color='w', label='허브(강조)',
               markerfacecolor="#e74c3c", markeredgecolor="#333", markersize=10)
    ]
    plt.legend(handles=legend_elements, loc="lower left", frameon=False)

    plt.title("기업 경쟁/협력 네트워크 (핵심 관계망)", fontsize=14, fontname=font_name)
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
