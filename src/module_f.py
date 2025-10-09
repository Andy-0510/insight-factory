import os
import json
import glob
import re
import datetime
from pathlib import Path
from scripts.plot_font import set_kr_font, get_kr_font_path
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
import networkx as nx
from src.utils import load_json, save_json, latest


# 1) 일반 그래프 폰트 설정(다른 플롯도 한글 OK)
_ = set_kr_font()

# 2) 워드클라우드 전용 폰트 경로
font_path = get_kr_font_path()
print(f"[INFO] wordcloud font_path: {font_path}")


def load_data():
    keywords = load_json("outputs/keywords.json", {"keywords": [], "stats": {}})
    topics = load_json("outputs/topics.json", {"topics": []})
    ts = load_json("outputs/trend_timeseries.json", {"daily": []})
    insights = load_json("outputs/trend_insights.json", {"summary": "", "top_topics": [], "evidence": {}})
    opps = load_json("outputs/biz_opportunities.json", {"ideas": []})
    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, []) if meta_path else []
    return keywords, topics, ts, insights, opps, meta_items

def simple_tokenize_ko(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text or "")
    toks = [t.lower() for t in toks if len(t) >= 2]
    return toks

def ensure_fonts():
    import os
    import matplotlib
    from matplotlib import font_manager
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    if font_path:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
    else:
        font_name = "NanumGothic"
    matplotlib.rcParams["font.family"] = font_name
    matplotlib.rcParams["font.sans-serif"] = [font_name, "NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    try:
        from matplotlib import font_manager as fm
        fm._rebuild()
    except Exception:
        pass
    return font_name

def apply_plot_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#999",
        "axes.linewidth": 0.8,
    })

def plot_wordcloud_from_keywords(keywords_obj, out_path="outputs/fig/wordcloud.png"):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import os
    ensure_fonts()
    apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    items = (keywords_obj or {}).get("keywords") or []
    if not items:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "워드클라우드 데이터 없음", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    freqs = {}
    for it in items[:200]:
        w = (it.get("keyword") or "").strip()
        s = float(it.get("score", 0) or 0)
        if w:
            freqs[w] = freqs.get(w, 0.0) + max(s, 0.0)
    font_path = get_kr_font_path()
    if not font_path:
        print("[WARN] font_path를 찾지 못했어요. assets/fonts에 TTF/OTF 넣거나 scripts/plot_font.py 확인 플리즈.")
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        font_path=font_path,
        colormap="tab20",
        prefer_horizontal=0.9,
        min_font_size=10,
        max_words=200,
        relative_scaling=0.5,
        normalize_plurals=False
    ).generate_from_frequencies(freqs)
    wc.to_file(out_path)

def plot_top_keywords(keywords, out_path="outputs/fig/top_keywords.png", topn=15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_fonts()
    apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = keywords.get("keywords", [])[:topn]
    if not data:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "키워드 데이터 없음", ha="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    labels = [d["keyword"] for d in data][::-1]
    scores = [d["score"] for d in data][::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=labels, color="#3b82f6")
    plt.title("Top Keywords")
    plt.xlabel("Score")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_topics(topics, out_path="outputs/fig/topics.png", topn_words=6):
    import os, math
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_fonts()
    apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tps = topics.get("topics", [])
    if not tps:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "토픽 데이터 없음", ha="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    k = len(tps)
    cols = 2
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if k > 1 else [axes]
    for i, t in enumerate(tps):
        ax = axes[i]
        words = (t.get("top_words") or [])[:topn_words]
        labels = [str((w.get("word") or "")) for w in words][::-1]
        probs = []
        for w in words[::-1]:
            pw = w.get("prob", 1.0)
            try:
                p = float(pw)
                if p <= 0:
                    p = 1.0
            except Exception:
                p = 1.0
            probs.append(p)
        sns.barplot(x=probs, y=labels, ax=ax, color="#10b981")
        ax.set_title(f"Topic #{t.get('topic_id')}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_timeseries(ts, out_path="outputs/fig/timeseries.png"):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    ensure_fonts()
    apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    daily = ts.get("daily", [])
    if not daily:
        plt.figure(figsize=(10, 5))
        plt.title("Articles per Day (no data)")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    df = pd.DataFrame(daily).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df["count"] = pd.to_numeric(df.get("count", 0), errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["date"]).sort_values("date")
    now_year = datetime.now().year
    y_min, y_max = now_year - 3, now_year + 1
    df = df[(df["date"].dt.year >= y_min) & (df["date"].dt.year <= y_max)]
    if df.empty:
        plt.figure(figsize=(10, 5))
        plt.title("Articles per Day (empty after filtering)")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    df = df.set_index("date")
    full_idx = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
    df = df.reindex(full_idx).fillna(0)
    df.index.name = "date"
    df["count"] = df["count"].astype(int)
    if len(df) == 1:
        d0 = df.index[0]
        y = float(df["count"].iloc[0])
        plt.figure(figsize=(12, 4.5))
        plt.xlim(d0 - timedelta(days=1), d0 + timedelta(days=1))
        ypad = max(1, y * 0.15)
        plt.ylim(0, y + ypad)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.plot([d0], [y], marker="o", color="#6366f1", label="Daily")
        plt.annotate(f"{int(y)}", (d0, y), textcoords="offset points", xytext=(0, -14), ha="center",
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        plt.title(f"Articles per Day ({d0.strftime('%Y-%m-%d')})")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.25, linestyle="--")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    plt.figure(figsize=(12, 4.5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.plot(df.index, df["count"], marker="o", markersize=3, linewidth=1, color="#6366f1", label="Daily")
    if len(df) >= 7:
        df["ma7"] = df["count"].rolling(window=7, min_periods=1).mean()
        plt.plot(df.index, df["ma7"], linestyle="--", linewidth=2, color="#f43f5e", label="7-day MA")
    plt.title("Articles per Day")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_keyword_network(keywords, docs, out_path="outputs/fig/keyword_network.png",
                         topn=50, min_cooccur=2, max_edges=100, label_top=25):
    """ 키워드 네트워크 생성 (PageRank 기반 라벨링 + 커뮤니티 탐지 시각화 버전) """
    import matplotlib.pyplot as plt
    import networkx as nx
    import os
    from networkx.algorithms import community
    from adjustText import adjust_text
    from matplotlib.patches import Patch

    ensure_fonts()
    apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1. 키워드 점수 가져오기
    keyword_scores = {
        it.get("keyword"): float(it.get("score", 0))
        for it in keywords.get("keywords", [])[:topn]
        if it.get("keyword")
    }

    if not keyword_scores or not docs:
        return {"nodes": 0, "edges": 0}

    # 2. 공동 출현 계산
    cooccur = {}
    for doc in docs:
        words_in_doc = set(simple_tokenize_ko(doc)).intersection(keyword_scores.keys())
        words_in_doc = sorted(words_in_doc)
        for i in range(len(words_in_doc)):
            for j in range(i + 1, len(words_in_doc)):
                pair = tuple(sorted((words_in_doc[i], words_in_doc[j])))
                cooccur[pair] = cooccur.get(pair, 0) + 1

    # 3. 상위 엣지 선택
    edges = sorted(
        [(u, v, w) for (u, v), w in cooccur.items() if w >= min_cooccur],
        key=lambda x: x[2],
        reverse=True
    )[:max_edges]

    # 4. 그래프 구성
    G = nx.Graph()
    for word, score in keyword_scores.items():
        G.add_node(word, score=score)
    G.add_weighted_edges_from(edges)

    # 고립 노드 제거
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0}

    # 5. 커뮤니티 탐지
    communities = list(community.greedy_modularity_communities(G))
    for i, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]['community'] = i

    # 6. PageRank 계산
    pagerank = nx.pagerank(G, weight='weight')

    # 7. 커뮤니티 대표 라벨 설정
    community_labels = {}
    for i, comm in enumerate(communities):
        top_node = max(comm, key=lambda node: pagerank.get(node, 0))
        community_labels[i] = top_node

    # 8. 시각화
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=5.0, iterations=50, seed=42)

    node_sizes = [100 + 2500 * G.nodes[n]['score'] for n in G.nodes()]
    community_ids = [G.nodes[n]['community'] for n in G.nodes()]
    node_colors = [plt.cm.tab20(c % 20) for c in community_ids]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='#000000',   # 더 진한 외곽선
        linewidths=1.0,
        alpha=0.9,
        ax=ax
    )


    edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    w_max = max(edge_weights, default=1)
    edge_widths = [0.3 + 2.0 * (w / w_max) for w in edge_weights]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='grey', alpha=0.5, ax=ax)

    # 9. 상위 PageRank 노드 라벨링
    top_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:label_top]
    texts = [
        ax.text(pos[n][0], pos[n][1], n,
                fontsize=9, ha='center', va='center',
                fontweight='bold', color="#111111",  # 진한 글씨
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8))
        for n in top_nodes if n in pos
    ]

    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # 10. 범례
    legend_elements = [
        Patch(facecolor=plt.cm.tab20(i % 20), edgecolor='black', label=f'Group: {comm_label}')
        for i, comm_label in community_labels.items()
    ]
    ax.legend(
        handles=legend_elements,
        title="주제 그룹 (Topic Groups)",
        loc='lower left',           # ← 내부로 이동
        bbox_to_anchor=(0.01, 0.01),  # ← 좌측 하단 살짝 안쪽
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        fontsize=9,
        title_fontsize=10
    )

    ax.set_title("키워드 네트워크 및 주제 그룹 분석", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("[INFO] Saved keyword_network.png")
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}


def export_csvs(ts_obj, keywords_obj, topics_obj, out_dir="outputs/export"):
    import pandas as pd
    import os
    os.makedirs(out_dir, exist_ok=True)
    daily = (ts_obj or {}).get("daily", [])
    df_ts = pd.DataFrame(daily) if daily else pd.DataFrame(columns=["date", "count"])
    df_ts.to_csv(os.path.join(out_dir, "timeseries_daily.csv"), index=False, encoding="utf-8")
    kws = (keywords_obj or {}).get("keywords", [])[:20]
    df_kw = pd.DataFrame(kws) if kws else pd.DataFrame(columns=["keyword", "score"])
    df_kw.to_csv(os.path.join(out_dir, "keywords_top20.csv"), index=False, encoding="utf-8")
    topics = (topics_obj or {}).get("topics", [])
    rows = []
    for t in topics:
        tid = t.get("topic_id")
        for w in (t.get("top_words") or [])[:10]:
            pw = w.get("prob", None)
            try:
                p = float(pw)
                if p == 0.0:
                    p = 1e-6
            except Exception:
                p = 1e-6
            rows.append({"topic_id": tid, "word": w.get("word", ""), "prob": p})
    df_tw = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["topic_id", "word", "prob"])
    df_tw.to_csv(os.path.join(out_dir, "topics_top_words.csv"), index=False, encoding="utf-8")
    print("[INFO] export CSVs -> outputs/export/*.csv")

# ===== [PATCH] 관계·경쟁 심화 분석 섹션 =====
def _generate_relationship_competition_section(fig_dir="fig",
                                               net_path="outputs/company_network.json",
                                               summary_path="outputs/analysis_summary.json"):
    import os, json
    import pandas as pd
    lines = []
    lines.append("\n## 관계·경쟁 심화 분석\n")

    # 네트워크 로드
    net = {}
    try:
        with open(net_path, "r", encoding="utf-8") as f:
            net = json.load(f) or {}
    except Exception:
        lines.append("- (네트워크 데이터가 없어 본 섹션을 생략합니다.)\n")
        return "\n".join(lines)

    edges = net.get("edges", [])
    nodes = net.get("nodes", [])
    top_pairs = net.get("top_pairs", [])
    central = net.get("centrality", [])
    betw = net.get("betweenness", [])
    comms = net.get("communities", [])

    # 핵심 요약
    num_nodes = len(nodes)
    num_edges = len(edges)
    lines.append("**핵심 요약**\n")
    lines.append(f"- **관계망 규모:** 노드 {num_nodes}개 / 엣지 {num_edges}개\n")
    if top_pairs:
        tp = top_pairs[0]
        lines.append(f"- **가장 강한 관계:** {tp.get('source')} ↔ {tp.get('target')} (가중치 {tp.get('weight')}, 유형 {tp.get('rel_type')})\n")
    if central:
        lines.append(f"- **허브 후보:** {central[0].get('org')} (Degree {central[0].get('degree_centrality')})\n")
    if betw:
        lines.append(f"- **브로커 후보:** {betw[0].get('org')} (Betweenness {betw[0].get('betweenness')})\n")

    # 상위 관계쌍
    if top_pairs:
        df_pairs = pd.DataFrame(top_pairs)[["source","target","weight","rel_type"]]
        lines.append("\n### 상위 관계쌍(Edge)\n")
        # 설명 인용 추가
        lines.append("> 동일 문서/문장 내에서 함께 언급된 기업 쌍이며, 가중치는 동시출현 빈도입니다. 값이 높을수록 상호 관련성이 강하고, 유형은 키워드 규칙으로 경쟁/협력/중립을 추정합니다.\n")
        lines.append(df_pairs.rename(columns={
            "source": "Source", "target": "Target", "weight": "Weight", "rel_type": "Type"
        }).to_markdown(index=False))
        lines.append("\n")

    # 중심성 상위
    if central:
        df_c = pd.DataFrame(central)[["org","degree_centrality"]]
        lines.append("\n### 중심성 상위(연결 허브)\n")
        # 설명 인용 추가
        lines.append("> Degree 중심성은 한 노드가 연결된 상대 수의 비율로, 값이 높을수록 다수의 기업과 직접 연결된 허브 성격을 가집니다. 허브는 이슈 확산과 정보 접근성이 높습니다.\n")
        lines.append(df_c.rename(columns={"org": "Org", "degree_centrality": "DegreeCentrality"}).to_markdown(index=False))
        lines.append("\n")
    if betw:
        df_b = pd.DataFrame(betw)[["org","betweenness"]]
        lines.append("\n### 매개 중심성 상위(정보 브로커)\n")
        # 설명 인용 추가
        lines.append("> Betweenness는 네트워크 경로의 ‘다리’ 역할 정도를 의미합니다. 값이 높을수록 서로 다른 집단을 연결하는 중개자(브로커)로 해석되며, 거래·협상력과 정보 흐름 장악력이 큽니다.\n")
        lines.append(df_b.rename(columns={"org": "Org", "betweenness": "Betweenness"}).to_markdown(index=False))
        lines.append("\n")

    # 커뮤니티 미리보기
    if comms:
        lines.append("\n### 커뮤니티(관계 클러스터)\n")
        # 설명 인용 추가
        lines.append("> 모듈러리티 기반으로 자동 추출한 관계 집단입니다. 같은 집단 내 기업들은 유사 주제나 공급망 활동을 공유할 가능성이 높습니다.\n")
        preview = []
        for c in comms[:5]:
            members = c.get("members", [])[:6]
            theme = (c.get("interpretation", "") or "").strip()

            # 해석이 비어 있으면 자동 생성: 대표 멤버 키워드 기반 요약
            if not theme:
                # 간단 규칙: 대표 멤버명을 나열해 요약(필요 시 도메인 키워드와 결합 가능)
                if members:
                    theme = f"{members[0]} 중심의 연관 클러스터"
                else:
                    theme = "주요 기업 연관 클러스터"

            preview.append(f"- C{c.get('community_id')}: {', '.join(members)} | 해석: {theme}")
        lines.extend(preview)
        lines.append("\n")

    # 이미지 첨부(이미 D/generate_visuals에서 생성했다고 가정)
    img_rel = f"outputs/{fig_dir}/company_network.png"
    if os.path.exists(img_rel):
        lines.append("### 네트워크 시각화\n")
        lines.append(f"![Company Network]({fig_dir}/company_network.png)\n")

    # 종합 코멘트
    lines.append("> 동시출현이 높은 쌍은 직접 경쟁 또는 공급망 핵심 협력 가능성을 시사하며, 허브/브로커는 시장 영향력 및 중개 포지션을 의미합니다. 커뮤니티는 전략·밸류체인 단위의 동조 클러스터일 수 있습니다.\n")

    return "\n".join(lines)


def build_docs_from_meta(meta_items):
    docs = []
    for it in meta_items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        desc = (it.get("description") or it.get("description_og") or "").strip()
        doc = (title + " " + desc).strip()
        if doc:
            docs.append(doc)
    return docs

def _fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def _fmt_score(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _truncate(s, n=80):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n-1] + "…"

def build_markdown(keywords, topics, ts, insights, opps, fig_dir="fig", out_md="outputs/report.md"):
    import pandas as pd
    import glob

    # --- ✨✨✨ 신규 헬퍼 함수 1: 기술 성숙도 섹션 생성 ✨✨✨ ---
    def _generate_tech_maturity_section(fig_dir="fig"):
        section_lines = ["\n## 기술 성숙도 분석 (Technology Maturity Analysis)\n"]
        try:
            maturity_data = load_json("outputs/tech_maturity.json", {"results": []})
            if not maturity_data.get("results"):
                section_lines.append("- (분석된 기술 성숙도 데이터가 없습니다.)\n")
                return "\n".join(section_lines)

            # --- ✨✨✨ 차트 이미지와 설명 추가 ✨✨✨ ---
            section_lines.append(f"![Technology Maturity Map]({fig_dir}/tech_maturity_map.png)\n")
            section_lines.append("> 각 기술의 시장 내 위치(X축: 관심도, Y축: 긍정성)와 사업 활발도(버블 크기)를 보여줍니다.\n")

            section_lines.append("| 기술 (Technology) | 성숙도 단계 (Stage) | 판단 근거 (Rationale) |")
            section_lines.append("|:---|:---|:---|")
            for item in maturity_data["results"]:
                tech = item.get("technology", "N/A")
                analysis = item.get("analysis", {})
                stage = analysis.get("stage", "N/A")
                reason = analysis.get("reason", "-")
                section_lines.append(f"| {tech} | **{stage}** | {reason} |")
            section_lines.append("\n")
        except Exception as e:
            section_lines.append(f"- (기술 성숙도 데이터를 불러오는 중 오류 발생: {e})\n")
        
        return "\n".join(section_lines)

    # --- ✨✨✨ 신규 헬퍼 함수 2: 약한 신호 섹션 생성 ✨✨✨ ---
    def _generate_weak_signal_section():
        section_lines = ["\n## 포착된 약한 신호 및 해석 (Emerging Signals & Interpretation)\n"]
        try:
            weak_signal_data = load_json("outputs/weak_signal_insights.json", {"results": []})
            if not weak_signal_data.get("results"):
                section_lines.append("- (포착된 약한 신호가 없습니다.)\n")
                return "\n".join(section_lines)

            for item in weak_signal_data["results"]:
                signal = item.get("signal", "N/A")
                interpretation = item.get("interpretation", "-")
                section_lines.append(f"- **{signal}**")
                section_lines.append(f"  - **해석:** {interpretation}")
            section_lines.append("\n")
        except Exception as e:
            section_lines.append(f"- (약한 신호 데이터를 불러오는 중 오류 발생: {e})\n")

        return "\n".join(section_lines)

    def _generate_matrix_section(topics_obj):
        try:
            csv_path = "outputs/export/company_topic_matrix_wide.csv"
            if not os.path.exists(csv_path):
                return "\n## 기업×토픽 집중도 매트릭스 (주간)\n\n- (분석할 유효 데이터가 없습니다.)\n"
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            topic_cols = [col for col in df.columns if col.startswith('topic_')]
            if df.empty or not topic_cols:
                return "\n## 기업×토픽 집중도 매트릭스 (주간)\n\n- (분석할 유효 데이터가 없습니다.)\n"
            df_numeric = df.copy()
            for col in topic_cols:
                df_numeric[col] = df[col].fillna('').astype(str).str.split(' ').str[0].replace('', '0').astype(float)
            competitive_scores = df_numeric[topic_cols].gt(0).sum()
            top_competitive_topic_id = competitive_scores.idxmax() if not competitive_scores.empty and competitive_scores.max() > 0 else "N/A"
            df_numeric['total_score'] = df_numeric[topic_cols].sum(axis=1)
            top_focused_org = df_numeric.loc[df_numeric['total_score'].idxmax()]['org'] if not df_numeric['total_score'].empty and df_numeric['total_score'].max() > 0 else "N/A"
            max_score = 0
            rising_star_info = "N/A"
            if df_numeric[topic_cols].max().max() > 0:
                for col in topic_cols:
                    if df_numeric[col].max() > max_score:
                        max_score = df_numeric[col].max()
                        org_name = df_numeric.loc[df_numeric[col].idxmax()]['org']
                        rising_star_info = f"{org_name} @ {col}"
            topic_map = {f"topic_{t.get('topic_id')}": ", ".join([w.get('word', '') for w in t.get('top_words', [])[:2]]) for t in topics_obj.get('topics', [])}
            section_lines = ["\n## 기업×토픽 집중도 매트릭스 (주간)\n"]
            section_lines.append("**핵심 요약:**\n")
            section_lines.append(f"- **가장 경쟁이 치열한 토픽:** **{topic_map.get(top_competitive_topic_id, top_competitive_topic_id)}** (가장 많은 기업들이 주목)\n")
            section_lines.append(f"- **가장 집중도가 높은 기업:** **{top_focused_org}** (다양한 토픽에 걸쳐 높은 관련성)\n")
            section_lines.append(f"- **주목할 만한 조합:** **{rising_star_info}** (가장 높은 단일 연관 점수 기록)\n")
            section_lines.append("각 기업별 상위 8개 토픽의 연관 점수와 해당 토픽 내에서의 점유율(%)을 나타냅니다.\n")
            section_lines.append(df.to_markdown(index=False))
            section_lines.append("\n**코멘트 및 액션 힌트:**\n")
            section_lines.append(f"> 특정 토픽에서 높은 점유율을 보이는 기업은 해당 분야의 '주도자(Leader)'일 가능성이 높습니다. 반면, 특정 기업이 소수의 토픽에 높은 점수를 집중하고 있다면, 이는 해당 기업의 '핵심 전략 분야'를 시사합니다. 경쟁사 및 파트너사의 집중 분야를 파악하여 우리의 전략을 점검해볼 수 있습니다.\n")
            return "\n".join(section_lines)
        except Exception as e:
            return f"\n## 기업×토픽 집중도 매트릭스 (주간)\n\n- (데이터 처리 중 예외 오류가 발생했습니다: {e})\n"

    def _generate_visual_analysis_section(fig_dir="fig"):
        section_lines = ["\n## 기업×토픽 시각적 분석\n"]
        has_content = False
        heatmap_path = f"outputs/{fig_dir}/matrix_heatmap.png"
        if os.path.exists(heatmap_path):
            section_lines.append("### 전체 시장 구도 (Heatmap)\n")
            section_lines.append(f"![Heatmap]({fig_dir}/matrix_heatmap.png)\n")
            section_lines.append("> 전체 기업과 토픽 간의 관계를 한눈에 보여줍니다. 색이 진할수록 연관성이 높습니다.\n")
            has_content = True
        share_images = sorted(glob.glob(f"outputs/{fig_dir}/topic_share_*.png"))
        if share_images:
            section_lines.append("### 주요 토픽별 경쟁 구도 (Pie Charts)\n")
            section_lines.append("> 가장 뜨거운 주제를 두고 어떤 기업들이 경쟁하는지 점유율을 보여줍니다.\n")
            for img_path in share_images:
                img_name = os.path.basename(img_path)
                section_lines.append(f"![Topic Share]({fig_dir}/{img_name})")
            section_lines.append("\n")
            has_content = True
        focus_images = sorted(glob.glob(f"outputs/{fig_dir}/company_focus_*.png"))
        if focus_images:
            section_lines.append("### 주요 기업별 전략 분석 (Bar Charts)\n")
            section_lines.append("> 시장을 주도하는 주요 기업들이 어떤 토픽에 집중하고 있는지 보여줍니다.\n")
            for img_path in focus_images:
                img_name = os.path.basename(img_path)
                section_lines.append(f"![Company Focus]({fig_dir}/{img_name})")
            section_lines.append("\n")
            has_content = True
        if not has_content:
            return ""
        return "\n".join(section_lines)
    
    def _generate_signals_section(fig_dir="fig"): # fig_dir 인자 추가
        section_lines = ["\n## 주요 시그널 분석 (Key Signal Analysis)\n"]
        has_content = False

        # 1. 강한 신호 (Strong Signals) 테이블 및 차트 생성
        try:
            strong_df = pd.read_csv("outputs/export/trend_strength.csv")
            if not strong_df.empty:
                section_lines.append("### 강한 신호 (Strong Signals)\n")
                section_lines.append("> 최근 뉴스에서 가장 주목받은 상위 키워드들입니다.\n")
                
                # --- ✨✨✨ 차트 이미지 삽입 ✨✨✨ ---
                section_lines.append(f"![Strong Signals Chart]({fig_dir}/strong_signals_barchart.png)\n")
                
                report_df = strong_df.head(10)[['term', 'cur', 'z_like']].copy()
                report_df.rename(columns={'term': '강한 신호 (Term)', 'cur': '최근 언급량 (cur)', 'z_like': '임팩트 (z_like)'}, inplace=True)
                report_df.insert(0, '순위', range(1, 1 + len(report_df)))

                section_lines.append(report_df.to_markdown(index=False))
                section_lines.append("\n")
                has_content = True
        except FileNotFoundError:
            pass
        except Exception as e:
            section_lines.append(f"- (강한 신호 데이터를 처리하는 중 오류 발생: {e})\n")

        # 2. 약한 신호 (Weak Signals) 테이블 및 차트 생성
        try:
            weak_df = pd.read_csv("outputs/export/weak_signals.csv")
            weak_insights = load_json("outputs/weak_signal_insights.json", {"results": []})
            
            if not weak_df.empty:
                section_lines.append("### 약한 신호 (Weak Signals)\n")
                section_lines.append("> 총 언급량은 적지만 최근 급부상하여 미래가 기대되는 '틈새 키워드'들입니다.\n")

                # --- ✨✨✨ 차트 이미지 삽입 ✨✨✨ ---
                section_lines.append(f"![Weak Signal Radar]({fig_dir}/weak_signal_radar.png)\n")

                if weak_insights.get("results"):
                    insights_map = {item['signal']: item['interpretation'] for item in weak_insights["results"]}
                    report_rows = []
                    for _, row in weak_df.iterrows():
                        term = row['term']
                        report_rows.append({
                            "약한 신호 (Signal)": term,
                            "지표 (cur / z_like)": f"{row['cur']} / {row['z_like']:.2f}",
                            "LLM의 1줄 요약 (Interpretation)": insights_map.get(term, "-")
                        })
                    section_lines.append(pd.DataFrame(report_rows).to_markdown(index=False))
                
                section_lines.append("\n")
                has_content = True
        except FileNotFoundError:
            pass
        except Exception as e:
            section_lines.append(f"- (약한 신호 데이터를 처리하는 중 오류 발생: {e})\n")

        if not has_content:
            return ""
            
        return "\n".join(section_lines)

    klist = keywords.get("keywords", [])[:15]
    tlist = topics.get("topics", [])
    daily = ts.get("daily", [])
    summary = (insights.get("summary", "") or "").strip()
    n_days = len(daily)
    total_cnt = sum(int(x.get("count", 0)) for x in daily)
    date_range = f"{daily[0].get('date', '?')} ~ {daily[-1].get('date', '?')}" if n_days > 0 else "-"
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"# Weekly/New Biz Report ({today})\n")
    lines.append("## Executive Summary\n")
    lines.append("- 이번 기간 핵심 토픽과 키워드, 주요 시사점을 요약합니다.\n")
    if summary:
        lines.append(summary + "\n")
    lines.append("## Key Metrics\n")
    num_docs = keywords.get("stats", {}).get("num_docs", "N/A")
    num_docs_disp = _fmt_int(num_docs) if isinstance(num_docs, (int, float)) or str(num_docs).isdigit() else str(num_docs)
    lines.append(f"- 기간: {date_range}")
    lines.append(f"- 총 기사 수: {_fmt_int(total_cnt)}")
    lines.append(f"- 문서 수: {num_docs_disp}")
    lines.append(f"- 키워드 수(상위): {len(klist)}")
    lines.append(f"- 토픽 수: {len(tlist)}")
    lines.append(f"- 시계열 데이터 일자 수: {n_days}\n")
    lines.append("## Top Keywords\n")
    lines.append(f"![Word Cloud]({fig_dir}/wordcloud.png)\n")
    if klist:
        kw_all = sorted((keywords.get("keywords") or []), key=lambda x: x.get("score", 0), reverse=True)
        lines.append("| Rank | Keyword | Score |")
        lines.append("|---:|---|---:|")
        for i, k in enumerate(kw_all[:15], 1):
            kw = (k.get("keyword", "") or "").replace("|", r"\|")
            sc = _fmt_score(k.get("score", 0), nd=3)
            lines.append(f"| {i} | {kw} | {sc} |")
    else:
        lines.append("- (데이터 없음)")
    lines.append(f"\n![Top Keywords]({fig_dir}/top_keywords.png)\n")
    lines.append(f"![Keyword Network]({fig_dir}/keyword_network.png)\n")
    lines.append("## Topics\n")
    if tlist:
        for t in tlist:
            tid = t.get("topic_id")
            top_words = [w.get("word", "") for w in t.get("top_words", []) if w.get("word")]
            head = (t.get("topic_name") or ", ".join(top_words[:3]) or f"Topic #{tid}")
            words_preview = ", ".join(top_words[:6])
            lines.append(f"- {head} (#{tid})")
            if words_preview:
                lines.append(f"  - 대표 단어: {words_preview}")
            if t.get("insight"):
                one_liner = (t.get("insight") or "").replace("\n", " ").strip()
                lines.append(f"  - 요약: {one_liner}")
    else:
        lines.append("- (데이터 없음)")
    lines.append(f"\n![Topics]({fig_dir}/topics.png)\n")
    lines.append(_generate_matrix_section(topics))
    lines.append(_generate_visual_analysis_section(fig_dir))
  
    lines.append(_generate_relationship_competition_section(fig_dir))

    lines.append("\n## Trend\n")
    lines.append("- 최근 기사 수 추세와 7일 이동평균선을 제공합니다.")
    lines.append(f"\n![Timeseries]({fig_dir}/timeseries.png)\n")
    lines.append("## Insights\n")
    if summary:
        lines.append(summary + "\n")
    else:
        lines.append("- (요약 없음)\n")

    lines.append(_generate_signals_section())

    lines.append(_generate_tech_maturity_section())
    lines.append(_generate_weak_signal_section())

    lines.append("## Opportunities (Top 5)\n")
    ideas_all = (opps.get("ideas", []) or [])
    if ideas_all:
        ideas_sorted = sorted(
            ideas_all,
            key=lambda it: float(it.get("score", it.get("priority_score", 0)) or 0),
            reverse=True
        )[:5]
        _do_trunc = os.getenv("TRUNCATE_OPP", "").lower() in ("1", "true", "yes", "y")
        lines.append("| Idea | Target | Value Prop | Score (Market / Urgency / Feasibility / Risk) |")
        lines.append("|---|---|---|---|")
        for it in ideas_sorted:
            idea_raw = (it.get('idea', '') or it.get('title', '') or '')
            tgt_raw = it.get('target_customer', '') or ''
            vp_raw = (it.get('value_prop', '') or '').replace("\n", " ")
            if _do_trunc:
                idea = _truncate(idea_raw, 120).replace("|", r"\|")
                tgt = _truncate(tgt_raw, 80).replace("|", r"\|")
                vp = _truncate(vp_raw, 280).replace("|", r"\|")
            else:
                idea = idea_raw.replace("|", r"\|")
                tgt = tgt_raw.replace("|", r"\|")
                vp = vp_raw.replace("|", r"\|")
            score_val = it.get("score", "")
            bd = it.get("score_breakdown", {})
            mkt = bd.get("market", "")
            urg = bd.get("urgency", "")
            feas = bd.get("feasibility", "")
            risk = bd.get("risk", "")
            score_str = f"{score_val} ({mkt} / {urg} / {feas} / {risk})" if score_val != "" else ""
            lines.append(f"| {idea} | {tgt} | {vp} | {score_str} |")
    else:
        lines.append("- (아이디어 없음)")
    chart_path = "outputs/fig/idea_score_distribution.png"
    if os.path.exists(chart_path):
        lines.append("\n### 📊 아이디어 점수 분포")
        lines.append(f"![아이디어 점수 분포](fig/idea_score_distribution.png)\n")
    else:
        print(f"[WARN] Chart image not found at {chart_path}")
    lines.append("\n## Appendix\n")
    lines.append("- 데이터: keywords.json, topics.json, trend_timeseries.json, trend_insights.json, biz_opportunities.json")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def build_html_from_md(md_path="outputs/report.md", out_html="outputs/report.html"):
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
<title>Auto Report</title>
<link rel="preconnect" href="https://fonts.gstatic.com">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif; line-height: 1.6; padding: 24px; color: #222; }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
  th {{ background: #f7f7f7; }}
  code {{ background: #f1f5f9; padding: 2px 4px; border-radius: 4px; }}
  td, th {{ overflow-wrap: anywhere; word-break: break-word; white-space: normal; }}
</style>
</head>
<body>
{html}
</body>
</html>"""
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html_tpl)
    except Exception as e:
        print("[WARN] HTML 변환 실패:", e)

def main():
    keywords, topics, ts, insights, opps, meta_items = load_data()
    os.makedirs("outputs/fig", exist_ok=True)
    try:
        plot_top_keywords(keywords)
    except Exception as e:
        print("[WARN] top_keywords 그림 실패:", e)
    try:
        plot_topics(topics)
    except Exception as e:
        print("[WARN] topics 그림 실패:", e)
    try:
        plot_wordcloud_from_keywords(keywords)
    except Exception as e:
        print("[WARN] wordcloud 생성 실패:", e)
    try:
        plot_timeseries(ts)
    except Exception as e:
        print("[WARN] timeseries 그림 실패:", e)
    try:
        docs = build_docs_from_meta(meta_items)
        kw_list = keywords.get("keywords", [])
        n_kw = len(kw_list)
        label_cap = 25
        plot_keyword_network(
            keywords, docs,
            out_path="outputs/fig/keyword_network.png",
            topn=n_kw,
            min_cooccur=1,
            max_edges=200,
            label_top=(None if n_kw <= label_cap else label_cap)
        )
    except Exception as e:
        print("[WARN] 키워드 네트워크 실패:", e)
    try:
        export_csvs(ts, keywords, topics)
    except Exception as e:
        print("[WARN] CSV 내보내기 실패:", e)
    try:
        build_markdown(keywords, topics, ts, insights, opps)
        build_html_from_md()
    except Exception as e:
        print("[WARN] 리포트 생성 실패(폴백 생성으로 대체):", e)
        try:
            skeleton = """# Weekly/New Biz Report (fallback)
## Executive Summary
- (생성 실패 폴백) 요약 데이터를 불러오지 못했습니다.
## Key Metrics
- 기간: -
- 총 기사 수: 0
- 문서 수: 0
- 키워드 수(상위): 0
- 토픽 수: 0
- 시계열 데이터 일자 수: 0
## Top Keywords
- (데이터 없음)
## Topics
- (데이터 없음)
## Trend
- (데이터 없음)
## Insights
- (요약 없음)
## Opportunities (Top 5)
- (아이디어 없음)
## Appendix
- 데이터: keywords.json, topics.json, trend_timeseries.json, trend_insights.json, biz_opportunities.json
"""
            with open("outputs/report.md", "w", encoding="utf-8") as f:
                f.write(skeleton)
            try:
                build_html_from_md()
            except Exception as e2:
                print("[WARN] HTML 폴백 변환 실패:", e2)
        except Exception as e3:
            print("[ERROR] 폴백 리포트 생성도 실패:", e3)
    print("[INFO] Module F 완료 | report.md, report.html 생성(또는 폴백 생성)")

if __name__ == "__main__":
    main()