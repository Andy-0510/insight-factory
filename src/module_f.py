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

# 1) 일반 그래프 폰트 설정(다른 플롯도 한글 OK)
_ = set_kr_font()

# 2) 워드클라우드 전용 폰트 경로
font_path = get_kr_font_path()
print(f"[INFO] wordcloud font_path: {font_path}")

def load_json(path, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

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
    import json

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





[generative_visual.py]
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
