import os
import json
import re
import glob
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import seaborn as sns
from wordcloud import WordCloud
from adjustText import adjust_text
import networkx as nx

from src.utils import load_json

# --- 1. 설정 및 헬퍼 함수 ---
ROOT_OUTPUT_DIR = "outputs"
DAILY_ARCHIVE_DIR = os.path.join(ROOT_OUTPUT_DIR, "daily")
EXPORT_DIR = os.path.join(ROOT_OUTPUT_DIR, "export")
FIG_DIR = os.path.join(ROOT_OUTPUT_DIR, "fig")

def _safe_read_csv(path, **kwargs):
    try:
        if path and os.path.exists(path): return pd.read_csv(path, **kwargs)
    except Exception as e: print(f"[WARN] Failed to read {path}: {e}")
    return pd.DataFrame()

def _savefig(figure, path):
    """그래프(figure)를 주어진 경로(path)에 저장하는 헬퍼 함수"""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches='tight', facecolor="white")
    plt.close(figure)

def _setup_fonts():
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    font_path = next((path for path in font_paths if 'NanumGothic' in path or 'NotoSansKR' in path), None)
    if font_path:
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    else:
        print("[WARN] NanumGothic or NotoSansKR font not found.")
    plt.rcParams['axes.unicode_minus'] = False

def _ensure_dirs():
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)

# --- 2. 데이터 로딩 함수 ---
def load_all_data():
    """모든 시각화에 필요한 데이터 소스를 한 번에 로드합니다."""
    print("[INFO] Loading all data sources for visualization...")
    return {
        "keywords": load_json(os.path.join(ROOT_OUTPUT_DIR, "keywords.json"), {"keywords": []}),
        "topics": load_json(os.path.join(ROOT_OUTPUT_DIR, "topics.json"), {"topics": []}),
        "ts": load_json(os.path.join(ROOT_OUTPUT_DIR, "trend_timeseries.json"), {"daily": []}),
        "biz_opps": load_json(os.path.join(ROOT_OUTPUT_DIR, "biz_opportunities.json"), {"ideas": []}),
        "tech_maturity": load_json(os.path.join(ROOT_OUTPUT_DIR, "tech_maturity.json"), {"results": []}),
        "company_network": load_json(os.path.join(ROOT_OUTPUT_DIR, "company_network.json"), {}),
        "signal_counts": _safe_read_csv(os.path.join(EXPORT_DIR, "daily_signal_counts.csv")),
        "trend_strength": _safe_read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv")),
        "weak_signals": _safe_read_csv(os.path.join(EXPORT_DIR, "weak_signals.csv")),
        "company_matrix": _safe_read_csv(os.path.join(EXPORT_DIR, "company_topic_matrix_long.csv")),
    }

# --- 3. 개별 시각화 함수들 ---
def plot_enhanced_timeseries(df_display, spike_threshold=2.0):
    """
    주어진 데이터프레임을 사용하여 '전체 기사량'과 '신호 기사 비율'을 시각화합니다.
    (데이터 로딩 로직 제거, 0으로 나누기 오류 방지 추가)
    """
    print("[INFO] Generating enhanced timeseries chart...")
    
    # --- ▼▼▼ [수정] 함수는 데이터 시각화에만 집중하도록 구조 변경 ▼▼▼ ---
    df = df_display.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 스파이크 탐지
    all_spikes_dfs = []
    for metric, name in [('count', '전체 기사량'), ('signal_ratio', '신호 기사 비율')]:
        if metric not in df.columns:
            print(f"[WARN] Metric '{metric}' not found in DataFrame. Skipping spike detection.")
            continue

        rolling = df[metric].rolling(window=7, min_periods=7)
        df[f'{metric}_ma'] = rolling.mean()
        df[f'{metric}_std'] = rolling.std()
        
        # [안정성 수정] 0으로 나누는 것을 방지하기 위해 분모에 작은 값(epsilon)을 더함
        epsilon = 1e-9
        df[f'{metric}_z'] = (df[metric] - df[f'{metric}_ma']) / (df[f'{metric}_std'] + epsilon)
        
        spikes = df[df[f'{metric}_z'] >= spike_threshold].copy()
        if not spikes.empty:
            spikes['metric'] = name
            spikes['value'] = spikes[metric]
            spikes['z_score'] = spikes[f'{metric}_z']
            all_spikes_dfs.append(spikes[['date', 'metric', 'value', 'z_score']])
    
    # 스파이크 결과 CSV 저장
    out_spike_csv = os.path.join(EXPORT_DIR, "timeseries_spikes_enhanced.csv")
    if all_spikes_dfs:
        df_all_spikes = pd.concat(all_spikes_dfs).sort_values('date')
        df_all_spikes['date'] = df_all_spikes['date'].dt.strftime('%Y-%m-%d')
        df_all_spikes['value'] = df_all_spikes.apply(
            lambda row: f"{row['value']:.2%}" if '비율' in row['metric'] else f"{int(row['value'])}건", axis=1)
        df_all_spikes.to_csv(out_spike_csv, index=False, encoding="utf-8-sig", float_format='%.2f')
        print(f"[INFO] Detected {len(df_all_spikes)} spikes. Saved to {out_spike_csv}")

    # 차트 생성 로직 (기존과 거의 동일)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['date'], df['count'], color='#3b82f6', linestyle='-', linewidth=2, label='전체 기사량')
    ax1.plot(df['date'], df.get('count_ma'), color='#343a40', linestyle=':', linewidth=1, label='기사량 7일 이동평균')
    ax1.set_ylabel('전체 기사량 (건)', color='#343a40')
    ax1.tick_params(axis='y', labelcolor='#343a40'); ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.bar(df['date'], df['signal_ratio'], color='#e9ecef', label='신호 기사 비율', zorder=1)
    ax2.set_ylabel('신호 기사 비율 (%)', color='#6c757d')
    ax2.tick_params(axis='y', labelcolor='#6c757d')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}')); ax2.set_ylim(bottom=0)

    if all_spikes_dfs:
        spikes_count = df[df.get('count_z', 0) >= spike_threshold]
        spikes_ratio = df[df.get('signal_ratio_z', 0) >= spike_threshold]
        if not spikes_count.empty:
            ax1.scatter(spikes_count['date'], spikes_count['count'], color='#0b5ed7', s=100, zorder=6, label='기사량 스파이크')
        if not spikes_ratio.empty:
            ax2.scatter(spikes_ratio['date'], spikes_ratio['signal_ratio'], color='#dc3545', s=100, zorder=6, label='비율 스파이크')

    ax1.set_zorder(2); ax1.patch.set_visible(False)
    plt.title('일일 기사량 및 신호 기사 비율 추이 (스파이크 탐지)', fontsize=16)
    ax1.set_xlabel('날짜'); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.74, 0.87))
    
    _savefig(fig, os.path.join(FIG_DIR, "timeseries.png"))
    print(f"[INFO] Enhanced timeseries chart saved.")

def plot_wordcloud(freqs, output_path):
    if not freqs:
        print(f"[WARN] No frequency data for wordcloud: {output_path}")
        return
    font_path = next((path for path in fm.findSystemFonts(fontpaths=None, fontext='ttf') if 'NanumGothic' in path or 'NotoSansKR' in path), None)
    if font_path is None: print(f"[WARN] Korean font not found for {output_path}.")
    try:
        wc = WordCloud(width=1600, height=900, background_color="white", colormap="tab20c",
                       font_path=font_path, relative_scaling=0.4, random_state=42,
                       collocations=False).generate_from_frequencies(freqs)
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        wc.to_file(output_path)
        print(f"[INFO] Wordcloud saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate wordcloud for {output_path}: {e}")

def plot_topics_bubble(topics_data, output_path, min_bubble=50, jitter=0.015):
    """
    (월간용) 토픽 데이터를 받아 버블 차트를 생성합니다.
    """
    print("[INFO] Generating topics bubble chart...")
    tlist = topics_data.get("topics", [])
    if not tlist:
        print("[WARN] No topics data for bubble chart.")
        return

    # 1. 데이터 추출 및 변환
    xs, ys, ss, labels = [], [], [], []
    for t in tlist:
        # .get()을 사용하여 안전하게 값 추출 및 float으로 변환
        x = float(t.get("interest", t.get("score", 0)) or 0)
        y = float(t.get("positive", t.get("sentiment", 0.5)) or 0.5)
        s = float(t.get("activity", len(t.get("top_words", [])) * 5) or 1) * 15

        # 최소 버블 크기 보장 및 겹침 방지
        s = max(min_bubble, s)
        x += np.random.uniform(-jitter, jitter)
        y += np.random.uniform(-jitter, jitter)

        xs.append(x)
        ys.append(y)
        ss.append(s)
        labels.append(t.get("topic_name") or f"Topic #{t.get('topic_id')}")

    # 2. 시각화
    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(xs, ys, s=ss, c=ys, cmap="coolwarm", alpha=0.6, edgecolors="#343a40", linewidths=0.5)

    # 라벨 추가 (adjustText로 겹침 최소화)
    texts = [ax.text(xs[i], ys[i], lab, fontsize=9) for i, lab in enumerate(labels)]
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
    
    fig.colorbar(sc, ax=ax, label="긍정성 (Positivity)")
    ax.set_xlabel("관심도/관련도 (Interest / Relevance)")
    ax.set_ylabel("긍정성 (Positivity / Sentiment)")
    ax.set_title("전략적 토픽 맵 (Strategic Topic Map)", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 3. 그래프 저장 (수정된 _savefig 호출)
    _savefig(fig, output_path)
    print(f"[INFO] Topics bubble chart saved to {output_path}")

def plot_tech_maturity_map(maturity_data):
    """(월간용) 기술 성숙도 맵을 생성합니다."""
    print("[INFO] Generating tech maturity map...")
    results = maturity_data.get("results", [])
    if not results: return
    
    records = []
    for item in results:
        records.append({
            "technology": item.get("technology"),
            "frequency": item.get("metrics", {}).get("frequency", 0),
            "sentiment": item.get("metrics", {}).get("sentiment", 0.0),
            "events": sum(item.get("metrics", {}).get("events", {}).values()),
            "stage": item.get("analysis", {}).get("stage", "N-A")
        })
    df = pd.DataFrame(records)
    df = df[df['stage'] != 'Error']
    if df.empty: return

    stage_palette = {"Emerging": "#9CA3AF", "Growth": "#10B981", "Maturity": "#3B82F6", "N-A": "#D1D5DB"}
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.scatterplot(data=df, x="frequency", y="sentiment", size="events", hue="stage",
                    sizes=(300, 2500), alpha=0.7, palette=stage_palette, ax=ax, legend='auto')
    
    # --- ▼▼▼▼▼ [수정] 평균값을 계산하고 사분면 보조선 추가 ▼▼▼▼▼ ---
    if not df.empty:
        # 1. x축, y축 평균 계산
        x_mean = df['frequency'].mean()
        y_mean = df['sentiment'].mean()

        # 2. 평균값 위치에 회색 점선으로 보조선 추가
        ax.axvline(x=x_mean, color='grey', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=y_mean, color='grey', linestyle='--', linewidth=1, alpha=0.7)

        # (선택) 각 축의 평균값 텍스트 표시
        ax.text(x_mean, ax.get_ylim()[0], f'관심도 평균\n({x_mean:.1f})',
                ha='center', va='bottom', color='grey', fontsize=9)
        ax.text(ax.get_xlim()[0], y_mean, f'긍정성 평균\n({y_mean:.2f})',
                ha='left', va='center', color='grey', fontsize=9)
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    
    # --- ▼▼▼▼▼ [수정] 라벨 및 값 표시 로직 최종 수정 ▼▼▼▼▼ ---
    # 기술명 라벨은 adjust_text로 위치를 최적화
    texts = []
    for i in range(len(df)):
        x_pos, y_pos = df.iloc[i]['frequency'], df.iloc[i]['sentiment']
        
        texts.append(ax.text(x_pos, y_pos, df.iloc[i]['technology'],
            fontdict=dict(color='black', size=11, weight='bold', ha='center', va='bottom')
        ))
    
    adjust_text(texts, ax=ax, expand_points=(1.5, 1.5),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # 이벤트 값은 각 원의 바깥 하단에 고정 위치
    for i in range(len(df)):
        x_pos, y_pos = df.iloc[i]['frequency'], df.iloc[i]['sentiment']
        
        ax.annotate(f"event=({df.iloc[i]['events']})",
            xy=(x_pos, y_pos),
            xytext=(0, -11), # y축으로 -11포인트 이동하여 원과 간격 확보
            textcoords="offset points",
            ha='center', va='top',
            fontsize=9,
            color='black',
            weight='normal',
            bbox=dict(boxstyle="round,pad=0.0", fc="white", ec="none", alpha=0.6)
        )
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    plt.title('기술 성숙도 맵 (Technology Maturity Map)', fontsize=16)
    plt.xlabel('시장 관심도 (뉴스 빈도)', fontsize=12)
    plt.ylabel('시장 긍정성 (감성 점수)', fontsize=12)

    # 범례에서 'events'(size) 항목 제거 (기존과 동일)
    handles, labels = ax.get_legend_handles_labels()
    hue_handles, hue_labels = [], []
    size_legend_started = False
    for h, l in zip(handles, labels):
        if l == 'stage' or l == 'events':
            if l == 'events': size_legend_started = True
            continue
        if not size_legend_started:
            hue_handles.append(h)
            hue_labels.append(l)
    ax.legend(hue_handles, hue_labels, title='성숙도 단계', loc='best', frameon=True, framealpha=0.8)
    
    _savefig(fig, os.path.join(FIG_DIR, 'tech_maturity_map.png'))
    print("[INFO] Tech maturity map updated and saved.")

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

    # 범례 추가 (그래프 내부 빈 공간에 위치, 테두리 포함)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", lw=2, label="경쟁"),       # 빨간 선
        Line2D([0], [0], color="#27ae60", lw=2, label="협력"),       # 초록 선
        Line2D([0], [0], color="#7a7a7a", lw=2, label="중립"),       # 회색 선
        Line2D([0], [0], marker='o', color='w', label='허브 기업',   # 빨간 노드
               markerfacecolor="#e74c3c", markeredgecolor="#333", markersize=10),
        Line2D([0], [0], marker='o', color='w', label='일반 기업',    # 파란 노드
               markerfacecolor="#86b6f6", markeredgecolor="#333", markersize=8)
    ]

    # 범례 추가 (그래프 안쪽 좌하단 + 테두리 추가)
    legend = plt.legend(handles=legend_elements,
                        loc="lower left",
                        frameon=True,
                        framealpha=1,
                        edgecolor="#333",
                        fontsize=9)
    legend.get_frame().set_linewidth(0.8)

    # 그래프 전체 테두리 추가
    ax = plt.gca()
    ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        fill=False, edgecolor="#555", linewidth=1.2
    ))


    plt.title("기업 경쟁/협력 네트워크 (핵심 관계망)", fontsize=14, fontname=font_name)
    plt.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved simplified company_network.png with {len(G.nodes())} nodes and {len(G.edges())} edges")

def plot_idea_score_distribution(biz_opps_data):
    """(월간용) 신사업 아이디어 점수 분포를 생성합니다."""
    print("[INFO] Generating idea score distribution chart...")
    ideas = sorted(biz_opps_data.get("ideas", []), key=lambda it: it.get("score", 0), reverse=True)[:5]
    if not ideas:
        print("[WARN] No business opportunity data for score chart.")
        return

    labels = [idea.get("idea", "")[:15] + "..." if len(idea.get("idea", "")) > 15 else idea.get("idea", "") for idea in ideas]
    
    market = [idea.get("score_breakdown", {}).get("market", 0) for idea in ideas]
    urgency = [idea.get("score_breakdown", {}).get("urgency", 0) for idea in ideas]
    feasibility = [idea.get("score_breakdown", {}).get("feasibility", 0) for idea in ideas]
    risk = [idea.get("score_breakdown", {}).get("risk", 0) for idea in ideas]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - width*1.5, market, width, label='시장성(Market)', color='#20c997')
    bars2 = ax.bar(x - width*0.5, urgency, width, label='시급성(Urgency)', color='#3b82f6')
    bars3 = ax.bar(x + width*0.5, feasibility, width, label='실현가능성(Feasibility)', color='#ffc107')
    bars4 = ax.bar(x + width*1.5, risk, width, label='리스크(Risk)', color='#6c757d')

    ax.set_ylabel('Score (0.0 ~ 1.0)')
    ax.set_title('Top 5 신사업 아이디어 점수 분포', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- ▼▼▼▼▼ [수정] 범례 위치 변경 ▼▼▼▼▼ ---
    ax.legend(loc='upper right')
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    # --- ▼▼▼▼▼ [수정] 막대 위 값 텍스트를 굵게(bold) 표시 ▼▼▼▼▼ ---
    for bars in [bars1, bars2, bars3, bars4]:
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9, weight='bold')
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    
    _savefig(fig, os.path.join(FIG_DIR, 'idea_score_distribution.png'))
    print("[INFO] Idea score distribution chart saved.")

# --- 4. 주기별 시각화 실행 함수 ---
def run_daily_visuals():
    """일간 리포트에 필요한 시각화만 실행합니다."""
    print("\n--- Generating Daily Visuals ---")
    
    ts_json_path = os.path.join(ROOT_OUTPUT_DIR, "trend_timeseries.json")
    signal_csv_path = os.path.join(EXPORT_DIR, "daily_signal_counts.csv")
    
    print(f"Loading daily data: '{ts_json_path}', '{signal_csv_path}'")
    ts_data = load_json(ts_json_path, {"daily": []})
    df_total = pd.DataFrame(ts_data.get("daily", []))
    df_signal = _safe_read_csv(signal_csv_path)
    print(f"  -> Loaded {len(df_total)} timeseries records, {len(df_signal)} signal records.")

    if df_total.empty:
        print("[WARN] Timeseries data is empty. Skipping daily chart generation.")
        return

    df_merged = pd.merge(df_total, df_signal, on="date", how="left").fillna(0)
        
    # 'signal_ratio'가 없는 경우를 대비
    if 'signal_article_count' in df_merged.columns and 'count' in df_merged.columns:
        df_merged['signal_ratio'] = (df_merged['signal_article_count'] / df_merged['count']).where(df_merged['count'] > 0, 0)
    else:
        # 필요한 컬럼이 없으면 0으로 채움
        df_merged['signal_ratio'] = 0

    try:
        plot_enhanced_timeseries(df_merged.tail(30))
    except Exception as e:
        print(f"[WARN] Failed to generate daily visuals: {e}")

def run_weekly_visuals():
    """주간 리포트에 필요한 시각화를 위해 데이터를 집계하고 실행합니다."""
    print("\n--- Aggregating and Generating Weekly Visuals ---")
    
    # 1. 주간 데이터 집계
    all_keywords, all_trends = [], pd.DataFrame()
    for i in range(7):
        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        date_folders = sorted(glob.glob(os.path.join(DAILY_ARCHIVE_DIR, date_str, "*")))
        if not date_folders: continue
        latest_daily_folder = date_folders[-1]
        
        kw_path = os.path.join(latest_daily_folder, "keywords.json")
        if os.path.exists(kw_path):
            all_keywords.extend(load_json(kw_path, {"keywords": []}).get("keywords", []))
            
        trends_path = os.path.join(latest_daily_folder, "export", "trend_strength.csv")
        if os.path.exists(trends_path):
            df = pd.read_csv(trends_path)
            all_trends = pd.concat([all_trends, df], ignore_index=True)
    print(f"  -> Aggregated {len(all_keywords)} keywords and {len(all_trends)} trend entries over 7 days.")

    # 2. 주간 워드클라우드 생성
    if all_keywords:
        weekly_scores = defaultdict(float)
        for k in all_keywords: weekly_scores[k['keyword']] += k.get('score', 0.0)
        plot_wordcloud(dict(weekly_scores), os.path.join(FIG_DIR, "weekly_wordcloud.png"))

    # 3. 주간 상승/하강 신호 바차트 생성
    if not all_trends.empty:
        weekly_trends_df = all_trends.groupby('term')['z_like'].mean().reset_index().rename(columns={'z_like': 'weekly_avg_z_like'})
        
        rising = weekly_trends_df[weekly_trends_df['weekly_avg_z_like'] > 0].head(5)
        falling = weekly_trends_df[weekly_trends_df['weekly_avg_z_like'] < 0].tail(5)
        combined = pd.concat([rising, falling])
        
        if not combined.empty:
            fig = plt.figure(figsize=(12, 8))
            sns.barplot(data=combined, y="term", x="weekly_avg_z_like",
                        palette=["#3b82f6" if x > 0 else "#ef4444" for x in combined['weekly_avg_z_like']])
            plt.title('주간 핵심 신호 모멘텀 (상승/하강 Top 5)', fontsize=16)
            plt.xlabel('주간 평균 모멘텀 (z_like)', fontsize=12)
            plt.ylabel('')
            _savefig(fig, os.path.join(FIG_DIR, "weekly_strong_signals_barchart.png"))
            print(f"[INFO] Weekly strong signals barchart saved.")

def run_monthly_visuals():
    """월간 리포트에 필요한 시각화를 실행합니다."""
    print("\n--- Generating Monthly Visuals ---")
    all_data = load_all_data() # 월간용 전체 데이터 로드

    if all_data["topics"].get("topics"):
        try:
            plot_topics_bubble(all_data["topics"], os.path.join(FIG_DIR, "topics_bubble.png"))
        except Exception as e:
            print(f"[WARN] plot_topics_bubble failed: {e}")
    
    # 각 시각화 함수를 안전하게 호출
    try: plot_topics_bubble(all_data["topics"], os.path.join(FIG_DIR, "topics_bubble.png"))
    except Exception as e: print(f"[WARN] plot_topics_bubble failed: {e}")
    
    try: plot_tech_maturity_map(all_data['tech_maturity'])
    except Exception as e: print(f"[WARN] plot_tech_maturity_map failed: {e}")
        
    try: plot_company_network_from_json()
    except Exception as e: print(f"[WARN] plot_company_network_from_json failed: {e}")

    try: plot_idea_score_distribution(all_data['biz_opps'])
    except Exception as e: print(f"[WARN] plot_idea_score_distribution failed: {e}")

# --- 5. Main 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for different report types.")
    parser.add_argument("--report-type", required=True, choices=['daily', 'weekly', 'monthly'])
    args = parser.parse_args()
    
    _setup_fonts()
    
    if args.report_type == 'daily':
        run_daily_visuals()
    elif args.report_type == 'weekly':
        run_weekly_visuals()
    elif args.report_type == 'monthly':
        run_monthly_visuals()

    print("\n[SUCCESS] Visualizations generated.")


if __name__ == '__main__':
    main()