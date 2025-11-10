import os
import json
import re
import glob
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import seaborn as sns
from wordcloud import WordCloud
from adjustText import adjust_text
import networkx as nx
import matplotlib

from src.utils import load_json
from src.timeutil import to_date # <-- 이 줄을 추가합니다.

# --- ▼▼▼ [수정] ensure_fonts 함수 (캐시 재구성 추가) ▼▼▼ ---
def ensure_fonts():
    """시스템 폰트를 찾아 Matplotlib 한글 설정을 보장하고, 캐시를 재구성합니다."""
    # --- ▼▼▼ [추가] 캐시 재구성 시도 ▼▼▼ ---
    try:
        # Check if cache needs rebuilding based on font list length
        # This is a heuristic and might not always be necessary,
        # but can help if fonts were installed after first matplotlib import.
        initial_font_count = len(fm.fontManager.ttflist)
        fm._load_fontmanager(try_read_cache=False) # Force re-scan
        if len(fm.fontManager.ttflist) > initial_font_count:
             print("[INFO] Matplotlib font cache rebuilt.")
    except Exception as e:
        print(f"[WARN] Failed to force font cache rebuild: {e}")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_gothic = next((path for path in font_paths if 'NanumGothic' in path), None)
    noto_sans_cjk = next((path for path in font_paths if 'NotoSansKR' in path or 'NotoSansCJK' in path), None)

    font_path = nanum_gothic or noto_sans_cjk

    if font_path:
        # Check if font is already known to fontManager
        font_name_prop = fm.FontProperties(fname=font_path).get_name()
        if font_name_prop not in [f.name for f in fm.fontManager.ttflist]:
             fm.fontManager.addfont(font_path) # Add only if not already known
             print(f"[INFO] Added font: {font_path}")

        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_name
        print(f"[INFO] Matplotlib font set to: {plt.rcParams['font.family']} (from {font_path})")
    else:
        # ... (기존 fallback 로직) ...
        print("[WARN] NanumGothic or NotoSansKR font not found. Please install it for proper Korean display.") #
        # Fallback for different OS if primary fonts fail #
        if os.name == 'posix': # Linux/Mac #
             # Try common system fonts #
             available_fonts = [f.name for f in fm.fontManager.ttflist] #
             if 'AppleGothic' in available_fonts: #
                 plt.rcParams['font.family'] = 'AppleGothic' #
             elif 'Malgun Gothic' in available_fonts: # Might be available on some Linux #
                 plt.rcParams['font.family'] = 'Malgun Gothic' #
             else: #
                 plt.rcParams['font.family'] = 'sans-serif' # Last resort #
        elif os.name == 'nt': # Windows #
             plt.rcParams['font.family'] = 'Malgun Gothic' # Usually available #
        else: #
             plt.rcParams['font.family'] = 'sans-serif' # Default fallback #
        print(f"[INFO] Matplotlib font fallback to: {plt.rcParams['font.family']}") #


    plt.rcParams['axes.unicode_minus'] = False
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---


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
        # "tech_maturity": load_json(os.path.join(ROOT_OUTPUT_DIR, "tech_maturity.json"), {"results": []}),
        "company_network": load_json(os.path.join(ROOT_OUTPUT_DIR, "company_network.json"), {}),
        "growth": _safe_read_csv(os.path.join(EXPORT_DIR, "topic_growth.csv")),
        "signal_counts": _safe_read_csv(os.path.join(EXPORT_DIR, "daily_signal_counts.csv")),
        "trend_strength": _safe_read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv")),
        "weak_signals": _safe_read_csv(os.path.join(EXPORT_DIR, "weak_signals.csv")),
        "company_matrix": _safe_read_csv(os.path.join(EXPORT_DIR, "company_topic_matrix_long.csv")),
    }

# --- 3. 개별 시각화 함수들 ---
def plot_enhanced_timeseries(df_display, spike_threshold=2.0):
    """
    주어진 데이터프레임을 사용하여 '전체 기사량'과 '관심 기사 비율'을 시각화합니다.
    (범례, 막대 색상, 마커 수정됨)
    """
    print("[INFO] Generating enhanced timeseries chart (updated)...")
    
    df = df_display.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 스파이크 탐지 (기존과 동일)
    all_spikes_dfs = []
    for metric, name in [('count', '전체 기사량'), ('signal_ratio', '관심 기사 비율')]:
        if metric not in df.columns:
             print(f"[WARN] Metric '{metric}' not found in DataFrame. Skipping spike detection.")
             continue

        rolling = df[metric].rolling(window=7, min_periods=7)
        df[f'{metric}_ma'] = rolling.mean()
        df[f'{metric}_std'] = rolling.std()
        
        epsilon = 1e-9
        df[f'{metric}_z'] = (df[metric] - df[f'{metric}_ma']) / (df[f'{metric}_std'] + epsilon)
        
        spikes = df[df[f'{metric}_z'] >= spike_threshold].copy()
        if not spikes.empty:
            spikes['metric'] = name
            spikes['value'] = spikes[metric]
            spikes['z_score'] = spikes[f'{metric}_z']
            all_spikes_dfs.append(spikes[['date', 'metric', 'value', 'z_score']])
    
    # 스파이크 결과 CSV 저장 (기존과 동일)
    out_spike_csv = os.path.join(EXPORT_DIR, "timeseries_spikes_enhanced.csv")
    if all_spikes_dfs:
        df_all_spikes = pd.concat(all_spikes_dfs).sort_values('date')
        df_all_spikes['date'] = df_all_spikes['date'].dt.strftime('%Y-%m-%d')
        df_all_spikes['value'] = df_all_spikes.apply(
            lambda row: f"{row['value']:.2%}" if '비율' in row['metric'] else f"{int(row['value'])}건", axis=1)
        df_all_spikes.to_csv(out_spike_csv, index=False, encoding="utf-8-sig", float_format='%.2f')
        print(f"[INFO] Detected {len(df_all_spikes)} spikes. Saved to {out_spike_csv}")

    # --- 차트 생성 (수정됨) ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['date'], df['count'], color='#3b82f6', linestyle='-', linewidth=2, label='일간 기사량')
    ax1.plot(df['date'], df.get('count_ma'), color='#343a40', linestyle=':', linewidth=1, label='기사량 7일 이동평균')
    ax1.set_ylabel('일간 기사량 (건)', color='#343a40')
    ax1.tick_params(axis='y', labelcolor='#343a40'); ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    # [수정] color를 '#e9ecef'에서 '#ced4da' (더 진한 회색)로 변경
    ax2.bar(df['date'], df['signal_ratio'], color='#ced4da', label='고관여 기사 비율', zorder=1)
    ax2.set_ylabel('고관여 기사 비율 (%)', color='#6c757d')
    ax2.tick_params(axis='y', labelcolor='#6c757d')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}')); ax2.set_ylim(bottom=0)

    if all_spikes_dfs:
        spikes_count = df[df.get('count_z', 0) >= spike_threshold]
        spikes_ratio = df[df.get('signal_ratio_z', 0) >= spike_threshold]
        if not spikes_count.empty:
            # [수정] marker='^' (위쪽 삼각형) 추가
            ax1.scatter(spikes_count['date'], spikes_count['count'], color='#0b5ed7', s=100, marker='P', zorder=4, label='기사량 스파이크')
        if not spikes_ratio.empty:
            # [수정] marker='*' (별), s=150 (크기 증가) 추가
            ax2.scatter(spikes_ratio['date'], spikes_ratio['signal_ratio'], color='#dc3545', s=150, marker='X', zorder=4, label='비율 스파이크')

    ax1.set_zorder(2); ax1.patch.set_visible(False)
    plt.title('일간 기사량(좌) 및 고관여 기사 비율(우) 추이', fontsize=16)
    ax1.set_xlabel('DATE'); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # [수정] fig.legend 대신 ax1.legend를 사용하여 그래프 내부에 범례 배치
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize='small')
    
    _savefig(fig, os.path.join(FIG_DIR, "timeseries.png"))
    print(f"[INFO] Enhanced timeseries chart saved (legend, color, markers updated).")

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

def plot_topics_bubble(topics_data, output_path, min_bubble=100, max_bubble=8000, jitter=0.015):
    """
    (월간용) 토픽 데이터를 받아 버블 차트를 생성합니다.
    [수정됨]
    - X축: Interest (Log Scale)
    - Y축: Activity (Z-Score)
    - 크기: Positivity (Sentiment) (정규화 스케일링 적용)
    - 색상: 단색 (중복 정보 제거)
    - 축 마진 추가로 원 잘림 방지
    - X, Y축 평균선(mean) 및 라벨 추가 (순서 수정됨)
    """
    print("[INFO] Generating topics bubble chart (Revised)...")
    tlist = topics_data.get("topics", [])
    if not tlist:
        print("[WARN] No topics data for bubble chart.")
        return

    # 1. 데이터 추출 (X=Interest, Y=Activity, Size=Positivity)
    xs_raw, ys_raw = [], [] # Jitter가 적용되지 않은 원본값 (평균 계산용)
    xs, ys, positivity_scores, labels = [], [], [], [] # Jitter가 적용된 값 (플로팅용)
    
    for t in tlist:
        x_val = float(t.get("interest", 0) or 0) + 1.0 # Log scale을 위해 1.0 더하기
        y_val = float(t.get("activity", 0) or 0)
        s_val = float(t.get("positivity", 0.5) or 0.5)
        
        xs_raw.append(x_val)
        ys_raw.append(y_val)
        
        # Jitter (겹침 방지)
        x_jittered = x_val * (1 + np.random.uniform(-jitter, jitter)) # Log 스케일용 Jitter
        y_jittered = y_val + np.random.uniform(-jitter, jitter)

        xs.append(x_jittered)
        ys.append(y_jittered)
        positivity_scores.append(s_val)
        # [수정] topic_name이 없으면 semantic_key를, 그것도 없으면 topic_id를 사용
        label_name = t.get("topic_name")
        label_semantic = t.get("semantic_key") # 1단계에서 저장한 키
        label_id = f"Topic #{t.get('topic_id')}"

        labels.append(label_name or label_semantic or label_id)

    # 2. 원 크기 정규화 (스케일링)
    s_arr = np.array(positivity_scores)
    if s_arr.max() - s_arr.min() > 1e-6:
        s_norm = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min())
        ss = (s_norm * (max_bubble - min_bubble)) + min_bubble
    else:
        ss = [min_bubble] * len(s_arr)

    # 3. 시각화
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 4. [수정] 스캐터 플롯 (긍정/부정 색상 및 크기 적용)
    # c=positivity_scores: 긍정/부정 점수를 색상(c)으로 매핑
    # cmap="RdYlGn": (부정)Red -> (중립)Yellow -> (긍정)Green 컬러맵 사용
    sc = ax.scatter(xs, ys, s=ss, 
                    c=positivity_scores, cmap="RdYlGn", vmin=0, vmax=1, 
                    alpha=0.7, edgecolors="#343a40", linewidths=0.5)
    # --- ▼▼▼ 5. [수정] 축 스케일 및 평균선 (순서 변경) ▼▼▼ ---
    # 축 스케일을 먼저 설정합니다.
    ax.set_xscale('log')
    
    x_mean_val, y_mean_val = None, None
    if xs_raw and ys_raw:
        try:
            x_mean_val = np.mean(xs_raw) # Jitter가 없는 원본 데이터로 평균 계산
            y_mean_val = np.mean(ys_raw)

            # 점선 그리기
            ax.axvline(x=x_mean_val, color='grey', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=y_mean_val, color='grey', linestyle='--', linewidth=1, alpha=0.7)
        except Exception as e:
            print(f"[WARN] Failed to calculate or plot mean lines: {e}")
    # --- ▲▲▲ 축 설정 완료 ▲▲▲ ---

    # --- ▼▼▼ 6. [수정] 축 마진 (순서 변경) ▼▼▼ ---
    if xs and ys:
        try:
            min_y, max_y = np.min(ys), np.max(ys)
            y_range = max_y - min_y
            if y_range < 1e-6: y_range = 1.0 
            y_margin = y_range * 0.15
            ax.set_ylim(min_y - y_margin, max_y + y_margin)

            min_x, max_x = np.min(xs), np.max(xs)
            log_min_x = np.log10(min_x)
            log_max_x = np.log10(max_x)
            log_range = log_max_x - log_min_x
            if log_range < 1e-6: log_range = 1.0 
            log_margin = log_range * 0.15 
            ax.set_xlim(10**(log_min_x - log_margin), 10**(log_max_x + log_margin))
        except Exception as e:
            print(f"[WARN] Failed to apply axis margins: {e}")
    # --- ▲▲▲ 마진 설정 완료 ▲▲▲ ---

    # --- ▼▼▼ 7. [수정] 텍스트 추가 (가장 마지막에 실행) ▼▼▼ ---
    
    # 7a. 버블 이름 (겹침 방지)
    texts = [ax.text(xs[i], ys[i], lab, fontsize=9) for i, lab in enumerate(labels)]
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    # 7b. Y축 평균선 라벨 (축 범위가 설정된 후)
    if y_mean_val is not None:
        current_xlim = ax.get_xlim() # 최종 X축 범위 가져오기
        ax.text(current_xlim[0], y_mean_val, f' 활동성 평균\n ({y_mean_val:.2f})', 
                ha='left', va='center', color='grey', fontsize=9,
                bbox=dict(boxstyle='square,pad=0.1', fc='white', alpha=0.7, ec='none'))

    # --- ▼▼▼ 7c. [신규] X축 평균선 라벨 추가 ▼▼▼ ---
    if x_mean_val is not None:
        current_ylim = ax.get_ylim() # 최종 Y축 범위 가져오기
        ax.text(x_mean_val, current_ylim[0], f'관심도 평균 ({x_mean_val:.1f})', 
                ha='center', va='bottom', color='grey', fontsize=9,
                bbox=dict(boxstyle='square,pad=0.1', fc='white', alpha=0.7, ec='none'))
    # --- ▲▲▲ X축 라벨 추가 완료 ▲▲▲ ---

    # --- ▼▼▼ [신규] 5. 컬러바(범례) 추가 ▼▼▼ ---
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('감성 점수 (Positivity)', fontsize=12)
    # 컬러바 라벨을 0, 0.5, 1.0에만 표시
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['0.0 (부정)', '0.5 (중립)', '1.0 (긍정)'])
    cbar.ax.tick_params(labelsize=10)
    # --- ▲▲▲ 컬러바 추가 완료 ▲▲▲ ---

    # 8. 최종 라벨 및 저장
    ax.set_xlabel("관심도 (Interest / Log Scale)")
    ax.set_ylabel("활동성 (Activity / Z-Score)")
    ax.set_title("전략적 토픽 맵 (Strategic Topic Map)", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)

    _savefig(fig, output_path)
    print(f"[INFO] Revised topics bubble chart saved to {output_path} (with axis margins and mean lines)")

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

def plot_company_network_from_json(json_path=os.path.join(ROOT_OUTPUT_DIR, "company_network.json"), # 경로 수정
                                   output_path=os.path.join(FIG_DIR, "company_network.png"), # 경로 수정
                                   top_edges=30, top_nodes=10):

    if not os.path.exists(json_path):
        print(f"[WARN] company_network.json not found at {json_path}") # 경로 명시
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
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", lw=2, label="경쟁"),
        Line2D([0], [0], color="#27ae60", lw=2, label="협력"),
        Line2D([0], [0], color="#7a7a7a", lw=2, label="중립"),
        Line2D([0], [0], marker='o', color='w', label='허브(강조)',
               markerfacecolor="#e74c3c", markeredgecolor="#333", markersize=10)
    ]

    # 범례 추가 (그래프 내부 빈 공간에 위치, 테두리 포함)
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

def plot_heatmap(company_matrix_df, topics_data, output_path):
    """(월간용) 기업-토픽 매트릭스 히트맵을 생성합니다."""
    print("[INFO] Generating company-topic heatmap...")
    if company_matrix_df.empty or not topics_data.get("topics"):
        print("[WARN] No data for heatmap.")
        return

    # 토픽 ID와 이름을 매핑
    topics_map = {t['topic_id']: t.get('topic_name', f"Topic {t['topic_id']}") for t in topics_data.get("topics", [])}
    
    # pivot_table을 사용하여 히트맵 데이터 구성
    try:
        heatmap_data = company_matrix_df.pivot_table(index="org", columns="topic", values="hybrid_score", aggfunc="sum").fillna(0)
    except Exception as e:
        print(f"[WARN] Failed to pivot data for heatmap: {e}")
        return
        
    if heatmap_data.empty:
        print("[WARN] Pivoted heatmap data is empty.")
        return

    # 상위 15개 기업만 선택
    top_orgs = heatmap_data.sum(axis=1).nlargest(15).index
    heatmap_data = heatmap_data.loc[top_orgs]

    # 컬럼 이름을 토픽 ID에서 토픽명으로 변경 (없는 경우 대비)
    heatmap_data.columns = [topics_map.get(int(col), f"Topic {col}") for col in heatmap_data.columns]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # --- ▼▼▼ [수정] cmap 인자를 'crest'로 변경 ▼▼▼ ---
    sns.heatmap(
        heatmap_data,
        cmap="crest", # 👈 추천 색상 팔레트 적용
        linewidths=.5,
        ax=ax,
        annot=True, # 각 셀에 값 표시 (선택 사항)
        fmt=".1f"   # 소수점 첫째 자리까지 표시
    )
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    
    ax.set_title('기업별 토픽 집중도 (Hybrid Score)', fontsize=16, weight='bold')
    ax.set_xlabel('[토픽]', fontsize=12, weight='bold')
    ax.set_ylabel('[기업]', fontsize=12, weight='bold')
    plt.xticks(rotation=45, ha='right')
    
    _savefig(fig, output_path)
    print(f"[INFO] Heatmap saved to {output_path}")

def _create_empty_plot(output_path, title="데이터 없음"):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.5, title, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='gray')
    ax.axis('off')
    ax.set_title("키워드 네트워크", fontsize=18)
    _savefig(fig, output_path)
    plt.close(fig)

def plot_keyword_network(keywords_data, meta_items, output_path, top_n=20, min_weight=5):
    """
    키워드 동시 발생 네트워크를 시각화합니다.
    노드 크기는 키워드 스코어, 노드 색상은 키워드 감성 평균, 엣지 두께는 동시 발생 빈도를 나타냅니다.
    """
    print(f"[INFO] Generating keyword network graph to {output_path} (top {top_n} keywords)...")

    fig = None # finally 블록에서 사용하기 위해 fig 초기화

    try:
        if not keywords_data or not keywords_data.get("keywords"):
            print("[WARN] No keyword data available for network visualization.")
            _create_empty_plot(output_path, "키워드 네트워크: 데이터 없음")
            return

        keywords_df = pd.DataFrame(keywords_data["keywords"])
        if keywords_df.empty:
            print("[WARN] Keyword DataFrame is empty for network visualization.")
            _create_empty_plot(output_path, "키워드 네트워크: 데이터 없음")
            return

        # 상위 N개 키워드 선택
        top_keywords = keywords_df.nlargest(top_n, 'score')
        selected_keywords = set(top_keywords['keyword'].tolist()) # 빠른 조회를 위해 set 사용
        if len(selected_keywords) < 2:
            print("[WARN] Not enough top keywords to form a meaningful network.")
            _create_empty_plot(output_path, "키워드 네트워크: 키워드 부족")
            return

        # 키워드별 평균 감성 점수 계산 (선택 사항 - meta_items 구조에 따라 조정 필요)
        keyword_sentiments = defaultdict(lambda: {'sum_sentiment': 0.0, 'count': 0})
        # meta_items에 문서별 또는 키워드별 감성 점수가 포함되어 있다고 가정
        # 실제 'meta_items' 구조에 맞게 이 부분 조정 필요 (감성 색상 적용 원할 시)
        # 예시: item에 'avg_sentiment'가 있다고 가정
        for item in meta_items:
            if isinstance(item, dict):
                 avg_sentiment = item.get('avg_sentiment', 0.5) # 없으면 중립(0.5)으로 처리
                 # 'analyzed_keywords' 키가 있고 그 값이 리스트라고 가정
                 analyzed_kws = item.get('analyzed_keywords', [])
                 if isinstance(analyzed_kws, list):
                     for kw_info in analyzed_kws:
                         # kw_info가 딕셔너리고 'keyword' 키를 가지고 있는지 확인
                         if isinstance(kw_info, dict):
                             keyword = kw_info.get('keyword')
                             if keyword in selected_keywords: # 선택된 키워드에 대해서만 집계
                                 keyword_sentiments[keyword]['sum_sentiment'] += avg_sentiment
                                 keyword_sentiments[keyword]['count'] += 1

        keyword_avg_sentiment = {
            kw: data['sum_sentiment'] / data['count']
            for kw, data in keyword_sentiments.items() if data['count'] > 0
        }

        # 키워드 동시 발생 빈도 계산 (문서 기준)
        pair_counts = Counter() # Counter 사용
        items_with_keywords_count = 0 # 디버그용 카운터
        for item in meta_items:
            if not isinstance(item, dict): continue # 딕셔너리가 아니면 건너뛰기

            # 현재 아이템의 텍스트 콘텐츠에서 키워드 추출
            # meta_items에 텍스트가 포함되어 있어야 함
            text_content = (item.get('body') or item.get('description') or "").lower()
            if not text_content: continue # 텍스트 없으면 건너뛰기

            # 선택된 키워드 중 이 문서 텍스트에 포함된 키워드 찾기
            doc_keywords_set = {kw for kw in selected_keywords if kw.lower() in text_content}

            if doc_keywords_set: items_with_keywords_count += 1 # 디버그

            # 집합을 정렬된 리스트로 변환하여 쌍 만들기
            doc_keywords = sorted(list(doc_keywords_set))

            # 문서 내 키워드가 2개 이상일 때만 쌍 계산
            if len(doc_keywords) >= 2:
                for i in range(len(doc_keywords)):
                    for j in range(i + 1, len(doc_keywords)):
                        # 쌍 순서 고정 (알파벳 순)
                        pair = tuple(sorted((doc_keywords[i], doc_keywords[j])))
                        pair_counts[pair] += 1

        # 디버그 로그 추가
        print(f"[DEBUG-plot_kw] Items processed: {len(meta_items)}, Items with any selected keywords: {items_with_keywords_count}")
        print(f"[DEBUG-plot_kw] Total unique co-occurring pairs found (before filtering): {len(pair_counts)}")
        if pair_counts: print(f"[DEBUG-plot_kw] Top 10 pairs: {pair_counts.most_common(10)}")


        # 네트워크 그래프 생성
        G = nx.Graph()
        # 노드 추가 (속성 포함)
        for _, row in top_keywords.iterrows():
            keyword = row['keyword']
            score = row['score']
            sentiment = keyword_avg_sentiment.get(keyword, 0.5) # 없으면 중립
            G.add_node(keyword, score=score, sentiment=sentiment)


        # 엣지 추가
        edges_to_add = [] # 추가할 엣지 임시 저장
        max_weight = 0 # 최대 가중치 추적
        edges_before_filter = 0 # 필터링 전 엣지 카운터
        for (u, v), weight in pair_counts.items():
            edges_before_filter += 1
            if weight >= min_weight: # 최소 가중치 필터링
                # 노드가 그래프에 있는지 확인 (위에서 이미 추가했으므로 있어야 함)
                if u in G and v in G:
                    edges_to_add.append((u, v, weight))
                    if weight > max_weight:
                        max_weight = weight

        # 필터링 전후 엣지 수 로그
        print(f"[DEBUG-plot_kw] Edges before min_weight ({min_weight}) filter: {edges_before_filter}")
        print(f"[DEBUG-plot_kw] Edges after min_weight ({min_weight}) filter: {len(edges_to_add)}")

        # 필터링된 엣지가 없으면 경고 출력 (그래프는 노드만 그림)
        if not edges_to_add:
             print(f"[WARN] No edges met the minimum weight requirement ({min_weight}). Graph will only show nodes.")
             # 노드만 있는 그래프 그리기로 진행

        # 가중치와 함께 엣지 추가
        G.add_weighted_edges_from(edges_to_add)


        # 노드가 없으면 빈 그래프 생성 후 종료
        if not G.nodes:
            print("[WARN] Network graph has no nodes after filtering. Check keyword data and min_weight.")
            _create_empty_plot(output_path, "키워드 네트워크: 노드 부족")
            return

        # --- 시각화 설정 ---
        fig, ax = plt.subplots(figsize=(20, 18)) # Figure 크기 조정

        # 레이아웃 알고리즘 선택 및 파라미터 조정
        try:
            # Fruchterman-Reingold 레이아웃 시도 (중앙 집중 경향)
            pos = nx.fruchterman_reingold_layout(G, k=0.5, iterations=80, seed=42)
        except Exception as layout_e:
            print(f"[WARN] Preferred layout failed: {layout_e}. Falling back to spring_layout.")
            pos = nx.spring_layout(G, seed=42) # 예비 레이아웃

        # 노드 크기 스케일링 증가
        # 노드 속성 'score'가 없을 경우 KeyError 방지
        node_sizes = [max(800, G.nodes[node].get('score', 0.0) * 2500) for node in G.nodes()] # 기본 800, 스케일 2500

        # 노드 색상 (감성 점수 - RdYlGn 컬러맵)
        # 노드 속성 'sentiment'가 없을 경우 KeyError 방지, 기본값 0.5 사용
        sentiments = [G.nodes[node].get('sentiment', 0.5) for node in G.nodes()]
        cmap = plt.cm.RdYlGn # Red-Yellow-Green (부정-중립-긍정)
        node_colors = [cmap(s) for s in sentiments]

        # 엣지 두께 (동시 발생 빈도)
        edge_weights = [G.edges[u, v]['weight'] for u, v in G.edges()]
        scaled_edge_widths = [1.0] * len(edge_weights) # 기본 두께
        if max_weight > 0:
            # 최대 두께 8, 최소 1.0으로 스케일링
            scaled_edge_widths = [w / max_weight * 7 + 1.0 for w in edge_weights]

        # --- 네트워크 요소 그리기 ---
        # 노드 그리기
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85, ax=ax)

        # 엣지 그리기 (엣지가 있을 경우에만)
        if G.edges:
            nx.draw_networkx_edges(G, pos, width=scaled_edge_widths, alpha=0.35, edge_color='gray', ax=ax)

        # 레이블 그리기 (폰트 설정 및 adjustText 사용 시도)
        font_name = plt.rcParams.get('font.family', 'sans-serif') # 설정된 폰트 이름 가져오기
        if isinstance(font_name, list): font_name = font_name[0] # 리스트면 첫번째 이름 사용
        label_font_size = 16 # 레이블 폰트 크기

        try:
            # adjustText 라이브러리로 레이블 겹침 방지 시도
            texts = []
            for node, (x, y) in pos.items():
                # ax.text를 사용하여 텍스트 객체 생성
                texts.append(ax.text(x, y, node, fontsize=label_font_size, fontfamily=font_name,
                                     ha='center', va='center', color='#222222')) # 폰트 이름 적용

            # adjustText import 및 실행 (라이브러리 없으면 경고)
            try:
                 from adjustText import adjust_text
                 adjust_text(texts, ax=ax) # adjustText 실행
            except ImportError:
                 print("[WARN] adjustText library not found. Labels might overlap.")
                 # adjustText 없으면 ax.text로 그린 상태 유지

        except Exception as e_font:
             print(f"[WARN] Failed to draw labels with font '{font_name}'. Check font installation/cache. Error: {e_font}")
             # 폰트 에러 시 NetworkX 기본 레이블 함수로 fallback (겹침 발생 가능)
             try:
                nx.draw_networkx_labels(G, pos, font_size=label_font_size, font_weight='normal', font_color='#222222', ax=ax)
             except Exception as e_fallback: # Fallback도 실패할 경우 대비
                  print(f"[ERROR] Fallback label drawing also failed: {e_fallback}")


        # 그래프 제목 및 축 설정
        ax.set_title(f'키워드 동시 발생 네트워크 (상위 {top_n}개, 최소 빈도 {min_weight})', fontsize=24, pad=20)
        ax.axis('off') # 축 숨기기

        # --- 범례 (텍스트만, 위치 조정) ---
        # 범례용 가상 핸들 생성 (마커 없이 텍스트만)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='작음', markersize=0),
            Line2D([0], [0], marker='o', color='w', label='중간', markersize=0),
            Line2D([0], [0], marker='o', color='w', label='큼', markersize=0)
        ]
        # 범례 생성 및 위치 지정 (그래프 우측 상단 약간 안쪽)
        size_legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.97, 0.95), title="키워드 스코어\n(원의 크기)", fontsize=12, title_fontsize=14, frameon=True, handlelength=0, handletextpad=0)
        # 범례의 가상 마커 숨기기 (텍스트만 보이도록)
        if hasattr(size_legend, 'legend_handles'):
            for handle in size_legend.legend_handles:
                handle.set_visible(False)
        ax.add_artist(size_legend) # 범례를 그래프에 추가

        # 노드 색상 범례 (sentiment)
        norm = plt.Normalize(vmin=0, vmax=1) # 감성 점수 범위 0~1
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) # 더미 배열
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.6)
        cbar.set_label('평균 감성 점수 (노드 색상: 부정 🔴 ~ 🟢 긍정)', fontsize=14, labelpad=10)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.set_ticklabels(['매우 부정', '부정', '중립', '긍정', '매우 긍정'])
        cbar.ax.tick_params(labelsize=10)

        # 엣지 두께 범례 (weight)
        legend_weights = [1, 5, 10] # 예시 동시 발생 빈도
        legend_labels_weight = [f'{w}회 이상' for w in legend_weights]
        if max_weight > 0:
            # Create dummy lines for edge width legend
            for i, w_val in enumerate(legend_weights):
                ax.plot([], [], color='gray', linewidth=w_val / max_weight * 5 + 0.5, label=legend_labels_weight[i], alpha=0.7)
            edge_legend = ax.legend(loc='upper left', bbox_to_anchor=(0.8, 0.95), title="동시 발생 빈도 (엣지 두께)", fontsize=12, title_fontsize=14, frameon=True)
            ax.add_artist(edge_legend)

        # --- 범례 끝 ---

        # 그래프 레이아웃 조정 (범례 공간 확보)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        # 그래프 이미지 파일 저장
        _savefig(fig, output_path)
        print(f"[SUCCESS] Keyword network graph saved to {output_path}")

    except Exception as e:
        # 오류 발생 시 로깅 및 빈 그래프 생성
        print(f"[ERROR] An error occurred during keyword network plotting: {e}")
        import traceback
        traceback.print_exc()
        _create_empty_plot(output_path, "키워드 네트워크 생성 오류")

    finally:
        # Figure 객체가 생성되었으면 닫아서 메모리 해제
        if fig:
             plt.close(fig)

# --- ▼▼▼ [NEW] Function to plot Topic Mini Trends ▼▼▼ ---
def plot_topic_mini_trends(topics_data, growth_df, meta_items, output_path, top_n=2, keyword_n=10):
    """(Monthly) Generates ONE chart for Top 2 Rising (Solid) and Bottom 2 Falling (Dashed) topics (WEEKLY)."""
    print(f"[INFO] Generating ACTUAL topic mini trends (Top {top_n} 상위/하위, Weekly, 1 Chart)...")
    
    fig = None # finally 블록용 초기화
    try:
        if not topics_data or not topics_data.get("topics") or not meta_items or growth_df.empty:
            print("[WARN] Insufficient data for actual topic mini trends (topics, growth, or meta missing).")
            _create_empty_plot(output_path, "토픽 트렌드: 데이터 부족")
            return

        # 1. Top/Bottom N 토픽 ID 및 이름 찾기 (growth_df 기준)
        growth_df = growth_df.dropna(subset=['momentum_score', 'topic_id'])
        rising_topics_df = growth_df.nlargest(top_n, 'momentum_score')
        falling_topics_df = growth_df.nsmallest(top_n, 'momentum_score')
        
        target_topic_ids = set(rising_topics_df['topic_id']) | set(falling_topics_df['topic_id'])
        
        if not target_topic_ids:
            print("[WARN] No rising or falling topics found from growth_df.")
            _create_empty_plot(output_path, "토픽 트렌드: 성장률 데이터 없음")
            return

        # 2. 상위/하위 토픽의 키워드 맵 생성
        topic_map = {t.get("topic_id"): t for t in topics_data.get("topics", [])}
        topic_info = {}

        for topic_id in target_topic_ids:
            topic = topic_map.get(topic_id)
            if not topic:
                print(f"[WARN] Topic ID {topic_id} (from growth_df) not found in topics.json, skipping.")
                continue
                
            # [수정] topic_name이 없으면 topic_id 사용
            topic_name = topic.get("topic_name") or f"Topic {topic_id}"
            keywords = [w.get("word") for w in topic.get("top_words", [])[:keyword_n]]
            if not keywords:
                print(f"[WARN] Topic '{topic_name}' has no keywords, skipping.")
                continue
                
            topic_info[topic_id] = {
                "name": topic_name,
                "pattern": re.compile('|'.join(re.escape(kw) for kw in keywords), re.IGNORECASE)
            }

        if not topic_info:
            print("[WARN] No valid topics with keywords to plot.")
            _create_empty_plot(output_path, "토픽 트렌드: 키워드 없음")
            return

        # 3. [수정] 30일간의 일별 기사 카운트 집계 (D-1 기준)
        daily_counts = defaultdict(lambda: defaultdict(int))
        end_date = datetime.now().date() - timedelta(days=1) # [수정] D-1
        start_date = end_date - timedelta(days=29) # 30일치 데이터

        for item in meta_items:
            d_raw = item.get("published_time") or item.get("pubDate_raw") or ""
            item_date_str = to_date(d_raw) # 1단계에서 import한 to_date 사용
            
            try:
                item_date = datetime.strptime(item_date_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            
            if not (start_date <= item_date <= end_date):
                continue

            content = (item.get("title", "") + " " + (item.get("body") or item.get("description", ""))).strip()
            if not content:
                continue

            # 각 토픽의 키워드(패턴)와 매칭
            for topic_id, info in topic_info.items():
                if info["pattern"].search(content):
                    daily_counts[item_date_str][topic_id] += 1

        if not daily_counts:
            print("[WARN] No articles matched any top/bottom topics in the last 30 days.")
            _create_empty_plot(output_path, "토픽 트렌드: 매칭 없음")
            return

        # 4. DataFrame 변환 및 Plotting 준비
        df_counts = pd.DataFrame(daily_counts).T.fillna(0)
        df_counts.index = pd.to_datetime(df_counts.index)
        all_dates_index = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'))
        df_counts = df_counts.reindex(all_dates_index, fill_value=0)
        
        # [수정] 주간 단위로 리샘플링
        df_weekly = df_counts.resample('W-MON', label='left', closed='left').sum()

        # 5. [신규] 1x1 차트 생성
        fig, ax = plt.subplots(1, 1, figsize=(10, 5)) # 1행 1열

        # 6. 상위 2개 그래프 (실선, 색상 지정)
        colors = plt.cm.get_cmap('Dark2', 8)
        for i, (_, row) in enumerate(rising_topics_df.iterrows()):
            topic_id = row['topic_id']
            if topic_id in df_weekly.columns:
                topic_name = topic_info[topic_id]['name']
                # [수정] 범례(label)에서 'Mom:' 제거, 색상 지정
                ax.plot(df_weekly.index, df_weekly[topic_id], marker='.', linestyle='-', color=colors(i), label=f"[상위] {topic_name}")
        
        # 7. 하위 2개 그래프 (점선, 다른 색상)
        for i, (_, row) in enumerate(falling_topics_df.iterrows()):
            topic_id = row['topic_id']
            if topic_id in df_weekly.columns:
                topic_name = topic_info[topic_id]['name']
                # [수정] 범례(label)에서 'Mom:' 제거, 색상/스타일 지정
                ax.plot(df_weekly.index, df_weekly[topic_id], marker='.', linestyle='--', color=colors(i + top_n), label=f"[하위] {topic_name}")
        
        ax.set_title(f'Top {top_n} 상위(실선) vs 하위(점선) 토픽 트렌드', fontsize=12) # [수정] 이름
        ax.set_ylabel('Weekly Mentions')
        ax.legend(fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 8. 공통 X축 설정
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xlabel('Date', fontsize=10) # [수정] X축 이름
        
        fig.suptitle('토픽 별 언급량 추이 (주 단위)', fontsize=16, y=1.03) # [수정] 제목
        plt.tight_layout()
        _savefig(fig, output_path)
        print(f"[INFO] ACTUAL Top/Bottom {top_n} topic trends (Weekly, 1 Chart) saved to {output_path}")

    except Exception as e:
        print(f"[ERROR] An error occurred during actual topic trend plotting: {e}")
        import traceback
        traceback.print_exc()
        _create_empty_plot(output_path, "토픽 트렌드: 생성 오류")
    finally:
        if fig:
            plt.close(fig)

# --- ▼▼▼ [NEW] Function to plot Risk Keyword Network ▼▼▼ ---
# Note: This requires defining risk keywords and calculating their co-occurrence.
# We'll use a placeholder list and simple co-occurrence for demonstration.
RISK_KEYWORDS_SAMPLE = ["규제", "리스크", "우려", "논란", "지연", "하락", "부진", "문제", "보안", "취약점"]

def plot_risk_negative_spikes(sentiment_df, topics_data, output_path, top_n_topics=5, window=7, threshold=1.5):
    """(Monthly) Plots sentiment trends for topics with recent negative spikes."""
    print("[INFO] Generating risk negative spikes chart...")
    fig = None # finally 블록용 초기화

    try:
        if sentiment_df is None or sentiment_df.empty or 'semantic_key' not in sentiment_df.columns:
            print("[WARN] Insufficient or invalid sentiment data for risk spikes.")
            # _create_empty_plot(output_path, "부정 감성 급등: 데이터 없음") # [삭제]
            return

        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df = sentiment_df.sort_values('date')

        # --- ▼▼▼ [신규] 최근 30일 데이터만 필터링 ▼▼▼ ---
        end_date = sentiment_df['date'].max()
        start_date = end_date - pd.Timedelta(days=29) # 30일간의 데이터
        sentiment_df = sentiment_df[sentiment_df['date'] >= start_date].copy()
        # --- ▲▲▲ 필터링 완료 ▲▲▲ ---

        spiking_topics = []
        analyzed_topics = 0
        all_groups_data = {} # Store processed groups for plotting

        for key, group in sentiment_df.groupby('semantic_key'):
            if key == "Uncategorized" or len(group) < window + 1: continue # 최소 기간 충족하는지 확인
            analyzed_topics += 1

            # --- ▼▼▼ [수정] 롤링 계산 및 NaN 처리 ▼▼▼ ---
            # closed='left' : 현재 날짜 제외하고 이전 window일 평균 계산
            group['ma'] = group['avg_sentiment'].rolling(window=window, closed='left', min_periods=window).mean()
            group['std'] = group['avg_sentiment'].rolling(window=window, closed='left', min_periods=window).std()

            # Z-Score 계산 (ma나 std가 NaN이면 Z-score도 NaN이 됨)
            group['z_score'] = (group['avg_sentiment'] - group['ma']) / (group['std'].fillna(0) + 1e-6) # std가 NaN이면 0으로 처리

            all_groups_data[key] = group.copy() # Store the group with calculations

            # 마지막 데이터 포인트 확인
            last_point = group.iloc[-1]

            # Check if ma, std, z_score are valid (not NaN) before spike check
            if pd.notna(last_point['ma']) and pd.notna(last_point['std']) and pd.notna(last_point['z_score']):
                # 조건: Z-score가 임계치 미만
                if last_point['z_score'] < -threshold:
                    spiking_topics.append({'key': key, 'last_z': last_point['z_score'], 'last_date': last_point['date']})
            # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        if not spiking_topics:
            print("[INFO] No significant negative sentiment spikes detected recently.")
            # _create_empty_plot(output_path, "부정 감성 급등: 해당 없음") # [삭제]
            return

        # Z-score 하락폭 기준으로 상위 N개 토픽 선정
        spiking_topics.sort(key=lambda x: x['last_z']) # 가장 부정적인 Z-score 순
        top_spiking_keys = [t['key'] for t in spiking_topics[:top_n_topics]]

        # --- ▼▼▼ [신규] semantic_key를 topic_name으로 매핑 ▼▼▼ ---
        topics_list = topics_data.get("topics", [])
        semkey_to_name_map = {
            t.get("semantic_key"): t.get("topic_name")
            for t in topics_list
            if t.get("semantic_key") and t.get("topic_name")
        }
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        num_plot_topics = len(top_spiking_keys)
        fig, axes = plt.subplots(num_plot_topics, 1, figsize=(10, 2.5 * num_plot_topics), sharex=True)
        if num_plot_topics == 1: axes = [axes] # 단일 토픽일 경우 배열로 만듦

        for i, key in enumerate(top_spiking_keys):
            group_to_plot = all_groups_data[key] # Use stored group data
            group_to_plot = group_to_plot.set_index('date') # <-- 이 줄을 추가하세요
            ax = axes[i]
            # Plot only non-NaN values for MA
            valid_ma = group_to_plot.dropna(subset=['ma'])
            ax.plot(group_to_plot.index, group_to_plot['avg_sentiment'], marker='.', linestyle='-', label='Sentiment Score')
            ax.plot(valid_ma.index, valid_ma['ma'], linestyle='--', color='gray', alpha=0.7, label=f'{window}-day MA')

            # 스파이크 지점 강조
            spike_info = next(t for t in spiking_topics if t['key'] == key) # Find corresponding spike info
            last_date = spike_info['last_date']
            last_score = group_to_plot.loc[last_date, 'avg_sentiment']
            # Check if last_score is valid before plotting
            if pd.notna(last_score):
                 ax.scatter(last_date, last_score, color='red', s=100, zorder=5, label=f'Spike (Z={spike_info["last_z"]:.2f})')
        
            # --- ▼▼▼ [신규] Y축 범위를 0.0에서 1.0으로 고정 ▼▼▼ ---
            ax.set_ylim(0, 1)
            # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
            
            # [수정] topic_name 사용, 없으면 key(semantic_key) 사용
            display_name = semkey_to_name_map.get(key, key)
            ax.set_title(f"Topic: {display_name}", fontsize=11)
            ax.tick_params(axis='x', labelsize=9, rotation=30) # Rotate labels slightly
            ax.tick_params(axis='y', labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            if i == 0: ax.legend(fontsize=8) # 첫 번째 그래프에만 범례 표시

        plt.xlabel('Date', fontsize=10)
        fig.suptitle('Topics with Recent Negative Sentiment Spikes', fontsize=15, y=1.03)
        plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout if needed
        _savefig(fig, output_path)
        print(f"[INFO] Risk negative spikes chart saved to {output_path} ({len(spiking_topics)}/{analyzed_topics} topics analyzed)")

    except KeyError as ke:
         print(f"[ERROR] KeyError during risk spike plotting: {ke}. DataFrame columns might be missing.")
         import traceback
         traceback.print_exc()
         if fig: plt.close(fig) # Close figure if created
         # _create_empty_plot(output_path, "부정 감성 급등: 처리 오류 (KeyError)") # [삭제]
    except Exception as e:
        print(f"[ERROR] An error occurred during risk spike plotting: {e}")
        import traceback
        traceback.print_exc()
        if fig: plt.close(fig) # Close figure if created
        # _create_empty_plot(output_path, "부정 감성 급등: 생성 오류") # [삭제]
    finally:
        if fig:
             plt.close(fig)

# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# 파일 경로: scripts/generate_visuals.py
# ... (imports, ensure_fonts, 다른 함수들) ...

# --- ▼▼▼ [수정] plot_risk_keyword_network 함수 (폰트 및 범례 추가) ▼▼▼ ---
def plot_risk_keyword_network(meta_items, output_path, min_weight=1):
    """(Monthly) Generates a network of co-occurring risk keywords."""
    print("[INFO] Generating risk keyword network...")
    fig = None # finally 블록용 초기화

    try:
        if not meta_items:
            print("[WARN] No meta items to analyze for risk keyword network.")
            _create_empty_plot(output_path, "리스크 키워드 네트워크: 데이터 없음") # 빈 그래프 생성
            return

        risk_kw_set = set(RISK_KEYWORDS_SAMPLE) # 사용할 리스크 키워드 목록
        pair_counts = Counter()
        items_with_risk_kws = 0 # 디버그용

        for item in meta_items:
            if not isinstance(item, dict): continue

            text_content = (item.get('body') or item.get('description') or "").lower()
            if not text_content: continue

            # 문서 내 리스크 키워드 찾기
            present_risk_keywords = sorted([kw for kw in risk_kw_set if kw in text_content])

            if present_risk_keywords: items_with_risk_kws += 1 # 디버그

            # 동시 발생 계산
            if len(present_risk_keywords) >= 2:
                for i in range(len(present_risk_keywords)):
                    for j in range(i + 1, len(present_risk_keywords)):
                        pair = tuple(sorted((present_risk_keywords[i], present_risk_keywords[j])))
                        pair_counts[pair] += 1

        print(f"[DEBUG-plot_risk] Items processed: {len(meta_items)}, Items with any risk keywords: {items_with_risk_kws}") # 디버그
        print(f"[DEBUG-plot_risk] Total unique risk pairs found (before filtering): {len(pair_counts)}") # 디버그
        if pair_counts: print(f"[DEBUG-plot_risk] Top 10 risk pairs: {pair_counts.most_common(10)}") # 디버그


        # 그래프 생성
        G = nx.Graph()
        nodes_added = set() # 추가된 노드 추적
        edges_to_add = [] # 추가할 엣지 임시 저장
        max_weight = 0 # 최대 가중치 추적
        edges_before_filter = 0 # 필터링 전 엣지 카운터

        for (u, v), weight in pair_counts.items():
            edges_before_filter += 1
            if weight >= min_weight: # 최소 가중치 필터링
                # 노드가 아직 없으면 추가
                if u not in nodes_added: G.add_node(u); nodes_added.add(u)
                if v not in nodes_added: G.add_node(v); nodes_added.add(v)
                # 엣지 추가 리스트에 추가
                edges_to_add.append((u, v, weight))
                if weight > max_weight:
                    max_weight = weight

        print(f"[DEBUG-plot_risk] Edges before min_weight ({min_weight}) filter: {edges_before_filter}")
        print(f"[DEBUG-plot_risk] Edges after min_weight ({min_weight}) filter: {len(edges_to_add)}")

        # 필터링된 엣지 추가
        G.add_weighted_edges_from(edges_to_add)

        # 노드가 없으면 빈 그래프 생성 후 종료
        if not G.nodes:
            print("[WARN] No risk keywords found or connected.")
            _create_empty_plot(output_path, "리스크 키워드 네트워크: 노드 없음")
            return
        if not G.edges:
             print(f"[WARN] No risk keyword edges met the minimum weight ({min_weight}). Graph will only show nodes.")
             # 노드만 있는 그래프 그리기로 진행

        # --- 시각화 설정 ---
        fig, ax = plt.subplots(figsize=(15, 12)) # Figure 크기 조정

        # 레이아웃
        pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42) # spring layout 사용

        # 노드 크기 (연결된 엣지의 가중치 합 또는 degree 기준)
        # weighted_degree = dict(G.degree(weight='weight')) # 가중치 degree 계산
        # node_sizes = [max(300, weighted_degree.get(n, 0) * 80 + 300) for n in G.nodes()] # 가중치 degree 기반 크기
        # 또는 단순 degree 기반
        degree = dict(G.degree())
        node_sizes = [max(300, degree.get(n, 0) * 200 + 300) for n in G.nodes()]


        # 엣지 두께 (동시 발생 빈도)
        edge_weights = [G.edges[u, v]['weight'] for u, v in G.edges()]
        scaled_edge_widths = [0.5] * len(edge_weights)
        if max_weight > 0:
            scaled_edge_widths = [w / max_weight * 5 + 0.5 for w in edge_weights] # 최대 5.5, 최소 0.5

        # --- 네트워크 요소 그리기 ---
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#f87171", alpha=0.8, ax=ax) # 리스크는 붉은색 계열
        if G.edges:
            nx.draw_networkx_edges(G, pos, width=scaled_edge_widths, alpha=0.4, edge_color='#fda4af', ax=ax) # 연한 붉은색 엣지

        # --- ▼▼▼ [수정] 레이블 폰트 설정 적용 ▼▼▼ ---
        font_name = plt.rcParams.get('font.family', 'sans-serif') # 설정된 폰트 이름 가져오기
        if isinstance(font_name, list): font_name = font_name[0]
        label_font_size = 12 # 폰트 크기

        try:
            nx.draw_networkx_labels(G, pos, font_size=label_font_size, font_weight='normal', font_color='#333333', font_family=font_name, ax=ax)
        except Exception as e_font:
             print(f"[WARN] Failed to draw risk labels with font '{font_name}'. Check font installation/cache. Error: {e_font}")
             nx.draw_networkx_labels(G, pos, font_size=label_font_size, font_weight='normal', font_color='#333333', ax=ax) # Fallback
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        # 그래프 제목 및 축 설정
        ax.set_title(f'리스크 키워드 동시 발생 네트워크 (최소 빈도 {min_weight})', fontsize=18, pad=15)
        ax.axis('off')

        # --- ▼▼▼ [추가] 범례 추가 (노드 크기, 엣지 두께) ▼▼▼ ---
        # 노드 크기 범례 (Degree)
        legend_labels_size = ['연결성-작음', '연결성-중간', '연결성-큼']
        handles_size = [
            Line2D([0], [0], marker='o', color='w', label=label, markersize=0) # Use Line2D, markersize=0
            for label in legend_labels_size
        ]
        size_legend = ax.legend(handles=handles_size, loc='upper left', bbox_to_anchor=(0.85, 0.95), title="키워드 연결성\n(원의 크기)", fontsize=9, title_fontsize=11, frameon=True, handlelength=0, handletextpad=0) # handle options to minimize space
        # Hide markers just in case (optional belt-and-suspenders)
        if hasattr(size_legend, 'legend_handles'):
            for handle in size_legend.legend_handles:
                handle.set_visible(False)
        ax.add_artist(size_legend)


        # 엣지 두께 범례 (Weight)
        if G.edges and max_weight > 0:
            legend_weights_raw = [min_weight, max(min_weight, max_weight // 2), max_weight] # Min, Mid, Max weight
            legend_labels_weight = [f'{w}회 이상' for w in legend_weights_raw]
            handles_edge = []
            for w_val, label_val in zip(legend_weights_raw, legend_labels_weight):
                 handles_edge.append(Line2D([0], [0], color='#fda4af', linewidth=w_val / max_weight * 5 + 0.5, label=label_val, alpha=0.7)) # 스케일링 동일하게 적용
            edge_legend = ax.legend(handles=handles_edge, loc='lower left', bbox_to_anchor=(0.85, 0.73), title="동시 발생 빈도\n(선의 두께)", fontsize=9, title_fontsize=11, frameon=True)
            ax.add_artist(edge_legend)
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        # 그래프 레이아웃 조정 (범례 공간 확보)
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # 오른쪽 여백 조정
        # 그래프 이미지 파일 저장
        _savefig(fig, output_path)
        print(f"[SUCCESS] Risk keyword network graph saved to {output_path}")

    except Exception as e:
        # 오류 처리
        print(f"[ERROR] An error occurred during risk keyword network plotting: {e}")
        import traceback
        traceback.print_exc()
        _create_empty_plot(output_path, "리스크 키워드 네트워크 생성 오류")

    finally:
        # Figure 닫기
        if fig:
             plt.close(fig)

# --- End of plot_risk_keyword_network function ---

def plot_topics(topics, out_path="outputs/fig/topics.png", topn_words=6):
    """
    topics.json 데이터를 받아 각 토픽별 상위 키워드 막대 그래프를 생성합니다.
    (내림차순 정렬, topic_name 사용, 최대 6개 토픽 제한)
    """
    print(f"[INFO] Generating topics bar charts to {out_path} (Max 6 topics)...")
    import os, math
    import matplotlib.pyplot as plt
    import seaborn as sns
    # ensure_fonts() 와 apply_plot_style()은 main() 또는 _setup_fonts()에서 호출됨
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tps_all = topics.get("topics", [])
    
    # --- ▼▼▼ [신규 수정] 토픽 수를 최대 6개로 제한 ▼▼▼ ---
    tps = tps_all[:6]
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    if not tps:
        print("[WARN] No topics data for bar chart.")
        fig, ax = plt.subplots(figsize=(8,5))
        ax.text(0.5,0.5,"토픽 데이터 없음", ha="center", fontsize=14, color='gray')
        ax.axis("off")
        _savefig(fig, out_path) # 정의된 헬퍼 함수 사용
        plt.close(fig)
        return

    k = len(tps); cols = 2; rows = math.ceil(k / cols)
    # figsize을 960px 너비에 맞게 조정 (예: 12)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if k > 1 else [axes]

    for i, t in enumerate(tps):
        ax = axes[i]
        words = (t.get("top_words") or [])[:topn_words]
        
        # [수정 1] 내림차순 정렬 (top_words가 이미 내림차순이라 가정, [::-1] 제거)
        labels = [str((w.get("word") or "")) for w in words]
        probs = []
        for w in words:
            pw = w.get("prob", 1.0)
            try:
                p = float(pw)
                if p <= 0: p = 1.0
            except Exception:
                p = 1.0
            probs.append(p)
        
        sns.barplot(x=probs, y=labels, ax=ax, color="#10b981")
        
        # [수정 2] topic_name을 우선 사용
        title = t.get('topic_name', f"Topic #{t.get('topic_id')}")
        ax.set_title(title, fontsize=14, weight='bold') # 제목 굵게
        
        ax.set_xlabel("Weight", fontsize=10)
        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=10) # 키워드 폰트 크기

    # 나머지 빈 subplot 숨기기
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    _savefig(fig, out_path) # 정의된 헬퍼 함수 사용
    plt.close(fig)


# --- 4. 주기별 시각화 실행 함수 ---

def run_daily_visuals():
    """
    일간 리포트에 필요한 시각화를 실행합니다.
    [수정] 'daily_article_ratios.csv'에서 비율을 로드하고,
           'trend_timeseries.json'에서 총 기사량을 로드하여 차트를 생성합니다.
    """
    print("\n--- Generating Daily Visuals ---")

    # --- ▼▼▼ 수정: 로드 파일 및 로직 변경 ▼▼▼ ---
    
    # 1. 비율 데이터 로드 (select_top_articles.py가 생성)
    ratio_csv_path = os.path.join(EXPORT_DIR, "daily_article_ratios.csv")
    print(f"Loading pre-calculated ratios from: '{ratio_csv_path}'")
    df_ratios = _safe_read_csv(ratio_csv_path)

    if df_ratios.empty or 'signal_ratio' not in df_ratios.columns:
        print(f"[WARN] '{os.path.basename(ratio_csv_path)}' is empty or missing 'signal_ratio' column.")
        print("[WARN] Skipping daily chart generation.")
        return # 비율 파일 없으면 시각화 불가
        
    print(f"  -> Loaded {len(df_ratios)} ratio records.")

    # 2. 총 기사량 데이터 로드 (module_c가 생성)
    ts_json_path = os.path.join(ROOT_OUTPUT_DIR, "trend_timeseries.json")
    print(f"Loading total article counts from: '{ts_json_path}'")
    ts_data = load_json(ts_json_path, {"daily": []})
    df_total = pd.DataFrame(ts_data.get("daily", []))

    if df_total.empty or 'count' not in df_total.columns:
         print("[WARN] trend_timeseries.json is empty or missing 'count' column.")
         print("[WARN] Skipping daily chart generation.")
         return # 총 기사량 없으면 시각화 불가

    print(f"  -> Loaded {len(df_total)} total count records.")

    # 3. 데이터 병합 (차트 생성을 위해)
    try:
        df_plot_data = pd.merge(
            df_total[['date', 'count']],       # trend_timeseries의 'count'
            df_ratios[['date', 'signal_ratio']], # daily_article_ratios의 'signal_ratio'
            on="date",
            how="left" # 왼쪽(df_total) 기준 병합
        ).fillna({'count': 0, 'signal_ratio': 0}) # 비율 없는 날짜는 0으로 채움
        
        if df_plot_data.empty:
            print("[WARN] Merged data for plotting is empty. Skipping chart.")
            return

        # 4. 시계열 차트 생성 (최근 30일)
        # plot_enhanced_timeseries는 'count'와 'signal_ratio' 컬럼을 기대함
        plot_enhanced_timeseries(df_plot_data.tail(30))
        print("[INFO] Daily timeseries chart generation complete.")

    except Exception as e:
        print(f"[ERROR] Failed to merge data or generate daily visuals: {e}")
        import traceback
        traceback.print_exc()
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def run_weekly_visuals():
    """
    주간 리포트에 필요한 시각화를 위해, 미리 집계된 통합 파일을 사용합니다.
    """
    print("\n--- Generating Weekly Visuals using Aggregated Data ---")
    
    # 1. 주간 통합 데이터 로드
    keywords_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "keywords.json"), {"keywords": []})
    all_keywords = keywords_data.get("keywords", [])
    all_trends = _safe_read_csv(os.path.join(EXPORT_DIR, "trend_strength.csv"))
    
    # --- ▼▼▼ [추가] 주간 topics.json 로드 ▼▼▼ ---
    topics_data = load_json(os.path.join(ROOT_OUTPUT_DIR, "topics.json"), {"topics": []})
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    
    meta_items_weekly = load_json(os.path.join(ROOT_OUTPUT_DIR, "debug", "weekly_meta_agg.json"), [])
    
    print(f"  -> Loaded {len(all_keywords)} aggregated keywords, {len(all_trends)} trend entries, {len(topics_data.get('topics',[]))} topics.")
    
    # 2. 주간 워드클라우드 생성
    if all_keywords:
        weekly_scores = {k['keyword']: k.get('score', 0.0) for k in all_keywords}
        plot_wordcloud(weekly_scores, os.path.join(FIG_DIR, "weekly_wordcloud.png"))

    # 3. 주간 상승/하강 신호 바차트 생성
    if not all_trends.empty:
        weekly_trends_df = all_trends.groupby('term')['z_like'].mean().reset_index().rename(columns={'z_like': 'weekly_avg_z_like'})
        
        rising = weekly_trends_df[weekly_trends_df['weekly_avg_z_like'] > 0].nlargest(5, 'weekly_avg_z_like')
        falling = weekly_trends_df[weekly_trends_df['weekly_avg_z_like'] < 0].nsmallest(5, 'weekly_avg_z_like')
        combined = pd.concat([rising, falling])
        
        if not combined.empty:
            fig = plt.figure(figsize=(12, 8))
            sns.barplot(data=combined, y="term", x="weekly_avg_z_like",
                     palette=["#3b82f6" if x > 0 else "#ef4444" for x in combined['weekly_avg_z_like']])
            plt.title('주간 핵심 신호 모멘텀 (상위 상승/하강 term)', fontsize=16)
            plt.xlabel('주간 평균 모멘텀 (z_like)', fontsize=12)
            plt.ylabel('')
            _savefig(fig, os.path.join(FIG_DIR, "weekly_strong_signals_barchart.png"))
            print("[INFO] Weekly strong signals barchart saved.")
    
    # 4. 주간 키워드 네트워크 생성
    try:
        plot_keyword_network(keywords_data, meta_items_weekly, os.path.join(FIG_DIR, "keyword_network.png"), top_n=20, min_weight=1)
    except Exception as e:
        print(f"[WARN] plot_keyword_network (weekly) failed: {e}")

    # --- ▼▼▼ 주간 토픽 막대 그래프 생성 호출 ▼▼▼ ---
    try:
        # topics_data를 전달하고, 파일 이름을 weekly_topics_barchart.png 등으로 지정
        plot_topics(topics_data, out_path=os.path.join(FIG_DIR, "weekly_topics_barchart.png"), topn_words=6)
        print(f"[INFO] Weekly topics barchart saved.")
    except Exception as e:
        print(f"[WARN] plot_topics (weekly) failed: {e}")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def run_monthly_visuals():
    """월간 리포트에 필요한 시각화를 실행합니다."""
    print("\n--- Generating Monthly Visuals ---")
    all_data = load_all_data() # 월간용 전체 데이터 로드

    # 각 시각화 함수를 안전하게 호출
    try: plot_topics_bubble(all_data["topics"], os.path.join(FIG_DIR, "topics_bubble.png"))
    except Exception as e: print(f"[WARN] plot_topics_bubble failed: {e}")
    
    # --- ▼▼▼ 이 줄을 주석 처리 ▼▼▼ ---
    # try: plot_tech_maturity_map(all_data['tech_maturity'])
    # except Exception as e: print(f"[WARN] plot_tech_maturity_map failed: {e}")
    # --- ▲▲▲ 주석 처리 완료 ▲▲▲ ---
        
    try: plot_company_network_from_json()
    except Exception as e: print(f"[WARN] plot_company_network_from_json failed: {e}")

    try: plot_idea_score_distribution(all_data['biz_opps'])
    except Exception as e: print(f"[WARN] plot_idea_score_distribution failed: {e}")

    # --- ▼▼▼ [추가] 히트맵 생성 함수 호출 ▼▼▼ ---
    try:
        plot_heatmap(all_data['company_matrix'], all_data['topics'], os.path.join(FIG_DIR, "matrix_heatmap.png"))
    except Exception as e:
        print(f"[WARN] plot_heatmap failed: {e}")

    # --- ▼▼▼ [추가] 월간 리포트용 누락 시각화 생성 호출 ▼▼▼ ---

    # 1. 월간 메타 데이터 로드 (키워드 네트워크 및 토픽 트렌드 생성에 모두 필요)
    meta_items_monthly = load_json(os.path.join(ROOT_OUTPUT_DIR, "debug", "monthly_meta_agg.json"), [])

    # 2. 키워드 네트워크 생성
    try:
        plot_keyword_network(all_data['keywords'], meta_items_monthly, os.path.join(FIG_DIR, "keyword_network.png"))
    except Exception as e:
        print(f"[WARN] plot_keyword_network failed: {e}")

    # 3. 토픽 미니 트렌드 생성 (실제 데이터 기반)
    try:
        # [수정] growth_df, meta_items_monthly를 전달, top_n=2로 변경
        plot_topic_mini_trends(
            all_data['topics'], 
            all_data['growth'], 
            meta_items_monthly, 
            os.path.join(FIG_DIR, "topics_mini_trends.png"), 
            top_n=2
        )
    except Exception as e:
        print(f"[WARN] plot_topic_mini_trends failed: {e}")

    # 3. 리스크 관련 시각화 생성
    try:
        # --- ▼▼▼ [수정] 데이터 로딩 로직 변경 ▼▼▼ ---
        # 1. 아카이브에서 가장 최신 daily_topic_sentiment.csv 찾기
        archive_files = sorted(glob.glob(os.path.join(DAILY_ARCHIVE_DIR, "*", "*", "export", "daily_topic_sentiment.csv")))
        sentiment_csv_path = None
        if archive_files:
            sentiment_csv_path = archive_files[-1] # 가장 최신 파일

        # 1b. 아카이브에 없으면 루트 export 폴더 확인
        if not sentiment_csv_path or not os.path.exists(sentiment_csv_path):
            sentiment_csv_path = os.path.join(EXPORT_DIR, "daily_topic_sentiment.csv")

        print(f"[INFO] Loading sentiment data for risk chart from: {sentiment_csv_path}")
        sentiment_df_monthly = _safe_read_csv(sentiment_csv_path)

        # 2. 토픽 이름 매핑을 위해 topics.json 로드
        # (all_data 딕셔너리에서 이미 로드됨)
        topics_data = all_data["topics"] 
        # --- ▲▲▲ 로딩 로직 변경 완료 ▲▲▲ ---

        plot_risk_negative_spikes(sentiment_df_monthly, topics_data, os.path.join(FIG_DIR, "risk_negative_spikes.png"))
    except Exception as e:
        print(f"[WARN] plot_risk_negative_spikes failed: {e}")

    try:
        meta_items_monthly = load_json(os.path.join(ROOT_OUTPUT_DIR, "debug", "monthly_meta_agg.json"), [])
        plot_risk_keyword_network(meta_items_monthly, os.path.join(FIG_DIR, "risk_keyword_network.png"))
    except Exception as e:
        print(f"[WARN] plot_risk_keyword_network failed: {e}")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- 5. Main 함수 ---
if __name__ == '__main__':
    # ... (argparse 및 main 함수 호출 로직) ...
    parser = argparse.ArgumentParser(description="Generate visualizations for different report types.")
    parser.add_argument("--report-type", required=True, choices=['daily', 'weekly', 'monthly'])
    args = parser.parse_args()

    ensure_fonts()
    _ensure_dirs() 

    if args.report_type == 'daily':
        run_daily_visuals()
    elif args.report_type == 'weekly':
        run_weekly_visuals()
    elif args.report_type == 'monthly':
        run_monthly_visuals()

    print("\n[SUCCESS] Visualizations generation attempt complete.")