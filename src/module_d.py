# -*- coding: utf-8 -*-
import os
import re
import json
import csv
import glob
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Any
from collections import defaultdict, Counter
from src.config import load_config

CFG = load_config()

# ========== 유틸리티 ==========
def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def load_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _load_lines(p: str) -> set:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return {x.strip() for x in f if x.strip()}
    except Exception:
        return set()

# ========== 회사×토픽 매트릭스: ORG 잡음 제거 ==========
ORG_BAD_PATTERNS = [
    r"^\d{1,4}(년|월|분기|일)$",
    r"^\d+(hz|w|mah|nm|mm|cm|kg|g|인치|형|세대|위|종|개국|명|가지)$",
    r"^\d+-\w+-\d+",
    r"^\d{1,3}(천|만|억|조)?(원|달러|위안|엔)$",
    r"^\d+$"
]

def norm_org_token(t: str) -> str:
    t = (t or "").strip()
    if t.endswith("의") and len(t) >= 3:
        t = t[:-1]
    if len(t) >= 3 and t[-1] in ("은","는","이","가","을","를","과","와"):
        t = t[:-1]
    return t

def is_bad_org_token(t: str, org_stop_words: set) -> bool:
    if not t or len(t) < 2:
        return True
    
    s_lower = t.lower()
    if s_lower in org_stop_words:
        return True
    
    if re.fullmatch(r"^[0-9\W_]+$", s_lower):
        return True
        
    for pat in ORG_BAD_PATTERNS:
        if re.fullmatch(pat, s_lower, re.I):
            return True
    return False

def extract_orgs(text: str, alias_map: Dict[str, str], whitelist: set, org_stop_words: set) -> List[str]:
    if not text:
        return []
    
    toks = re.findall(r"[가-힣A-Za-z0-9\-\+\.]{2,}", text)
    
    normalized_toks = []
    for t in toks:
        normalized = alias_map.get(t, t)
        normalized = alias_map.get(normalized.lower(), normalized)
        normalized_toks.append(normalized)

    cand = []
    for t in normalized_toks:
        if t in whitelist:
            cand.append(t)
            continue
        if is_bad_org_token(t, org_stop_words):
            continue
        cand.append(t)

    cnt = Counter(cand)
    out = [w for w, c in cnt.most_common(50) if c >= 2 and w not in whitelist]
    
    return list(whitelist.intersection(set(normalized_toks))) + out

def load_topic_labels(topics_obj: dict, topn: int) -> list:
    labels = []
    for t in (topics_obj.get("topics") or []):
        words = [w.get("word","") for w in (t.get("top_words") or []) if w.get("word")][:topn]
        labels.append({"topic_id": int(t.get("topic_id", 0)), "words": words})
    return labels

# ========== 기업×토픽 매트릭스 분석 ==========
def export_company_topic_matrix(meta_items: List[Dict[str, Any]], topics_obj: dict, cfg: dict) -> None:
    print("[INFO] Generating Company-Topic Matrix...")
    os.makedirs("outputs/export", exist_ok=True)

    # 사전 로드
    alias_map = cfg.get("alias", {})
    brand_to_company = load_json("data/dictionaries/brand_to_company.json", {})
    topic_like_entities = _load_lines("data/dictionaries/topic_like_entities.txt")
    
    ent_org = _load_lines("data/dictionaries/entities_org.txt")
    whitelist = {alias_map.get(w.lower(), w) for w in ent_org} - set(topic_like_entities)

    # 트렌드 신호 로드
    trend_signals = {}
    try:
        trends_df = pd.read_csv("outputs/export/trend_strength.csv")
        for _, row in trends_df.iterrows():
            trend_signals[row['term']] = {
                'z_like': row.get('z_like', 0.0),
                'diff': row.get('diff', 0)
            }
        print(f"[DEBUG] Loaded {len(trend_signals)} trend signals")
    except Exception:
        print("[WARN] trend_strength.csv not found")

    # 토픽 워드셋 구성
    topic_wordsets = {tl["topic_id"]: set(tl.get("words", [])) for tl in load_topic_labels(topics_obj, 30)}
    doc_results = []

    for it in meta_items:
        text = it.get("body") or it.get("description") or ""
        if not text:
            continue

        raw_toks = re.findall(r"[가-힣A-Za-z0-9\-\+\.]{2,}", text)
        mentioned_orgs = set()
        for t in raw_toks:
            norm_t = alias_map.get(t.lower(), t)
            mapped_org = brand_to_company.get(norm_t, norm_t)
            if mapped_org in whitelist and mapped_org not in topic_like_entities:
                mentioned_orgs.add(mapped_org)
        
        if not mentioned_orgs:
            continue

        low_text_words = set(text.lower().split())
        doc_topic_scores = {tid: len(ws.intersection(low_text_words)) for tid, ws in topic_wordsets.items()}
        
        for org in mentioned_orgs:
            doc_results.append({"org": org, **doc_topic_scores})

    if not doc_results:
        print("[WARN] No valid org-topic relationships found")
        return

    # 집계 및 IDF 보정
    df = pd.DataFrame(doc_results)
    matrix_df = df.groupby("org").sum()

    N = len(matrix_df)
    df_topics = (matrix_df > 0).sum(axis=0)
    idf = np.log(1 + N / (df_topics + 1))
    base_score_df = matrix_df * idf

    # 하이브리드 점수 계산
    melted_df = base_score_df.reset_index().melt(id_vars='org', var_name='topic_id', value_name='base_score')
    melted_df = melted_df[melted_df['base_score'] > 0].copy()

    # Topic Share
    topic_total_scores = melted_df.groupby('topic_id')['base_score'].transform('sum')
    melted_df['topic_share'] = melted_df['base_score'] / (topic_total_scores + 1e-9)

    # Company Focus
    org_total_scores_sq = melted_df.groupby('org')['base_score'].transform(lambda x: np.linalg.norm(x, 2))
    melted_df['company_focus'] = melted_df['base_score'] / (org_total_scores_sq + 1e-9)
    
    # Hybrid Score (최근성 보정)
    def get_hybrid_score(row):
        base_score = row['base_score']
        term = row['org']
        z_like = trend_signals.get(term, {}).get('z_like', 0.0)
        diff = trend_signals.get(term, {}).get('diff', 0)
        lambda1, lambda2 = 0.20, 0.05
        recency_boost = (1 + lambda1 * max(0, z_like) + lambda2 * (1 if diff > 0 else 0))
        return base_score * recency_boost

    melted_df['hybrid_score'] = melted_df.apply(get_hybrid_score, axis=1)

    # Long 포맷 저장
    melted_df.rename(columns={'topic_id': 'topic'}, inplace=True)
    melted_df.sort_values(by=["org", "hybrid_score"], ascending=[True, False], inplace=True)
    melted_df.to_csv("outputs/export/company_topic_matrix_long.csv", index=False, float_format='%.4f', encoding="utf-8-sig")
    print("[INFO] Saved company_topic_matrix_long.csv")

    # Wide 포맷 저장
    TOP_K_TOPICS = 8
    wide_df = melted_df.groupby('org').apply(lambda x: x.nlargest(TOP_K_TOPICS, 'hybrid_score')).reset_index(drop=True)
    wide_df['score_with_share'] = wide_df.apply(lambda row: f"{row['hybrid_score']:.2f} ({row['topic_share']:.0%})", axis=1)
    wide_df['topic'] = 'topic_' + wide_df['topic'].astype(str)
    final_wide_df = wide_df.pivot(index='org', columns='topic', values='score_with_share').fillna("")
    final_wide_df.to_csv("outputs/export/company_topic_matrix_wide.csv", encoding="utf-8-sig")
    print(f"[INFO] Saved company_topic_matrix_wide.csv (Top-{TOP_K_TOPICS} per org)")

# ========== 기업 네트워크 분석 ==========
def compute_company_network(max_files: int = 5, min_edge_weight: int = 5) -> dict:
    print("[INFO] Computing company network...")
    
    companies = list(_load_lines("data/dictionaries/entities_org.txt"))
    if not companies:
        return {"edges": [], "top_pairs": [], "centrality": []}

    meta_files = sorted(glob.glob("data/news_meta_*.json"))[-max_files:]
    co = {}
    
    for fp in meta_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                items = json.load(f) or []
        except Exception:
            continue
            
        for it in items:
            text = ((it.get("title") or it.get("title_og") or "") + " " +
                    (it.get("body") or it.get("description") or it.get("description_og") or ""))
            present = [c for c in companies if c and c in text]
            present = sorted(set(present))
            
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = present[i], present[j]
                    co[(a, b)] = co.get((a, b), 0) + 1

    edges = [{"source": a, "target": b, "weight": w}
             for (a, b), w in co.items() if w >= min_edge_weight]

    top_pairs = sorted(edges, key=lambda e: e["weight"], reverse=True)[:5]

    # 중심성 계산
    import networkx as nx
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"])

    centrality = []
    if G.number_of_nodes() > 0:
        deg = nx.degree_centrality(G)
        btw = nx.betweenness_centrality(G, normalized=True, weight="weight")
        nodes_sorted = sorted(G.nodes(), key=lambda n: (deg.get(n, 0.0), btw.get(n, 0.0)), reverse=True)[:5]
        
        for n in nodes_sorted:
            centrality.append({
                "org": n,
                "degree_centrality": round(float(deg.get(n, 0.0)), 3),
                "betweenness": round(float(btw.get(n, 0.0)), 3)
            })

    print(f"[INFO] Network: {len(edges)} edges, {len(centrality)} central orgs")
    return {"edges": edges, "top_pairs": top_pairs, "centrality": centrality}

# ========== 분석 요약 ==========
def generate_analysis_summary(matrix_path: str, network_path: str) -> dict:
    summary = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "matrix_stats": {},
        "network_stats": {}
    }
    
    if os.path.exists(matrix_path):
        try:
            df = pd.read_csv(matrix_path, encoding='utf-8-sig')
            summary["matrix_stats"] = {
                "num_orgs": len(df),
                "num_topics": len([c for c in df.columns if c.startswith('topic_')]),
                "top_org": df.iloc[0]['org'] if not df.empty else None
            }
        except Exception:
            pass
    
    if os.path.exists(network_path):
        try:
            with open(network_path, 'r', encoding='utf-8') as f:
                net = json.load(f)
            summary["network_stats"] = {
                "num_edges": len(net.get("edges", [])),
                "num_pairs": len(net.get("top_pairs", [])),
                "top_hub": net.get("centrality", [{}])[0].get("org") if net.get("centrality") else None
            }
        except Exception:
            pass
    
    return summary

# ========== 메인 ==========
def main():
    print("[INFO] Module D - Analysis 시작")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/export", exist_ok=True)
    
    meta_items = load_json(latest("data/news_meta_*.json"), [])
    topics_obj = load_json("outputs/topics.json", {"topics": []})
    
    # 1. 기업×토픽 매트릭스
    try:
        export_company_topic_matrix(meta_items, topics_obj, CFG)
    except Exception as e:
        print(f"[ERROR] Matrix export failed: {e}")
    
    # 2. 기업 네트워크
    try:
        network_result = compute_company_network(max_files=5, min_edge_weight=5)
        with open("outputs/company_network.json", "w", encoding="utf-8") as f:
            json.dump(network_result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Network analysis failed: {e}")
    
    # 3. 분석 요약
    summary = generate_analysis_summary(
        "outputs/export/company_topic_matrix_wide.csv",
        "outputs/company_network.json"
    )
    with open("outputs/analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("[INFO] Module D 완료")

if __name__ == "__main__":
    main()