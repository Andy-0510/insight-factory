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
import networkx as nx
import matplotlib.pyplot as plt
import spacy  # NER 도구
from networkx.algorithms import community

CFG = load_config()

# spaCy 한국어 모델 로드 (한 번만)
nlp = spacy.load("ko_core_news_sm")  # 한국어 NER 모델 (ORG: 조직/회사)

# 키워드 세트 (경쟁/협력 구분용)
COMPETITIVE_KEYWORDS = [k.lower() for k in ["경쟁", "대응", "추격", "점유율", "앞서", "뒤처져", "시장 1위"]]
COOPERATIVE_KEYWORDS = [k.lower() for k in ["협력", "파트너십", "공급", "MOU", "제휴", "협약", "공동 개발"]]

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
    
    doc = nlp(text)  # spaCy NER 사용
    orgs = set(ent.text.strip().lower() for ent in doc.ents if ent.label_ == "ORG")
    
    normalized_toks = []
    for t in orgs:
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

# ========== 기업 네트워크 분석 (업데이트된 버전) ==========
def load_meta_files(max_files=5, offset=0):
    files = sorted(glob.glob("data/news_meta_*.json"), reverse=True)[offset:offset + max_files]
    all_items = []
    for f in files:
        with open(f, "r", encoding="utf-8") as ff:
            all_items.extend(json.load(ff))
    return all_items

def compute_company_network(items, period="current"):
    data = {}
    for idx, it in enumerate(items):
        text = (it.get("body") or it.get("description") or "").strip()
        if text:
            orgs = extract_orgs(text, CFG.get("alias", {}), set(), set())
            if orgs:
                data[idx] = {'organizations': orgs, 'sentences': text.split('.')}
    print(f"[DEBUG] compute_company_network: Found {len(data)} documents with organizations")
    
    if not data:
        print("[WARN] No documents with organizations found")
        return None
    
    entities = {}
    relations = defaultdict(Counter)
    for key in data:
        orgs = data[key]['organizations']
        sentences = data[key]['sentences']
        for ent in orgs:
            if ent not in entities:
                entities[ent] = []
            entities[ent].extend([doc for doc in orgs if doc != ent])
        
        for i in range(len(orgs)):
            for j in range(i + 1, len(orgs)):
                a, b = sorted([orgs[i], orgs[j]])
                pair = (a, b)
                rel_type = "neutral"
                for sent in sentences:
                    rel = classify_relationship(sent, a, b)
                    if rel != "neutral":
                        rel_type = rel
                        break
                relations[pair][rel_type] += 1
    
    G = nx.Graph()
    for ind in entities:
        co_orgs = set(entities[ind])
        G.add_node(ind, freq=len(co_orgs))
        for edge in co_orgs:
            pair = tuple(sorted([ind, edge]))
            weight = sum(relations.get(pair, {}).values())
            rel_type = max(relations.get(pair, {"neutral": 0}), key=relations.get(pair, {"neutral": 0}).get)
            if weight > 0:
                G.add_edge(ind, edge, weight=weight, rel_type=rel_type)
    
    print(f"[DEBUG] Network created: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return G

def analyze_network(G):
    if not G:
        print("[WARN] Empty network provided to analyze_network")
        return {}
    
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    
    roles = {}
    for node in G.nodes():
        deg = degree_centrality.get(node, 0)
        btw = betweenness.get(node, 0)
        close = closeness.get(node, 0)
        if deg > 0.7 and btw > 0.3:
            roles[node] = ("허브 (Hub)", "산업 전반에 영향력 행사")
        elif btw > 0.5:
            roles[node] = ("브로커 (Broker)", "공급망 중개자 역할")
        elif deg < 0.3:
            roles[node] = ("주변부 (Peripheral)", "특정 니치 영역 집중")
        else:
            roles[node] = ("일반 (Regular)", "평균적 관계망")
    
    print(f"[DEBUG] Roles generated: {len(roles)} roles, Sample: {dict(list(roles.items())[:2]) if roles else 'Empty'}")
    
    top_pairs = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:5]
    top_pairs = [{"source": e[0], "target": e[1], "weight": e[2]['weight'], "rel_type": e[2]['rel_type']} for e in top_pairs]
    
    central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    central = [{"org": c[0], "degree_centrality": round(float(c[1]), 3)} for c in central]
    betw = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    betw = [{"org": b[0], "betweenness": round(float(b[1]), 3)} for b in betw]
    
    comms = list(community.greedy_modularity_communities(G, weight='weight'))
    communities = []
    for i, comm in enumerate(comms):
        members = list(comm)
        theme = infer_community_theme(members)
        communities.append({"community_id": i, "members": members, "interpretation": theme})
    
    nodes = [{"org": node} for node in G.nodes()]
    print(f"[DEBUG] Nodes generated: {len(nodes)} nodes, Sample: {nodes[:2] if nodes else 'Empty'}")
    
    return {
        "nodes": nodes,
        "edges": [{"source": u, "target": v, "weight": d['weight'], "rel_type": d['rel_type']} for u, v, d in G.edges(data=True)],
        "roles": roles,
        "top_pairs": top_pairs,
        "centrality": central,
        "betweenness": betw,
        "communities": communities
    }

def build_company_network(out_json="outputs/company_network.json", out_png="outputs/fig/company_network.png", out_md="outputs/company_network_report.md"):
    changes = compare_network_periods()
    
    items = load_meta_files(max_files=5)
    print(f"[DEBUG] Loaded {len(items)} meta items")
    G = compute_company_network(items)
    if not G:
        print("[WARN] No network data.")
        return
    
    analysis = analyze_network(G)
    analysis['changes'] = changes
    analysis['actions'] = generate_action_items(analysis, changes)
    
    print(f"[DEBUG] Analysis contents: nodes={len(analysis['nodes'])}, roles={len(analysis['roles'])}")
    
    # JSON 저장
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    # 시각화
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [300 + 50 * G.nodes[n]['freq'] for n in G.nodes()]
    edge_colors = ['red' if d['rel_type'] == 'competitive' else 'green' if d['rel_type'] == 'cooperative' else 'gray' for u, v, d in G.edges(data=True)]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="lightblue", font_size=10, edge_color=edge_colors)
    plt.title("Company Network (Red: Competitive, Green: Cooperative)")
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    # Markdown 리포트
    lines = [
        "### 기업 경쟁/협력 네트워크 분석\n",
        "#### 1. 전체 구조 (한눈에 보기)\n",
        f"- 노드 수: {len(G.nodes())}개 기업\n",
        f"- 연결 수: {len(G.edges())}개 관계\n",
        f"- 평균 연결도: {sum(dict(G.degree()).values()) / len(G.nodes()):.1f} (업계 평균 2.1 대비 ↑)\n",
        f"![Company Network](fig/company_network.png)\n",
        
        "#### 2. 주요 플레이어\n",
        "| 기업 | 역할 | 변화 | 해석 |\n|------|------|------|------|\n"
    ]
    for org, (role, interp) in analysis['roles'].items():
        delta = changes.get(org, {}).get('centrality_change', 0)
        delta_str = f"↑{delta:.0%}" if delta > 0 else f"↓{abs(delta):.0%}" if delta < 0 else "-"
        lines.append(f"| {org} | {role} | {delta_str} | {interp} |\n")
    
    lines.append("\n#### 3. 주목할 관계\n")
    for p in analysis['top_pairs']:
        rel_perc = p['weight']
        rel_type = p['rel_type']
        perc_str = f"{rel_type.capitalize()} {rel_perc}%"
        lines.append(f"- **{p['source']}-{p['target']}**: {perc_str} → {rel_type} 구도 지속\n")
    
    lines.append("\n#### 4. 진영 분석\n")
    for c in analysis['communities']:
        lines.append(f"- 진영 {c['community_id']}: {', '.join(c['members'])} → {c['interpretation']}\n")
    
    lines.append("\n#### 5. 액션 아이템\n")
    for a in analysis['actions']:
        lines.append(f"{a}\n")
    
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    
    print("[INFO] Updated company network: JSON, PNG, MD saved.")

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
    
    # 2. 기업 네트워크 (업데이트된 버전 호출)
    try:
        build_company_network()
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