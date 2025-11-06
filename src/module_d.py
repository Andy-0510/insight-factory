"""
Module D - Company x Topic Matrix + Relationship & Competition Network
- 기업×토픽 매트릭스 산출 (기존 로직 보강)
- 기업-기업 동시출현 네트워크 산출 및 요약 저장
- outputs/company_network.json, outputs/analysis_summary.json 생성/갱신
- 시각화(이미지)는 별도 generate_visuals 단계에서 처리
"""

import os
import re
import json
import glob
import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import unicodedata # <-- 이 줄을 추가하세요

from src.config import load_config, llm_config
import networkx as nx
from networkx.algorithms import community
from src.utils import load_json, save_json, latest


# 선택적 의존: spaCy 한국어 NER (환경에 있을 때만 사용)
try:
    import spacy  # NER
    _SPACY_OK = True
except Exception:
    spacy = None
    _SPACY_OK = False

CFG = load_config()
DICT_DIR = "data/dictionaries"

# --- ▼▼▼ [신규 추가] 관계 규칙 파일 로드 ▼▼▼ ---
def _load_relationship_rules():
    rules_path = os.path.join(DICT_DIR, "relationship_rules.json")
    print(f"[INFO] [module_d] Loading relationship rules from {rules_path}...")
    rules = load_json(rules_path, {})
    # 키를 정규화 (알파벳순 정렬 + 소문자)
    normalized_rules = {}
    for key, rel_type in rules.items():
        parts = sorted([part.strip().lower() for part in key.split('|')])
        if len(parts) == 2:
            normalized_key = f"{parts[0]}|{parts[1]}"
            normalized_rules[normalized_key] = rel_type
    print(f"[INFO] [module_d] {len(normalized_rules)} relationship rules loaded and normalized.")
    return normalized_rules

RELATIONSHIP_RULES = _load_relationship_rules()
# --- ▲▲▲ 추가 완료 ▲▲▲ ---

# ====== 키워드(관계 분류) ======
COMPETITIVE_KEYWORDS = [k.lower() for k in ["경쟁", "추격", "점유율", "앞서", "뒤처져", "시장 1위", "소송", "분쟁", "입찰", "견제", "제치고", "따돌리고", "맞서"]]
COOPERATIVE_KEYWORDS = [k.lower() for k in ["협력", "파트너십", "공급", "mou", "제휴", "협약", "공동 개발", "agreement", "contract", "납품", "수주", "공동 투자", "공동투자", "컨소시엄", "체결", "구매", "채택"]]



# ====== 유틸 ======
def today_utc_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _load_lines(p: str) -> set:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return {x.strip() for x in f if x.strip()}
    except Exception:
        return set()
    
# --- ▼▼▼ [신규] norm_tok 함수 정의 추가 ▼▼▼ ---
def norm_tok(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s
# --- ▲▲▲ 추가 완료 ▲▲▲ ---

# ====== ORG 토큰 필터 ======
ORG_BAD_PATTERNS = [
    r"^\d{1,4}(년|월|분기|일)$",
    r"^\d+(hz|w|mah|nm|mm|cm|kg|g|인치|형|세대|위|종|개국|명|가지)$",
    r"^\d+-\w+-\d+",
    r"^\d{1,3}(천|만|억|조)?(원|달러|위안|엔)$",
    r"^\d+$",
]

# ====== ORG 정규화/검증 ======
def norm_org_token(t: str) -> str:
    t = (t or "").strip()
    if t.endswith("의") and len(t) >= 3:
        t = t[:-1]
    if len(t) >= 3 and t[-1] in ("은", "는", "이", "가", "을", "를", "과", "와"):
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

# ====== spaCy 로더(옵션) ======
_NLP = None
def _get_nlp():
    global _NLP
    if _NLP is None and _SPACY_OK:
        try:
            # 설치되어 있을 때만 사용
            _NLP = spacy.load("ko_core_news_sm")
        except Exception:
            _NLP = None
    return _NLP

# ====== ORG 추출 ======
def extract_orgs_with_spacy(text: str) -> List[str]:
    nlp = _get_nlp()
    if not nlp or not text:
        return []
    doc = nlp(text)
    return [ent.text.strip() for ent in doc.ents if getattr(ent, "label_", "") == "ORG"]

def extract_orgs(text: str,
                 alias_map: Dict[str, str],
                 whitelist: set,
                 org_stop_words: set) -> List[str]:
    if not text:
        return []
    # 1) spaCy NER 우선 시도
    orgs = set()
    for v in extract_orgs_with_spacy(text):
        orgs.add(v.strip().lower())

    # 2) 토큰 매칭(보완)
    raw_toks = re.findall(r"[가-힣A-Za-z0-9\-\+\.]{2,}", text)
    for t in raw_toks:
        tt = t.strip().lower()
        orgs.add(tt)

    # 3) 정규화 + 필터
    normalized = []
    for t in orgs:
        base = alias_map.get(t, t)
        base = alias_map.get(base.lower(), base)
        base = norm_org_token(base)
        normalized.append(base)

    cand = []
    for w in normalized:
        if w in whitelist:
            cand.append(w)
            continue
        if is_bad_org_token(w, org_stop_words):
            continue
        cand.append(w)

    # 4) 빈도 기준 필터(너무 희박한 후보 제거)
    cnt = Counter(cand)
    out = [w for w, c in cnt.most_common(50) if c >= 2 and w not in whitelist]

    # 최종: 화이트리스트 교집합을 앞으로 + 나머지
    return list(whitelist.intersection(set(normalized))) + out

# ====== 토픽 라벨 ======
def load_topic_labels(topics_obj: dict, topn: int) -> list:
    labels = []
    for t in (topics_obj.get("topics") or []):
        words = [w.get("word", "") for w in (t.get("top_words") or []) if w.get("word")][:topn]
        labels.append({"topic_id": int(t.get("topic_id", 0)), "words": words})
    return labels

# ====== 기업×토픽 매트릭스 ======
def export_company_topic_matrix(meta_items: List[Dict[str, Any]], topics_obj: dict, cfg: dict) -> None:
    print("[INFO] Generating Company-Topic Matrix...")
    os.makedirs("outputs/export", exist_ok=True)

    # 사전 로드
    alias_map = cfg.get("alias", {})
    brand_to_company = load_json("data/dictionaries/brand_to_company.json", {})
    topic_like_entities = _load_lines("data/dictionaries/topic_like_entities.txt")

    ent_org = _load_lines("data/dictionaries/entities_org.txt")
    whitelist = {alias_map.get(w.lower(), w) for w in ent_org} - set(topic_like_entities)

    # 트렌드 신호 로드(옵션)
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

    # 토픽 워드셋
    topic_wordsets = {tl["topic_id"]: set(tl.get("words", [])) for tl in load_topic_labels(topics_obj, 30)}

    doc_results = []
    for it in meta_items:
        text = it.get("body") or it.get("description") or ""
        if not text:
            continue

        # 브랜드→회사 매핑 + 화이트리스트 검증
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

    # 집계 + IDF 보정
    df = pd.DataFrame(doc_results)
    matrix_df = df.groupby("org").sum()

    N = len(matrix_df)
    df_topics = (matrix_df > 0).sum(axis=0)
    idf = np.log(1 + N / (df_topics + 1))
    base_score_df = matrix_df * idf

    # Long 포맷
    melted_df = base_score_df.reset_index().melt(id_vars='org', var_name='topic', value_name='base_score')
    melted_df = melted_df[melted_df['base_score'] > 0].copy()

    # Topic Share
    topic_total_scores = melted_df.groupby('topic')['base_score'].transform('sum')
    melted_df['topic_share'] = melted_df['base_score'] / (topic_total_scores + 1e-9)

    # Company Focus (L2 정규화)
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
    melted_df.sort_values(by=["org", "hybrid_score"], ascending=[True, False], inplace=True)
    melted_df.to_csv("outputs/export/company_topic_matrix_long.csv", index=False, float_format='%.4f', encoding="utf-8-sig")
    print("[INFO] Saved company_topic_matrix_long.csv")

    # Wide 포맷 (상위 K 토픽)
    TOP_K_TOPICS = 8
    wdf = (melted_df.groupby('org', group_keys=False).apply(lambda x: x.nlargest(TOP_K_TOPICS, 'hybrid_score')))
    wdf['score_with_share'] = wdf.apply(lambda row: f"{row['hybrid_score']:.2f} ({row['topic_share']:.0%})", axis=1)
    wdf['topic'] = 'topic_' + wdf['topic'].astype(str)
    final_wide_df = wdf.pivot(index='org', columns='topic', values='score_with_share').fillna("")
    final_wide_df.to_csv("outputs/export/company_topic_matrix_wide.csv", encoding="utf-8-sig")
    print(f"[INFO] Saved company_topic_matrix_wide.csv (Top-{TOP_K_TOPICS} per org)")

# ====== 메타 로더 ======
def load_meta_files(max_files=5, offset=0):
    # 기존 경로 패턴 유지
    files = sorted(glob.glob("data/news_meta_*.json"), reverse=True)[offset:offset + max_files]
    all_items = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as ff:
                all_items.extend(json.load(ff))
        except Exception:
            pass
    return all_items

# ====== 관계 유형 분류 ======
def classify_relationship(context_texts: List[str]) -> str:
    ctx = " ".join(context_texts).lower()
    r_score = sum(1 for w in COMPETITIVE_KEYWORDS if w in ctx)
    p_score = sum(1 for w in COOPERATIVE_KEYWORDS if w in ctx)
    if r_score > p_score:
        return "rivalry"
    if p_score > r_score:
        return "partnership"
    return "neutral"

# ===== Optimized co-occurrence builder (config-driven) =====
# ===== Optimized co-occurrence builder (config-driven) =====
def build_cooccurrence_edges(items: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, str, int, str, str]], List[str]]: # 반환 타입에 str 추가
    # 0) config + dictionaries
    net_cfg = CFG.get("network", {})
    whitelist_only = bool(net_cfg.get("whitelist_only", True))
    use_regex_fallback = bool(net_cfg.get("use_regex_fallback", False))
    edge_min_weight = int(net_cfg.get("edge_min_weight", 3))
    cooccur_level = str(net_cfg.get("cooccur_level", "sentence")).lower()
    domain_hints = [s.lower() for s in CFG.get("domain_hints", [])]

    alias_map = CFG.get("alias", {})
    brand_to_company = load_json("data/dictionaries/brand_to_company.json", {})
    topic_like_entities = _load_lines("data/dictionaries/topic_like_entities.txt")
    ent_org = _load_lines("data/dictionaries/entities_org.txt")
    
    # [수정] whitelist를 소문자로 정규화
    whitelist_base = {w.lower() for w in ent_org} - {t.lower() for t in topic_like_entities}
    whitelist = {norm_tok(alias_map.get(w, w)) for w in whitelist_base}
    org_stop_words = set()

    pair_counter: Counter = Counter()
    pair_ctx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    nodes: set = set()

    for it in items:
        title = (it.get("title") or "").strip()
        text = (it.get("body") or it.get("description") or "").strip()
        if not text:
            continue

        # 1) 도메인 힌트 필터(선택)
        if domain_hints:
            content_low = (title + " " + text).lower()
            if not any(h in content_low for h in domain_hints):
                continue

        # 2) ORG 추출 (소문자로 정규화됨)
        if use_regex_fallback:
            orgs_raw = extract_orgs(text, alias_map, whitelist, org_stop_words)
        else:
            orgs_raw_spacy = extract_orgs_with_spacy(text)
            orgs_raw = [alias_map.get(o.lower(), o.lower()) for o in orgs_raw_spacy]
            orgs_raw = [norm_org_token(o) for o in orgs_raw if o]

        # 3) 화이트리스트 제한
        if whitelist_only:
            orgs_raw = [o for o in orgs_raw if o in whitelist]

        # 4) 브랜드→회사 표준화 (소문자 유지)
        orgs_norm = sorted(set(brand_to_company.get(o, o) for o in orgs_raw if o))
        if len(orgs_norm) == 0:
            continue

        # 5) 동시출현 계산
        if cooccur_level == "sentence":
            sentences = re.split(r"(?<=[.!?])[ \t\n\r]+", text)
            for sent in sentences:
                s_low = sent.lower()
                s_orgs = [o for o in orgs_norm if o.lower() in s_low]
                s_orgs = sorted(set(s_orgs))
                if len(s_orgs) < 2:
                    continue
                for i in range(len(s_orgs)):
                    for j in range(i+1, len(s_orgs)):
                        a, b = s_orgs[i], s_orgs[j]
                        pair = (a, b) if a < b else (b, a)
                        pair_counter[pair] += 1
                        snippet = " | ".join([title, sent[:200]])
                        pair_ctx[pair].append(snippet)
                        nodes.add(a); nodes.add(b)
        else:
            # document level
            if len(orgs_norm) == 1:
                nodes.add(orgs_norm[0])
                continue
            for i in range(len(orgs_norm)):
                for j in range(i+1, len(orgs_norm)):
                    a, b = orgs_norm[i], orgs_norm[j]
                    pair = (a, b) if a < b else (b, a)
                    pair_counter[pair] += 1
                    snippet = " | ".join([title, text[:300]])
                    pair_ctx[pair].append(snippet)
                    nodes.add(a); nodes.add(b)

    # 6) 엣지 생성: 최소 가중치 + 관계 분류
    edges = []
    for (a, b), w in pair_counter.items():
        if w < edge_min_weight:
            continue
        
        # --- ▼▼▼ [수정] 규칙 기반 관계 분류 (Override) ▼▼▼ ---
        key_a = a
        key_b = b
        rule_key = f"{key_a}|{key_b}" if key_a < key_b else f"{key_b}|{key_a}"

        if rule_key in RELATIONSHIP_RULES:
            rel = RELATIONSHIP_RULES[rule_key] # 1. 규칙 파일에서 관계
        else:
            rel = classify_relationship(pair_ctx[(a, b)]) # 2. 규칙 없으면 텍스트 분석
        # --- ▲▲▲ 수정 완료 ▲▲▲ ---

        # --- ▼▼▼ [수정] 근거 문장 1개 추출 ▼▼▼ ---
        context_snippets = pair_ctx.get((a, b), [""])
        first_snippet = context_snippets[0].split(" | ")[-1].strip() # 기사 제목 제외
        key_evidence = first_snippet[:150] + "..." if len(first_snippet) > 150 else first_snippet
        if not key_evidence: key_evidence = "N/A" # 빈 스니펫 방지
        # --- ▲▲▲ 수정 완료 ▲▲▲ ---

        edges.append((a, b, int(w), rel, key_evidence)) # 5번째 요소로 evidence 추가

    return edges, sorted(nodes)


def compute_company_network(items): # items 인자 추가
    edges, nodes = build_cooccurrence_edges(items)
    if not edges and not nodes:
        return None
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    
    # --- ▼▼▼ [수정] evidence 포함하여 엣지 추가 ▼▼▼ ---
    for a, b, w, rel, evidence in edges:
       G.add_edge(a, b, weight=w, rel_type=rel, evidence=evidence)
    # --- ▲▲▲ 수정 완료 ▲▲▲ ---

    print(f"[DEBUG] Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ====== 네트워크 분석 ======
# ====== 네트워크 분석 ======
def analyze_network(G: nx.Graph, top_n: int = 10) -> Dict[str, Any]:
    if not G or G.number_of_nodes() == 0:
        return {
            "timestamp": today_utc_iso(),
            "nodes": [],
            "edges": [],
            "top_pairs": [],
            "centrality": [],
            "betweenness": [],
            "communities": [],
        }

    nodes = [{"org": n, "degree": int(G.degree(n))} for n in G.nodes()]
    
    # --- ▼▼▼ [수정] 최종 JSON에 evidence 포함 ▼▼▼ ---
    edges = [{"source": u, "target": v, 
              "weight": int(d.get("weight", 1)), 
              "rel_type": d.get("rel_type", "neutral"),
              "evidence": d.get("evidence", "") # <-- evidence 필드 추가
             }
             for u, v, d in G.edges(data=True)]
    # --- ▲▲▲ 수정 완료 ▲▲▲ ---

    # 상위 엣지
    top_pairs = sorted(edges, key=lambda x: (x["weight"], x["source"], x["target"]), reverse=True)[:top_n]

    # (이하 중심성, 커뮤니티 분석 등은 기존과 동일)
    # ...
    degc = nx.degree_centrality(G)
    cent = sorted(
        [{"org": n, "degree_centrality": round(float(degc.get(n, 0.0)), 4)} for n in G.nodes()],
        key=lambda x: x["degree_centrality"],
        reverse=True
    )[:top_n]

    betw = nx.betweenness_centrality(G, normalized=True)
    betw_out = sorted(
        [{"org": n, "betweenness": round(float(betw.get(n, 0.0)), 4)} for n in G.nodes()],
        key=lambda x: x["betweenness"],
        reverse=True
    )[:top_n]
    
    comms = []
    try:
        gm = community.greedy_modularity_communities(G, weight="weight")
        for cid, cset in enumerate(gm):
            members = sorted(list(cset))
            comms.append({
                "community_id": int(cid),
                "size": int(len(members)),
                "members": members,
                "interpretation": "",
            })
    except Exception:
        pass

    return {
        "timestamp": today_utc_iso(),
        "nodes": nodes,
        "edges": edges,
        "top_pairs": top_pairs,
        "centrality": cent,
        "betweenness": betw_out,
        "communities": comms,
    }

# ====== 분석 요약 갱신 ======
def generate_analysis_summary(matrix_path: str, network_obj: Dict[str, Any]) -> dict:
    summary = {
        "timestamp": today_utc_iso(),
        "matrix_stats": {},
        "network_stats": {}
    }

    # 매트릭스 통계
    if os.path.exists(matrix_path):
        try:
            df = pd.read_csv(matrix_path, encoding='utf-8-sig')
            summary["matrix_stats"] = {
                "num_orgs": int(len(df)),
                "num_topics": int(len([c for c in df.columns if str(c).startswith('topic_')])),
                "top_org": df.iloc[0]['org'] if (not df.empty and 'org' in df.columns) else None
            }
        except Exception:
            pass

    # 네트워크 통계
    try:
        edges = network_obj.get("edges", [])
        summary["network_stats"] = {
            "num_nodes": int(len(network_obj.get("nodes", []))),
            "num_edges": int(len(edges)),
            "top_pairs": network_obj.get("top_pairs", [])[:5],
            "top_hub": network_obj.get("centrality", [{}])[0].get("org") if network_obj.get("centrality") else None
        }
    except Exception:
        pass

    return summary


def build_company_network(meta_items: List[Dict[str, Any]], out_json="outputs/company_network.json"):
    # items = load_meta_files(max_files=5) # 👈 이 부분을 삭제하고 인자로 받은 meta_items 사용
    print(f"[DEBUG] Building network from {len(meta_items)} meta items")
    G = compute_company_network(meta_items) # compute_company_network에 meta_items 전달
    if not G:
        print("[WARN] No network data.")
        save_json(out_json, {"timestamp": today_utc_iso(), "nodes": [], "edges": [], "top_pairs": [], "centrality": [], "betweenness": [], "communities": []})
        return

    analysis = analyze_network(G, top_n=10)
    save_json(out_json, analysis)
    print(f"[INFO] Saved {out_json} (nodes={len(analysis.get('nodes', []))}, edges={len(analysis.get('edges', []))})")
    

# ====== 메인 파이프라인 ======
def main():
    print("[INFO] Module D - Analysis 시작")

    # 1. 실행 주기에 맞는 메타 데이터 경로 설정
    is_monthly_run = os.getenv("MONTHLY_RUN", "false").lower() == "true"
    if is_monthly_run:
        meta_path = "outputs/debug/monthly_meta_agg.json"
        print(f"[INFO] Monthly Run: Using aggregated meta file for {__name__}.")
    else:
        meta_path = "outputs/debug/news_meta_latest.json"
        if not os.path.exists(meta_path):
            meta_path = latest("data/news_meta_*.json")

    if not meta_path or not os.path.exists(meta_path):
        raise SystemExit("Input meta file not found for Module D.")
        
    # 2. 모든 분석에 사용할 데이터 로드 (단 한 번만)
    print(f"[INFO] Loading meta data from: {meta_path}")
    meta_items = load_json(meta_path, [])
    topics_obj = load_json("outputs/topics.json", {"topics": []})
    
    # 3. 기업×토픽 매트릭스 생성
    try:
        export_company_topic_matrix(meta_items, topics_obj, CFG)
    except Exception as e:
        print(f"[ERROR] Matrix export failed: {e}")

    # 4. 기업 네트워크 생성
    try:
        build_company_network(meta_items, out_json="outputs/company_network.json")
    except Exception as e:
        print(f"[ERROR] Network analysis failed: {e}")

    # 5. 분석 요약 저장
    net_obj = load_json("outputs/company_network.json", {})
    summary = generate_analysis_summary(
        "outputs/export/company_topic_matrix_wide.csv",
        net_obj
    )
    save_json("outputs/analysis_summary.json", summary)
    print("[INFO] Module D 완료")


if __name__ == "__main__":
    main()
