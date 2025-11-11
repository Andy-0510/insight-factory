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
    """
    [수정] spaCy(NER) 의존도를 낮추고, Whitelist 기반 토큰 매칭을 우선하는 로직
    """
    if not text:
        return []
        
    orgs = set()
    text_low = text.lower() # 텍스트 전체를 소문자로

    # 1. [신규] Whitelist 매칭 (가장 중요)
    #    (spaCy보다 먼저, 사전에 등록된 모든 기업명을 텍스트에서 검색)
    for w in whitelist:
        # (w는 이미 norm_tok 처리된 소문자 상태라고 가정)
        if w in text_low:
            orgs.add(w)

    # 2. [신규] spaCy NER (보조)
    #    (Whitelist에 없는 기업을 추가로 탐지하기 위함)
    spacy_orgs = extract_orgs_with_spacy(text)
    for v in spacy_orgs:
        t = v.strip().lower()
        base = alias_map.get(t, t)
        base = alias_map.get(base.lower(), base)
        base = norm_org_token(base)
        
        if base and not is_bad_org_token(base, org_stop_words):
             # (whitelist_only가 켜져있으면 어차피 걸러지지만,
             #  꺼져있을 경우를 대비해 중복 추가)
            orgs.add(base)

    # 3. [신규] 브랜드 -> 회사명 변환 (최종 단계)
    #    (예: "galaxy" -> "삼성전자")
    brand_to_company = load_json("data/dictionaries/brand_to_company.json", {})
    final_orgs = set()
    for o in orgs:
        # whitelist에 있는 이름(예: '삼성디스플레이')은 그대로 사용하고,
        # whitelist에 없는 이름(예: 'galaxy')만 brand_to_company에서 변환 시도
        if o in whitelist:
            final_orgs.add(o)
        else:
            final_orgs.add(brand_to_company.get(o, o))

    # 4. Whitelist 필터링 (config.json 설정에 따름)
    net_cfg = CFG.get("network", {})
    whitelist_only = bool(net_cfg.get("whitelist_only", True))
    
    if whitelist_only:
        return list(final_orgs.intersection(whitelist))
    else:
        return list(final_orgs)

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

# ===== Optimized co-occurrence builder (config-driven) =====
def build_cooccurrence_edges(items: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, str, int, str, str]], List[str]]:
    # 0) config + dictionaries (기존과 동일)
    net_cfg = CFG.get("network", {})
    whitelist_only = bool(net_cfg.get("whitelist_only", True))
    use_regex_fallback = bool(net_cfg.get("use_regex_fallback", False))
    edge_min_weight = int(net_cfg.get("edge_min_weight", 3))
    # [수정] cooccur_level은 이제 'sentence'로 강제됨
    domain_hints = [s.lower() for s in CFG.get("domain_hints", [])]

    alias_map = CFG.get("alias", {})
    brand_to_company = load_json("data/dictionaries/brand_to_company.json", {})
    topic_like_entities = _load_lines("data/dictionaries/topic_like_entities.txt")
    ent_org = _load_lines("data/dictionaries/entities_org.txt")
    
    whitelist_base = {w.lower() for w in ent_org} - {t.lower() for t in topic_like_entities}
    whitelist = {norm_tok(alias_map.get(w, w)) for w in whitelist_base}
    org_stop_words = set() # (기존 is_bad_org_token에서 사용)

    # --- ▼▼▼ [신규] 1. 문장 기반 카운터 및 증거 저장소 ▼▼▼ ---
    pair_comp_counter: Counter = Counter()
    pair_part_counter: Counter = Counter()
    pair_neutral_counter: Counter = Counter()
    
    pair_comp_evidence: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    pair_part_evidence: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    pair_neutral_evidence: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    
    nodes: set = set()
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    for it in items:
        title = (it.get("title") or "").strip()
        text = (it.get("body") or it.get("description") or "").strip()
        url = it.get("url", "#") # 증거 수집용
        if not text:
            continue

        # (Optional: 도메인 힌트 필터 - 기존 유지)
        if domain_hints:
            content_low = (title + " " + text).lower()
            if not any(h in content_low for h in domain_hints):
                continue

        # --- ▼▼▼ [신규] 2. 문장 단위로 순회하며 로직 실행 ▼▼▼ ---
        sentences = re.split(r"(?<=[.!?다])\s+", text)

        for sent in sentences:
            sent_low = sent.lower()
            
            # A. 이 문장에 언급된 Org 찾기
            orgs_in_sent_raw = extract_orgs(sent, alias_map, whitelist, org_stop_words)
            orgs_in_sent = sorted(set(brand_to_company.get(o, o) for o in orgs_in_sent_raw if o))

            if len(orgs_in_sent) < 2: # 문장 안에 2개 미만 기업 시 스킵
                continue
            
            # B. 이 문장에 언급된 긍/부정 키워드 찾기
            found_comp_kws = [kw for kw in COMPETITIVE_KEYWORDS if kw in sent_low]
            found_part_kws = [kw for kw in COOPERATIVE_KEYWORDS if kw in sent_low]

            is_comp = len(found_comp_kws) > 0
            is_part = len(found_part_kws) > 0
            
            # 증거 객체 생성
            evidence_obj = {"title": title, "url": url, "text": sent, "comp_score": len(found_comp_kws), "part_score": len(found_part_kws)}

            # C. 이 문장 기준으로 카운터 증가
            for i in range(len(orgs_in_sent)):
                for j in range(i+1, len(orgs_in_sent)):
                    a, b = orgs_in_sent[i], orgs_in_sent[j]
                    pair = (a, b) if a < b else (b, a)
                    nodes.add(a); nodes.add(b)

                    # 경쟁 키워드가 협력 키워드보다 우선권
                    if is_comp:
                        pair_comp_counter[pair] += 1
                        pair_comp_evidence[pair].append(evidence_obj)
                    elif is_part: # 경쟁이 아닐 때만 협력 카운트
                        pair_part_counter[pair] += 1
                        pair_part_evidence[pair].append(evidence_obj)
                    else: # 둘 다 없으면 중립 카운트
                        pair_neutral_counter[pair] += 1
                        pair_neutral_evidence[pair].append(evidence_obj)
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    # 4. 엣지 생성 (최종 결과 집계)
    edges = []
    all_pairs = set(pair_comp_counter.keys()) | set(pair_part_counter.keys()) | set(pair_neutral_counter.keys())

    for pair in all_pairs:
        a, b = pair
        comp_score = pair_comp_counter.get(pair, 0)
        part_score = pair_part_counter.get(pair, 0)
        neutral_score = pair_neutral_counter.get(pair, 0)
        
        # 총 동시언급 횟수 (문장 기준)
        total_weight = comp_score + part_score + neutral_score
        
        if total_weight < edge_min_weight: # 최소 가중치 필터
            continue

        # 4a. 관계 유형(rel_type) 결정
        tentative_rel_type = "neutral"
        if comp_score > part_score:
            tentative_rel_type = "rivalry"
        elif part_score > comp_score:
            tentative_rel_type = "partnership"

        # 4b. 규칙(Rules) 오버라이드
        key_a = a.lower()
        key_b = b.lower()
        rule_key = f"{key_a}|{key_b}" if key_a < key_b else f"{key_b}|{key_a}"
        final_rel_type = RELATIONSHIP_RULES.get(rule_key, tentative_rel_type)

        # 4c. 최종 관계 유형에 따라 근거 기사 목록(evidence) 결정
        top_articles_for_evidence = []
        if final_rel_type == "rivalry":
            pair_comp_evidence[pair].sort(key=lambda x: x["comp_score"], reverse=True)
            top_articles_for_evidence = pair_comp_evidence[pair][:3] # 경쟁 상위 3개
        elif final_rel_type == "partnership":
            pair_part_evidence[pair].sort(key=lambda x: x["part_score"], reverse=True)
            top_articles_for_evidence = pair_part_evidence[pair][:3] # 협력 상위 3개
        else: # "neutral"
            top_articles_for_evidence = pair_neutral_evidence[pair][:3] # 중립 3개

        # 4d. 근거 기사 HTML 생성 (이전과 동일, 점수 제거)
        links = []
        for art in top_articles_for_evidence:
            title_clean = art['title'].replace('<', '&lt;').replace('>', '&gt;')
            if final_rel_type == "rivalry":
                links.append(f'<a href="{art["url"]}" target="_blank" style="font-weight:600; color:#c41e3a;">{title_clean}</a>')
            elif final_rel_type == "partnership":
                links.append(f'<a href="{art["url"]}" target="_blank" style="font-weight:600; color:#065f46;">{title_clean}</a>')
            else: # "neutral"
                links.append(f'<a href="{art["url"]}" target="_blank" style="color:#555;">{title_clean}</a>')

        key_evidence_html = "<br>".join(links) if links else "N/A"

        edges.append((a, b, int(total_weight), final_rel_type, key_evidence_html))

    # 5. 디버그 로그 (이전 단계에서 추가했던 코드)
    try:
        os.makedirs("outputs/debug", exist_ok=True)
        debug_log_path = "outputs/debug/company_pair_evidence_scores.txt"
        
        # [수정] 엣지 리스트를 디버깅용으로 변환
        debug_all_evidence_scores = []
        for a, b, w, rel, ev in edges:
             debug_all_evidence_scores.append({
                 "pair": f"{a}|{b}",
                 "weight": w,
                 "rel_type": rel,
                 "scored_articles_html": ev # (HTML을 그대로 저장)
             })

        debug_all_evidence_scores.sort(key=lambda x: x["weight"], reverse=True)
        top_10_pairs = debug_all_evidence_scores[:10]

        with open(debug_log_path, "w", encoding="utf-8") as f:
            f.write(f"--- Top 10 Company Pairs Evidence Scoring Debug Log (Sentence-Based) ---\n")
            f.write(f"Generated at: {datetime.datetime.utcnow().isoformat() + 'Z'}\n")
            
            for i, pair_data in enumerate(top_10_pairs, 1):
                f.write("\n" + "="*80 + "\n")
                f.write(f"Rank #{i}: {pair_data['pair']} (Total Sentences: {pair_data['weight']}, Type: {pair_data['rel_type']})\n")
                f.write(f"Top 3 Evidence Links (HTML formatted):\n")
                f.write("---------------------\n")
                f.write(pair_data['scored_articles_html'].replace("<br>", "\n") + "\n")
                    
        print(f"[INFO] [module_d] Saved evidence scoring debug log to {debug_log_path}")
        
    except Exception as e:
        print(f"[WARN] [module_d] Failed to write evidence scoring debug log: {e}")

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
    # --- ▼▼▼ [신규] 2-2-1 선별 로직 (Top 5 테이블용) ▼▼▼ ---
    rivalry_edges = sorted([e for e in edges if e["rel_type"] == "rivalry"], key=lambda x: x["weight"], reverse=True)
    partnership_edges = sorted([e for e in edges if e["rel_type"] == "partnership"], key=lambda x: x["weight"], reverse=True)
    neutral_edges = sorted([e for e in edges if e["rel_type"] == "neutral"], key=lambda x: x["weight"], reverse=True)

    # Top 2 Rivalry, Top 2 Partnership, 1 Neutral
    top_pairs = []
    top_pairs.extend(rivalry_edges[:2])
    top_pairs.extend(partnership_edges[:2])
    if neutral_edges:
        top_pairs.append(neutral_edges[0])

    # 만약 개수가 모자라면(5개 미만) 다른 타입에서 채움 (단, 5개 제한)
    if len(top_pairs) < 5:
        # 이미 추가된 엣지 (source, target) 확인
        added_pairs = set((e['source'], e['target']) for e in top_pairs)

        # 남은 엣지 풀 (중복 제외)
        remaining_edges = [
            e for e in (rivalry_edges[2:] + partnership_edges[2:] + neutral_edges[1:])
            if (e['source'], e['target']) not in added_pairs
        ]
        remaining_edges.sort(key=lambda x: x["weight"], reverse=True)

        needed = 5 - len(top_pairs)
        top_pairs.extend(remaining_edges[:needed])

    # 최종적으로 5개로 자르고, 가중치 순으로 정렬
    top_pairs = sorted(top_pairs, key=lambda x: x["weight"], reverse=True)[:5]
    # --- ▲▲▲ 선별 로직 완료 ▲▲▲ ---

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
