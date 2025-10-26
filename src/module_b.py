# -*- coding: utf-8 -*-
# Module B - Keywords (Integrated & Quality-boosted)
# - config.json + data/dictionaries 리소스 병합
# - KR-WordRank + TF-IDF 기반 (라이트), 문서별 KeyBERT MMR 재랭킹 및 BERTopic 보정(프로)
# - 숫자/날짜/통화/단위 필터 + 행정지명/인명/일반어 디버프 + 하드 드롭
# - 값 정렬은 itemgetter(1) 공용 헬퍼로 통일(오타 방지)

import os
import re
import glob
import json
import math
import time
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from src.config import load_config

import numpy as np
from operator import itemgetter  # 값 정렬 키
from src.utils import load_json, save_json, latest


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------
# Optional deps (graceful fallback)
# -------------------------
try:
    from krwordrank.word import KRWordRank
    from krwordrank.hangle import normalize as kr_normalize
except Exception:
    KRWordRank = None
    def kr_normalize(x, english=False, number=True): return (x or "").strip()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer, cosine_similarity = None, None

try:
    from keybert import KeyBERT
except Exception:
    KeyBERT = None

try:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
except Exception:
    BERTopic, UMAP, HDBSCAN = None, None, None

# -------------------------
# Optional deps # Load Korean NER model globally (if available)
# -------------------------
try:
    import spacy
    try:
        # Load the model once when the module is imported
        NLP = spacy.load("ko_core_news_sm")
        print("[INFO] [module_b] spaCy Korean model 'ko_core_news_sm' loaded successfully.")
    except OSError:
        print("[WARN] [module_b] spaCy model 'ko_core_news_sm' not found. Download it using: python -m spacy download ko_core_news_sm. NER features will be limited.")
        NLP = None
    except Exception as e:
        print(f"[WARN] [module_b] Failed to load spaCy model: {e}. NER features disabled.")
        NLP = None
except ImportError:
    print("[WARN] [module_b] spaCy library not found. Install it using: pip install spacy. NER features disabled.")
    spacy = None
    NLP = None


# -------------------------
# Utilities / IO
# -------------------------
def load_lines(path: str) -> List[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []

def latest(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

# 공용: dict를 값(value) 기준 내림차순 정렬
def sort_items_by_value_desc(d: Dict[str, float]):
    return sorted(d.items(), key=itemgetter(1), reverse=True)  # (key,value) 2-튜플의 인덱스 1이 값


# -------------------------
# Build docs from meta
# -------------------------
def build_docs(items: List[dict]) -> List[str]:
    docs = []
    for it in items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        body  = (it.get("body") or it.get("description") or it.get("description_og") or "").strip()
        txt = (title + " " + body).strip()
        if txt:
            docs.append(txt)
    return docs

def dedup_docs_by_cosine(docs: List[str], threshold: float = 0.90) -> List[str]:
    if TfidfVectorizer is None or cosine_similarity is None or len(docs) <= 1:
        return docs
    vec = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X, dense_output=False)
    keep_indices = []
    removed = set()
    for i in range(len(docs)):
        if i in removed: continue
        keep_indices.append(i)
        for j in range(i + 1, len(docs)):
            if j in removed: continue
            if sim[i, j] >= threshold:
                removed.add(j)
    return [docs[i] for i in keep_indices]


# -------------------------
# Normalization / noun-ish cleanup
# -------------------------
_HANGUL_ALNUM = re.compile(r"[가-힣A-Za-z0-9]+")

def basic_normalize(txt: str) -> str:
    t = kr_normalize((txt or "").strip(), english=False, number=True)
    toks = _HANGUL_ALNUM.findall(t)
    return " ".join(toks)


_JOSA = ("은","는","이","가","을","를","에","에서","으로","로","과","와","에게","한테","께","이나","나","든지","까지","부터","라도","마저","밖에","뿐","의","처럼","만큼","보다")
_EOMI = ("했다","하였다","한다","했다가","하며","해서","하는","되다","되면","되었","된다","되니","되어","됐다","됐다가","했다며","이다","였다","이다가","이며","이어서","인","일","입니다","습니다","으니까","으니까요","는데요","고요","구요","네요","군요","시오","십시오")

def nounish_strip(sentence: str) -> str:
    toks = sentence.split()
    out = []
    for tk in toks:
        t = tk
        for suf in _JOSA:
            if t.endswith(suf) and len(t) >= len(suf) + 2:
                t = t[:-len(suf)]
                break
        for suf in _EOMI:
            if t.endswith(suf) and len(t) >= len(suf) + 2:
                t = t[:-len(suf)]
                break
        if len(t) >= 2:
            out.append(t)
    return " ".join(out)


# -------------------------
# Patterns / Locations
# -------------------------
def compile_patterns(CFG: dict):
    rp = CFG.get("regex_patterns", {}) or {} 
    def _comp(key, default): 
        try: 
            return re.compile(rp.get(key, default)) 
        except Exception: 
            return re.compile(default) 
    pats = { 
        "NUMERIC_ONLY": _comp("NUMERIC_ONLY", r"^\d+$"), 
        "DATE_PAT": _comp("DATE_PAT", r"^\d{1,2}일$|^\d{1,2}월$|^\d{4}년$|^\d{4}$"), 
        "CURRENCY_PAT": _comp("CURRENCY_PAT", r"^[0-9,\.]+(원|달러|유로|엔|위안|억원|조원)$"), 
        "PERSON_NAME_PAT": _comp("PERSON_NAME_PAT", r"^[가-힣]{2,4}$"), 
    }
    # config.json에 정의된 UNIT_TOKEN_PAT을 로드 시도, 없으면 기본 빈 패턴
    pats["UNIT_TOKEN_PAT"] = _comp("UNIT_TOKEN_PAT", r"^\d+([.,]\d+)?(PLACEHOLDER_UNIT)$") # Placeholder, 실제 패턴은 config에서 로드됨
    return pats #

_LOCATION_CORE = {
    "서울","부산","대구","인천","광주","대전","울산","세종",
    "경기","경기도","강원","강원도","충북","충남","전북","전남","경북","경남",
    "제주","제주도","수원","용인","성남","고양","화성","부천","안산","안양","남양주"
}
_LOCATION_SUFFIX = {"도","시","군","구","읍","면","동","리"}

def is_location_token(tok: str) -> bool:
    if not tok: return False
    if tok in _LOCATION_CORE:
        return True
    if len(tok) >= 2 and tok[-1] in _LOCATION_SUFFIX:
        return True
    return False


# -------------------------
# Preprocess with strict filters
# -------------------------
def preprocess_docs(docs: List[str], phrase_stop: List[str], stopwords: List[str],
                    use_nounish: bool=True, patterns: Optional[dict]=None) -> List[str]:
    ps = set(phrase_stop or [])
    sw = set(stopwords or [])
    P = patterns or {}

    out = []
    for d in docs:
        if not d:
            continue
        t = basic_normalize(d)
        for ph in ps:
            if ph:
                t = t.replace(ph, " ")

        if use_nounish:
            t = nounish_strip(t)

        toks = []
        for w in t.split():
            # 사전 불용어 제거
            if w in sw:
                continue
            # 숫자/날짜/통화/단위 제거
            if P.get("NUMERIC_ONLY") and P["NUMERIC_ONLY"].match(w):  # 숫자만
                continue
            if P.get("DATE_PAT") and P["DATE_PAT"].match(w):          # 날짜/연도
                continue
            if P.get("CURRENCY_PAT") and P["CURRENCY_PAT"].match(w):  # 통화
                continue
            if P.get("UNIT_TOKEN_PAT") and P["UNIT_TOKEN_PAT"].match(w):  # 숫자+단위
                continue
            if len(w) < 2:
                continue
            toks.append(w)

        t2 = " ".join(toks)
        if t2:
            out.append(t2)
    return out


# -------------------------
# Alias / brand-entity resources
# -------------------------
def build_alias_map(config_alias: Dict[str, str], product_alias_path: str) -> Dict[str, str]:
    alias = dict(config_alias or {})
    pa = load_json(product_alias_path, {})
    for can, variants in pa.items():
        for v in variants:
            alias[v] = can
    return alias

def normalize_alias(token: str, alias_map: Dict[str, str]) -> str:
    if token in alias_map:
        return alias_map[token]
    low = token.lower()
    for k, v in alias_map.items():
        if k.lower() == low:
            return v
    return token

def load_brand_entity_lists() -> Tuple[set, set]:
    brands = set(load_lines("data/dictionaries/brands.txt"))
    entities = set()
    for p in ("data/dictionaries/entities.txt", "data/dictionaries/entities_org.txt", "data/dictionaries/entites.txt"):
        entities.update(load_lines(p))
    brands = {b.strip() for b in brands if b.strip()}
    entities = {e.strip() for e in entities if e.strip()}
    return brands, entities


# -------------------------
# Domain weighting with debuffs
# -------------------------
def apply_domain_weights(scores: Dict[str, float],
                         domain_hints: List[str],
                         common_debuff: List[str],
                         alias_map: Dict[str, str],
                         weight_CFG: Dict[str, float],
                         brands: Optional[set]=None,
                         entities: Optional[set]=None,
                         patterns: Optional[dict]=None) -> Dict[str, float]:
    if not scores:
        return {}
    P = patterns or {}
    boosted = {}
    # Load weights from config
    dh_boost = float(weight_CFG.get("domain_hint_boost", 1.6)) 
    cd_debuff = float(weight_CFG.get("common_debuff", 0.55)) 
    entity_boost_config = float(weight_CFG.get("entity_boost", 1.4))
    brand_boost_config = float(weight_CFG.get("brand_boost", 1.2)) 
    person_debuff = float(weight_CFG.get("person_name_debuff", 0.8))
    loc_debuff = float(weight_CFG.get("location_debuff", 0.6)) 
    num_debuff = float(weight_CFG.get("number_debuff", 0.5))

    # --- ▼▼▼ NER 기반 가중치 추가 ▼▼▼ ---
    ner_org_boost = 1.7 # spaCy가 ORG로 인식한 경우 가중치 (기존 entity_boost보다 약간 높게 설정 가능)
    ner_product_boost = 1.5 # spaCy가 PRODUCT로 인식한 경우 (모델이 지원한다면)
    # --- ▲▲▲ NER 기반 가중치 추가 ▲▲▲ ---

    # Prepare sets for faster lookup
    dh = set(h.lower() for h in (domain_hints or [])) # lowercase comparison
    cm = set(c.lower() for c in (common_debuff or [])) # lowercase comparison
    brands_set = brands or set()
    entities_set = entities or set()

    # Process items only once to avoid duplicate lookups
    keywords_to_process = list(scores.keys())
    ner_results = {} # Cache NER results

    # --- ▼▼▼ NER 분석 (Optional) ▼▼▼ ---
    if NLP: # Only run if spaCy model loaded successfully
        # Process keywords in batches for efficiency if list is very large
        # For moderate lists, processing one by one is fine
        print(f"[INFO] [module_b] Running NER on {len(keywords_to_process)} candidate keywords...")
        # Note: NLP.pipe is more efficient for large numbers, but simple loop is okay here
        for k in keywords_to_process:
            try:
                doc = NLP(k) # Process keyword with spaCy
                # Store the first recognized entity label (if any)
                if doc.ents:
                    ner_results[k] = doc.ents[0].label_ # Get label like 'ORG', 'PERSON'
                else:
                     ner_results[k] = None # No entity found
            except Exception as ner_err:
                # Log error for the specific keyword and continue
                # print(f"[WARN] [module_b] NER failed for keyword '{k}': {ner_err}")
                ner_results[k] = None # Mark as failed/None

        print("[INFO] [module_b] NER analysis complete.")
    # --- ▲▲▲ NER 분석 완료 ▲▲▲ ---

    for k, v in scores.items():
        k_normalized = normalize_alias(k, alias_map) # Apply alias normalization
        s = v # Initial score
        ner_label = ner_results.get(k) # Get cached NER result for original keyword k

        # --- ▼▼▼ 가중치 적용 로직 통합 ▼▼▼ ---
        final_boost = 1.0
        final_debuff = 1.0

        # 1. Domain Hints Boost (using normalized keyword)
        if any(h in k_normalized.lower() for h in dh): # Use lowercase for hints check
            final_boost = max(final_boost, dh_boost)

        # 2. NER-based Boost (using original keyword k for NER label)
        if ner_label == 'ORG':
            final_boost = max(final_boost, ner_org_boost)
        elif ner_label == 'PRODUCT': # If your spaCy model supports PRODUCT
            final_boost = max(final_boost, ner_product_boost)
        # Add more elif for other entity types if needed

        # 3. List-based Boost (using normalized keyword)
        # Apply only if NER didn't already give a stronger boost
        if k_normalized in entities_set and final_boost < entity_boost_config:
            final_boost = max(final_boost, entity_boost_config)
        if k_normalized in brands_set and final_boost < brand_boost_config:
            final_boost = max(final_boost, brand_boost_config)

        # 4. Debuffs (using normalized keyword)
        if k_normalized.lower() in cm: # Use lowercase for common debuff check
            final_debuff = min(final_debuff, cd_debuff)

        # NER Person Debuff (using original keyword k for NER label)
        # Apply person debuff if NER identified as PERSON, *unless* it's also in entities/brands list
        if ner_label == 'PERSON' and k_normalized not in entities_set and k_normalized not in brands_set:
             final_debuff = min(final_debuff, person_debuff)
        # Fallback Regex Person Debuff (if NER didn't run or didn't identify)
        elif ner_label is None and P.get("PERSON_NAME_PAT") and P["PERSON_NAME_PAT"].fullmatch(k_normalized):
             if k_normalized not in entities_set and k_normalized not in brands_set:
                 final_debuff = min(final_debuff, person_debuff)

        # Location Debuff (normalized keyword)
        if is_location_token(k_normalized):
            final_debuff = min(final_debuff, loc_debuff)

        # Number/Date/Currency/Unit Debuff (normalized keyword)
        if (P.get("NUMERIC_ONLY") and P["NUMERIC_ONLY"].match(k_normalized)) \
           or (P.get("DATE_PAT") and P["DATE_PAT"].match(k_normalized)) \
           or (P.get("CURRENCY_PAT") and P["CURRENCY_PAT"].match(k_normalized)) \
           or (P.get("UNIT_TOKEN_PAT") and P["UNIT_TOKEN_PAT"].match(k_normalized)):
            final_debuff = min(final_debuff, num_debuff)

        # Apply final combined boost and debuff
        s = s * final_boost * final_debuff
        # --- ▲▲▲ 가중치 적용 로직 통합 완료 ▲▲▲ ---

        if s <= 0:
            continue

        # Store the potentially updated score for the NORMALIZED keyword
        # Use max to handle cases where different original keywords map to the same normalized one
        boosted[k_normalized] = max(boosted.get(k_normalized, 0.0), s)

    return boosted


# -------------------------
# Stats / autotune
# -------------------------
def compute_doc_stats(docs: List[str]) -> Tuple[int, float]:
    n = len(docs)
    avg = np.mean([len(x) for x in docs]) if docs else 0.0
    return n, float(avg)

def autotune_kr(n_docs: int, avg_len: float, min_count_base: int=3) -> Tuple[int,int]:
    mc = max(min_count_base, int(round(math.log1p(n_docs) + avg_len/800)))
    mc = min(max(3, mc), 12)
    max_len = 12 if avg_len < 400 else 15
    return mc, max_len


# -------------------------
# KR-WordRank + TF-IDF (Light)
# -------------------------
def extract_krwordrank(docs: List[str], beta: float=0.85, max_iter: int=20, min_count: Optional[int]=None,
                       max_length: Optional[int]=None, topk: int=200) -> Dict[str, float]:
    if KRWordRank is None:
        return {}
    n_docs, avg_len = compute_doc_stats(docs)
    if min_count is None or max_length is None:
        mc, ml = autotune_kr(n_docs, avg_len)
        if min_count is None: min_count = mc
        if max_length is None: max_length = ml
    extractor = KRWordRank(min_count=min_count, max_length=max_length, verbose=False)
    keywords, rank, _ = extractor.extract(docs, beta=beta, max_iter=max_iter)
    sorted_items = sort_items_by_value_desc(keywords)  # 값 기준 정렬
    return dict(sorted_items[: max(1, int(topk or 1))])

def tfidf_weights(docs: List[str], vocab: List[str]) -> Dict[str, float]:
    # --- ▼▼▼▼▼ [수정] 문서 개수 확인 로직 추가 ▼▼▼▼▼ ---
    if TfidfVectorizer is None or not docs or len(docs) < 3:
        # 문서 수가 3개 미만이면 TF-IDF 계산을 건너뛰고 기본 가중치 반환
        return {v: 1.0 for v in vocab}
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=3, max_df=0.9)
    X = vec.fit_transform(docs)
    idf = dict(zip(vec.get_feature_names_out(), vec.idf_))
    return {v: float(idf.get(v, 1.0)) for v in vocab}

def hybrid_rank(docs: List[str], beta: float=0.85, max_iter: int=20, topk: int=200,
                w_kr: float=0.7, w_tfidf: float=0.3) -> Dict[str, float]:
    kr = extract_krwordrank(docs, beta=beta, max_iter=max_iter, topk=topk)
    if not kr:
        return {}
    vocab = list(kr.keys())
    idf = tfidf_weights(docs, vocab)
    def norm(d):
        vals = list(d.values()) if d else [0.0]
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return {k: 1.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}
    kr_n = norm(kr)
    idf_n = norm(idf)
    blended = {k: w_kr*kr_n.get(k,0.0) + w_tfidf*idf_n.get(k,0.0) for k in vocab}
    sorted_items = sort_items_by_value_desc(blended)  # 값 기준 정렬
    return dict(sorted_items[: max(1, int(topk or 1))])

def tfidf_only(docs: List[str], topk: int=200) -> Dict[str, float]:
    # --- ▼▼▼▼▼ [수정] 문서 개수 확인 로직 추가 ▼▼▼▼▼ ---
    if TfidfVectorizer is None or not docs or len(docs) < 3:
        # 문서 수가 3개 미만이면 빈 딕셔너리 반환
        return {}
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=3, max_df=0.9)
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    avg = np.asarray(X.mean(axis=0)).ravel()
    pairs = list(zip(terms, avg))
    pairs.sort(key=itemgetter(1), reverse=True)  # 값 기준 정렬
    return dict(pairs[: max(1, int(topk or 1))])


# -------------------------
# KeyBERT MMR reranking (Pro, per-document)
# -------------------------
def keybert_rerank_doc(doc: str, candidates: List[str], model_name: str, topn: int,
                       use_mmr: bool=True, diversity: float=0.5, ngram_range: Tuple[int,int]=(1,3), stopwords: Optional[set]=None) -> Dict[str,float]:
    if KeyBERT is None or not doc or not candidates:
        return {}
    try:
        kb = KeyBERT(model=model_name)
        extracted = kb.extract_keywords(
            doc,
            keyphrase_ngram_range=ngram_range,
            stop_words=list(stopwords) if stopwords else None,
            use_mmr=use_mmr,
            diversity=diversity,
            top_n=max(topn, len(candidates))
        )
        cand_set = set(candidates)
        rer = [(p, s) for (p, s) in extracted if p in cand_set]
        return dict(rer[:topn]) if rer else {}
    except Exception:
        return {}



# -------------------------
# BERTopic topic context (optional)
# -------------------------
def topic_context_keywords(docs: List[str], model_name: str, umap_neighbors: int=15,
                           min_cluster_size: int=12, topn_per_topic: int=10) -> Dict[int, List[str]]:
    if BERTopic is None:
        return {}
    try:
        umap_model = UMAP(n_neighbors=umap_neighbors, n_components=10, metric="cosine", low_memory=True, random_state=42)
        hdb_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", prediction_data=True)
        tm = BERTopic(umap_model=umap_model, hdbscan_model=hdb_model,
                      embedding_model=model_name, calculate_probabilities=False, verbose=False)
        topics, _ = tm.fit_transform(docs)
        info = tm.get_topic_info()
        out = {}
        for t in info["Topic"].tolist():
            if int(t) < 0:
                continue
            words = [w for w, _ in tm.get_topic(int(t))[:topn_per_topic]]
            out[int(t)] = words
        return out
    except Exception:
        return {}


# -------------------------
# Main
# -------------------------
def main():
    print("[INFO] [module_b] KICK-OFF: 키워드 추출을 시작합니다.") # 시작 로그
    t0 = time.time() # 시간 측정 시작
    
    CFG = load_config()
    weights = CFG.get("weights", {}) or {}

    # Merge config + data/dictionaries resources
    defaults = CFG.get("keyword_extraction_defaults", {}) or {}

    phrase_stop = sorted(
        set(CFG.get("phrase_stop", []) or [])
        | set(load_lines("data/dictionaries/phrase_stopwords.txt"))
    )

    stopwords = sorted(
        set(CFG.get("stopwords", []) or [])
        | set(load_lines("data/dictionaries/stopwords_ext.txt"))
        | set(defaults.get("MORE_STOP", []))
        | set(defaults.get("EN_STOP", []))
    )

    alias_seed = {}
    alias_seed.update(defaults.get("FIX_MAP", {}) or {})
    alias_seed.update(CFG.get("alias", {}) or {})
    alias_map = build_alias_map(alias_seed, product_alias_path="data/dictionaries/product_alias.json")

    brands, entities = load_brand_entity_lists()

    # Patterns
    patterns = compile_patterns(CFG)

    topn_keywords = int(CFG.get("top_n_keywords", 50))
    
    # --- Pro/Lite 모드 결정 로직 (환경변수(.yml, .env) 우선 -> config.json) ---
    env_pro = os.getenv("USE_PRO", "").lower()
    if env_pro in ("1", "true", "yes", "y"):
        use_pro = True
    elif env_pro in ("0", "false", "no", "n"):
        use_pro = False
    else:
        # 환경 변수가 없거나 인식할 수 없는 값이면 config.json 확인
        use_pro = bool(CFG.get("use_pro", False))
    print(f"[INFO] [module_b] 실행 모드: {'PRO' if use_pro else 'LITE'}")
    
    # --- ▼▼▼▼▼ [수정된 부분] 데이터 로드 로직 ▼▼▼▼▼ ---
    is_monthly_run = os.getenv("MONTHLY_RUN", "false").lower() == "true"
    
    if is_monthly_run:
        meta_path = "outputs/debug/monthly_meta_agg.json"
        print(f"[INFO] Monthly Run: Using aggregated meta file for {__name__}.")
    else:
        # 일간 실행 시에는 디버깅용 최신 복사본을 우선 사용
        meta_path = "outputs/debug/news_meta_latest.json"
        if not os.path.exists(meta_path):
            meta_path = latest("data/news_meta_*.json")

    if not meta_path or not os.path.exists(meta_path):
        # `no data/news_meta_*.json found` 메시지 대신 더 명확한 에러 메시지
        raise SystemExit(f"Input meta file not found at expected path: {meta_path}")
        
    print(f"[INFO] Loading meta data from: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        items = json.load(f)
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    
    raw_docs = build_docs(items)
    if not raw_docs:
        raise SystemExit("no documents")
    print(f"[INFO] [module_b] 총 {len(raw_docs)}개 문서 로드 완료. 문서 정제 및 전처리를 시작합니다.")


    # Dedup (optional)
    raw_docs = dedup_docs_by_cosine(raw_docs, threshold=0.93)

    # Preprocess with strict filters
    pre_docs = preprocess_docs(raw_docs, phrase_stop=phrase_stop, stopwords=stopwords,
                               use_nounish=True, patterns=patterns)
    if not pre_docs:
        raise SystemExit("no valid docs after preprocessing")

    beta = float(weights.get("beta", 0.85))
    max_iter = int(weights.get("max_iter", 20))

    # Base extraction
    if KRWordRank is not None:
        print("[INFO] [module_b] 기본 키워드 추출 실행 (KR-WordRank + TF-IDF Hybrid)")
        base_scores = hybrid_rank(pre_docs, beta=beta, max_iter=max_iter, topk=max(200, topn_keywords))
    else:
        print("[INFO] [module_b] 기본 키워드 추출 실행 (TF-IDF Only)")
        base_scores = tfidf_only(pre_docs, topk=max(200, topn_keywords))
    print(f"[INFO] [module_b] 기본 추출 완료. 후보 키워드 {len(base_scores)}개 생성.")

    combined = base_scores.copy()

    # Pro: per-document KeyBERT MMR reranking and aggregation (Modified with TF-IDF weighting)
    if use_pro and KeyBERT is not None and combined:
        print("[INFO] [module_b] [PRO] KeyBERT MMR 재랭킹 (TF-IDF 가중치 적용)을 시작합니다...") # 로그 수정
        cand = list(combined.keys())
        model_name = CFG.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") # 
        diversity = float(weights.get("mmr_diversity", 0.65)) # config.json에서 가져오도록 수정 
        max_docs_rerank = int(CFG.get("max_docs_rerank", 135)) # 
        sel_docs = pre_docs[:max_docs_rerank] #

        # --- ▼▼▼ TF-IDF 가중치 계산 추가 ▼▼▼ ---
        doc_term_weights = defaultdict(lambda: defaultdict(float))
        if TfidfVectorizer is not None and len(sel_docs) >= 3: # TF-IDF 계산 조건 확인
            try:
                # Use parameters consistent with tfidf_only or hybrid_rank
                tfidf_vec = TfidfVectorizer(ngram_range=(1,3), min_df=3, max_df=0.9, stop_words=list(stopwords)) # stopwords 추가
                X_tfidf = tfidf_vec.fit_transform(sel_docs) #
                feature_names = tfidf_vec.get_feature_names_out() #
                # Map feature names to their indices for quick lookup
                feature_indices = {name: idx for idx, name in enumerate(feature_names)}

                # Store TF-IDF score for each term in each document
                rows, cols = X_tfidf.nonzero()
                for row, col in zip(rows, cols):
                    term = feature_names[col]
                    # Check if the term is in our candidate list to reduce memory usage
                    if term in cand:
                         doc_term_weights[row][term] = X_tfidf[row, col]
                print(f"[INFO] [module_b] [PRO] TF-IDF 가중치 계산 완료 ({len(doc_term_weights)} 문서)")
            except Exception as tfidf_err:
                 print(f"[WARN] [module_b] [PRO] TF-IDF 가중치 계산 실패: {tfidf_err}. 가중치 없이 진행합니다.")
                 # Clear weights if calculation failed
                 doc_term_weights.clear() #
        else:
             print("[INFO] [module_b] [PRO] 문서 수가 적거나 TfidfVectorizer 없음. TF-IDF 가중치 없이 진행.")
        # --- ▲▲▲ TF-IDF 가중치 계산 완료 ▲▲▲ ---


        agg_weighted_score, agg_weight_sum = defaultdict(float), defaultdict(float) # 가중 합계, 가중치 합계

        # Initialize KeyBERT model once
        try:
            kb_model = KeyBERT(model=model_name)
        except Exception as kb_init_err:
             print(f"[ERROR] [module_b] [PRO] KeyBERT 모델 초기화 실패: {kb_init_err}. Pro 모드 중단.")
             kb_model = None

        if kb_model: # Proceed only if KeyBERT model initialized successfully
            print(f"[INFO] [module_b] [PRO] 문서별 KeyBERT 추출 시작 (문서 수: {len(sel_docs)})")
            # Process documents one by one
            for doc_idx, d in enumerate(sel_docs):
                try:
                    # Extract keywords using KeyBERT MMR
                    rer = kb_model.extract_keywords( # Renamed variable from kb
                        d, #
                        keyphrase_ngram_range=(1,3), #
                        stop_words=list(stopwords) if stopwords else None, # Use consistent stopwords
                        use_mmr=True, #
                        diversity=diversity, #
                        top_n=min(len(cand), topn_keywords * 2) # Extract slightly more candidates initially
                    )
                except Exception as kb_extract_err:
                    # Log error for the specific document and continue
                    print(f"[WARN] [module_b] [PRO] 문서 {doc_idx} KeyBERT 추출 실패: {kb_extract_err}")
                    rer = [] # Set to empty list on failure


                if not rer:
                    continue

                # Filter results to only include candidates from the initial list (cand)
                filtered_rer = {p: s for (p, s) in rer if p in cand}

                if not filtered_rer:
                    continue

                # Normalize KeyBERT scores within the document (0 to 1)
                vals = list(filtered_rer.values()) #
                mn, mx = min(vals), max(vals) #
                norm_factor = (mx - mn + 1e-12) # Add epsilon for stability

                # Get TF-IDF weights for this document (or default to 1.0)
                current_doc_tfidf = doc_term_weights.get(doc_idx, {})

                # Aggregate scores weighted by TF-IDF
                for k, v in filtered_rer.items():
                    keybert_score_norm = (v - mn) / norm_factor # Normalized KeyBERT score
                    # Use TF-IDF weight if available, otherwise default weight is 1.0
                    tfidf_weight = current_doc_tfidf.get(k, 1.0)

                    # Weighted score contribution from this document
                    agg_weighted_score[k] += keybert_score_norm * tfidf_weight
                    agg_weight_sum[k] += tfidf_weight # Sum of weights (TF-IDF or 1.0)

            # Calculate the final weighted average score
            if agg_weighted_score:
                final_reranked_scores = {
                    k: agg_weighted_score[k] / agg_weight_sum[k]
                    for k in agg_weighted_score if agg_weight_sum[k] > 0
                }

                # Blend the original base scores with the new reranked scores
                all_keys = list(set(list(combined.keys()) + list(final_reranked_scores.keys())))
                def norm_dict(d, keys): # Helper to normalize scores
                    valid_scores = [d.get(k, 0.0) for k in keys]
                    mn, mx = min(valid_scores), max(valid_scores)
                    norm_factor = (mx - mn + 1e-12)
                    return {k: (d.get(k, 0.0) - mn) / norm_factor for k in keys}

                base_n = norm_dict(combined, all_keys) # Normalize original scores
                rer_n = norm_dict(final_reranked_scores, all_keys) # Normalize new scores

                # Combine scores (adjust weights 0.4/0.6 as needed)
                combined = {k: 0.4 * base_n.get(k, 0.0) + 0.6 * rer_n.get(k, 0.0) for k in all_keys}
                print("[INFO] [module_b] [PRO] KeyBERT 재랭킹 및 TF-IDF 가중 평균 적용 완료.")
            else:
                 print("[INFO] [module_b] [PRO] KeyBERT 재랭킹 결과가 없어 기본 점수 유지.")

    # ... (Rest of the code: Optional BERTopic boost, apply_domain_weights, hard drop, Output) ...

    # Optional: topic context boost (Pro)
    if use_pro and BERTopic is not None and len(pre_docs) >= 20:
        print("[INFO] [module_b] [PRO] BERTopic 컨텍스트 가중치 적용 시작...")
        model_name = CFG.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        umap_neighbors = int(CFG.get("umap_neighbors", 15))
        min_cluster_size = int(CFG.get("min_cluster_size", 12))
        topic_kw = topic_context_keywords(pre_docs, model_name=model_name,
                                          umap_neighbors=umap_neighbors,
                                          min_cluster_size=min_cluster_size,
                                          topn_per_topic=topn_keywords)
        topic_set = set([w for lst in topic_kw.values() for w in lst])
        for k in list(combined.keys()):
            if k in topic_set:
                combined[k] *= 1.05
                print("[INFO] [module_b] [PRO] BERTopic 컨텍스트 가중치 적용 완료.")

    # Domain/alias/brand/entity weights + debuffs
    print("[INFO] [module_b] 도메인 사전 가중치 및 디버프를 적용합니다.")
    combined = apply_domain_weights(
        combined,
        domain_hints=CFG.get("domain_hints", []),
        common_debuff=CFG.get("common_debuff", []),
        alias_map=alias_map,
        weight_CFG=weights,
        brands=brands,
        entities=entities,
        patterns=patterns
    )

    # Hard drop: 숫자/날짜/통화/단위 최종 제거
    def _hard_drop(tok: str) -> bool:
        return (patterns["NUMERIC_ONLY"].match(tok)
                or patterns["DATE_PAT"].match(tok)
                or patterns["CURRENCY_PAT"].match(tok)
                or patterns["UNIT_TOKEN_PAT"].match(tok))
    combined = {k: v for k, v in combined.items() if not _hard_drop(k)}

    # Output
    top_items = sort_items_by_value_desc(combined)[: topn_keywords]
    print(f"[INFO] [module_b] 최종 키워드 {len(top_items)}개 선정. 파일로 저장합니다.")
    print(f"[SUCCESS] [module_b] 키워드 추출 완료 | 최종 키워드 수: {len(top_items)} | 소요시간: {round(time.time()-t0, 2)}초")


    os.makedirs("outputs", exist_ok=True)
    with open("outputs/keywords.json", "w", encoding="utf-8") as f:
        json.dump({"keywords": [{"keyword": k, "score": float(s)} for k, s in top_items]}, f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] [module_b] 키워드 추출 완료 | 최종 키워드 수: {len(top_items)} | 소요시간: {round(time.time()-t0, 2)}초")
    
    os.makedirs("outputs/debug", exist_ok=True)
    with open("outputs/debug/run_meta_b.json", "w", encoding="utf-8") as f:
        json.dump({
            "use_pro": use_pro,
            "docs": len(pre_docs),
            "deps": {"krwordrank": KRWordRank is not None, "keybert": KeyBERT is not None, "bertopic": BERTopic is not None},
            "resources_dir": "data/dictionaries"
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
