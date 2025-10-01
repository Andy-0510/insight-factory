# -*- coding: utf-8 -*-
import os
import re
import json
import csv
import glob
import unicodedata
import datetime
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from src.config import load_config, llm_config

# ========== 공통 로드/유틸 ==========
def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def load_json(path: str, default=None):
    if default is None:
        default = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _load_lines(p: str) -> set:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return {x.strip() for x in f if x.strip()}
    except Exception:
        return set()

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def to_kst_date_str(s: str) -> str:
    from email.utils import parsedate_to_datetime
    try:
        if not s:
            raise ValueError
        s2 = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(s2)
        d = dt.date()
    except Exception:
        try:
            dt = parsedate_to_datetime(s)
            d = dt.date()
        except Exception:
            m = re.search(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})", s or "")
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                d = datetime.date(y, mm, dd)
            else:
                d = datetime.date.today()
    today = datetime.date.today()
    if d > today:
        d = today
    return d.strftime("%Y-%m-%d")

# ========== 회사×토픽 매트릭스: ORG 잡음 제거 ==========
# 패턴은 코드에 유지하되, 단어 목록은 config.json에서 관리 (유지보수성)

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
        # 별칭을 먼저 적용하여 'LG디스플레' -> 'LG디스플레이' 등으로 표준화
        normalized = alias_map.get(t, t)
        normalized = alias_map.get(normalized.lower(), normalized)
        normalized_toks.append(normalized)

    cand = []
    for t in normalized_toks:
        # 화이트리스트에 있으면 노이즈 필터를 통과
        if t in whitelist:
            cand.append(t)
            continue
        # 화이트리스트에 없으면 노이즈 필터 적용
        if is_bad_org_token(t, org_stop_words):
            continue
        cand.append(t)

    cnt = Counter(cand)
    # 화이트리스트 외 단어는 2번 이상 등장해야 후보로 인정
    out = [w for w, c in cnt.most_common(50) if c >= 2 and w not in whitelist]
    
    # 최종 결과: (문서 내 화이트리스트 단어) + (자주 등장한 비-노이즈 단어)
    return list(whitelist.intersection(set(normalized_toks))) + out

def load_topic_labels(topics_obj: dict, topn: int) -> list[dict]:
    labels = []
    for t in (topics_obj.get("topics") or []):
        words = [w.get("word","") for w in (t.get("top_words") or []) if w.get("word")][:topn]
        labels.append({"topic_id": int(t.get("topic_id", 0)), "words": words})
    return labels

def export_company_topic_matrix(meta_items: List[Dict[str,Any]], topics_obj: dict, cfg: dict) -> None:
    os.makedirs("outputs/export", exist_ok=True)
    
    # --- 사전 및 설정 로드 ---
    alias_cfg = cfg.get("alias", {})
    alias_product = load_json("data/dictionaries/product_alias.json", {})
    full_alias_map = dict(alias_cfg)
    for can, variants in alias_product.items():
        for v in variants:
            full_alias_map[v] = can
            full_alias_map[v.lower()] = can
            
    ent_org = set(_load_lines("data/dictionaries/entities_org.txt"))
    brands  = set(_load_lines("data/dictionaries/brands.txt"))
    whitelist = ent_org | brands
    org_stop_words = set(cfg.get("org_filter_stop_words", []))

    # --- 토픽 데이터 준비 ---
    topn_words = int(cfg.get("matrix_topic_top_n_words", 30))
    topic_labels = load_topic_labels(topics_obj, topn=topn_words)
    topic_wordsets = {tl["topic_id"]: {w.lower() for w in tl.get("words", [])} for tl in topic_labels}

    # --- 매트릭스 생성 ---
    matrix = defaultdict(lambda: defaultdict(int))
    
    for it in meta_items:
        text = (it.get("body") or it.get("description") or "") or ""
        if not text:
            continue
        
        orgs = extract_orgs(text, full_alias_map, whitelist, org_stop_words)
        low_text = text.lower()
        
        for org in set(orgs):
            for tid, ws in topic_wordsets.items():
                hit = sum(1 for w in ws if w and w in low_text)
                if hit > 0:
                    matrix[org][tid] += hit

    # --- CSV 파일 작성 ---
    all_tids = sorted(topic_wordsets.keys())
    
    with open("outputs/export/company_topic_matrix.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["org"] + [f"topic_{tid}" for tid in all_tids])
        
        for org in sorted(matrix.keys()):
            # 최종 단계에서 한번 더 노이즈 필터링
            if is_bad_org_token(org, org_stop_words):
                continue
            row = matrix[org]
            w.writerow([org] + [row.get(tid, 0) for tid in all_tids])
            
# ========== 스코어링/증거 ==========
def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

def normalize_score(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp01((x - lo) / (hi - lo))

NEG_WORDS = ["논란", "우려", "리스크", "규제", "지연", "지적", "하락", "부진", "적자", "연기"]

def extract_keywords_from_idea(idea_text: str, keywords_obj: dict) -> List[str]:
    """아이디어 문장에서 핵심 키워드를 추출합니다."""
    top_keywords = {k.get("keyword", "") for k in keywords_obj.get("keywords", [])}
    found_keywords = []
    for kw in top_keywords:
        if kw and kw.lower() in idea_text.lower():
            found_keywords.append(kw)
    
    nouns = re.findall(r"[가-힣A-Za-z]{2,}", idea_text)
    if not found_keywords and nouns:
        return nouns[:3]
    return found_keywords

def pick_evidence(idea_keywords: List[str], items: List[Dict[str,Any]], limit=3) -> List[Dict[str,Any]]:
    ev = []
    if not idea_keywords:
        return ev
    
    for it in items:
        base = (it.get("raw_body") or it.get("body") or it.get("description") or "")
        if not base:
            continue
        
        low = base.lower()
        if any(kw.lower() in low for kw in idea_keywords):
            sents = re.split(r"(?<=[\.!?다])\s+", base)
            for s in sents:
                if any(kw.lower() in (s or "").lower() for kw in idea_keywords):
                    ev.append({
                        "sentence": (s or "").strip()[:400],
                        "url": it.get("url") or "",
                        "date": to_kst_date_str(it.get("published_time") or it.get("pubDate_raw") or "")
                    })
                    if len(ev) >= limit:
                        return ev
    return ev

# ========== LLM 프롬프트/파서/정규화 ==========
def build_schema_hint() -> Dict[str, Any]:
    return {
        "idea": "아이디어 한 줄 제목",
        "problem": "해결하려는 문제(2-3문장, 220자 이내)",
        "target_customer": "핵심 타깃(산업/직군/조직 규모 명확히)",
        "value_prop": "핵심 가치제안(차별점, 180자 이내)",
        "solution": ["핵심 기능 bullet 최대 4개"],
        "risks": ["리스크/규제 bullet 3개 내"],
        "priority_score": "우선순위 점수(0.0~5.0, 숫자)"
    }

def build_prompt(context: Dict[str, Any], want: int = 5) -> str:
    schema = build_schema_hint()
    return (
        f"당신은 최상위 디스플레이 제조 기업의 '사업 전략 전문가'입니다. "
        f"아래 컨텍스트(최신 기술 뉴스 키워드, 토픽, 트렌드)를 기반으로 우리 회사가 추진할 만한 구체적인 신사업 아이디어를 제안해 주세요.\n"
        f"- 아이디어 개수: 정확히 {want}개\n"
        f"- JSON 배열 형식만 출력하세요. 설명은 필요 없습니다.\n"
        f"- 각 아이템은 아래 스키마 키를 정확히 사용하세요: {json.dumps(schema, ensure_ascii=False)}\n"
        f"- 제약 조건:\n"
        f"  1) 아이디어는 반드시 '차량용 디스플레이', 'AR/VR/XR용 마이크로디스플레이', 'IT용 차세대 패널(OLED, MicroLED)', '신소재/부품', '공정 자동화/수율 개선' 중 최소 3개 이상의 카테고리를 포함해야 합니다.\n"
        f"  2) 각 아이디어의 'problem' 항목에는 반드시 최신 트렌드나 'Why now' 관점을 1문장 이상 포함하여 문제의 시의성을 강조하세요.\n"
        f"  3) 'target_customer'는 '글로벌 완성차 OEM', '북미 빅테크 기업'처럼 구체적으로 명시하세요.\n"
        f"  4) 'priority_score'는 시장 잠재력, 기술 실현 가능성, 경쟁 강도를 고려하여 객관적으로 평가해주세요.\n"
        f"컨텍스트:\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )

def strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[\t ]*\w*[\t ]*\n", "", t, flags=re.M)
    t = re.sub(r"\n```[\t ]*$", "", t, flags=re.M)
    return t.strip()

def clean_json_text2(t: str) -> str:
    t = strip_code_fence(t or "")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u2028\u2029]", "", t)
    t = re.sub(r",\s*(\}|\])", r"\1", t)
    t = t.lstrip("\ufeff")
    return t.strip()

def parse_json_array_or_object(t: str):
    s = clean_json_text2(t)
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("ideas"), list):
            return obj["ideas"]
    except Exception:
        return None
    return None

def extract_balanced_array(t: str):
    s = clean_json_text2(t)
    start = s.find("[")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(s[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    payload = s[start:end+1]
    try:
        arr = json.loads(payload)
        return arr if isinstance(arr, list) else None
    except Exception:
        return None

def extract_objects_sequence(t: str, max_items=10):
    s = clean_json_text2(t)
    out, i, n = [], 0, len(s)
    while i < n and len(out) < max_items:
        start = s.find("{", i)
        if start == -1:
            break
        depth, end = 0, -1
        j = start
        while j < n:
            ch = s[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
            j += 1
        if end == -1:
            break
        chunk = s[start:end+1]
        try:
            obj = json.loads(clean_json_text2(chunk))
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            pass
        i = end + 1
    return out

def extract_ndjson_lines(t: str, max_items=10):
    ideas = []
    for line in (t or "").splitlines():
        if len(ideas) >= max_items:
            break
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
        try:
            obj = json.loads(clean_json_text2(line))
            if isinstance(obj, dict):
                ideas.append(obj)
                continue
        except Exception:
            objs = extract_objects_sequence(line, max_items=2)
            for o in objs:
                if isinstance(o, dict):
                    ideas.append(o)
                    if len(ideas) >= max_items:
                        break
    return ideas

def extract_ideas_any(text: str, want=5):
    arr = parse_json_array_or_object(text)
    if isinstance(arr, list) and arr:
        return arr
    arr2 = extract_balanced_array(text)
    if isinstance(arr2, list) and arr2:
        return arr2
    objs = extract_objects_sequence(text, max_items=want)
    if objs:
        return objs
    nd = extract_ndjson_lines(text, max_items=want)
    if nd:
        return nd
    return None

def as_list(x, max_len=4) -> List[str]:
    if isinstance(x, list):
        out = []
        for v in x[:max_len]:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    if x is None:
        return []
    s = str(x).strip()
    return [s] if s else []

def clip_text(s: Optional[str], max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"

def to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def normalize_item(it: Dict[str, Any]) -> Dict[str, Any]:
    idea = it.get("idea") or it.get("title") or it.get("name") or ""
    problem = it.get("problem") or it.get("pain") or ""
    target = it.get("target_customer") or it.get("target") or it.get("audience") or ""
    value = it.get("value_prop") or it.get("value") or it.get("description") or ""
    solution = it.get("solution") or it.get("solutions") or []
    risks = it.get("risks") or it.get("risk") or []
    score = it.get("priority_score", it.get("score", 0))
    score = max(0.0, min(5.0, to_float(score, 0.0)))

    idea = clip_text(idea, 100)
    problem = clip_text(problem, 300)
    target = clip_text(target, 120)
    value = clip_text(value, 220)
    solution = as_list(solution, max_len=4)
    risks = as_list(risks, max_len=3)

    out = {
        "idea": idea,
        "problem": problem,
        "target_customer": target,
        "value_prop": value,
        "solution": solution,
        "risks": risks,
        "priority_score": round(score, 1),
        "title": idea,
        "score": round(score, 1)
    }
    return out

# ========== LLM 호출 ==========
CFG = load_config()
LLM = llm_config(CFG)

def load_context_for_prompt() -> Dict[str, Any]:
    keywords = load_json("outputs/keywords.json", default={"keywords": []}) or {"keywords": []}
    topics = load_json("outputs/topics.json", default={"topics": []}) or {"topics": []}
    insights = load_json("outputs/trend_insights.json", default={"summary": "", "top_topics": [], "evidence": {}}) or {"summary": "", "top_topics": [], "evidence": {}}
    trend_strength_path = "outputs/export/trend_strength.csv"
    events_path = "outputs/export/events.csv"

    summary = (insights.get("summary") or "").strip()
    if len(summary) > 1200:
        summary = summary[:1200] + "…"

    kw_simple = [{"keyword": k.get("keyword",""), "score": k.get("score",0)} for k in (keywords.get("keywords") or [])[:20]]
    max_topics = int(CFG.get("llm_context_max_topics", 12))
    tp_simple = []
    for t in (topics.get("topics") or [])[:max_topics]:
        words = [w.get("word","") for w in (t.get("top_words") or [])][:6]
        tp_simple.append({"topic_id": t.get("topic_id"), "words": words})

    trend_rows = []
    try:
        with open(trend_strength_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    trend_rows.append({"term": r["term"], "cur": int(r.get("cur",0) or 0), "z_like": float(r.get("z_like",0.0) or 0.0)})
                except Exception:
                    continue
        trend_rows = sorted(trend_rows, key=lambda x: (x["z_like"], x["cur"]), reverse=True)[:30]
    except Exception:
        trend_rows = []

    evt_summary = Counter()
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                # 'types'와 'type'을 모두 안전하게 처리
                types_str = r.get("types") or r.get("type") or ""
                if types_str:
                    for etype in types_str.split(','):
                        evt_summary[etype.strip()] += 1
    except Exception:
        pass
    events_simple = dict(evt_summary)

    return {"summary": summary, "keywords": kw_simple, "topics": tp_simple, "trends": trend_rows, "events": events_simple}

def call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경 변수가 없습니다.")
    genai.configure(api_key=api_key)
    model_name = str(LLM.get("model", "gemini-2.0-flash-001"))
    max_tokens = int(LLM.get("max_output_tokens", 2048))
    temperature = float(LLM.get("temperature", 0.3))
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature, "top_p": 0.9}
    )
    text = (getattr(resp, "text", None) or "").strip()
    return text

def make_opportunities_llm(meta_items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    want = 5
    context = load_context_for_prompt()
    prompt = build_prompt(context, want=want)
    try:
        raw_text = call_gemini(prompt)
    except Exception as e:
        print(f"[ERROR] Gemini 호출 실패: {e}")
        return []
    ideas_raw = extract_ideas_any(raw_text, want=want) or []
    items = []
    for it in ideas_raw:
        try:
            norm = normalize_item(it)
            if norm["idea"] and norm["value_prop"]:
                items.append(norm)
        except Exception:
            continue
    items.sort(key=lambda x: x.get("priority_score", 0.0), reverse=True)
    return items[:want]

# ========== 신호 보강 ==========
def load_trend_strength_csv(path: str) -> List[Dict[str,Any]]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    r["cur"] = int(r.get("cur", 0) or 0)
                    r["prev"] = int(r.get("prev", 0) or 0)
                    r["diff"] = int(r.get("diff", 0) or 0)
                    r["total"] = int(r.get("total", 0) or 0)
                    r["ma7"] = float(r.get("ma7", 0.0) or 0.0)
                    r["z_like"] = float(r.get("z_like", 0.0) or 0.0)
                    rows.append(r)
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return rows

def load_events_csv(path: str) -> List[Dict[str,str]]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append(r)
    except Exception:
        pass
    return rows

def enrich_with_signals(ideas: List[Dict[str,Any]],
                        meta_items: List[Dict[str,Any]],
                        trend_rows: List[Dict[str,Any]],
                        events_rows: List[Dict[str,str]],
                        cfg: Dict[str, Any],
                        keywords_obj: dict) -> List[Dict[str,Any]]:
    
    weights = cfg.get("score_weights", {})
    prio_w = float(weights.get("priority_llm_weight", 0.25))
    mkt_w = float(weights.get("market_weight", 0.40))
    urg_w = float(weights.get("urgency_weight", 0.35))
    feas_w = float(weights.get("feasibility_weight", 0.25))
    risk_p = float(weights.get("risk_penalty", 1.0))
    
    trend_idx = {r.get("term",""): r for r in trend_rows}
    event_hit = defaultdict(int)
    for r in events_rows:
        types_str = r.get("types", "")
        if types_str:
            for etype in types_str.split(','):
                event_hit[etype.strip()] += 1

    out = []
    for it in ideas:
        idea_text = (it.get("idea") or "").strip()
        idea_keywords = extract_keywords_from_idea(idea_text, keywords_obj)
        
        cur, z = 0, 0.0
        if idea_keywords:
            related_trends = [trend_idx.get(kw, {}) for kw in idea_keywords]
            if related_trends:
                cur_list = [int(t.get("cur", 0)) for t in related_trends]
                z_list = [float(t.get("z_like", 0.0)) for t in related_trends]
                if cur_list: cur = max(cur_list)
                if z_list: z = max(z_list)

        s_market = ((1.0 - prio_w) * (cur / 10.0)) + (prio_w * normalize_score(it.get("priority_score", 0.0), 0.0, 5.0))
        s_urg = z / 5.0 
        evt_boost = sum(0.10 for etype in ["LAUNCH", "PARTNERSHIP"] if event_hit.get(etype, 0) > 0)
        s_urg = clamp01(s_urg + evt_boost)
        s_feas = 0.6
        
        it["evidence"] = pick_evidence(idea_keywords, meta_items, limit=3)
        risk = sum(0.10 for e in it["evidence"] if any(nw in (e.get("sentence") or "") for nw in NEG_WORDS))
        risk = clamp01(risk)
        
        score100 = 100.0 * (mkt_w * s_market + urg_w * s_urg + feas_w * s_feas) - (100.0 * risk_p * risk)
        it["score"] = round(max(0.0, min(100.0, score100)), 2)
        it["score_breakdown"] = {"market": round(s_market, 3), "urgency": round(s_urg, 3), "feasibility": round(s_feas, 3), "risk": round(risk, 3), "notes": {"cur": cur, "z_like": z}}
        
        it["title"] = it.get("title") or it.get("idea") or ""
        it["problem"] = it.get("problem") or f"{idea_text} 관련 과제가 상존함."
        it["target_customer"] = it.get("target_customer") or "기업(B2B)"
        it["value_prop"] = it.get("value_prop") or f"{it['idea']} 도입 가치(비용/품질/경험 개선)."
        it["solution"] = it.get("solution") or ["파일럿→제휴→인증 확보"]
        it["risks"] = it.get("risks") or ["규제/표준 불확실성", "비용/ROI 불확실성"]
        out.append(it)
        
    return out

# ========== Top5 보장 ==========
def fill_opportunities_to_five(ideas: list, keywords_obj: dict, want: int = 5) -> list:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    existing = ideas[:]
    if len(existing) >= want:
        return existing[:want]

    cands = [(k.get("keyword",""), float(k.get("score", 0))) for k in (keywords_obj.get("keywords") or [])[:50]]
    cands = [c for c in cands if c[0] and len(c[0]) >= 2]

    titles = [it.get("idea") or it.get("title") or "" for it in existing if (it.get("idea") or it.get("title"))]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
    if titles:
        M = vec.fit_transform(titles + [c[0] for c in cands])
        base = M[:len(titles)]
        pool = M[len(titles):]
    else:
        M = vec.fit_transform([c[0] for c in cands])
        base = None
        pool = M

    used = set(titles)
    for i, (term, _) in enumerate(cands):
        if len(existing) >= want:
            break
        if term in used:
            continue
        if base is not None:
            sim = cosine_similarity(pool[i:i+1], base).max() if base.shape[0] > 0 else 0.0
            if sim >= 0.6:
                continue
        sk = {
            "idea": term, "title": term,
            "problem": f"{term} 관련 시장/도입/규격 이슈를 해결할 기회.",
            "target_customer": "기업(B2B)",
            "value_prop": f"{term} 도입으로 비용/품질/경험을 개선.",
            "solution": ["파일럿", "파트너십", "인증/규격 검토", "조달/유통 테스트"],
            "risks": ["규제/표준 불확실성", "ROI 불확실성"],
            "priority_score": 3.0,
            "score": 60.0,
            "score_breakdown": {"market":0.5,"urgency":0.5,"feasibility":0.6,"risk":0.0,"notes":{}},
            "evidence": []
        }
        existing.append(sk)
        used.add(term)
    return existing[:want]

# ========== 메인 ==========
def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/export", exist_ok=True)

    meta_items = load_json(latest("data/news_meta_*.json"), [])
    keywords_obj = load_json("outputs/keywords.json", {"keywords":[]})
    topics_obj   = load_json("outputs/topics.json", {"topics":[]})
    trend_rows = load_trend_strength_csv("outputs/export/trend_strength.csv")
    events_rows = load_events_csv("outputs/export/events.csv")

    # 로그 보강
    print(f"[INFO] Loaded context data | trend_rows={len(trend_rows)}, events_rows={len(events_rows)}")

    topic_labels = load_topic_labels(topics_obj, topn=5) 
    try:
        export_company_topic_matrix(meta_items, topics_obj, CFG)
        print("[INFO] company_topic_matrix.csv exported")
    except Exception as e:
        print("[WARN] company_topic_matrix export failed:", repr(e))

    try:
        ideas_llm = make_opportunities_llm(meta_items)
    except Exception as e:
        print("[ERROR] LLM stage failed:", repr(e))
        ideas_llm = []

    # enrich_with_signals 호출 시 keywords_obj를 추가로 전달합니다.
    ideas_with_scores = enrich_with_signals(ideas_llm, meta_items, trend_rows, events_rows, CFG, keywords_obj)
    ideas_with_scores.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    ideas_final = fill_opportunities_to_five(ideas_with_scores, keywords_obj, want=5)

    save_json("outputs/biz_opportunities.json", {"ideas": ideas_final})
    print("[INFO] Module D done | ideas=%d" % len(ideas_final))

if __name__ == "__main__":
    main()
