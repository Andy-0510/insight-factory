import os
import json
import glob
import re
import sys
import datetime
from email.utils import parsedate_to_datetime
from src.timeutil import to_date, now_kst, kst_date_str, kst_run_suffix
from src.utils import load_json, save_json, latest


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_existing_urls(path):
    urls = set()
    if not os.path.exists(path):
        return urls
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                u = obj.get("url")
                if u:
                    urls.add(u)
            except Exception:
                continue
    return urls

def main():
    print("[INFO] [warehouse_append] KICK-OFF: 데이터 웨어하우스 저장을 시작합니다.")
    
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        print("[ERROR] meta 파일이 없습니다. Module A부터 실행하세요.")
        sys.exit(1)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    ensure_dir("data/warehouse")
    
    # --- ▼▼▼ [수정] 파일 이름 생성을 반복문 밖으로 이동 ▼▼▼ ---
    date_part = kst_date_str()
    run_part  = kst_run_suffix()
    out_path = f"data/warehouse/{date_part}-{run_part}.jsonl"
    print(f"[INFO] Appending data to a single file for this run: {out_path}")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    
    appended, skipped = 0, 0
    
    # [수정] 파일 내 중복은 한 번만 체크하면 되므로, 반복문 밖에서 로드
    existing_urls = load_existing_urls(out_path)

    for it in items:
        url = (it.get("url") or "").strip()
        if not url:
            skipped += 1
            continue
        
        # [수정] 매번 파일 시스템을 확인하는 대신, 메모리에 로드된 세트에서 확인
        if url in existing_urls:
            skipped += 1
            continue

        d_raw = it.get("published_time") or it.get("pubDate_raw") or ""
        published = to_date(d_raw)
        
        row = {
            "url": url,
            "title": it.get("title"),
            "site_name": it.get("site_name"),
            "_query": it.get("_query") or it.get("query"),
            "published": published,
            "created_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        }
        
        with open(out_path, "a", encoding="utf-8") as wf:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        # [수정] 방금 추가한 URL을 세트에도 추가하여 다음 중복 체크에 사용
        existing_urls.add(url)
        appended += 1
    
    print(f"[INFO] warehouse append | appended={appended} skipped={skipped}")
    
if __name__ == "__main__":
    main()
