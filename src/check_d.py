import json
import sys
import os

def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def main():
    # 1. 분석 요약 파일 확인
    if not os.path.exists("outputs/analysis_summary.json"):
        fail("outputs/analysis_summary.json 없음")
    
    try:
        with open("outputs/analysis_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        fail("analysis_summary.json 로드 실패")
    
    # 2. 매트릭스 파일 확인
    if not os.path.exists("outputs/export/company_topic_matrix_wide.csv"):
        fail("company_topic_matrix_wide.csv 없음")
    
    if not os.path.exists("outputs/export/company_topic_matrix_long.csv"):
        fail("company_topic_matrix_long.csv 없음")
    
    # 3. 네트워크 파일 확인
    if not os.path.exists("outputs/company_network.json"):
        fail("company_network.json 없음")
    
    try:
        with open("outputs/company_network.json", "r", encoding="utf-8") as f:
            network = json.load(f)
    except Exception:
        fail("company_network.json 로드 실패")
    
    # 4. 기본 검증
    if "edges" not in network:
        fail("company_network.json에 edges 필드 없음")
    
    print(f"[INFO] Check D OK | matrix_orgs={summary.get('matrix_stats', {}).get('num_orgs', 0)} "
          f"network_edges={summary.get('network_stats', {}).get('num_edges', 0)}")

if __name__ == "__main__":
    main()