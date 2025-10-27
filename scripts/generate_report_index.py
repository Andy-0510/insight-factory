import os
import json
import glob
from collections import defaultdict

report_index = {"daily": defaultdict(list), "weekly": defaultdict(list), "monthly": defaultdict(list)}
output_base = "outputs"
index_path = "report_index.json"  # 루트에 저장

print(f"Generating report index to {index_path}...")

for report_type in ["daily", "weekly", "monthly"]:
    type_dir = os.path.join(output_base, report_type)
    if not os.path.isdir(type_dir): 
        continue

    for date_folder in sorted(os.listdir(type_dir), reverse=True):  # 최신 날짜 우선 (필요시)
        date_dir = os.path.join(type_dir, date_folder)
        if not os.path.isdir(date_dir): 
            continue

        # 각 날짜 폴더에서 가장 최신 시간 폴더 하나만 선택
        latest_time_folder = None
        time_folders = [f for f in os.listdir(date_dir) if os.path.isdir(os.path.join(date_dir, f))]
        
        if not time_folders:
            continue  # 시간 폴더가 없으면 건너뜀

        # 시간 폴더가 HHMM 형식이라면 정수로 변환하여 가장 큰 값 선택
        try:
            # HHMM 형식의 경우
            latest_time_folder = max(time_folders, key=lambda x: int(x[:4]) if x[:2].isdigit() and x[2:4].isdigit() else -1)
        except:
            # 일반 문자열 비교로 대체
            latest_time_folder = max(time_folders)

        time_dir = os.path.join(date_dir, latest_time_folder)
        if not os.path.isdir(time_dir): 
            continue

        found_reports = []
        # HTML 리포트 우선 검색
        html_files = glob.glob(os.path.join(time_dir, "*.html"))
        # MD 리포트 (HTML 없을 경우 대비)
        md_files = glob.glob(os.path.join(time_dir, "*.md"))

        # HTML 파일 경로 추가 (outputs/type/date/time/report.html 형식)
        for f_path in html_files:
            # report_index.json에 저장할 경로는 outputs부터 시작하는 상대 경로
            relative_path = os.path.join(output_base, report_type, date_folder, latest_time_folder, os.path.basename(f_path)).replace("\\", "/")
            found_reports.append({
                "name": os.path.basename(f_path),
                "path": relative_path
            })

        # HTML이 없을 경우 MD 파일 추가
        if not found_reports and md_files:
            for f_path in md_files:
                relative_path = os.path.join(output_base, report_type, date_folder, latest_time_folder, os.path.basename(f_path)).replace("\\", "/")
                found_reports.append({
                    "name": os.path.basename(f_path),
                    "path": relative_path
                })

        if found_reports:
            # 해당 날짜의 최신 시간 리포트만 저장
            report_index[report_type][date_folder] = [{
                "time": latest_time_folder,
                "reports": found_reports
            }]

# 날짜 키를 최신순으로 정렬
final_index = {}
for report_type, dates in report_index.items():
    # 날짜를 YYYY-MM-DD 형식으로 해석하여 정렬
    sorted_dates = dict(sorted(dates.items(), key=lambda item: item[0], reverse=True))
    final_index[report_type] = sorted_dates

# Save to root or /docs
with open(index_path, "w", encoding="utf-8") as f:
    json.dump(final_index, f, ensure_ascii=False, indent=2)

print(f"Generated report index at {index_path} with {sum(len(v) for v in final_index.values())} dates.")
