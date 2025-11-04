import os
import sys
import json
import shutil
import glob
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

KST = timezone(timedelta(hours=9))
PY = sys.executable # Use system's Python executable

def log(msg: str):
    print(f"[MAIN] {msg}", flush=True)

def run_step(name: str, cmd: list[str], env: dict):
    log(f"=== [{name}] 실행: {' '.join(cmd)} ===")
    # Make sure env values are strings
    safe_env = {k: str(v) for k, v in env.items()}
    proc = subprocess.run(cmd, env=safe_env, capture_output=True, text=True) # Capture output
    if proc.stdout:
        print(proc.stdout.strip()) # Print stdout
    if proc.returncode != 0:
        print(f"[ERROR] Step failed: {name}", file=sys.stderr)
        if proc.stderr:
             print(proc.stderr.strip(), file=sys.stderr) # Print stderr on error
        raise SystemExit(f"[중단] 단계 실패: {name}")
    log(f"[완료] {name}")


def load_env_file():
    if load_dotenv is None:
        return
    candidates = [Path(".env"), Path(__file__).resolve().parent / ".env"]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p)
            log(f".env 로드: {p}")
            break

def merge_config_env(env: dict):
    cfg_path = Path("config.json")
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    def pick(key_env: str, key_cfg: str, default=None):
        if env.get(key_env) is not None:
            return env[key_env]
        if key_cfg in cfg and cfg[key_cfg] is not None:
            return str(cfg[key_cfg]).lower() if isinstance(cfg[key_cfg], bool) else str(cfg[key_cfg])
        return default

    env.setdefault("DRY_RUN", pick("DRY_RUN", "dry_run", "false"))
    env.setdefault("USE_PRO", pick("USE_PRO", "use_pro", "false"))
    env.setdefault("TZ", "Asia/Seoul")
    return env

def require_key(env: dict, name: str):
    v = env.get(name) or os.getenv(name)
    if not v:
        raise SystemExit(f"[중단] 필수 키 누락: {name}")
    return v

# --- ▼▼▼ Archiving Functions (Separated by Type) ▼▼▼ ---
def _archive_outputs(pipeline_type: str, out_base="outputs"):
    """Archives outputs to the appropriate daily/weekly/monthly folder."""
    if pipeline_type not in ["daily", "weekly", "monthly"]:
        log(f"[WARN] Unknown pipeline type '{pipeline_type}' for archiving. Skipping.")
        return

    date_kst = datetime.now(KST).strftime("%Y-%m-%d")
    time_kst = datetime.now(KST).strftime("%H%M-KST")
    outdir = os.path.join(out_base, pipeline_type, date_kst, time_kst)

    try:
        os.makedirs(os.path.join(outdir, "fig"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "export"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "debug"), exist_ok=True)

        copied_files = 0
        # Copy top-level files
        for pat in ["outputs/*.json", "outputs/*.html", "outputs/*.md"]:
            for p in glob.glob(pat):
                try:
                    shutil.copy(p, outdir)
                    copied_files += 1
                except Exception as e:
                    log(f"[WARN] Failed to copy {p} to {outdir}: {e}")

        # Copy subdirectories
        for sub in ["export", "fig", "debug"]:
            src = os.path.join("outputs", sub)
            dst_sub = os.path.join(outdir, sub)
            if os.path.isdir(src):
                try:
                    # Copy entire directory tree
                    shutil.copytree(src, dst_sub, dirs_exist_ok=True)
                    # Count files in the source subdir for logging purposes
                    copied_files += sum([len(files) for r, d, files in os.walk(src)])
                except Exception as e:
                    log(f"[WARN] Failed to copy directory {src} to {dst_sub}: {e}")

        log(f"[아카이브 완료] {copied_files} files/dirs copied to {outdir}")

    except Exception as e:
        log(f"[ERROR] Archiving failed for {outdir}: {e}")

# Separate functions remain for potential direct calls if needed, but _archive_outputs is primary
def do_archive_daily(out_base="outputs"):
     _archive_outputs("daily", out_base)

def do_archive_weekly(out_base="outputs"):
     _archive_outputs("weekly", out_base)

def do_archive_monthly(out_base="outputs"):
     _archive_outputs("monthly", out_base)
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

# --- ▼▼▼ Step Definitions (Separated by Pipeline) ▼▼▼ ---
def build_daily_steps():
    """Defines the sequence of steps for the daily pipeline."""
    return [
        ("a",                [PY, "-m", "src.module_a"]),
        ("wh",               [PY, "-m", "src.warehouse_append"]),
        ("body",             [PY, "-m", "scripts.fetch_article_bodies"]),
        ("b",                [PY, "-m", "src.module_b"]),
        ("c",                [PY, "-m", "src.module_c"]),
        ("sentiment",        [PY, "-m", "scripts.calculate_daily_sentiment"]),
        ("export",           [PY, "-m", "scripts.signal_export"]),
        ("select_articles",  [PY, "-m", "scripts.select_top_articles"]),
        # ("future",         [PY, "-m", "scripts.future_insights"]), # future_insights not in daily.yml
        ("gen_visual_daily", [PY, "-m", "scripts.generate_visuals", "--report-type", "daily"]), # Added arg
        # ("preflight",      [PY, "-m", "scripts.preflight"]), # Not explicitly in daily.yml execution block
        ("f_daily",          [PY, "-m", "src.module_f.daily_report"]),
        ("f_cmt_daily",      [PY, "-m", "src.module_f.daily_commentary_report"]), # Keep commented out
        ("report_index",     [PY, "-m", "scripts.generate_report_index"]),
        ("html_daily_report",     [PY, "-m", "scripts.generate_daily_html_report"]),
    ]

def build_weekly_steps():
    """Defines the sequence of steps for the weekly pipeline."""
    return [
        ("aggregate_weekly", [PY, "-m", "scripts.aggregate_weekly_data"]),
        ("future",           [PY, "-m", "scripts.future_insights"]), # Uses aggregated data
        ("c",                [PY, "-m", "src.module_c"]), # Uses aggregated data
        ("gen_visual_weekly",[PY, "-m", "scripts.generate_visuals", "--report-type", "weekly"]), # Added arg
        # ("f_weekly",         [PY, "-m", "src.module_f.weekly_report"]),
        ("f_cmt_weekly",   [PY, "-m", "src.module_f.weekly_commentary_report"]), # Keep commented out
        ("html_weekly_report",   [PY, "-m", "scripts.generate_weekly_html_report"]),
        
    ]

def build_monthly_steps():
    """Defines the sequence of steps for the monthly pipeline."""
    return [
        ("aggregate_monthly",[PY, "-m", "scripts.aggregate_monthly_data"]),
        ("b",                [PY, "-m", "src.module_b"]), # Uses aggregated data
        ("c",                [PY, "-m", "src.module_c"]), # Uses aggregated data
        ("export",           [PY, "-m", "scripts.signal_export"]), # Uses aggregated data
        ("topic_growth",     [PY, "-m", "scripts.calculate_topic_growth"]),
        ("enrich_topics",    [PY, "-m", "scripts.enrich_monthly_topics"]),
        ("d",                [PY, "-m", "src.module_d"]),
        ("e",                [PY, "-m", "src.module_e"]),
        ("future",           [PY, "-m", "scripts.future_insights"]), # Uses aggregated data
        ("g_risk",           [PY, "-m", "src.module_g_risk"]),
        ("h_planning",       [PY, "-m", "src.module_h_planning"]),
        ("gen_visual_monthly",[PY, "-m", "scripts.generate_visuals", "--report-type", "monthly"]), # Added arg
        # ("f_monthly",        [PY, "-m", "src.module_f.monthly_report"]),
        # ("f_cmt_monthly",  [PY, "-m", "src.module_f.monthly_commentary_report"]), # Keep commented out
        ("html_monthly_report",  [PY, "-m", "scripts.generate_monthly_html_report"]), # Keep commented out

        
    ]
# --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

def main():
    parser = argparse.ArgumentParser(description="Local runner (GitHub Actions-like)")
    # --- ▼▼▼ Pipeline Selection Argument ▼▼▼ ---
    parser.add_argument(
        "--pipeline",
        choices=["daily", "weekly", "monthly"],
        required=True, # Make it required as there's no default full run anymore
        help="실행할 파이프라인 (daily, weekly, monthly)"
    )
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    parser.add_argument("--dry-run", choices=["true", "false"], help="드라이런 실행 여부")
    parser.add_argument("--pro-mode", choices=["true", "false"], help="Pro 모드")
    parser.add_argument("--body-min-len", type=int, default=None, help="본문 최소 길이")
    parser.add_argument("--only", nargs="*", help="선택한 파이프라인 내에서 실행할 단계 지정") # Clarified help text
    # --- ▼▼▼ Archive Argument (Optional, default is based on pipeline) ▼▼▼ ---
    parser.add_argument("--archive", action="store_true", help="파이프라인 실행 후 결과 아카이브 강제 실행")
    parser.add_argument("--no-archive", action="store_true", help="파이프라인 실행 후 결과 아카이브 비활성화")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
    parser.add_argument(
        "--generate-visuals",
        type=str,
        choices=['daily', 'weekly', 'monthly'],
        help="지정된 리포트 타입의 시각화만 생성하고 종료합니다."
    )

    args = parser.parse_args()

    # --- Visuals-only mode remains the same ---
    if args.generate_visuals:
        report_type = args.generate_visuals
        log(f"--- Running visuals only for report type: {report_type} ---")
        command = [
            PY, "-m", "scripts.generate_visuals", "--report-type", report_type
        ]
        # Environment setup for visuals-only might be needed depending on the script
        load_env_file()
        env_visuals = os.environ.copy()
        env_visuals = merge_config_env(env_visuals) # Ensure config settings like USE_PRO are read
        run_step(f"gen_visual_{report_type}", command, env_visuals)
        log("--- Visuals generation successful ---")
        sys.exit(0)
    # --- End Visuals-only mode ---

    # --- Full Pipeline Execution ---
    log(f"--- Running full pipeline: {args.pipeline} ---")

    load_env_file()

    env = os.environ.copy()
    if args.dry_run is not None:
        env["DRY_RUN"] = args.dry_run
    if args.pro_mode is not None:
        env["USE_PRO"] = args.pro_mode
    if args.body_min_len is not None:
        env["BODY_MIN_LEN"] = str(args.body_min_len)

    env = merge_config_env(env)

    # --- ▼▼▼ 수정된 부분: 환경 변수 명시적 설정 ▼▼▼ ---
    if args.pipeline == "daily":
        all_pipeline_steps = build_daily_steps()
        env["USE_PRO"] = env.get("USE_PRO", "false") # 일간 기본값 유지
        env["WEEKLY_RUN"] = "false"  # 명시적으로 false 설정
        env["MONTHLY_RUN"] = "false" # 명시적으로 false 설정
    elif args.pipeline == "weekly":
        all_pipeline_steps = build_weekly_steps()
        env["USE_PRO"] = "true"
        env["WEEKLY_RUN"] = "true"   # 명시적으로 true 설정
        env["MONTHLY_RUN"] = "false" # 명시적으로 false 설정
    elif args.pipeline == "monthly":
        all_pipeline_steps = build_monthly_steps()
        env["USE_PRO"] = "true"
        env["WEEKLY_RUN"] = "false"  # 명시적으로 false 설정
        env["MONTHLY_RUN"] = "true"  # 명시적으로 true 설정
    else:
        raise SystemExit(f"[중단] 알 수 없는 파이프라인 타입: {args.pipeline}")
    # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

    steps_to_run = all_pipeline_steps
    if args.only:
        wanted = {s.lower() for s in args.only}
        # Filter steps from the *selected pipeline's* list
        steps_to_run = [(name, cmd) for name, cmd in all_pipeline_steps if name.lower() in wanted]
        if not steps_to_run:
            raise SystemExit(f"[중단] --only에 해당하는 단계가 {args.pipeline} 파이프라인에 없습니다.")
        log(f"Running selected steps from {args.pipeline}: {[s[0] for s in steps_to_run]}")


    # --- Check required keys (example for Gemini) ---
    # Determine if any step requires Gemini based on the selected pipeline steps
    steps_requiring_llm = {"c", "e", "future", "g_risk", "h_planning",
                           "f_daily", "f_weekly", "f_monthly", # Report generation might use LLM
                           "f_cmt_daily", "f_cmt_weekly", "f_cmt_monthly"} # Commentary definitely uses LLM
    will_run_names = {name for name, _ in steps_to_run}

    if not will_run_names.isdisjoint(steps_requiring_llm):
        log("Checking for GEMINI_API_KEY as an LLM step is included.")
        require_key(env, "GEMINI_API_KEY")
    # Add checks for NAVER keys if module_a is run, etc.

    # --- Execute Steps ---
    last_ok = None
    log(f"Starting {args.pipeline} pipeline with USE_PRO={env.get('USE_PRO')}")
    for name, cmd in steps_to_run:
        run_step(name, cmd, env)
        last_ok = name

    log(f"[성공] {args.pipeline} 파이프라인 완료. 마지막 성공 단계: {last_ok}")

    '''
    # --- Archiving Logic ---
    should_archive = False
    if args.archive:
        should_archive = True
    elif args.no_archive:
        should_archive = False
    else:
        # Default: Archive unless --no-archive is specified
        should_archive = True

    if should_archive:
        log(f"Archiving results for {args.pipeline} pipeline...")
        _archive_outputs(args.pipeline) # Use the unified archive function
    else:
        log("Skipping archiving based on --no-archive flag.")
    '''

if __name__ == "__main__":
    main()