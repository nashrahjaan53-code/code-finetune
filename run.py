import argparse
import subprocess
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def banner():
    print(f"""
{BLUE}{BOLD}
╔══════════════════════════════════════════════════╗
║   Code Generation Assistant -- QLoRA Pipeline    ║
║      Mistral-7B x CodeAlpaca-20k                 ║
╚══════════════════════════════════════════════════╝
{RESET}""")

def run_step(name, script_path):
    print(f"\n{YELLOW}{BOLD}>> Starting: {name}{RESET}")
    print(f"{BLUE}{'─' * 50}{RESET}")

    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        env=env,
        check=False
    )
    elapsed = round(time.time() - start, 1)

    if result.returncode == 0:
        print(f"\n{GREEN}{BOLD}[DONE] {name} completed in {elapsed}s{RESET}")
        return True
    else:
        print(f"\n{RED}{BOLD}[FAILED] {name} failed! Check errors above.{RESET}")
        return False

def run_all():
    steps = [
        ("Data Preparation", "src/prepare_data.py"),
        ("Model Training",   "src/train.py"),
        ("Evaluation",       "src/evaluate.py"),
        ("Inference Chat",   "src/inference.py"),
    ]

    banner()
    print(f"{BOLD}Running all {len(steps)} steps in sequence...{RESET}\n")

    results = {}
    for name, script in steps:
        success = run_step(name, script)
        results[name] = success

        # stop pipeline if a critical step fails
        if not success and name != "Evaluation":
            print(f"\n{RED}Pipeline stopped due to failure in: {name}{RESET}")
            break

    print(f"\n{BLUE}{BOLD}{'═' * 50}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'═' * 50}{RESET}")
    for name, success in results.items():
        status = f"{GREEN}[PASS]" if success else f"{RED}[FAIL]"
        print(f"  {status}  {name}{RESET}")
    print(f"{BLUE}{'═' * 50}{RESET}\n")

def run_single(step):
    banner()
    steps = {
        "data"      : ("Data Preparation", "src/prepare_data.py"),
        "train"     : ("Model Training",   "src/train.py"),
        "evaluate"  : ("Evaluation",       "src/evaluate.py"),
        "inference" : ("Inference Chat",   "src/inference.py"),
    }

    if step not in steps:
        print(f"{RED}❌ Unknown step '{step}'. Choose from: data, train, evaluate, inference{RESET}")
        sys.exit(1)

    name, script = steps[step]
    run_step(name, script)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Pipeline Runner")
    parser.add_argument(
        "--step",
        type=str,
        choices=["data", "train", "evaluate", "inference"],
        help="Run a specific step only (default: run all)"
    )
    args = parser.parse_args()

    if args.step:
        run_single(args.step)
    else:
        run_all()