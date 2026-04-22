#!/usr/bin/env python3
"""
Local MLX-based auto-research agent.

Loads an instruction-tuned model via mlx_lm and runs the Karpathy autoresearch
experiment loop locally without external coding agents.

Usage:
    uv run agent.py

The agent:
1. Loads mlx-community/gemma-4-26b-a4b-it-4bit (or another MLX model)
2. Reads program.md, train.py, and results.tsv for context
3. Uses the local LLM to propose experimental changes to train.py
4. Runs train.py, evaluates val_bpb, keeps or reverts
5. Repeats indefinitely
"""

import os
import re
import subprocess
import sys
import time
import traceback

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("AGENT_MODEL", "mlx-community/gemma-4-26b-a4b-it-4bit")
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "8192"))
TEMP = float(os.environ.get("AGENT_TEMP", "0.3"))
TRAIN_TIMEOUT = int(os.environ.get("AGENT_TRAIN_TIMEOUT", "900"))  # 15 minutes
TOP_P = float(os.environ.get("AGENT_TOP_P", "0.95"))

TRAIN_FILE = "train.py"
PROGRAM_FILE = "program.md"
RESULTS_FILE = "results.tsv"
LOG_FILE = "run.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def run_git(args, check=True):
    cmd = ["git"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return result


def get_last_kept_commit():
    """Return the hash of the most recent 'keep' commit from results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return None
    kept = [line for line in lines[1:] if line.strip() and line.split("\t")[3].strip() == "keep"]
    if not kept:
        return None
    return kept[-1].split("\t")[0].strip()


def get_best_val_bpb():
    """Return the best (lowest) val_bpb from results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        return float("inf")
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return float("inf")
    best = float("inf")
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        status = parts[3].strip()
        if status == "keep":
            try:
                val = float(parts[1].strip())
                if val < best:
                    best = val
            except ValueError:
                pass
    return best


def parse_run_log():
    """Parse run.log for val_bpb and peak_vram_mb."""
    if not os.path.exists(LOG_FILE):
        return None
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    val_match = re.search(r"^val_bpb:\s+([0-9.]+)", text, re.MULTILINE)
    mem_match = re.search(r"^peak_vram_mb:\s+([0-9.]+)", text, re.MULTILINE)
    if not val_match:
        return None
    return {
        "val_bpb": float(val_match.group(1)),
        "peak_vram_mb": float(mem_match.group(1)) if mem_match else 0.0,
        "raw": text,
    }


def extract_code(text):
    """Extract the first Python code block from LLM output."""
    # Look for ```python ... ```
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Look for ``` ... ``` without language tag
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no code blocks, assume the entire response is code
    # (but only if it looks like Python)
    if "import " in text or "def " in text or "class " in text:
        return text.strip()
    return None


def run_train():
    """Run train.py with timeout. Returns True if it completed."""
    print(f"Running training (timeout={TRAIN_TIMEOUT}s)...")
    with open(LOG_FILE, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            [sys.executable, TRAIN_FILE],
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    try:
        proc.wait(timeout=TRAIN_TIMEOUT)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Training timed out after {TRAIN_TIMEOUT}s, killing...")
        proc.kill()
        proc.wait()
        return False


def build_prompt(program_md, train_py, results_tsv, crash_context=""):
    """Build the prompt for the LLM agent."""
    best_bpb = get_best_val_bpb()
    best_str = f"{best_bpb:.6f}" if best_bpb != float("inf") else "N/A (no experiments yet)"

    prompt_parts = [
        "You are an autonomous ML researcher running experiments on Apple Silicon via MLX.",
        "Your ONLY job is to edit train.py to improve val_bpb (validation bits per byte).",
        "Lower val_bpb is better.",
        "",
        "RULES:",
        "- You can ONLY modify train.py. Do not modify prepare.py or any other file.",
        "- Each training run has a fixed 5-minute wall-clock budget.",
        "- Output the COMPLETE new train.py file inside a single ```python code block.",
        "- Do NOT output explanations outside the code block.",
        "- Do NOT abbreviate or skip sections. The output must be a fully runnable train.py.",
        "- Keep changes simple and targeted. One change per experiment.",
        "- If the previous run crashed, fix the bug and try again.",
        "",
        f"Current best val_bpb: {best_str}",
        "",
        "=== program.md (experiment protocol) ===",
        program_md,
        "",
        "=== results.tsv (experiment history) ===",
        results_tsv if results_tsv else "(no results yet)",
        "",
    ]

    if crash_context:
        prompt_parts += [
            "=== PREVIOUS RUN CRASHED ===",
            crash_context,
            "",
            "Fix the bug and output the corrected complete train.py.",
            "",
        ]

    prompt_parts += [
        "=== Current train.py ===",
        train_py,
        "",
        "Propose ONE specific experimental change to train.py that might improve val_bpb.",
        "Output the COMPLETE new train.py file inside a ```python block.",
    ]

    return "\n".join(prompt_parts)


def log_result(commit, val_bpb, memory_mb, status, description):
    """Append a result line to results.tsv."""
    memory_gb = round(memory_mb / 1024, 1)
    line = f"{commit}\t{val_bpb:.6f}\t{memory_gb}\t{status}\t{description}\n"
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"Logged: {line.strip()}")


def get_git_description():
    """Get a short description from the last commit message."""
    result = run_git(["log", "-1", "--pretty=%s"], check=False)
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)
    print("Model loaded. Starting agent loop.")
    print()

    crash_context = ""
    attempt = 0

    while True:
        attempt += 1
        print(f"\n{'='*60}")
        print(f"EXPERIMENT ATTEMPT #{attempt}")
        print(f"{'='*60}")

        # Read current state
        program_md = read_file(PROGRAM_FILE)
        train_py = read_file(TRAIN_FILE)
        results_tsv = read_file(RESULTS_FILE)

        # Build prompt
        prompt_text = build_prompt(program_md, train_py, results_tsv, crash_context)

        # Tokenize with chat template
        messages = [{"role": "user", "content": prompt_text}]
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )

        print(f"Prompt tokens: {len(prompt_tokens)}")
        print("Generating new train.py...")

        t0 = time.time()
        try:
            response = generate(
                model,
                tokenizer,
                prompt=prompt_tokens,
                max_tokens=MAX_TOKENS,
                sampler=make_sampler(temp=TEMP, top_p=TOP_P),
                verbose=False,
            )
        except Exception as exc:
            print(f"Generation failed: {exc}")
            traceback.print_exc()
            time.sleep(10)
            continue

        gen_time = time.time() - t0
        print(f"Generation took {gen_time:.1f}s")

        # Extract code
        new_train_py = extract_code(response)
        if not new_train_py:
            print("No valid code block found in response. Retrying...")
            print("Raw response preview:")
            print(response[:500])
            crash_context = "Previous generation did not produce valid Python code."
            time.sleep(5)
            continue

        # Validate that it looks like train.py
        if "def " not in new_train_py and "class " not in new_train_py:
            print("Extracted code doesn't look like train.py. Retrying...")
            crash_context = "Previous generation produced code that doesn't look like train.py."
            time.sleep(5)
            continue

        # Write new train.py
        write_file(TRAIN_FILE, new_train_py)
        print(f"Wrote new {TRAIN_FILE} ({len(new_train_py)} chars)")

        # Commit
        desc = f"agent experiment #{attempt}"
        run_git(["add", TRAIN_FILE])
        commit_result = run_git(["commit", "-m", desc], check=False)
        if commit_result.returncode != 0:
            print("Git commit failed (no changes?). Skipping this attempt.")
            crash_context = "Git commit failed — model produced identical train.py."
            time.sleep(5)
            continue
        commit_result = run_git(["rev-parse", "--short", "HEAD"], check=False)
        commit_hash = commit_result.stdout.strip()
        print(f"Committed as {commit_hash}: {desc}")

        # Run experiment
        success = run_train()
        results = parse_run_log()

        if not success or results is None:
            print("Run failed or produced no valid results.")
            # Read tail of log for crash context
            tail = ""
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                tail = "".join(lines[-50:])
            print("Last 50 lines of run.log:")
            print(tail)

            log_result(commit_hash, 0.0, 0.0, "crash", desc)
            crash_context = f"Run crashed or timed out. Last log lines:\n{tail}"

            # Attempt to revert to last known good
            last_good = get_last_kept_commit()
            if last_good:
                print(f"Reverting to last kept commit: {last_good}")
                run_git(["reset", "--hard", last_good], check=False)
            else:
                print("No previous kept commit to revert to. Continuing with current code.")
            time.sleep(5)
            continue

        # Success - evaluate keep/revert
        val_bpb = results["val_bpb"]
        peak_vram_mb = results["peak_vram_mb"]
        best_bpb = get_best_val_bpb()

        print(f"\nResults: val_bpb={val_bpb:.6f}, peak_vram_mb={peak_vram_mb:.1f}")

        if best_bpb == float("inf"):
            # First successful run = baseline
            status = "keep"
            print("First successful run - keeping as baseline.")
        elif val_bpb < best_bpb:
            status = "keep"
            print(f"IMPROVED! {best_bpb:.6f} -> {val_bpb:.6f} -- KEEPING")
        else:
            status = "discard"
            print(f"No improvement. Best={best_bpb:.6f}, This={val_bpb:.6f} -- DISCARDING")

        log_result(commit_hash, val_bpb, peak_vram_mb, status, desc)

        if status == "keep":
            run_git(["add", RESULTS_FILE])
            run_git(["commit", "--amend", "--no-edit"], check=False)
            crash_context = ""
        else:
            last_good = get_last_kept_commit()
            if last_good:
                print(f"Reverting to {last_good}")
                run_git(["reset", "--hard", last_good], check=False)
            crash_context = f"Experiment discarded. val_bpb={val_bpb:.6f} did not improve over best={best_bpb:.6f}."

        print(f"Experiment #{attempt} complete. Status: {status}")
        time.sleep(2)


if __name__ == "__main__":
    main()
