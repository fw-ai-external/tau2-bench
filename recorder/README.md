# Tau2 Recording and Training Workflow

This directory contains tools for recording, analyzing, and preparing conversation traces from τ²-bench simulations for supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT).

## Study Documentation

For documentation on the text-based distillation study conducted using this infrastructure:
- **[Executive Summary](./executive_summary.md)** - Business-focused summary of findings and conclusions
- **[Technical Methods](./technical_methods.md)** - Detailed audit-ready documentation for reproducibility

## Overview: 5-Step Workflow

```
1. Generate       → 2. Record        → 3. Analyze       → 4. Export        → 5. Train
   execution          traces via          results         to RLOR           with RLOR
   scripts           tmux             with dialogs        format
   (parallel)        (parallel)       & logs
```

---

## Step 1: Generate Execution Scripts

Use `generate_run_scripts.py` to create tmux-compatible shell scripts for multiple models running in parallel.

### Usage

```bash
python recorder/generate_run_scripts.py
```

### Configuration

Edit the script to specify:
- **MODELS**: List of models to test (e.g., gpt-5, gemini-2.5-pro, claude-sonnet)
- **DOMAIN**: Domain to evaluate (e.g., airline)
- **NUM_TRIALS**: Trials per task (default: 4)
- **TEMPERATURE**: Agent temperature (default: 1.0)
- **USER_MODEL**: User simulator model (default: gemini-2.5-pro)
- **MAX_WORKERS**: Parallel task workers per model (default: 3)
- **BUDGET**: Token budget per call (default: 16384)

### Output

Generates shell scripts in `recorder/run_scripts/`:
- `run_<model_name>.sh` for each model
- `run_all_tmux.sh` to launch all in parallel

### Example

```bash
# View what will be generated
cat recorder/generate_run_scripts.py | head -30

# Generate scripts
python recorder/generate_run_scripts.py

# Launch all models in parallel tmux sessions
bash recorder/run_scripts/run_all_tmux.sh

# Or launch a single model
bash recorder/run_scripts/run_gpt-5.sh

# Monitor progress
tmux list-sessions
tmux attach-session -t tau2-gen-gpt-5
```

---

## Step 2: Record Traces

The `run_and_record.py` script records full conversation traces and LLM payloads during simulations.

### Key Features

- **Full transparency**: Captures every LLM call (agent + user simulator) in `tau2_payloads.jsonl`
- **Structured dialogs**: Normalized conversations in OpenAI format with evaluation metrics
- **Parallel execution**: ThreadPoolExecutor for concurrent task evaluation
- **Resilience**: Individual failures don't crash the run; errors are logged separately
- **Separate user model**: Reliable user simulator (e.g., gpt-4.1 @ T=0.0) ensures protocol compliance
- **Rich metrics**: Captures DB success, communication rates, reward breakdowns, termination reasons

### Output Files (per model/run)

```
recordings/run_<timestamp>_<domain>_<model>_<config>/
├── tau2_dialogs.jsonl        # Primary output: dialogs + metrics (one JSON per line)
├── tau2_payloads.jsonl       # Raw LLM API calls (for debugging/transparency)
├── run_manifest.json         # Metadata: model, temperature, seeds, commit hash
├── errors.log                # Warnings and infrastructure failures
└── litellm.yaml              # LiteLLM configuration snapshot
```

### Example Dialog Record (tau2_dialogs.jsonl)

```json
{
  "session_id": "20251021-002721_airline_gpt-5_temp1.0_tr4:airline:0:0",
  "domain": "airline",
  "task_id": "0",
  "messages": [
    {"role": "assistant", "content": "Hi! How can I help you today?", "tool_calls": null},
    {"role": "user", "content": "I want to cancel my reservation...", ...},
    ...
  ],
  "metrics": {
    "success": true,
    "score": 1.0,
    "db_success": true,
    "num_steps": 17,
    "termination_reason": "user_stop",
    "reward_breakdown": {"DB": 1.0, "COMMUNICATE": 1.0},
    "infra_failure": false
  }
}
```

### Configuration Parameters

- `--domains`: Comma-separated domain names (required)
- `--num-trials`: Trials per task (default: 1)
- `--model`: Agent LLM (default: gpt-4.1)
- `--temperature`: Agent temperature (default: 0.2)
- `--user-model`: User simulator LLM (default: gpt-4.1)
- `--user-temperature`: User simulator temperature (default: 0.0)
- `--reasoning-effort-agent`: Reasoning effort (low/medium/high)
- `--budget-agent`: Token budget for agent (default: None)
- `--max-workers`: Parallel workers (default: 6)
- `--llm-retries`: Retries per LLM call (default: 0)
- `--infra-retries`: Retries for infrastructure failures (default: 2)
- `--seed`: Random seed for reproducibility
- `--outdir`: Output directory (default: ./recordings)
- `--run-id`: Custom run identifier
- `--debug`: Enable debug mode

---

## Step 3: Analyze Results

### 3a. Quick Summary with analyze_dialogs.py

Get high-level statistics on your runs:

```bash
# Single run summary
python -m recorder.analyze_dialogs \
  recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4/tau2_dialogs.jsonl

# Multiple runs (e.g., all models)
for dialogs in recordings/run_*/tau2_dialogs.jsonl; do
  echo "=== $dialogs ==="
  python -m recorder.analyze_dialogs "$dialogs" | head -30
done

# Export to CSV for Excel/R analysis
python -m recorder.analyze_dialogs \
  recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4/tau2_dialogs.jsonl \
  --output metrics.csv

# Machine-readable JSON output
python -m recorder.analyze_dialogs \
  recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4/tau2_dialogs.jsonl \
  --json > metrics.json
```

**Statistics Provided:**
- Overall success rate, score distribution
- Per-task and per-trial performance
- Termination reasons (user_stop, transfer, max_steps, etc.)
- DB success rates, communication rates
- Worst-performing tasks for focused analysis
- Task success histograms (e.g., "20 tasks had 4/4 successes")

### 3b. Examine Logs for Issues

```bash
# Check for infrastructure/API errors
grep "Failed to run" recordings/run_*/errors.log | wc -l

# View specific warnings
grep "WARNING" recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4/errors.log | head -20

# Check for rate limits or specific errors
grep "list index out of range\|rate_limit\|timeout" recordings/run_*/errors.log

# Count unique error types
grep "Failed to extract\|Error" recordings/run_*/errors.log | cut -d: -f3- | sort | uniq -c
```

### 3c. Deep Dive with Python

See `EVALUATION_SYSTEM.md` for understanding:
- DB evaluation (strict function call matching)
- Communication scoring (substring matching in agent messages)
- NL assertions (LLM judge evaluations)
- Failure modes and what causes them

---

## Step 4: Export for Training

Use `prepare_tau2_data.py` to combine multiple runs and create RLOR-compatible training/test splits.

### Usage

```bash
python /home/yi/home/tau2-bench/recorder/prepare_tau2_data.py \
  --input-dirs \
    /home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user/run_20251021-184516_airline_claude-sonnet-4-5-20250929_temp1.0_tr4 \
    /home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user/run_20251021-184516_airline_deepseek-v3p1-terminus_temp1.0_tr4 \
    /home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user/run_20251021-184516_airline_gemini-2.5-pro_temp1.0_tr4 \
    /home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user/run_20251021-184516_airline_glm-4p5_temp1.0_tr4 \
    /home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user/run_20251021-184516_airline_gpt-5_temp1.0_tr4 \
    /home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user/run_20251021-184516_airline_gpt-5-mini_temp1.0_tr4 \
    /home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user/run_20251021-184516_airline_kimi-k2-instruct-0905_temp1.0_tr4 \
    /home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user/run_20251021-184516_airline_qwen3-235b-a22b_temp1.0_tr4 \
  --output-dir /mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-flash-til20pct \
  --domain airline \
  --test-fraction 0.2
```

Alternatively, if you want to break it down:
```bash 
# --- env ---
export BASE="/home/yi/home/tau2-bench"
export REC_BASE="/home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user"
export OUT_BASE="/mnt/datasets/tau2-bench/recordings/airline"
export AGG_OUT="$OUT_BASE/airline-user-gemini-2.5-flash-til20pct"
export SPLIT_MANIFEST="$AGG_OUT/split_manifest.json"

# Input directories (basenames)
export RUN_BASENAMES=(
  run_20251021-184516_airline_claude-sonnet-4-5-20250929_temp1.0_tr4
  run_20251021-184516_airline_deepseek-v3p1-terminus_temp1.0_tr4
  run_20251021-184516_airline_gemini-2.5-pro_temp1.0_tr4
  run_20251021-184516_airline_glm-4p5_temp1.0_tr4
  run_20251021-184516_airline_gpt-5_temp1.0_tr4
  run_20251021-184516_airline_gpt-5-mini_temp1.0_tr4
  run_20251021-184516_airline_kimi-k2-instruct-0905_temp1.0_tr4
  run_20251021-184516_airline_qwen3-235b-a22b_temp1.0_tr4
)

# Readable run names (map basename -> short name)
declare -A RUN_NAME_MAP=(
  [run_20251021-184516_airline_claude-sonnet-4-5-20250929_temp1.0_tr4]=claude-sonnet-4.5
  [run_20251021-184516_airline_deepseek-v3p1-terminus_temp1.0_tr4]=deepseek-v3.1-terminus
  [run_20251021-184516_airline_gemini-2.5-pro_temp1.0_tr4]=gemini-2.5-pro
  [run_20251021-184516_airline_glm-4p5_temp1.0_tr4]=glm-4.5
  [run_20251021-184516_airline_gpt-5_temp1.0_tr4]=gpt-5
  [run_20251021-184516_airline_gpt-5-mini_temp1.0_tr4]=gpt-5-mini
  [run_20251021-184516_airline_kimi-k2-instruct-0905_temp1.0_tr4]=kimi-k2-0905
  [run_20251021-184516_airline_qwen3-235b-a22b_temp1.0_tr4]=qwen3-235b-a22b
)

# Build args
RUN_DIRS=$(printf " %s" "${RUN_BASENAMES[@]/#/$REC_BASE/}")
RUN_NAME_MAP_ARGS=""
for key in "${!RUN_NAME_MAP[@]}"; do
  RUN_NAME_MAP_ARGS+=" $key:${RUN_NAME_MAP[$key]}"
done

python "$BASE/recorder/prepare_tau2_data.py" \
  --input-dirs $RUN_DIRS \
  --output-dir "$AGG_OUT" \
  --domain airline \
  --run-name-map $RUN_NAME_MAP_ARGS \
  --test-fraction 0.2

for key in "${RUN_BASENAMES[@]}"; do
  run_name="${RUN_NAME_MAP[$key]}"
  out_dir="$AGG_OUT/$run_name"
  python "$BASE/recorder/prepare_tau2_data.py" \
    --input-dirs "$REC_BASE/$key" \
    --output-dir "$out_dir" \
    --domain airline \
    --run-name-map "$key:$run_name" \
    --split-manifest "$SPLIT_MANIFEST"
done

```

### What It Does

1. **Combines multiple runs**: Merges dialogs from all input directories
2. **Cleans data**: Removes infrastructure failures and scoreless records
3. **Creates train/test split**: Splits by **task IDs** (not individual trials)
   - All trials of a task go together (train or test)
   - Ensures no data leakage between splits
4. **Remaps run names**: Maps long directory names to clean identifiers
5. **Generates split manifest**: `split_manifest.json` defines the split (reusable for future runs)
6. **Sets group_id field**: Each record gets a `group_id` = `{run_name}:{domain}:{task_id}`
   - Used by RLOR for grouping successes/failures in RFT (GRPO)

### Output

```
/mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-pro-til20pct/
├── train.jsonl              # 80% of tasks (all trials, all models)
├── test.jsonl               # 20% of tasks (all trials, all models)
├── agent_airline_tools.json # OpenAI-format agent tool schemas
└── split_manifest.json      # Defines train/test task split
```

Note: The `agent_{domain}_tools.json` file contains the tool schemas in OpenAI function calling format. Additionally, each record in the train/test JSONL files includes a `tools` field with the same tool schemas.

### Statistics (Example)

```
Total records read: 800 (4 models × 50 tasks × 4 trials)
Records processed: 800 (100% - no failures)
Train split: 640 records (40 tasks × 4 trials × 4 models)
Test split:  160 records (10 tasks × 4 trials × 4 models)
```

### Reusing Splits

To apply the same split to new model runs:

```bash
python recorder/prepare_tau2_data.py \
  --input-dirs recordings/run_new_model_* \
  --output-dir /mnt/datasets/tau2-bench/recordings/airline/airline-user-newmodel-til20pct \
  --domain airline \
  --split-manifest /mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-pro-til20pct/split_manifest.json
```

### Validating Tools in Output Data

Use the `check_tools_in_jsonl.py` script to validate that tools are properly embedded in your data:

```bash
# Check a single file
python recorder/check_tools_in_jsonl.py /mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-flash-til20pct/gpt-5/train.jsonl

# Check multiple files with summary
python recorder/check_tools_in_jsonl.py /mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-flash-til20pct/gpt-5/*.jsonl --summary

# Check all model subdirectories
python recorder/check_tools_in_jsonl.py /mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-flash-til20pct/*/train.jsonl --summary
```

The script validates:
- All records have the `tools` field
- Tools are in the correct format (list of OpenAI function schemas)
- Tool names are extracted and counted
- Reports any JSON parsing errors or format issues

---

## Step 5: Train Models (Next Steps)

Once you have prepared data, use the RLOR training workflow to fine-tune models.

### Reference

- **Main entry point**: `cookbook-internal/recipes/workflow/rlor/main.py`
- **Example config**: `cookbook-internal/recipes/workflow/rlor/conf/tau2_airline_multiturn.yaml`

### Typical Workflow

```yaml
# In tau2_airline_multiturn.yaml, reference your prepared data:
load_train_data:
  kwargs:
    input_files:
      - /mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-pro-til20pct/train.jsonl

load_test_data:
  kwargs:
    input_files:
      - /mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-pro-til20pct/test.jsonl
```

Then run:
```bash
python recipes/workflow/rlor/main.py --config-name tau2_airline_multiturn
```

This will:
1. Load your prepared training data
2. Filter successful examples for SFT (supervised fine-tuning)
3. Optionally prepare preference pairs for RFT (reinforcement fine-tuning)
4. Fine-tune the base model
5. Evaluate checkpoints

**Note**: RFT is supported in the workflow but was not part of the documented distillation study. The study focused on SFT with and without rejection sampling (filtering for successful trials only).

---

## Troubleshooting

### High Failure Rate During Recording

**Symptoms:** Many tasks fail with `list index out of range` or rate limit errors

**Solutions:**
1. **Reduce parallelism** to avoid rate limiting:
   ```bash
   --max-workers 2  # or even 1 for very strict limits
   ```

2. **Add retries** for transient failures:
   ```bash
   --llm-retries 3 --infra-retries 5
   ```

3. **Increase token budgets** if hitting limits:
   ```bash
   --budget-agent 32000 --budget-user 32000
   ```

### Analyzing Partial Failures

You can retry infrastructure failures separately without re-running successes:

```bash
python -m recorder.retry_failures \
  recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4/ \
  --max-retries 3 \
  --max-workers 1
```

### Data Quality Issues

**Inspect problematic records:**
```python
import json

with open("train.jsonl") as f:
    for line in f:
        record = json.loads(line)
        if record["metrics"]["score"] == 0.0:
            print(f"Task {record['task_id']}: Failed")
            print(f"  Reason: {record['metrics'].get('reward_basis')}")
```

**Filter by success for rejection sampling (SFT):**
```python
# Rejection sampling: keep only successful trials with reward 1.0
sft_success_data = [
    d for d in all_records
    if not d["metrics"].get("infra_failure", False)
    and d["metrics"]["score"] == 1.0
]
```

---

## Key Concepts

### Infrastructure Failures vs. Model Failures

Each record includes `metrics.infra_failure`:
- **false**: Task completed (success or legitimate failure)
- **true**: Infrastructure error (API failure, rate limit, timeout)

**Always exclude infrastructure failures from training data** — they don't represent model behavior.

### Reward Structure

See `EVALUATION_SYSTEM.md` for details, but briefly:

- **DB Score**: Deterministic check of database state after agent actions
- **Communication Score**: Semantic check if agent communicated required info
- **NL Assertions**: LLM judge evaluation of complex policies
- **Final Score**: Product of applicable components (both must pass to get 1.0)

### Training Approaches

- **SFT (Supervised Fine-Tuning)**: Train on demonstration data
  ```python
  # All demonstrations (exclude only infrastructure failures)
  sft_data = [d for d in dialogs if not d["metrics"].get("infra_failure")]
  ```

- **SFT with Rejection Sampling**: Train only on successful demonstrations
  ```python
  # Rejection sampling: filter for reward 1.0
  sft_success_data = [d for d in dialogs if d["metrics"]["score"] == 1.0]
  ```

- **RFT (Reinforcement Fine-Tuning)**: Use GRPO to learn from successes and failures
  ```python
  # Uses the same sft_data (all demonstrations with both successes and failures)
  # GRPO grouping and preference learning happens inside RLOR workflow
  # Supported in workflow but not used in the documented study
  rft_data = [d for d in dialogs if not d["metrics"].get("infra_failure")]
  ```
  
  **Note on RFT grouping:** RLOR uses the `group_id` field (set in `prepare_tau2_data.py`) to group successes and failures together for preference learning. The `group_id` is constructed as `{run_name}:{domain}:{task_id}`, meaning all trials of the same task from the same recording run are grouped together. GRPO requires that each group contains both successes and failures.

---

## Files in This Directory

### Entry Points

| File | Purpose |
|------|---------|
| `run_and_record.py` | Main script to record traces via tau2 simulations |
| `generate_run_scripts.py` | Generate tmux execution scripts for parallel model evaluation |

### Core Infrastructure

| File | Purpose |
|------|---------|
| `llm_recorder.py` | LiteLLM monkeypatching infrastructure for payload capture |
| `prepare_tau2_data.py` | Prepare data for RLOR training (train/test split, export tools) |
| `common_args.py` | Shared argument definitions across scripts |

### Diagnostic and Verification Tools

| File | Purpose |
|------|---------|
| `check_tools_in_jsonl.py` | Validate that tools field is properly embedded in JSONL files |
| `analyze_dialogs.py` | Compute summary statistics and performance metrics from recorded traces |

### Study-Specific Scripts and Documentation

| File | Purpose |
|------|---------|
| `generate_tau2_figures.py` | Generate visualizations from aggregated evaluation results for the distillation study |
| `prep-tau2-from-gemini-2.5-flash.sh` | Airline domain data preparation script used in the study |
| `prep-tau2-from-gemini-2.5-flash-retail.sh` | Retail domain data preparation script used in the study |
| `executive_summary.md` | Business-focused summary of distillation study findings |
| `technical_methods.md` | Detailed audit-ready documentation for study reproducibility |

### General Documentation

| File | Purpose |
|------|---------|
| `README.md` | This file - workflow documentation and infrastructure guide |
| `EVALUATION_SYSTEM.md` | Deep dive into τ²-bench evaluation mechanics |

---

**Updated**: October 30, 2025  
**Workflow Version**: 5-step (Generate → Record → Analyze → Export → Train)

