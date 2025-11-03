# Technical Methods: Text-Based Distillation on Tau2-Bench

**Related Documentation:**
- [Executive Summary](./executive_summary.md) - Business-focused summary of findings and conclusions
- [Recording Infrastructure README](./README.md) - Tools and workflows for data collection

---

## Overview

This study evaluated whether supervised fine-tuning (SFT) on visible text tokens from frontier models improves small student model performance on tau2-bench multi-turn tool-calling tasks. Two experiments were conducted: single-teacher (GPT-5) distillation and multi-teacher ensemble (GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro) distillation, each with and without rejection sampling.

## Code Repositories

**Tau2-bench with recording infrastructure:**
- Repository: https://github.com/fw-ai-external/tau2-bench
- Branch: user/yi/recorder
- Local path: `/home/yi/home/tau2-bench`
- Recording tools: `recorder/` subdirectory

**Training workflows:**
- Repository: https://github.com/fw-ai/cookbook-internal
- Branch: yi/tau2-distill
- Local path: `/home/yi/home/cookbook-internal`
- Workflow configs: `recipes/workflow/rlor/conf/`

**Host:** aws-dev2 (34.239.129.42)

## Models

**Base student model:**
- Qwen3-30B-A3B-Instruct-2507
- Path: `/shared/text-models/Qwen3-30B-A3B-Instruct-2507`

**Teacher models:**
- GPT-5 (OpenAI o1)
- Claude Sonnet 4.5 (Anthropic)
- Gemini 2.5 Pro (Google)

**User simulator:**
- Gemini 2.5 Flash
- Temperature: 0.0

## Data Collection

### Recording Infrastructure

Teacher demonstrations were recorded using:
- Script: `recorder/run_and_record.py`
- Documentation: `recorder/README.md`

**Recording parameters:**
- Agent temperature: 1.0
- Trials per task: 4
- Max workers: Varies by script
- User model: Gemini 2.5 Flash (temperature 0.0)

**Output format:**
- JSONL files with OpenAI-format conversations
- Tool specifications included in each record
- Metrics: `exact_match`, `task_id`, `domain`

### Raw Recording Locations

**Airline domain:**
- `/home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user/`

**Retail domain:**
- `/home/yi/home/tau2-bench/recordings/retail-gemini-2.5-flash-user/`

## Data Preparation

### Airline Domain

**Script:** `recorder/prep-tau2-from-gemini-2.5-flash.sh`

**Teachers recorded (8 total in airline):**
- claude-sonnet-4.5
- deepseek-v3.1-terminus
- gemini-2.5-pro
- glm-4.5
- gpt-5
- gpt-5-mini
- kimi-k2-0905
- qwen3-235b-a22b

**Teachers used in study:** gpt-5, claude-sonnet-4.5, gemini-2.5-pro

**Commands executed:**
```bash
export BASE="/home/yi/home/tau2-bench"
export REC_BASE="/home/yi/home/tau2-bench/recordings/airline-gemini-2.5-flash-user"
export OUT_BASE="/mnt/datasets/tau2-bench/recordings/airline"
export AGG_OUT="$OUT_BASE/airline-user-gemini-2.5-flash-til20pct"

python "$BASE/recorder/prepare_tau2_data.py" \
  --input-dirs $RUN_DIRS \
  --output-dir "$AGG_OUT" \
  --run-name-map $RUN_NAME_MAP_ARGS \
  --test-fraction 0.2 \
  --domain airline
```

**Output:**
- Train/test split (80/20 by task ID)
- Per-teacher subdirectories: `gpt-5/`, `claude-sonnet-4.5/`, `gemini-2.5-pro/`
- Files: `train.jsonl`, `test.jsonl`, `agent_airline_tools.json`, `split_manifest.json`

**Tool specifications:**
- Saved separately as `agent_airline_tools.json` (OpenAI function calling format)
- Also included in each JSONL record under the `tools` field for training convenience

### Retail Domain

**Script:** `recorder/prep-tau2-from-gemini-2.5-flash-retail.sh`

**Teachers recorded and used in study (3 total):**
- claude-sonnet-4.5
- gemini-2.5-pro
- gpt-5

**Commands executed:**
```bash
export BASE="/home/yi/home/tau2-bench"
export REC_BASE="/home/yi/home/tau2-bench/recordings/retail-gemini-2.5-flash-user"
export OUT_BASE="/mnt/datasets/tau2-bench/recordings/retail"
export AGG_OUT="$OUT_BASE/retail-user-gemini-2.5-flash-til20pct"

python "$BASE/recorder/prepare_tau2_data.py" \
  --input-dirs $RUN_DIRS \
  --output-dir "$AGG_OUT" \
  --domain retail \
  --run-name-map $RUN_NAME_MAP_ARGS \
  --test-fraction 0.2
```

**Output:**
- Train/test split (80/20 by task ID)
- Per-teacher subdirectories: `gpt-5/`, `claude-sonnet-4.5/`, `gemini-2.5-pro/`
- Files: `train.jsonl`, `test.jsonl`, `agent_retail_tools.json`, `split_manifest.json`

**Tool specifications:**
- Saved separately as `agent_retail_tools.json` (OpenAI function calling format)
- Also included in each JSONL record under the `tools` field for training convenience

### Prepared Data Locations

**Airline:**
- `/mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-flash-til20pct/`

**Retail:**
- `/mnt/datasets/tau2-bench/recordings/retail/retail-user-gemini-2.5-flash-til20pct/`

## Training Configuration

Training workflows executed via RLOR framework:
- Entry point: `cookbook-internal/recipes/workflow/rlor/main.py`
- Base config: `conf/base.yaml`

### Experiment 1: Single-Teacher (GPT-5) Distillation

**Configuration file:** `cookbook-internal/recipes/workflow/rlor/conf/tau2_airline_retail.yaml`

**Data sources:**
- Airline: `/mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-flash-til20pct/gpt-5/train.jsonl`
- Retail: `/mnt/datasets/tau2-bench/recordings/retail/retail-user-gemini-2.5-flash-til20pct/gpt-5/train.jsonl`

**Training variants:**
1. `sft_finetune_airline_gpt5`: All GPT-5 demonstrations
2. `sft_finetune_airline_gpt5_success`: Filtered for `exact_match == 1.0`
3. `sft_finetune_retail_gpt5`: All GPT-5 demonstrations
4. `sft_finetune_retail_gpt5_success`: Filtered for `exact_match == 1.0`

**Hyperparameters:**
- Precision: bf16
- Max context length: 32,000 tokens
- Batch size: 128,000 tokens
- Learning rate: 3.0e-5
- GPU ranks: [2, 3]

**Execution:**
```bash
python recipes/workflow/rlor/main.py --config-name tau2_airline_retail
```

**Output directory:**
- `/mnt/text/workflow/tau2_airline_retail/qwen3-30b-a3b-instruct-2507-base-gemini-2.5-flash-user-airline-retail-yi/`

### Experiment 2: Multi-Teacher Ensemble Distillation

**Configuration file:** `cookbook-internal/recipes/workflow/rlor/conf/tau2_ensemble_airline_retail.yaml`

**Data sources (per domain):**
- GPT-5: `gpt-5/train.jsonl`
- Claude Sonnet 4.5: `claude-sonnet-4.5/train.jsonl`
- Gemini 2.5 Pro: `gemini-2.5-pro/train.jsonl`

**Training variants:**
1. `sft_finetune_airline_ensemble`: Combined teacher demonstrations
2. `sft_finetune_airline_ensemble_success`: Filtered for `exact_match == 1.0`
3. `sft_finetune_retail_ensemble`: Combined teacher demonstrations
4. `sft_finetune_retail_ensemble_success`: Filtered for `exact_match == 1.0`

**Hyperparameters:**
- Same as Experiment 1

**Execution:**
```bash
python recipes/workflow/rlor/main.py --config-name tau2_ensemble_airline_retail
```

**Output directory:**
- `/mnt/text/workflow/tau2_ensemble_airline_retail/qwen3-30b-a3b-instruct-2507-base-gemini-2.5-flash-user-airline-retail-ensemble-yi/`

## Evaluation Protocol

**Framework:** Tau2-bench rollout evaluation

**Test sets:**
- Airline: Tasks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (10 tasks)
- Retail: Tasks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] (22 tasks)

**Evaluation parameters:**
- Agent temperature: 1.0
- User temperature: 0.0
- Trials per task: 3
- Conversation style: qwen3
- User LLM: gemini-2.5-flash
- GPU ranks: [2, 3]
- Reasoning effort: Medium (GPT-5, Gemini 2.5 Pro), Off (Claude Sonnet 4.5)

**Models evaluated per experiment:**
- Base (no fine-tuning)
- SFT (all data)
- SFT + rejection sampling (success only)

**Evaluation working directory:**
- `/home/yi/home/tau2-bench/`

## Results Processing

**Pipeline steps:**
1. Rollout evaluation produces `simulation_results.json`
2. Parser creates JSON, CSV, Markdown reports
3. Aggregator combines results across models into `aggregated_results.json`

**Aggregated results locations:**

**Experiment 1:**
- `/mnt/text/workflow/tau2_airline_retail/qwen3-30b-a3b-instruct-2507-base-gemini-2.5-flash-user-airline-retail-yi/0021_aggregate_evaluation_results/aggregated_results.json`

**Experiment 2:**
- `/mnt/text/workflow/tau2_ensemble_airline_retail/qwen3-30b-a3b-instruct-2507-base-gemini-2.5-flash-user-airline-retail-ensemble-yi/0027_aggregate_evaluation_results/aggregated_results.json`

## Visualization

**Script:** `/home/yi/home/tau2-bench/recorder/generate_tau2_figures.py`

**Inputs:**
- Aggregated results JSON files from both experiments
- Teacher baseline results from prepared data directories

**Outputs:**
- `figures/gpt5_distillation.png`: Single-teacher experiment visualizations
- `figures/ensemble_distillation.png`: Multi-teacher experiment visualizations

**Figure types (per experiment):**
- Bar charts: Overall success rates by domain (airline, retail)
- Heatmaps: Per-task success rates by model

**Execution:**
```bash
python /home/yi/home/tau2-bench/recorder/generate_tau2_figures.py
```

## Reproducibility Checklist

### Software Versions
- Base model: Qwen3-30B-A3B-Instruct-2507
- Python environment: See repository requirements
- LiteLLM: For API calls to teacher models

### File Paths

**Code:**
- Tau2-bench: https://github.com/fw-ai-external/tau2-bench (branch: user/yi/recorder)
- Training workflows: https://github.com/fw-ai/cookbook-internal (branch: yi/tau2-distill)
- Recording tools: `/home/yi/home/tau2-bench/recorder/`
- Training configs: `/home/yi/home/cookbook-internal/recipes/workflow/rlor/conf/`

**Data:**
- Raw recordings: `/home/yi/home/tau2-bench/recordings/`
- Prepared data: `/mnt/datasets/tau2-bench/recordings/`
- Training outputs: `/mnt/text/workflow/`

**Analysis:**
- Visualization script: `/home/yi/home/tau2-bench/recorder/generate_tau2_figures.py`
- Report outputs: `/home/yi/home/notes/tau2_scripts/`

### Key Scripts
1. Data recording: `recorder/run_and_record.py`
2. Data preparation: `recorder/prepare_tau2_data.py`
3. Airline prep: `recorder/prep-tau2-from-gemini-2.5-flash.sh`
4. Retail prep: `recorder/prep-tau2-from-gemini-2.5-flash-retail.sh`
5. Training: `recipes/workflow/rlor/main.py`
6. Visualization: `recorder/generate_tau2_figures.py`

### Configuration Files
1. Single-teacher: `tau2_airline_retail.yaml`
2. Multi-teacher: `tau2_ensemble_airline_retail.yaml`
3. Base config: `base.yaml`

### Model Naming Convention

Models follow pattern: `qwen3-30b-a3b-instruct-2507-base-gemini-2.5-flash-user-{domain}-{variant}-{user}`

**Variants:**
- `base`: No fine-tuning
- `sft-airline-gpt5` / `sft-retail-gpt5`: Single-teacher SFT
- `sft-airline-gpt5-success` / `sft-retail-gpt5-success`: Single-teacher SFT with rejection sampling
- `sft-airline-ensemble` / `sft-retail-ensemble`: Multi-teacher SFT
- `sft-airline-ensemble-success` / `sft-retail-ensemble-success`: Multi-teacher SFT with rejection sampling

### Evaluation Metrics

**Per trial:**
- Status: PASSED or FAILED
- Based on exact database state match and communication requirements

**Aggregation:**
- Success rate: Proportion of trials with status PASSED
- Reported per model and domain

