### Script to prepare retail data for cookbook-internal training
# Only includes gpt-5, gemini-2.5-pro, and claude-sonnet (skipping gpt-mini and open models)

# --- env ---
export BASE="/home/yi/home/tau2-bench"
export REC_BASE="/home/yi/home/tau2-bench/recordings/retail-gemini-2.5-flash-user"
export OUT_BASE="/mnt/datasets/tau2-bench/recordings/retail"
export AGG_OUT="$OUT_BASE/retail-user-gemini-2.5-flash-til20pct"
export SPLIT_MANIFEST="$AGG_OUT/split_manifest.json"

# Input directories (basenames)
# Note: Only including the 3 models that were actually run
export RUN_BASENAMES=(
  run_20251024-203554_retail_claude-sonnet-4-5-20250929_temp1.0_tr4
  run_20251024-203543_retail_gemini-2.5-pro_temp1.0_tr4
  run_20251024-203536_retail_gpt-5_temp1.0_tr4
  # Commented out: models that were not run for retail
  # run_TIMESTAMP_retail_gpt-5-mini_temp1.0_tr4
  # run_TIMESTAMP_retail_qwen3-235b-a22b_temp1.0_tr4
  # run_TIMESTAMP_retail_kimi-k2-instruct-0905_temp1.0_tr4
  # run_TIMESTAMP_retail_deepseek-v3p1-terminus_temp1.0_tr4
  # run_TIMESTAMP_retail_glm-4p5_temp1.0_tr4
)

# Readable run names (map basename -> short name)
declare -A RUN_NAME_MAP=(
  [run_20251024-203554_retail_claude-sonnet-4-5-20250929_temp1.0_tr4]=claude-sonnet-4.5
  [run_20251024-203543_retail_gemini-2.5-pro_temp1.0_tr4]=gemini-2.5-pro
  [run_20251024-203536_retail_gpt-5_temp1.0_tr4]=gpt-5
  # Commented out: models that were not run for retail
  # [run_TIMESTAMP_retail_gpt-5-mini_temp1.0_tr4]=gpt-5-mini
  # [run_TIMESTAMP_retail_qwen3-235b-a22b_temp1.0_tr4]=qwen3-235b-a22b
  # [run_TIMESTAMP_retail_kimi-k2-instruct-0905_temp1.0_tr4]=kimi-k2-0905
  # [run_TIMESTAMP_retail_deepseek-v3p1-terminus_temp1.0_tr4]=deepseek-v3.1-terminus
  # [run_TIMESTAMP_retail_glm-4p5_temp1.0_tr4]=glm-4.5
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
  --domain retail \
  --run-name-map $RUN_NAME_MAP_ARGS \
  --test-fraction 0.2

for key in "${RUN_BASENAMES[@]}"; do
  run_name="${RUN_NAME_MAP[$key]}"
  out_dir="$AGG_OUT/$run_name"
  python "$BASE/recorder/prepare_tau2_data.py" \
    --input-dirs "$REC_BASE/$key" \
    --output-dir "$out_dir" \
    --domain retail \
    --run-name-map "$key:$run_name" \
    --split-manifest "$SPLIT_MANIFEST"
done

