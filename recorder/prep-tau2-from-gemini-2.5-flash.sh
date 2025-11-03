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
  --run-name-map $RUN_NAME_MAP_ARGS \
  --test-fraction 0.2 \
  --domain airline

for key in "${RUN_BASENAMES[@]}"; do
  run_name="${RUN_NAME_MAP[$key]}"
  out_dir="$AGG_OUT/$run_name"
  python "$BASE/recorder/prepare_tau2_data.py" \
    --input-dirs "$REC_BASE/$key" \
    --output-dir "$out_dir" \
    --run-name-map "$key:$run_name" \
    --split-manifest "$SPLIT_MANIFEST" \
    --domain airline
done

