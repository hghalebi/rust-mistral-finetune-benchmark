#!/usr/bin/env bash
set -euo pipefail

MODEL=""
PROMPT=""
MAX_NEW_TOKENS="256"
TEMPERATURE="0.2"
SEED="42"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      # tolerate and ignore unused args
      shift
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "missing --model" >&2
  exit 2
fi

if [[ -z "$PROMPT" ]]; then
  echo "missing --prompt" >&2
  exit 2
fi

REQUEST=$(jq -cn \
  --arg model "$MODEL" \
  --arg prompt "$PROMPT" \
  --argjson max_new_tokens "$MAX_NEW_TOKENS" \
  --argjson temperature "$TEMPERATURE" \
  --argjson seed "$SEED" \
  '{model:$model,prompt:$prompt,stream:false,options:{num_predict:$max_new_tokens,temperature:$temperature,seed:$seed}}')

RESPONSE=$(curl -sS -H 'Content-Type: application/json' -d "$REQUEST" http://127.0.0.1:11434/api/generate)

TEXT=$(echo "$RESPONSE" | jq -r 'if (.response // "") != "" then .response else (.thinking // "") end')

if [[ -z "$TEXT" ]]; then
  echo "$RESPONSE" >&2
  exit 3
fi

jq -cn --arg text "$TEXT" '{text:$text}'
