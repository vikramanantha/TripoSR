#!/usr/bin/env bash
# S sweep: steps_per_outer stays 1 at every S (BATCH_SIZE=0 auto-derives).
# eval/vis/checkpoints disabled and wandb off so the measurement is pure training throughput.
cd /home/markiv/sdfer/TripoSR
mkdir -p logs sweep
RES=sweep/results.txt; : > "$RES"

step_of(){ tr '\r' '\n' < "$1" 2>/dev/null | grep -a "Epoch " | tail -1 \
  | sed -E 's/.*\| ([0-9]+)\/[0-9]+ \[([0-9:]+)<.*/\1 \2/'; }
secs(){ awk -F: '{n=NF; s=0; for(i=1;i<=n;i++) s=s*60+$i; print s}' <<< "$1"; }

for S in 4 6 8 10 12; do
  LOG=logs/sweep_S${S}.log; : > "$LOG"
  echo "=== S=$S starting ===" >&2
  docker exec -e WANDB_MODE=disabled -e SDFER_SAMPLES_PER_BATCH=$S \
    -e SDFER_EPOCHS=50 -e SDFER_EVAL_EVERY=100000 -e SDFER_SAVE_EVERY=100000 \
    -e SDFER_VIS_EVERY=0 markiv bash -lc 'cd /home/markiv/TripoSR && ./train_sdf.sh --train' \
    > "$LOG" 2>&1 &
  CLIENT=$!

  # wait for warmup (step>=30), fail-fast on crash, 8 min cap
  A=""; for i in $(seq 1 96); do
    sleep 5
    if grep -aqE "CUDA out of memory|Traceback|RuntimeError|Killed" "$LOG"; then A="OOM"; break; fi
    R=$(step_of "$LOG"); ST=${R%% *}
    [ -n "$ST" ] && [ "$ST" -ge 30 ] 2>/dev/null && { A="$R"; break; }
  done
  if [ "$A" = "OOM" ] || [ -z "$A" ]; then
    printf "S=%-3s FAILED (oom/no-start)\n" "$S" | tee -a "$RES"
  else
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    # measure over the next 60 steps
    B=""; T=$(( ${A%% *} + 60 ))
    for i in $(seq 1 120); do
      sleep 5
      grep -aqE "CUDA out of memory|Traceback" "$LOG" && break
      R=$(step_of "$LOG"); ST=${R%% *}
      [ -n "$ST" ] && [ "$ST" -ge "$T" ] 2>/dev/null && { B="$R"; break; }
    done
    if [ -n "$B" ]; then
      s1=${A%% *}; e1=$(secs "${A##* }"); s2=${B%% *}; e2=$(secs "${B##* }")
      printf "S=%-3s mem=%sMiB steps %s->%s in %ss\n" "$S" "$MEM" "$s1" "$s2" "$((e2-e1))" | tee -a "$RES"
      python3 -c "
sps=($e2-$e1)/($s2-$s1); print(f'   {sps:.3f} s/step  {sps/$S:.4f} s/sample  epoch(6400/4/$S steps)={6400//4//$S*sps/60:.2f} min')" | tee -a "$RES"
    else
      printf "S=%-3s INCOMPLETE\n" "$S" | tee -a "$RES"
    fi
  fi

  # teardown + wait for GPUs to actually release
  PG=$(ps -o pgid= -p "$(pgrep -f 'torch.distributed.run' | head -1)" 2>/dev/null | tr -d ' ')
  [ -n "$PG" ] && kill -INT -- "-$PG" 2>/dev/null
  kill "$CLIENT" 2>/dev/null
  for i in $(seq 1 60); do
    sleep 2
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 500 ] && break
  done
  sleep 5
done
echo "=== SWEEP DONE ==="; cat "$RES"
