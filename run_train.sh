#!/usr/bin/env bash
# Launch (or resume) training inside a detached tmux session.
# Closing RDP / SSH will NOT kill training.
#
# Usage:
#   ./run_train.sh              # start detached (default args below)
#   ./run_train.sh --attach     # start and attach to the session
#   ./run_train.sh --no-resume  # start fresh (ignore last_model.pth)
#   ./run_train.sh --status     # show session / STATUS / last log lines
#   ./run_train.sh --stop       # stop the tmux training session
#
# Extra args are forwarded to train.py, e.g.:
#   ./run_train.sh --batch_size 16

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SESSION="${TRAIN_TMUX_SESSION:-caption_train}"
CONDA_ENV="${TRAIN_CONDA_ENV:-image_caption}"

# Defaults (override by appending flags, or edit here)
PATIENCE="${TRAIN_PATIENCE:-10}"
EPOCHS="${TRAIN_EPOCHS:-80}"
LOG_EVERY="${TRAIN_LOG_EVERY:-1000}"
CHECKPOINT_EVERY="${TRAIN_CHECKPOINT_EVERY:-5000}"

DO_ATTACH=0
DO_RESUME=1
DO_STATUS=0
DO_STOP=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --attach) DO_ATTACH=1; shift ;;
    --no-resume) DO_RESUME=0; shift ;;
    --status) DO_STATUS=1; shift ;;
    --stop) DO_STOP=1; shift ;;
    --help|-h)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$DO_STATUS" -eq 1 ]]; then
  echo "=== tmux session ==="
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux list-sessions | grep "^${SESSION}:" || true
    echo "Attach with: tmux attach -t $SESSION"
  else
    echo "No tmux session named '$SESSION'"
  fi
  echo
  echo "=== runs/STATUS ==="
  if [[ -f runs/STATUS ]]; then cat runs/STATUS; else echo "(missing)"; fi
  echo
  echo "=== last 15 log lines ==="
  if [[ -f runs/train.log ]]; then tail -n 15 runs/train.log; else echo "(missing)"; fi
  exit 0
fi

if [[ "$DO_STOP" -eq 1 ]]; then
  stopped=0
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "Stopped tmux session '$SESSION'"
    stopped=1
  fi
  if [[ -f runs/train.pid ]]; then
    pid="$(cat runs/train.pid)"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      echo "Stopped nohup PID $pid"
      stopped=1
    fi
    rm -f runs/train.pid
  fi
  if [[ "$stopped" -eq 0 ]]; then
    echo "No session/PID to stop"
  fi
  exit 0
fi

TRAIN_CMD=(python train.py
  --patience "$PATIENCE"
  --epoch "$EPOCHS"
  --log_every "$LOG_EVERY"
  --checkpoint_every "$CHECKPOINT_EVERY"
)

if [[ "$DO_RESUME" -eq 1 ]]; then
  TRAIN_CMD+=(--resume)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "WARN: tmux not found — falling back to nohup (install later: sudo apt install tmux)"
  # shellcheck disable=SC1091
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
  mkdir -p runs
  nohup "${TRAIN_CMD[@]}" >> runs/nohup_stdout.txt 2>&1 &
  echo $! > runs/train.pid
  echo "Started with nohup. PID=$(cat runs/train.pid)"
  echo "  Command: ${TRAIN_CMD[*]}"
  echo "  tail -f runs/train.log"
  echo "  kill \$(cat runs/train.pid)   # to stop"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running."
  echo "  Attach : tmux attach -t $SESSION"
  echo "  Status : ./run_train.sh --status"
  echo "  Stop   : ./run_train.sh --stop"
  if [[ "$DO_ATTACH" -eq 1 ]]; then
    exec tmux attach -t "$SESSION"
  fi
  exit 0
fi

# Build an inner shell command: activate conda if available, then train.
INNER=$(cat <<EOF
cd "$ROOT"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
echo "[\$(date)] Starting: ${TRAIN_CMD[*]}"
echo "[\$(date)] Detach with Ctrl+b then d — training keeps running."
${TRAIN_CMD[*]}
echo "[\$(date)] Training exited with code \$?"
exec bash
EOF
)

tmux new-session -d -s "$SESSION" bash -lc "$INNER"

echo "Started detached tmux session: $SESSION"
echo "Command: ${TRAIN_CMD[*]}"
echo
echo "Useful:"
echo "  tmux attach -t $SESSION     # watch live (Ctrl+b d to detach)"
echo "  tail -f runs/train.log      # log only"
echo "  ./run_train.sh --status     # quick status"
echo "  ./run_train.sh --stop       # kill training session"

if [[ "$DO_ATTACH" -eq 1 ]]; then
  exec tmux attach -t "$SESSION"
fi
