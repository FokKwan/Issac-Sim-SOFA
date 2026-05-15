#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTION="${1:-}"

SOFA_SESSION="${SOFA_SESSION:-sofa-server}"
TRAIN_SESSION="${TRAIN_SESSION:-rl-train}"
LOG_DIR="${LOG_DIR:-logs}"
SOFA_LOG="$LOG_DIR/sofa_server.log"
TRAIN_LOG="$LOG_DIR/train.log"

TIMESTEPS="${TIMESTEPS:-100000}"
EVAL_STEPS="${EVAL_STEPS:-100}"
VENV_PATH="${VENV_PATH:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -x "$VENV_PATH/bin/python" ]]; then
  PYTHON_BIN="$VENV_PATH/bin/python"
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/train_task.sh start
  scripts/train_task.sh stop
  scripts/train_task.sh restart
  scripts/train_task.sh status
  scripts/train_task.sh logs [sofa|train]

Environment variables:
  TIMESTEPS    Training timesteps (default: 100000)
  EVAL_STEPS   Eval rollout steps (default: 100)
  VENV_PATH    Virtual env path (default: .venv)
  PYTHON_BIN   Python binary override (default: python3 or .venv/bin/python)
  SOFA_SESSION tmux session name for SOFA (default: sofa-server)
  TRAIN_SESSION tmux session name for training (default: rl-train)
  LOG_DIR      Log directory (default: logs)
EOF
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux not found. Please install tmux first." >&2
    exit 1
  fi
}

has_session() {
  local session_name="$1"
  tmux has-session -t "=${session_name}" 2>/dev/null
}

start_sofa() {
  if has_session "$SOFA_SESSION"; then
    echo "[INFO] SOFA session already running: $SOFA_SESSION"
    return
  fi

  mkdir -p "$LOG_DIR"
  tmux new-session -d -s "$SOFA_SESSION" \
    "cd \"$REPO_ROOT\" && \"$PYTHON_BIN\" sofa/soft_cable_scene.py 2>&1 | tee \"$SOFA_LOG\""
  echo "[OK] SOFA server started in tmux session: $SOFA_SESSION"
}

start_training() {
  if has_session "$TRAIN_SESSION"; then
    echo "[INFO] Training session already running: $TRAIN_SESSION"
    return
  fi

  mkdir -p "$LOG_DIR"
  tmux new-session -d -s "$TRAIN_SESSION" \
    "cd \"$REPO_ROOT\" && \"$PYTHON_BIN\" -m issac_sim.run_env --total-timesteps \"$TIMESTEPS\" --eval-steps \"$EVAL_STEPS\" 2>&1 | tee \"$TRAIN_LOG\""
  echo "[OK] Training started in tmux session: $TRAIN_SESSION"
}

stop_session() {
  local session_name="$1"
  if has_session "$session_name"; then
    tmux kill-session -t "$session_name"
    echo "[OK] Stopped tmux session: $session_name"
  else
    echo "[INFO] Session not running: $session_name"
  fi
}

show_status() {
  echo "Repository: $REPO_ROOT"
  echo "Python: $PYTHON_BIN"
  echo "SOFA session ($SOFA_SESSION): $(has_session "$SOFA_SESSION" && echo running || echo stopped)"
  echo "TRAIN session ($TRAIN_SESSION): $(has_session "$TRAIN_SESSION" && echo running || echo stopped)"
  echo "Logs:"
  echo "  - $SOFA_LOG"
  echo "  - $TRAIN_LOG"
}

show_logs() {
  local target="${2:-both}"
  case "$target" in
    sofa)
      tail -f "$SOFA_LOG"
      ;;
    train)
      tail -f "$TRAIN_LOG"
      ;;
    both)
      echo "==== $SOFA_LOG ===="
      tail -n 100 "$SOFA_LOG" || true
      echo
      echo "==== $TRAIN_LOG ===="
      tail -n 100 "$TRAIN_LOG" || true
      ;;
    *)
      echo "[ERROR] Unknown logs target: $target" >&2
      usage
      exit 1
      ;;
  esac
}

require_tmux

case "$ACTION" in
  start)
    start_sofa
    # Give SOFA some time to bind ZMQ port before training starts.
    sleep 5
    start_training
    show_status
    ;;
  stop)
    stop_session "$TRAIN_SESSION"
    stop_session "$SOFA_SESSION"
    show_status
    ;;
  restart)
    stop_session "$TRAIN_SESSION"
    stop_session "$SOFA_SESSION"
    start_sofa
    sleep 5
    start_training
    show_status
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
