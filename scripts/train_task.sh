#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACTION="${1:-}"

SOFA_SESSION="${SOFA_SESSION:-sofa-server}"
TRAIN_SESSION="${TRAIN_SESSION:-rl-train}"
LOG_DIR="${LOG_DIR:-logs}"
if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="$REPO_ROOT/$LOG_DIR"
fi
SOFA_LOG="$LOG_DIR/sofa_server.log"
TRAIN_LOG="$LOG_DIR/train.log"

TIMESTEPS="${TIMESTEPS:-100000}"
EVAL_STEPS="${EVAL_STEPS:-100}"
VENV_PATH="${VENV_PATH:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOFA_CONDA_ENV="${SOFA_CONDA_ENV:-sofa_rl}"
SOFA_PYTHON_BIN="${SOFA_PYTHON_BIN:-python}"
TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-}"
TRAIN_ENTRY="${TRAIN_ENTRY:-issac_sim.run_env}"
TRAIN_ENTRY_MODE="${TRAIN_ENTRY_MODE:-module}"
TRAIN_STRIP_CONDA="${TRAIN_STRIP_CONDA:-1}"
CONDA_BASE="${CONDA_BASE:-}"

if [[ -x "$VENV_PATH/bin/python" ]]; then
  PYTHON_BIN="$VENV_PATH/bin/python"
fi
if [[ -z "$TRAIN_PYTHON_BIN" && -x "$HOME/omniverse/python.sh" ]]; then
  TRAIN_PYTHON_BIN="$HOME/omniverse/python.sh"
fi
if [[ -z "$TRAIN_PYTHON_BIN" ]]; then
  TRAIN_PYTHON_BIN="$PYTHON_BIN"
fi
if [[ -z "$SOFA_PYTHON_BIN" ]]; then
  SOFA_PYTHON_BIN="$PYTHON_BIN"
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
  PYTHON_BIN   Base Python fallback (default: python3 or .venv/bin/python)
  SOFA_CONDA_ENV  Conda env for SOFA server (default: sofa_rl, empty disables conda activation)
  SOFA_PYTHON_BIN Python executable inside SOFA env (default: python)
  TRAIN_PYTHON_BIN Python executable for RL training (default: ~/omniverse/python.sh if exists)
  TRAIN_ENTRY      Training entry (default: issac_sim.run_env)
  TRAIN_ENTRY_MODE module|script (default: module)
  TRAIN_STRIP_CONDA 1 to unset conda vars before train launch (default: 1)
  CONDA_BASE       Conda base path override (e.g. ~/miniconda3)
  SOFA_SESSION tmux session name for SOFA (default: sofa-server)
  TRAIN_SESSION tmux session name for training (default: rl-train)
  LOG_DIR      Log directory (default: logs)
EOF
}

if [[ "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
  usage
  exit 0
fi

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

resolve_conda_base() {
  if [[ -z "$CONDA_BASE" ]]; then
    if command -v conda >/dev/null 2>&1; then
      CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    elif [[ -d "$HOME/miniconda3" ]]; then
      CONDA_BASE="$HOME/miniconda3"
    elif [[ -d "$HOME/anaconda3" ]]; then
      CONDA_BASE="$HOME/anaconda3"
    fi
  fi
}

build_sofa_command() {
  if [[ -n "$SOFA_CONDA_ENV" ]]; then
    resolve_conda_base
    if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
      echo "[ERROR] SOFA_CONDA_ENV=$SOFA_CONDA_ENV set, but conda.sh not found. Set CONDA_BASE manually." >&2
      return 1
    fi
    printf "%s" "source \"$CONDA_BASE/etc/profile.d/conda.sh\" && conda activate \"$SOFA_CONDA_ENV\" && cd \"$REPO_ROOT/sofa\" && \"$SOFA_PYTHON_BIN\" soft_cable_scene.py 2>&1 | tee \"$SOFA_LOG\""
  else
    printf "%s" "cd \"$REPO_ROOT\" && \"$SOFA_PYTHON_BIN\" sofa/soft_cable_scene.py 2>&1 | tee \"$SOFA_LOG\""
  fi
}

build_train_command() {
  local preamble="cd \"$REPO_ROOT\" && "
  if [[ "$TRAIN_STRIP_CONDA" == "1" ]]; then
    preamble+="unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL PYTHONHOME && "
  fi

  if [[ "$TRAIN_ENTRY_MODE" == "module" ]]; then
    printf "%s" "${preamble}\"$TRAIN_PYTHON_BIN\" -m \"$TRAIN_ENTRY\" --total-timesteps \"$TIMESTEPS\" --eval-steps \"$EVAL_STEPS\" 2>&1 | tee \"$TRAIN_LOG\""
  elif [[ "$TRAIN_ENTRY_MODE" == "script" ]]; then
    printf "%s" "${preamble}\"$TRAIN_PYTHON_BIN\" \"$TRAIN_ENTRY\" --total-timesteps \"$TIMESTEPS\" --eval-steps \"$EVAL_STEPS\" 2>&1 | tee \"$TRAIN_LOG\""
  else
    echo "[ERROR] TRAIN_ENTRY_MODE must be 'module' or 'script', got: $TRAIN_ENTRY_MODE" >&2
    return 1
  fi
}

ensure_session_alive_or_fail() {
  local session_name="$1"
  local log_file="$2"
  sleep 2
  if ! has_session "$session_name"; then
    echo "[ERROR] Session exited immediately: $session_name" >&2
    echo "[ERROR] Last logs from $log_file:" >&2
    tail -n 60 "$log_file" >&2 || true
    return 1
  fi
}

start_sofa() {
  if has_session "$SOFA_SESSION"; then
    echo "[INFO] SOFA session already running: $SOFA_SESSION"
    return
  fi

  local sofa_cmd
  sofa_cmd="$(build_sofa_command)"
  mkdir -p "$LOG_DIR"
  tmux new-session -d -s "$SOFA_SESSION" \
    "bash -lc '$sofa_cmd'"
  ensure_session_alive_or_fail "$SOFA_SESSION" "$SOFA_LOG"
  echo "[OK] SOFA server started in tmux session: $SOFA_SESSION"
}

start_training() {
  if has_session "$TRAIN_SESSION"; then
    echo "[INFO] Training session already running: $TRAIN_SESSION"
    return
  fi

  local train_cmd
  train_cmd="$(build_train_command)"
  mkdir -p "$LOG_DIR"
  tmux new-session -d -s "$TRAIN_SESSION" \
    "bash -lc '$train_cmd'"
  ensure_session_alive_or_fail "$TRAIN_SESSION" "$TRAIN_LOG"
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
  echo "SOFA env: ${SOFA_CONDA_ENV:-<disabled>}"
  echo "SOFA python: $SOFA_PYTHON_BIN"
  echo "TRAIN python: $TRAIN_PYTHON_BIN"
  echo "TRAIN entry: $TRAIN_ENTRY"
  echo "TRAIN entry mode: $TRAIN_ENTRY_MODE"
  echo "TRAIN strip conda: $TRAIN_STRIP_CONDA"
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
