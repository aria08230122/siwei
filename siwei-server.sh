#!/usr/bin/env bash
# siwei API 服务守护脚本: start / stop / restart / status / logs

set -u
DIR="${SIWEI_DIR:-$HOME/siwei}"
PID_FILE="$DIR/.siwei.pid"
LOG_FILE="$DIR/.siwei.log"
PERSONA="${SIWEI_PERSONA:-shenxi}"
PORT="${SIWEI_PORT:-5001}"
HOST="${SIWEI_HOST:-0.0.0.0}"

is_running() {
  [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null
}

cmd_start() {
  if is_running; then
    echo "siwei 已在运行 (pid=$(cat "$PID_FILE"), port=$PORT)"
    return 0
  fi
  cd "$DIR" || { echo "找不到目录: $DIR"; return 1; }
  nohup python3 siwei.py -p "$PERSONA" --serve --host "$HOST" --port "$PORT" \
    >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  sleep 1
  if is_running; then
    echo "siwei 已启动 (pid=$(cat "$PID_FILE"), port=$PORT, 人格=$PERSONA)"
    echo "日志: $LOG_FILE"
  else
    echo "启动失败,看日志最后几行:"
    tail -n 20 "$LOG_FILE" 2>/dev/null
    rm -f "$PID_FILE"
    return 1
  fi
}

cmd_stop() {
  if ! is_running; then
    echo "siwei 未在运行"
    rm -f "$PID_FILE"
    return 0
  fi
  local pid; pid=$(cat "$PID_FILE")
  kill "$pid" 2>/dev/null
  for _ in 1 2 3 4 5; do
    is_running || break
    sleep 1
  done
  if is_running; then
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$PID_FILE"
  echo "siwei 已停止"
}

cmd_status() {
  if is_running; then
    echo "siwei 运行中 (pid=$(cat "$PID_FILE"), port=$PORT)"
    curl -s --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null && echo
  else
    echo "siwei 未运行"
  fi
}

cmd_logs() {
  if [ -f "$LOG_FILE" ]; then
    tail -n "${1:-50}" "$LOG_FILE"
  else
    echo "尚无日志 ($LOG_FILE)"
  fi
}

case "${1:-status}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status)  cmd_status ;;
  logs)    cmd_logs "${2:-50}" ;;
  *)
    cat <<EOF
用法: $0 {start|stop|restart|status|logs [N]}

环境变量可覆盖默认值:
  SIWEI_DIR       siwei 仓库目录 (默认 ~/siwei)
  SIWEI_PERSONA   人格名 (默认 shenxi)
  SIWEI_PORT      监听端口 (默认 5001)
  SIWEI_HOST      监听地址 (默认 0.0.0.0)
EOF
    exit 1
    ;;
esac
