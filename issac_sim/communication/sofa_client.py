import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import zmq

LOGGER = logging.getLogger(__name__)


def _is_array_like(value):
    return isinstance(value, (list, tuple, np.ndarray))

class SofaCableClient:
    """
    ZMQ REQ client for SOFA cable-driven soft body (RL Env Interface)
    """
    def __init__(self, host="localhost", port=5555, timeout_ms=5000, reconnect_attempts=1):
        self.context = zmq.Context()
        self.sock = None
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self._connect()

    def _connect(self):
        if self.sock is not None:
            self.sock.close()
        self.sock = self.context.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        # 允许在超时后恢复 REQ socket 状态机，避免 send/recv 锁死
        if hasattr(zmq, "REQ_RELAXED"):
            self.sock.setsockopt(zmq.REQ_RELAXED, 1)
        if hasattr(zmq, "REQ_CORRELATE"):
            self.sock.setsockopt(zmq.REQ_CORRELATE, 1)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def _reconnect(self):
        LOGGER.warning("Reconnecting ZMQ socket to tcp://%s:%s", self.host, self.port)
        self._connect()

    def _send_and_recv(self, command_dict: dict) -> Optional[Dict[str, Any]]:
        """内部通信封装，处理序列化和异常"""
        attempts = self.reconnect_attempts + 1
        for attempt_idx in range(attempts):
            try:
                self.sock.send_string(json.dumps(command_dict))
                reply_str = self.sock.recv_string()
                parsed_reply = json.loads(reply_str)
                if not isinstance(parsed_reply, dict):
                    LOGGER.error("Unexpected SOFA response type: %s", type(parsed_reply).__name__)
                    return None
                return parsed_reply
            except zmq.error.Again:
                LOGGER.error("ZMQ timeout waiting for SOFA response.")
            except (json.JSONDecodeError, zmq.error.ZMQError) as exc:
                LOGGER.error("ZMQ communication error: %s", exc)

            if attempt_idx < attempts - 1:
                self._reconnect()
        return None

    def step(self, cable_disp):
        """
        发送控制量到 SOFA，执行一步物理仿真
        """
        cmd = {
            "type": "step",
            "cable_disp": np.asarray(cable_disp, dtype=float).reshape(-1).tolist()
            if _is_array_like(cable_disp)
            else float(cable_disp)
        }
        return self._send_and_recv(cmd)

    def reset(self):
        """
        通知 SOFA 重置仿真场景，返回初始状态观测值
        """
        cmd = {
            "type": "reset"
        }
        return self._send_and_recv(cmd)

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        self.context.term()
