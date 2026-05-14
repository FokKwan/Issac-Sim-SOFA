import zmq
import json
import logging

class SofaCableClient:
    """
    ZMQ REQ client for SOFA cable-driven soft body (RL Env Interface)
    """
    def __init__(self, host="localhost", port=5555, timeout_ms=5000):
        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.REQ)
        
        # 设置超时防死锁
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def _send_and_recv(self, command_dict: dict):
        """内部通信封装，处理序列化和异常"""
        try:
            self.sock.send_string(json.dumps(command_dict))
            reply_str = self.sock.recv_string()
            return json.loads(reply_str)
        except zmq.error.Again:
            logging.error("ZMQ Timeout: SOFA server is not responding! Did SOFA crash?")
            return None 

    def step(self, cable_disp: float):
        """
        发送控制量到 SOFA，执行一步物理仿真
        """
        cmd = {
            "type": "step",
            "cable_disp": float(cable_disp)
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
        self.sock.close()
        self.context.term()