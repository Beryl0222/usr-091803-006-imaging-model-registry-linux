"""医疗影像模型院内准入服务。

在原有健康检查之上提供准入领域的 JSON 接口，领域规则见 registry.py。
所有变更类接口要求 X-Actor 头（操作者工号/姓名），可选 X-Role（auditor/doctor/ops），
运维角色只能看到脱敏日志视图。
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from registry import (
    ModelRegistry,
    RegistryError,
    redact_for_ops,
)

SERVICE_ID = "imaging-model-registry"
SERVICE_NAME = "医疗影像模型准入"

# 进程级注册表（内存实现；更换持久化适配器时保持接口不变）。
registry = ModelRegistry()


def health_payload():
    """返回稳定的服务身份信息。"""
    return {"status": "ok", "service": SERVICE_ID, "name": SERVICE_NAME}


def _self_check() -> None:
    """启动前/巡检用的最小领域自检，不产生对外副作用。"""
    assert health_payload()["service"] == SERVICE_ID
    probe = ModelRegistry()
    payload = {
        "name": "自检模型", "version": "0.0.1",
        "package_hash": "a" * 64,
        "training_data_source": "院内脱敏数据，仅用于自检",
        "organs": ["腹部"], "conditions": ["自检病种"],
        "scenario_validation": {
            "常规检查": {"sensitivity": 0.9, "specificity": 0.9, "sample_size": 10},
            "急腹症": {"sensitivity": 0.8, "specificity": 0.8, "sample_size": 10},
            "设备分层": {"sensitivity": 0.8, "specificity": 0.8, "sample_size": 10},
        },
        "contraindications": [],
    }
    model = probe.register_model(payload, actor="selfcheck")
    probe.approve_model(model.model_id, actor="selfcheck")
    probe.enter_gray(model.model_id, actor="selfcheck", scenarios=["常规检查"])
    inf, reused = probe.infer(
        {"model_id": model.model_id, "image_hash": "b" * 64,
         "scenario": "常规检查", "device": "CT-1", "prompt": "自检", "parameters": {}},
        actor="selfcheck", idempotency_key="k1")
    assert not reused
    _, reused2 = probe.infer(
        {"model_id": model.model_id, "image_hash": "b" * 64,
         "scenario": "常规检查", "device": "CT-1", "prompt": "自检", "parameters": {}},
        actor="selfcheck", idempotency_key="k1")
    assert reused2, "幂等键重试必须复用同一病例结果"

    # 新版本登记 → 审批 → 灰度 → 切换 → 回退点必须可执行。
    payload_v2 = dict(payload, version="0.0.2", package_hash="c" * 64)
    model_v2 = probe.register_model(payload_v2, actor="selfcheck")
    probe.approve_model(model_v2.model_id, actor="selfcheck")
    probe.enter_gray(model_v2.model_id, actor="selfcheck", scenarios=["常规检查"])
    point = probe.switch_version(model_v2.model_id, actor="selfcheck")
    assert point.from_model_id == model.model_id
    restored = probe.rollback(point.rollback_id, actor="selfcheck")
    assert restored.model_id == model.model_id, "回退后必须由旧版本承接流量"


class Handler(BaseHTTPRequestHandler):
    """准入服务的 HTTP 适配层。"""

    server_version = "ImagingRegistry/1.0"

    # -- 基础工具 ----------------------------------------------------------

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RegistryError("请求体不是合法 JSON")
        if not isinstance(data, dict):
            raise RegistryError("请求体必须是 JSON 对象")
        return data

    def _actor(self) -> str:
        actor = (self.headers.get("X-Actor") or "").strip()
        if not actor:
            raise RegistryError("变更操作必须提供 X-Actor 头（操作者）", status=401)
        return actor

    def _role(self) -> str:
        return (self.headers.get("X-Role") or "doctor").strip().lower()

    # -- 路由 --------------------------------------------------------------

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/health":
                self._send_json(200, health_payload())
            elif path == "/v1/models":
                self._send_json(200, [m.public_dict() for m in registry.list_models()])
            elif path.startswith("/v1/models/"):
                self._send_json(200, registry.get_model(path.rsplit("/", 1)[1]).public_dict())
            elif path.startswith("/v1/inferences/"):
                view = registry.get_inference(path.rsplit("/", 1)[1]).to_dict()
                if self._role() == "ops":
                    view = redact_for_ops(view)
                self._send_json(200, view)
            elif path.startswith("/v1/cases/") and path.endswith("/trace"):
                case_id = path.split("/")[3]
                trace = registry.trace_case(case_id)
                if self._role() == "ops":
                    # 运维视图：只留技术与审批元数据，剥除提示/报告文本等。
                    trace = {
                        "case_id": trace["case_id"],
                        "events": redact_for_ops(trace["events"]),
                        "model": trace.get("model"),
                        "approval": redact_for_ops(trace.get("approval")),
                    }
                self._send_json(200, trace)
            elif path == "/v1/audit/events":
                events = [e.to_dict() for e in registry.audit_events()]
                if self._role() == "ops":
                    events = redact_for_ops(events)
                self._send_json(200, events)
            elif path == "/v1/rollback-points":
                name = (query.get("name") or [None])[0]
                self._send_json(200, [p.to_dict() for p in registry.list_rollback_points(name)])
            else:
                self._send_json(404, {"error": "not_found", "message": f"未知路径: {path}"})
        except RegistryError as exc:
            self._send_json(exc.status, {"error": "domain_error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - 防御性兜底，避免堆栈外泄
            self._send_json(500, {"error": "internal", "message": "服务内部错误"})
            raise exc

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            parts = [p for p in path.split("/") if p]
            body = self._read_json()

            # /v1/models
            if parts == ["v1", "models"]:
                record = registry.register_model(body, actor=self._actor())
                self._send_json(201, record.public_dict())
                return

            # /v1/models/{id}/<action>
            if len(parts) == 4 and parts[:2] == ["v1", "models"]:
                model_id, action = parts[2], parts[3]
                if action == "approve":
                    record = registry.approve_model(
                        model_id, actor=self._actor(), comment=str(body.get("comment", "")))
                elif action == "reject":
                    record = registry.reject_model(
                        model_id, actor=self._actor(), reason=str(body.get("reason", "")))
                elif action == "gray":
                    record = registry.enter_gray(
                        model_id, actor=self._actor(),
                        scenarios=body.get("scenarios"), devices=body.get("devices"))
                elif action == "switch":
                    point = registry.switch_version(model_id, actor=self._actor())
                    self._send_json(201, point.to_dict())
                    return
                elif action == "freeze":
                    record = registry.freeze_model(
                        model_id, actor=self._actor(), reason=str(body.get("reason", "")))
                elif action == "resume":
                    record = registry.resume_model(
                        model_id, actor=self._actor(), reason=str(body.get("reason", "")))
                else:
                    self._send_json(404, {"error": "not_found", "message": f"未知操作: {action}"})
                    return
                self._send_json(200, record.public_dict())
                return

            # /v1/rollbacks/{id}
            if len(parts) == 3 and parts[0] == "v1" and parts[1] == "rollbacks":
                record = registry.rollback(parts[2], actor=self._actor())
                self._send_json(200, record.public_dict())
                return

            # /v1/inferences
            if parts == ["v1", "inferences"]:
                key = self.headers.get("Idempotency-Key") or body.pop("idempotency_key", None)
                inference, reused = registry.infer(body, actor=self._actor(),
                                                   idempotency_key=key)
                payload = inference.to_dict()
                payload["reused"] = reused
                self._send_json(200 if reused else 201, payload)
                return

            # /v1/inferences/{id}/review
            if len(parts) == 4 and parts[:2] == ["v1", "inferences"] and parts[3] == "review":
                inference = registry.submit_review(
                    parts[2], actor=self._actor(),
                    conclusion=str(body.get("conclusion", "")),
                    note=str(body.get("note", "")),
                    report_id=body.get("report_id"))
                self._send_json(200, inference.to_dict())
                return

            self._send_json(404, {"error": "not_found", "message": f"未知路径: {path}"})
        except RegistryError as exc:
            self._send_json(exc.status, {"error": "domain_error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": "internal", "message": "服务内部错误"})
            raise exc

    def log_message(self, *_args):
        return


def main():
    parser = argparse.ArgumentParser(description=SERVICE_NAME)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        _self_check()
        print("基础检查通过")
        return
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
