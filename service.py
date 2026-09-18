"""医疗影像模型准入的运行入口与 HTTP 接口。

健康检查保持原有契约；业务接口按角色鉴权，
ops 角色只能访问 /health，临床数据与审计日志对其关闭。
"""

import argparse
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from admission import AdmissionService
from domain import DomainError

SERVICE_ID = "imaging-model-registry"
SERVICE_NAME = "医疗影像模型准入"

MAX_BODY_BYTES = 1024 * 1024

# 角色：registrar 准入登记、approver 科室审批、doctor 医生、auditor 审计、ops 运维。
READERS = ("registrar", "approver", "doctor", "auditor")

ROUTES = [
    ("GET", re.compile(r"^/health$"), "health", None),
    ("POST", re.compile(r"^/models$"), "register_model", ("registrar",)),
    ("GET", re.compile(r"^/models$"), "list_models", READERS),
    ("GET", re.compile(r"^/models/(?P<object_id>[^/]+)$"), "get_model", READERS),
    ("POST", re.compile(r"^/models/(?P<object_id>[^/]+)/approve$"), "approve_model", ("approver",)),
    ("POST", re.compile(r"^/models/(?P<object_id>[^/]+)/activate$"), "activate_model", ("registrar",)),
    ("POST", re.compile(r"^/models/(?P<object_id>[^/]+)/freeze$"), "freeze_model", ("registrar",)),
    ("POST", re.compile(r"^/models/(?P<object_id>[^/]+)/unfreeze$"), "unfreeze_model", ("registrar",)),
    ("POST", re.compile(r"^/models/(?P<object_id>[^/]+)/metrics$"), "record_metrics", ("registrar",)),
    ("GET", re.compile(r"^/channels/(?P<object_id>[^/]+)$"), "get_channel", ("registrar", "approver", "auditor")),
    ("POST", re.compile(r"^/channels/(?P<object_id>[^/]+)/rollback$"), "rollback_channel", ("registrar",)),
    ("POST", re.compile(r"^/inferences$"), "submit_inference", ("doctor",)),
    ("GET", re.compile(r"^/cases$"), "list_cases", ("auditor",)),
    ("GET", re.compile(r"^/cases/(?P<object_id>[^/]+)$"), "get_case", ("doctor", "auditor")),
    ("POST", re.compile(r"^/cases/(?P<object_id>[^/]+)/review$"), "review_case", ("doctor",)),
    ("GET", re.compile(r"^/cases/(?P<object_id>[^/]+)/trace$"), "trace_case", ("auditor",)),
    ("GET", re.compile(r"^/audit/events$"), "list_audit_events", ("auditor",)),
]


def health_payload():
    """返回稳定的服务身份信息。"""
    return {"status": "ok", "service": SERVICE_ID, "name": SERVICE_NAME}


class Handler(BaseHTTPRequestHandler):
    """业务接口入口；健康检查保持原有契约。"""

    admission = AdmissionService()

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def _dispatch(self):
        path = self.path.split("?", 1)[0]
        for method, pattern, action, roles in ROUTES:
            if method != self.command:
                continue
            match = pattern.match(path)
            if match:
                self._invoke(action, roles, match.groupdict())
                return
        self.send_error(404)

    def _invoke(self, action, roles, params):
        self._role = self.headers.get("X-Role", "")
        if roles is not None:
            if not self._role:
                return self._send_json(401, {"error": "缺少 X-Role 请求头"})
            if self._role not in roles:
                return self._send_json(403, {"error": "当前角色无权访问该接口"})
        actor = self.headers.get("X-Actor", "").strip()
        if self.command == "POST" and not actor:
            return self._send_json(401, {"error": "缺少 X-Actor 请求头"})
        try:
            status, payload = getattr(self, f"_action_{action}")(actor=actor, **params)
        except DomainError as exc:
            return self._send_json(exc.status, {"error": str(exc)})
        self._send_json(status, payload)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise DomainError("Content-Length 非法")
        if length > MAX_BODY_BYTES:
            raise DomainError("请求体超过大小限制", 413)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise DomainError("请求体不是合法 JSON")
        if not isinstance(data, dict):
            raise DomainError("请求体必须是 JSON 对象")
        return data

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        # 不写访问日志，避免路径与请求内容进入运维日志。
        return

    # ---- 接口动作 ----

    def _action_health(self, actor):
        return 200, health_payload()

    def _action_register_model(self, actor):
        package = self.admission.register_package(actor, self._role, self._read_json())
        return 201, package.to_dict()

    def _action_list_models(self, actor):
        return 200, {"models": self.admission.list_packages()}

    def _action_get_model(self, actor, object_id):
        return 200, self.admission.get_package(object_id)

    def _action_approve_model(self, actor, object_id):
        package = self.admission.approve_package(actor, self._role, object_id, self._read_json())
        return 200, package.to_dict()

    def _action_activate_model(self, actor, object_id):
        package, channel = self.admission.activate_package(actor, self._role, object_id)
        return 200, {
            "package": package.to_dict(),
            "channel": self.admission.channel_state(channel.name),
        }

    def _action_freeze_model(self, actor, object_id):
        reason = self._read_json().get("reason", "")
        package = self.admission.freeze_package(actor, self._role, object_id, reason)
        return 200, package.to_dict()

    def _action_unfreeze_model(self, actor, object_id):
        reason = self._read_json().get("reason", "")
        package = self.admission.unfreeze_package(actor, self._role, object_id, reason)
        return 200, package.to_dict()

    def _action_record_metrics(self, actor, object_id):
        return 200, self.admission.record_metrics(actor, self._role, object_id, self._read_json())

    def _action_get_channel(self, actor, object_id):
        return 200, self.admission.channel_state(object_id)

    def _action_rollback_channel(self, actor, object_id):
        reason = self._read_json().get("reason", "")
        channel, from_id, restored_id = self.admission.rollback_channel(actor, self._role, object_id, reason)
        return 200, {
            "channel": self.admission.channel_state(channel.name),
            "rolled_back_from": from_id,
            "restored": restored_id,
        }

    def _action_submit_inference(self, actor):
        case, replayed = self.admission.submit_inference(
            actor, self._role, self._read_json(),
            idempotency_key=self.headers.get("Idempotency-Key"),
        )
        return (200 if replayed else 201), {"case": case.to_dict(), "replayed": replayed}

    def _action_list_cases(self, actor):
        return 200, {"cases": self.admission.list_cases()}

    def _action_get_case(self, actor, object_id):
        return 200, self.admission.get_case(object_id)

    def _action_review_case(self, actor, object_id):
        case = self.admission.review_case(actor, self._role, object_id, self._read_json())
        return 200, case.to_dict()

    def _action_trace_case(self, actor, object_id):
        return 200, self.admission.trace_case(object_id)

    def _action_list_audit_events(self, actor):
        return 200, {"events": self.admission.audit_events()}


def make_handler(admission):
    """构造绑定指定准入实例的处理器，便于测试与多实例部署。"""

    class BoundHandler(Handler):
        pass

    BoundHandler.admission = admission
    return BoundHandler


def _smoke_check():
    """用一次性实例跑通登记到追溯的最小链路，供 --check 与巡检使用。"""
    service = AdmissionService()
    payload = {
        "name": "smoke-model",
        "version": "0.0.1",
        "package_hash": hashlib.sha256(b"smoke-package").hexdigest(),
        "training_data_source": "自检占位数据",
        "organ_coverage": ["肝"],
        "disease_coverage": ["占位病种"],
        "validation": {
            "routine": {"metrics": {"sensitivity": 0.9, "specificity": 0.9}, "sample_size": 10, "dataset": "自检集"},
            "acute_abdomen": {"metrics": {"sensitivity": 0.9, "specificity": 0.9}, "sample_size": 10, "dataset": "自检集"},
            "device:smoke": {"metrics": {"sensitivity": 0.9, "specificity": 0.9}, "sample_size": 10, "dataset": "自检集"},
        },
        "contraindications": [],
    }
    package = service.register_package("自检", "registrar", payload)
    service.approve_package("自检", "approver", package.package_id,
                            {"department": "放射科", "scope": {"scenarios": ["routine"]}})
    service.activate_package("自检", "registrar", package.package_id)
    image_hash = hashlib.sha256(b"smoke-image").hexdigest()
    request = {
        "model": "smoke-model",
        "scenario": "routine",
        "image_hash": image_hash,
        "prompt": "自检提示",
        "patient": {"name": "自检患者"},
    }
    case, replayed = service.submit_inference("自检", "doctor", request, idempotency_key="smoke-key")
    assert not replayed
    again, replayed = service.submit_inference("自检", "doctor", request, idempotency_key="smoke-key")
    assert replayed and again.case_id == case.case_id
    service.review_case("自检", "doctor", case.case_id, {"conclusion": "agree"})
    trace = service.trace_case(case.case_id)
    assert trace["model"]["package_id"] == package.package_id
    assert "自检患者" not in json.dumps(trace, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=SERVICE_NAME)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        assert health_payload()["service"] == SERVICE_ID
        _smoke_check()
        print("基础检查通过")
        return
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
