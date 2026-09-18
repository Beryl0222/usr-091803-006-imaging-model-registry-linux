"""覆盖登记、审批灰度、推理绑定、幂等、冻结回退与审计追溯的测试。"""

import hashlib
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from admission import AdmissionService
from domain import DomainError
from service import make_handler


def valid_validation():
    return {
        "routine": {"metrics": {"sensitivity": 0.91, "specificity": 0.88},
                    "sample_size": 1200, "dataset": "常规检查回顾集"},
        "acute_abdomen": {"metrics": {"sensitivity": 0.86, "specificity": 0.84},
                          "sample_size": 640, "dataset": "急腹症前瞻集"},
        "device:ct-a": {"metrics": {"sensitivity": 0.90, "specificity": 0.87},
                        "sample_size": 800, "dataset": "CT-A 设备集"},
    }


def package_payload(**overrides):
    payload = {
        "name": "abdomen-foundation",
        "version": "1.0.0",
        "package_hash": hashlib.sha256(b"abdomen-1.0.0").hexdigest(),
        "training_data_source": "三家中心脱敏腹部 CT，附授权与来源清单",
        "organ_coverage": ["肝", "胆", "胰", "脾"],
        "disease_coverage": ["肝细胞癌", "胆结石", "急性胰腺炎"],
        "validation": valid_validation(),
        "contraindications": ["孕早期", "碘造影剂过敏"],
    }
    payload.update(overrides)
    return payload


def image_hash(tag=b"img-1"):
    return hashlib.sha256(tag).hexdigest()


def inference_payload(**overrides):
    payload = {
        "model": "abdomen-foundation",
        "scenario": "routine",
        "image_hash": image_hash(),
        "params": {"window": "abdomen", "slice_mm": 5},
        "prompt": "提示：结合影像列出腹部异常征象",
    }
    payload.update(overrides)
    return payload


class RegistrationTest(unittest.TestCase):
    """模型包登记：必填项与分场景验证的完整性。"""

    def setUp(self):
        self.service = AdmissionService()

    def test_missing_required_fields_rejected(self):
        for key in ("name", "version", "package_hash", "training_data_source",
                    "organ_coverage", "disease_coverage", "validation", "contraindications"):
            payload = package_payload()
            del payload[key]
            with self.assertRaises(DomainError) as ctx:
                self.service.register_package("登记员", "registrar", payload)
            self.assertEqual(ctx.exception.status, 400, key)

    def test_package_hash_must_be_sha256(self):
        with self.assertRaises(DomainError):
            self.service.register_package("登记员", "registrar", package_payload(package_hash="abc"))

    def test_validation_must_cover_required_scenarios(self):
        validation = valid_validation()
        del validation["acute_abdomen"]
        with self.assertRaises(DomainError):
            self.service.register_package("登记员", "registrar", package_payload(validation=validation))

    def test_validation_must_include_device_scenario(self):
        validation = valid_validation()
        del validation["device:ct-a"]
        with self.assertRaises(DomainError):
            self.service.register_package("登记员", "registrar", package_payload(validation=validation))

    def test_validation_metric_range_and_dataset(self):
        validation = valid_validation()
        validation["routine"]["metrics"]["sensitivity"] = 1.2
        with self.assertRaises(DomainError):
            self.service.register_package("登记员", "registrar", package_payload(validation=validation))
        validation = valid_validation()
        del validation["routine"]["dataset"]
        with self.assertRaises(DomainError):
            self.service.register_package("登记员", "registrar", package_payload(validation=validation))

    def test_duplicate_hash_or_version_rejected(self):
        self.service.register_package("登记员", "registrar", package_payload())
        with self.assertRaises(DomainError) as ctx:
            self.service.register_package("登记员", "registrar", package_payload())
        self.assertEqual(ctx.exception.status, 409)
        with self.assertRaises(DomainError):
            self.service.register_package("登记员", "registrar", package_payload(version="1.1.0"))
        with self.assertRaises(DomainError):
            self.service.register_package(
                "登记员", "registrar",
                package_payload(package_hash=hashlib.sha256(b"abdomen-other").hexdigest()))


class FlowTestCase(unittest.TestCase):
    """提供登记、审批、激活的公共准备步骤。"""

    def setUp(self):
        self.service = AdmissionService()

    def _register(self, **overrides):
        return self.service.register_package("登记员甲", "registrar", package_payload(**overrides))

    def _approve_and_activate(self, package, scenarios=("routine",), max_cases=None):
        scope = {"scenarios": list(scenarios)}
        if max_cases is not None:
            scope["max_cases"] = max_cases
        self.service.approve_package("主任乙", "approver", package.package_id,
                                     {"department": "放射科", "scope": scope})
        self.service.activate_package("登记员甲", "registrar", package.package_id)
        return package


class ApprovalAndGrayscaleTest(FlowTestCase):
    """审批与限定灰度：场景范围与病例上限。"""

    def test_approval_scope_must_have_validation_evidence(self):
        package = self._register()
        with self.assertRaises(DomainError) as ctx:
            self.service.approve_package(
                "主任乙", "approver", package.package_id,
                {"department": "放射科", "scope": {"scenarios": ["routine", "device:ct-b"]}})
        self.assertEqual(ctx.exception.status, 400)

    def test_activate_requires_approval(self):
        package = self._register()
        with self.assertRaises(DomainError) as ctx:
            self.service.activate_package("登记员甲", "registrar", package.package_id)
        self.assertEqual(ctx.exception.status, 409)

    def test_inference_requires_active_grayscale(self):
        package = self._register()
        self.service.approve_package(
            "主任乙", "approver", package.package_id,
            {"department": "放射科", "scope": {"scenarios": ["routine"]}})
        with self.assertRaises(DomainError) as ctx:
            self.service.submit_inference("医生丙", "doctor", inference_payload())
        self.assertEqual(ctx.exception.status, 409)

    def test_grayscale_scope_enforced(self):
        self._approve_and_activate(self._register(), scenarios=("routine",))
        with self.assertRaises(DomainError) as ctx:
            self.service.submit_inference("医生丙", "doctor", inference_payload(scenario="acute_abdomen"))
        self.assertEqual(ctx.exception.status, 403)

    def test_grayscale_max_cases_enforced(self):
        self._approve_and_activate(self._register(), max_cases=1)
        self.service.submit_inference("医生丙", "doctor", inference_payload(), idempotency_key="k1")
        with self.assertRaises(DomainError) as ctx:
            self.service.submit_inference("医生丙", "doctor", inference_payload(image_hash=image_hash(b"img-2")))
        self.assertEqual(ctx.exception.status, 409)


class InferenceTest(FlowTestCase):
    """推理绑定：脱敏、幂等、去重。"""

    def setUp(self):
        super().setUp()
        self.package = self._approve_and_activate(self._register())

    def test_identity_stripped_before_binding(self):
        payload = inference_payload()
        payload["patient"] = {"name": "张三", "id_number": "110101199001011234", "phone": "13800000000"}
        payload["params"]["mrn"] = "MRN-7788"
        case, replayed = self.service.submit_inference("医生丙", "doctor", payload, idempotency_key="key-1")
        self.assertFalse(replayed)
        blob = json.dumps(case.to_dict(), ensure_ascii=False)
        for leaked in ("张三", "110101199001011234", "13800000000", "MRN-7788", "patient"):
            self.assertNotIn(leaked, blob)
        audit_blob = json.dumps(self.service.audit_events(), ensure_ascii=False)
        for leaked in ("张三", "110101199001011234", "13800000000", "MRN-7788"):
            self.assertNotIn(leaked, audit_blob)
        events = [e for e in self.service.audit_events() if e["action"] == "inference.completed"]
        self.assertEqual(events[0]["detail"]["deidentified_fields"], ["mrn", "patient"])

    def test_idempotent_retry_returns_single_case(self):
        case1, replayed1 = self.service.submit_inference(
            "医生丙", "doctor", inference_payload(), idempotency_key="retry-1")
        case2, replayed2 = self.service.submit_inference(
            "医生丙", "doctor", inference_payload(), idempotency_key="retry-1")
        self.assertFalse(replayed1)
        self.assertTrue(replayed2)
        self.assertEqual(case1.case_id, case2.case_id)
        self.assertEqual(len(self.service.list_cases()), 1)

    def test_idempotency_key_conflict_rejected(self):
        self.service.submit_inference("医生丙", "doctor", inference_payload(), idempotency_key="retry-2")
        with self.assertRaises(DomainError) as ctx:
            self.service.submit_inference(
                "医生丙", "doctor", inference_payload(image_hash=image_hash(b"img-other")),
                idempotency_key="retry-2")
        self.assertEqual(ctx.exception.status, 409)

    def test_duplicate_upload_deduplicated_without_key(self):
        case1, _ = self.service.submit_inference("医生丙", "doctor", inference_payload())
        case2, replayed2 = self.service.submit_inference("医生丁", "doctor", inference_payload())
        self.assertTrue(replayed2)
        self.assertEqual(case1.case_id, case2.case_id)
        self.assertEqual(len(self.service.list_cases()), 1)


class FreezeTest(FlowTestCase):
    """指标下降冻结：拦截新请求，不影响已签发报告。"""

    def setUp(self):
        super().setUp()
        self.package = self._approve_and_activate(self._register())

    def test_metric_drop_auto_freezes_and_preserves_issued_reports(self):
        case, _ = self.service.submit_inference(
            "医生丙", "doctor", inference_payload(), idempotency_key="k-1")
        self.service.review_case("医生丙", "doctor", case.case_id, {"conclusion": "agree"})
        report = self.service.record_metrics(
            "质控员", "registrar", self.package.package_id,
            {"scenario": "routine", "metrics": {"sensitivity": 0.80, "specificity": 0.87}})
        self.assertTrue(report["degraded"])
        self.assertEqual(report["package_status"], "frozen")
        with self.assertRaises(DomainError) as ctx:
            self.service.submit_inference("医生丙", "doctor", inference_payload(image_hash=image_hash(b"img-2")))
        self.assertEqual(ctx.exception.status, 409)
        stored = self.service.get_case(case.case_id)
        self.assertEqual(stored["status"], "reviewed")
        again, replayed = self.service.submit_inference(
            "医生丙", "doctor", inference_payload(), idempotency_key="k-1")
        self.assertTrue(replayed)
        self.assertEqual(again.case_id, case.case_id)

    def test_small_metric_drop_does_not_freeze(self):
        report = self.service.record_metrics(
            "质控员", "registrar", self.package.package_id,
            {"scenario": "routine", "metrics": {"sensitivity": 0.88, "specificity": 0.86}})
        self.assertFalse(report["degraded"])
        self.assertEqual(report["package_status"], "grayscale")

    def test_manual_freeze_and_unfreeze(self):
        self.service.freeze_package("登记员甲", "registrar", self.package.package_id, "人工复核中")
        with self.assertRaises(DomainError):
            self.service.submit_inference("医生丙", "doctor", inference_payload())
        self.service.unfreeze_package("登记员甲", "registrar", self.package.package_id, "复核通过")
        case, _ = self.service.submit_inference("医生丙", "doctor", inference_payload())
        self.assertEqual(case.package_id, self.package.package_id)

    def test_metrics_unknown_scenario_rejected(self):
        with self.assertRaises(DomainError):
            self.service.record_metrics(
                "质控员", "registrar", self.package.package_id,
                {"scenario": "device:ct-b", "metrics": {"sensitivity": 0.5}})


class RollbackTest(FlowTestCase):
    """版本切换留有可执行回退点。"""

    def _register_v2(self):
        return self._register(version="1.1.0",
                              package_hash=hashlib.sha256(b"abdomen-1.1.0").hexdigest())

    def test_rollback_restores_previous_version(self):
        v1 = self._approve_and_activate(self._register())
        v2 = self._approve_and_activate(self._register_v2())
        state = self.service.channel_state("abdomen-foundation")
        self.assertEqual(state["active_package_id"], v2.package_id)
        self.assertEqual(len(state["rollback_points"]), 2)
        case, _ = self.service.submit_inference("医生丙", "doctor", inference_payload())
        self.assertEqual(case.model_version, "1.1.0")

        self.service.rollback_channel("登记员甲", "registrar", "abdomen-foundation", "v2 指标波动")
        state = self.service.channel_state("abdomen-foundation")
        self.assertEqual(state["active_package_id"], v1.package_id)
        case2, _ = self.service.submit_inference(
            "医生丙", "doctor", inference_payload(image_hash=image_hash(b"img-2")))
        self.assertEqual(case2.model_version, "1.0.0")
        actions = [event["action"] for event in self.service.audit_events()]
        self.assertIn("channel.rollback", actions)

    def test_rollback_without_point_rejected(self):
        self._approve_and_activate(self._register())
        with self.assertRaises(DomainError) as ctx:
            self.service.rollback_channel("登记员甲", "registrar", "abdomen-foundation", "无点可退")
        self.assertEqual(ctx.exception.status, 409)


class TraceTest(FlowTestCase):
    """审计追溯与最终人工决定。"""

    def test_trace_links_model_evidence_approver_and_decision(self):
        package = self._approve_and_activate(self._register())
        case, _ = self.service.submit_inference("医生丙", "doctor", inference_payload())
        self.service.review_case("医生丙", "doctor", case.case_id,
                                 {"conclusion": "modified", "notes": "补充胆囊结石"})
        trace = self.service.trace_case(case.case_id)
        self.assertEqual(trace["model"]["package_hash"], package.package_hash)
        self.assertEqual(trace["model"]["version"], "1.0.0")
        self.assertEqual(trace["approval"]["approver"], "主任乙")
        self.assertEqual(trace["approval"]["department"], "放射科")
        self.assertEqual(set(trace["validation"]), {"routine", "acute_abdomen", "device:ct-a"})
        self.assertEqual(trace["review"]["conclusion"], "modified")
        self.assertEqual(trace["review"]["reviewer"], "医生丙")
        actions = [event["action"] for event in trace["audit_events"]]
        for expected in ("package.registered", "package.approved", "package.activated",
                         "inference.completed", "case.reviewed"):
            self.assertIn(expected, actions)

    def test_review_is_final(self):
        self._approve_and_activate(self._register())
        case, _ = self.service.submit_inference("医生丙", "doctor", inference_payload())
        self.service.review_case("医生丙", "doctor", case.case_id, {"conclusion": "agree"})
        with self.assertRaises(DomainError) as ctx:
            self.service.review_case("医生丁", "doctor", case.case_id, {"conclusion": "rejected"})
        self.assertEqual(ctx.exception.status, 409)

    def test_review_conclusion_must_be_valid(self):
        self._approve_and_activate(self._register())
        case, _ = self.service.submit_inference("医生丙", "doctor", inference_payload())
        with self.assertRaises(DomainError) as ctx:
            self.service.review_case("医生丙", "doctor", case.case_id, {"conclusion": "maybe"})
        self.assertEqual(ctx.exception.status, 400)


class HttpApiTest(unittest.TestCase):
    """角色权限、幂等键与端到端链路。"""

    def setUp(self):
        self.service = AdmissionService()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _call(self, method, path, payload=None, role=None, actor=None, headers=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(self.base + path, data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if role:
            request.add_header("X-Role", role)
        if actor:
            request.add_header("X-Actor", actor)
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with urlopen(request, timeout=2) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            body = error.read()
            error.close()
            return error.code, (json.loads(body) if body else {})

    def _activate_via_http(self):
        status, package = self._call("POST", "/models", package_payload(),
                                     role="registrar", actor="registrar-01")
        self.assertEqual(status, 201)
        package_id = package["package_id"]
        status, _ = self._call("POST", f"/models/{package_id}/approve",
                               {"department": "放射科",
                                "scope": {"scenarios": ["routine", "acute_abdomen"]}},
                               role="approver", actor="approver-01")
        self.assertEqual(status, 200)
        status, _ = self._call("POST", f"/models/{package_id}/activate", {},
                               role="registrar", actor="registrar-01")
        self.assertEqual(status, 200)
        return package_id

    def test_missing_role_and_actor_rejected(self):
        status, _ = self._call("POST", "/models", package_payload())
        self.assertEqual(status, 401)
        status, _ = self._call("POST", "/models", package_payload(), role="registrar")
        self.assertEqual(status, 401)

    def test_ops_cannot_reach_clinical_data(self):
        self._activate_via_http()
        status, body = self._call("POST", "/inferences", inference_payload(),
                                  role="doctor", actor="doctor-01", headers={"Idempotency-Key": "op-1"})
        self.assertEqual(status, 201)
        case_id = body["case"]["case_id"]
        for method, path in (("GET", f"/cases/{case_id}"), ("GET", "/audit/events"),
                             ("POST", "/inferences"), ("GET", f"/cases/{case_id}/trace"),
                             ("GET", "/models")):
            payload = inference_payload() if method == "POST" else None
            status, _ = self._call(method, path, payload, role="ops", actor="ops-01")
            self.assertEqual(status, 403, path)
        status, body = self._call("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["service"], "imaging-model-registry")

    def test_role_scope_enforced(self):
        status, package = self._call("POST", "/models", package_payload(),
                                     role="registrar", actor="registrar-01")
        self.assertEqual(status, 201)
        package_id = package["package_id"]
        status, _ = self._call("POST", f"/models/{package_id}/approve",
                               {"department": "放射科", "scope": {"scenarios": ["routine"]}},
                               role="doctor", actor="doctor-01")
        self.assertEqual(status, 403)
        status, _ = self._call("GET", "/audit/events", role="registrar", actor="registrar-01")
        self.assertEqual(status, 403)

    def test_end_to_end_with_idempotency_and_trace(self):
        package_id = self._activate_via_http()
        payload = inference_payload()
        payload["patient"] = {"name": "李四", "id_number": "310104198808084321"}
        status, first = self._call("POST", "/inferences", payload,
                                   role="doctor", actor="doctor-01", headers={"Idempotency-Key": "net-1"})
        self.assertEqual(status, 201)
        self.assertFalse(first["replayed"])
        status, second = self._call("POST", "/inferences", payload,
                                    role="doctor", actor="doctor-01", headers={"Idempotency-Key": "net-1"})
        self.assertEqual(status, 200)
        self.assertTrue(second["replayed"])
        self.assertEqual(first["case"]["case_id"], second["case"]["case_id"])
        case_id = first["case"]["case_id"]

        status, reviewed = self._call("POST", f"/cases/{case_id}/review",
                                      {"conclusion": "agree"}, role="doctor", actor="doctor-01")
        self.assertEqual(status, 200)
        self.assertEqual(reviewed["status"], "reviewed")

        status, trace = self._call("GET", f"/cases/{case_id}/trace", role="auditor", actor="auditor-01")
        self.assertEqual(status, 200)
        self.assertEqual(trace["model"]["package_id"], package_id)
        self.assertEqual(trace["approval"]["approver"], "approver-01")
        self.assertEqual(trace["review"]["reviewer"], "doctor-01")
        trace_blob = json.dumps(trace, ensure_ascii=False)
        for leaked in ("李四", "310104198808084321"):
            self.assertNotIn(leaked, trace_blob)

        status, events = self._call("GET", "/audit/events", role="auditor", actor="auditor-01")
        self.assertEqual(status, 200)
        audit_blob = json.dumps(events, ensure_ascii=False)
        for leaked in ("李四", "310104198808084321"):
            self.assertNotIn(leaked, audit_blob)

    def test_freeze_blocks_new_inference_over_http(self):
        package_id = self._activate_via_http()
        status, report = self._call("POST", f"/models/{package_id}/metrics",
                                    {"scenario": "routine",
                                     "metrics": {"sensitivity": 0.7, "specificity": 0.8}},
                                    role="registrar", actor="qc-01")
        self.assertEqual(status, 200)
        self.assertTrue(report["degraded"])
        status, body = self._call("POST", "/inferences", inference_payload(),
                                  role="doctor", actor="doctor-01")
        self.assertEqual(status, 409)
        self.assertIn("冻结", body["error"])


if __name__ == "__main__":
    unittest.main()
