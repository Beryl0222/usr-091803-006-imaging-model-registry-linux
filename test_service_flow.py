"""端到端 HTTP 流程测试：准入全链路、幂等、冻结、回退、审计与运维脱敏。"""

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import service
from registry import ModelRegistry

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def registration(**overrides):
    payload = {
        "name": overrides.pop("_name", "腹部通用影像模型HTTP"),
        "version": "1.0.0",
        "package_hash": HASH_A,
        "training_data_source": "合作医院脱敏 CT 8 万例，含数据使用许可说明",
        "organs": ["肝脏", "阑尾"],
        "conditions": ["急性阑尾炎", "肝占位"] + [f"病种{i}" for i in range(110)],
        "scenario_validation": {
            "常规检查": {"sensitivity": 0.90, "specificity": 0.92, "sample_size": 4000},
            "急腹症": {"sensitivity": 0.83, "specificity": 0.86, "sample_size": 2000},
            "设备分层": {"sensitivity": 0.79, "specificity": 0.84, "sample_size": 1600},
        },
        "contraindications": ["孕妇"],
    }
    payload.update(overrides)
    return payload


class HttpFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), service.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        # 每个用例使用全新的内存注册表。
        service.registry = ModelRegistry()

    def request(self, method, path, body=None, headers=None, expect=None):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = Request(f"{self.base}{path}", data=data, method=method, headers=headers or {})
        if data is not None:
            req.add_header("Content-Type", "application/json; charset=utf-8")
        try:
            with urlopen(req, timeout=3) as resp:
                payload = json.load(resp)
                if expect is not None:
                    self.assertEqual(resp.status, expect)
                return resp.status, payload
        except HTTPError as exc:
            payload = json.load(exc)
            if expect is not None:
                self.assertEqual(exc.code, expect, payload)
            return exc.code, payload

    def director(self):
        return {"X-Actor": "D1001", "X-Role": "doctor"}

    # ------------------------------------------------------------------

    def test_full_admission_to_audit_chain(self):
        # 1) 登记
        _, model = self.request("POST", "/v1/models", registration(), self.director(), 201)
        self.assertEqual(model["state"], "draft")
        self.assertEqual(model["condition_count"], 112)

        # 缺分场景证据不能登记
        bad = registration(package_hash=HASH_C, version="1.0.1")
        del bad["scenario_validation"]["急腹症"]
        status, err = self.request("POST", "/v1/models", bad, self.director())
        self.assertEqual(status, 400)
        self.assertIn("急腹症", err["message"])

        # 2) 审批（无 X-Actor 被拒）
        status, _ = self.request("POST", f"/v1/models/{model['model_id']}/approve",
                                 {"comment": "证据齐全"}, {"X-Role": "doctor"})
        self.assertEqual(status, 401)
        _, model = self.request("POST", f"/v1/models/{model['model_id']}/approve",
                                {"comment": "证据齐全，限定灰度"}, self.director(), 200)
        self.assertEqual(model["state"], "approved")

        # 3) 限定灰度（场景 + 设备）
        _, model = self.request("POST", f"/v1/models/{model['model_id']}/gray",
                                {"scenarios": ["常规检查", "急腹症"], "devices": ["CT-A"]},
                                self.director(), 200)
        self.assertEqual(model["state"], "gray")

        # 4) 推理：直接身份字段必须先剥离
        inf_body = {
            "model_id": model["model_id"], "image_hash": HASH_B,
            "scenario": "急腹症", "device": "CT-A",
            "prompt": "提示急腹症相关发现", "parameters": {"wl": 40},
            "patient_name": "张三",
        }
        status, err = self.request("POST", "/v1/inferences", inf_body,
                                   {"X-Actor": "D1024"})
        self.assertEqual(status, 400)
        self.assertIn("patient_name", err["message"])

        # 5) 幂等：重复上传/网络重试只产生一个病例结果
        inf_body.pop("patient_name")
        headers = {"X-Actor": "D1024", "Idempotency-Key": "upload-001"}
        s1, first = self.request("POST", "/v1/inferences", inf_body, headers, 201)
        s2, second = self.request("POST", "/v1/inferences", inf_body, headers)
        self.assertEqual(s2, 200)
        self.assertTrue(second["reused"])
        self.assertEqual(first["inference_id"], second["inference_id"])
        self.assertEqual(first["case_id"], second["case_id"])
        # 结果上绑定了模型版本与包哈希——医生知道提示来自哪版模型
        self.assertEqual(second["model_version"], "1.0.0")
        self.assertEqual(second["package_hash"], HASH_A)

        # 6) 医生复核（最终人工决定）并签发
        _, done = self.request(
            "POST", f"/v1/inferences/{first['inference_id']}/review",
            {"conclusion": "修改后采纳", "note": "建议复查", "report_id": "RPT-100"},
            {"X-Actor": "D1024"}, 200)
        self.assertTrue(done["issued"])

        # 7) 指标下降 → 冻结新请求，已签发报告仍可读
        _, frozen = self.request("POST", f"/v1/models/{model['model_id']}/freeze",
                                 {"reason": "急腹症敏感性周环比下降"},
                                 {"X-Actor": "Q2001"}, 200)
        self.assertEqual(frozen["state"], "suspended")
        status, _ = self.request("POST", "/v1/inferences",
                                 {**inf_body, "image_hash": HASH_C},
                                 {"X-Actor": "D1025", "Idempotency-Key": "upload-002"})
        self.assertEqual(status, 422)
        _, still = self.request("GET", f"/v1/inferences/{first['inference_id']}",
                                None, {"X-Role": "doctor"})
        self.assertEqual(still["review"]["report_id"], "RPT-100")

        # 8) 审计追溯：诊断提示 → 模型/验证证据/批准者/人工决定
        _, trace = self.request("GET", f"/v1/cases/{first['case_id']}/trace",
                                None, {"X-Role": "auditor"})
        self.assertEqual(trace["model"]["package_hash"], HASH_A)
        self.assertIn("sensitivity", trace["validation_evidence"])
        self.assertEqual(trace["approval"]["approved_by"], "D1001")
        self.assertEqual(trace["human_decision"]["conclusion"], "修改后采纳")
        self.assertEqual(trace["human_decision"]["reviewed_by"], "D1024")

        # 9) 普通运维看不到影像/身份/提示文本
        _, ops_trace = self.request("GET", f"/v1/cases/{first['case_id']}/trace",
                                    None, {"X-Role": "ops"})
        self.assertNotIn("human_decision", ops_trace)
        _, ops_inf = self.request("GET", f"/v1/inferences/{first['inference_id']}",
                                  None, {"X-Role": "ops"})
        self.assertEqual(ops_inf["prompt"], "<redacted>")
        self.assertEqual(ops_inf["review"]["note"], "<redacted>")
        self.assertEqual(ops_inf["image_hash"], HASH_B)  # 技术元数据保留

    def test_version_switch_and_rollback_over_http(self):
        # v1 上线
        _, m1 = self.request("POST", "/v1/models", registration(version="1.0.0",
                             package_hash=HASH_A), self.director(), 201)
        for mid in (m1["model_id"],):
            self.request("POST", f"/v1/models/{mid}/approve", {}, self.director(), 200)
            self.request("POST", f"/v1/models/{mid}/gray",
                         {"scenarios": ["常规检查"]}, self.director(), 200)

        # v2 审批进灰度后切换
        _, m2 = self.request("POST", "/v1/models", registration(version="2.0.0",
                             package_hash=HASH_C), self.director(), 201)
        self.request("POST", f"/v1/models/{m2['model_id']}/approve", {}, self.director(), 200)
        self.request("POST", f"/v1/models/{m2['model_id']}/gray",
                     {"scenarios": ["常规检查"]}, self.director(), 200)
        _, point = self.request("POST", f"/v1/models/{m2['model_id']}/switch",
                                {}, self.director(), 201)
        self.assertEqual(point["from_model_id"], m1["model_id"])
        _, points = self.request("GET", "/v1/rollback-points", None, {"X-Role": "auditor"})
        self.assertEqual(points[0]["rollback_id"], point["rollback_id"])

        # 执行回退点：v1 恢复承接流量
        _, restored = self.request("POST", f"/v1/rollbacks/{point['rollback_id']}",
                                   {}, {"X-Actor": "O3001"}, 200)
        self.assertEqual(restored["model_id"], m1["model_id"])
        self.assertEqual(restored["state"], "gray")

    def test_contraindication_and_scope_enforced(self):
        _, m = self.request("POST", "/v1/models",
                            registration(_name="急诊模型X", version="3.0.0",
                                         package_hash=HASH_C),
                            self.director(), 201)
        self.request("POST", f"/v1/models/{m['model_id']}/approve", {}, self.director(), 200)
        self.request("POST", f"/v1/models/{m['model_id']}/gray",
                     {"scenarios": ["急腹症"], "devices": ["CT-B"]}, self.director(), 200)
        body = {"model_id": m["model_id"], "image_hash": HASH_B, "scenario": "急腹症",
                "device": "CT-B", "prompt": "p", "parameters": {}}
        # 设备不在灰度范围
        status, _ = self.request("POST", "/v1/inferences", {**body, "device": "MR-9"},
                                 {"X-Actor": "D1024"})
        self.assertEqual(status, 422)
        # 命中禁用条件
        status, _ = self.request("POST", "/v1/inferences",
                                 {**body, "contraindication_hits": ["孕妇"]},
                                 {"X-Actor": "D1024"})
        self.assertEqual(status, 422)
        # 正常放行
        self.request("POST", "/v1/inferences", body, {"X-Actor": "D1024"}, 201)


if __name__ == "__main__":
    unittest.main()
