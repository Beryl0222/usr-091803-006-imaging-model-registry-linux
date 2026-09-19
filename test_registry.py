"""准入领域规则测试：登记、审批、灰度、版本回退、幂等、冻结、审计、脱敏。"""

import unittest

from registry import (
    ModelRegistry, RegistryError, StateError, ConflictError, NotFoundError,
    redact_for_ops, SCENARIOS,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def registration(**overrides):
    payload = {
        "name": "腹部通用影像模型",
        "version": "1.0.0",
        "package_hash": HASH_A,
        "training_data_source": "三家合作医院 2019-2024 脱敏 CT，共 12 万例，已取得数据使用许可",
        "organs": ["肝脏", "胆囊", "胰腺", "阑尾", "肠道"],
        "conditions": ["急性阑尾炎", "胆囊炎", "肠梗阻", "肝占位"] + [f"病种{i}" for i in range(100)],
        "scenario_validation": {
            "常规检查": {"sensitivity": 0.91, "specificity": 0.93, "sample_size": 5000},
            "急腹症": {"sensitivity": 0.84, "specificity": 0.88, "sample_size": 2200},
            "设备分层": {"sensitivity": 0.80, "specificity": 0.85, "sample_size": 1800},
        },
        "contraindications": ["孕妇", "非腹部扫描部位"],
    }
    payload.update(overrides)
    return payload


def approved_gray(reg, *, version="1.0.0", package_hash=HASH_A,
                  scenarios=("常规检查", "急腹症"), devices=None, make_active=True,
                  contraindications=("孕妇", "非腹部扫描部位")):
    rec = reg.register_model(
        registration(version=version, package_hash=package_hash,
                     contraindications=list(contraindications)),
        actor="科室主任")
    reg.approve_model(rec.model_id, actor="科室主任", comment="分场景证据齐全")
    # make_active=False 用于“新版本已进灰度但尚未切换”的场景；
    # 家族首个灰度版本会在 enter_gray 时自动承接流量（此前无版本可回退）。
    reg.enter_gray(rec.model_id, actor="科室主任", scenarios=list(scenarios), devices=devices)
    if make_active and reg.active_model(rec.name) is None:
        reg.switch_version(rec.model_id, actor="科室主任")
    return reg.get_model(rec.model_id)


def infer_request(model_id, *, image_hash=HASH_B, scenario="常规检查", device="CT-A",
                  prompt="请提示急腹症相关发现", parameters=None, hits=None):
    req = {
        "model_id": model_id, "image_hash": image_hash, "scenario": scenario,
        "device": device, "prompt": prompt, "parameters": parameters or {"窗宽": 400},
    }
    if hits is not None:
        req["contraindication_hits"] = hits
    return req


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.reg = ModelRegistry()

    def test_full_registration_card(self):
        rec = self.reg.register_model(registration(), actor="准入管理员")
        self.assertEqual(rec.state, "draft")
        self.assertEqual(rec.public_dict()["condition_count"], 104)
        self.assertEqual(set(rec.scenario_validation), set(SCENARIOS))
        card = rec.public_dict()
        for key in ("package_hash", "training_data_source", "organs", "conditions",
                    "scenario_validation", "contraindications"):
            self.assertIn(key, card)

    def test_overall_metrics_cannot_replace_scenarios(self):
        # 缺任一场景都不允许登记：整体指标不足以决定临床范围。
        payload = registration()
        del payload["scenario_validation"]["设备分层"]
        with self.assertRaisesRegex(RegistryError, "设备分层"):
            self.reg.register_model(payload, actor="准入管理员")

    def test_bad_hash_rejected(self):
        with self.assertRaisesRegex(RegistryError, "sha256"):
            self.reg.register_model(registration(package_hash="not-a-hash"), actor="x")

    def test_duplicate_package_hash_rejected(self):
        self.reg.register_model(registration(), actor="x")
        with self.assertRaises(ConflictError):
            self.reg.register_model(registration(version="1.0.1"), actor="x")

    def test_duplicate_family_version_rejected(self):
        self.reg.register_model(registration(), actor="x")
        with self.assertRaises(ConflictError):
            self.reg.register_model(registration(package_hash=HASH_C), actor="x")


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.reg = ModelRegistry()

    def test_approve_then_limited_gray(self):
        rec = self.reg.register_model(registration(), actor="管理员")
        with self.assertRaises(StateError):
            self.reg.enter_gray(rec.model_id, actor="主任", scenarios=["常规检查"])
        self.reg.approve_model(rec.model_id, actor="科室主任")
        gray = self.reg.enter_gray(rec.model_id, actor="科室主任",
                                   scenarios=["常规检查"], devices=["CT-A"])
        self.assertEqual(gray.state, "gray")
        self.assertEqual(gray.gray_devices, ["CT-A"])

    def test_gray_scenario_must_be_validated(self):
        rec = self.reg.register_model(registration(), actor="管理员")
        self.reg.approve_model(rec.model_id, actor="主任")
        with self.assertRaisesRegex(RegistryError, "灰度场景"):
            self.reg.enter_gray(rec.model_id, actor="主任", scenarios=["未验证场景"])

    def test_reject_requires_reason_and_blocks_approval(self):
        rec = self.reg.register_model(registration(), actor="管理员")
        with self.assertRaises(RegistryError):
            self.reg.reject_model(rec.model_id, actor="主任", reason=" ")
        self.reg.reject_model(rec.model_id, actor="主任", reason="急腹症敏感性不达标")
        with self.assertRaises(StateError):
            self.reg.approve_model(rec.model_id, actor="主任")


class VersionSwitchTest(unittest.TestCase):
    def setUp(self):
        self.reg = ModelRegistry()

    def test_switch_creates_executable_rollback_point(self):
        v1 = approved_gray(self.reg, version="1.0.0", package_hash=HASH_A)
        v2 = approved_gray(self.reg, version="1.1.0", package_hash=HASH_C, make_active=False)
        point = self.reg.switch_version(v2.model_id, actor="科室主任")
        self.assertEqual(point.from_model_id, v1.model_id)
        self.assertEqual(self.reg.active_model("腹部通用影像模型").model_id, v2.model_id)
        self.assertEqual(self.reg.get_model(v1.model_id).state, "archived")

        restored = self.reg.rollback(point.rollback_id, actor="值班负责人")
        self.assertEqual(restored.model_id, v1.model_id)
        self.assertEqual(self.reg.active_model("腹部通用影像模型").model_id, v1.model_id)
        self.assertEqual(self.reg.get_model(v2.model_id).state, "archived")

    def test_rollback_point_is_one_shot(self):
        v1 = approved_gray(self.reg, version="1.0.0", package_hash=HASH_A)
        v2 = approved_gray(self.reg, version="1.1.0", package_hash=HASH_C, make_active=False)
        point = self.reg.switch_version(v2.model_id, actor="主任")
        self.reg.rollback(point.rollback_id, actor="主任")
        with self.assertRaises(ConflictError):
            self.reg.rollback(point.rollback_id, actor="主任")

    def test_cannot_infer_with_archived_or_non_active_version(self):
        v1 = approved_gray(self.reg, version="1.0.0", package_hash=HASH_A)
        v2 = approved_gray(self.reg, version="1.1.0", package_hash=HASH_C, make_active=False)
        # v2 已在灰度但尚未切换为当前版本：不能承接流量。
        with self.assertRaisesRegex(StateError, "不是当前承接流量"):
            self.reg.infer(infer_request(v2.model_id), actor="医生甲")
        self.reg.switch_version(v2.model_id, actor="主任")
        # 切换后旧版本归档，不能再用于新推理。
        with self.assertRaises(StateError):
            self.reg.infer(infer_request(v1.model_id), actor="医生甲")


class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.reg = ModelRegistry()
        self.model = approved_gray(self.reg)

    def test_direct_identifiers_must_be_stripped(self):
        req = infer_request(self.model.model_id)
        req["patient_name"] = "张三"
        with self.assertRaisesRegex(RegistryError, "patient_name"):
            self.reg.infer(req, actor="医生甲")
        req.pop("patient_name")
        req["dicom_meta"] = {"住院号": "000123"}
        with self.assertRaisesRegex(RegistryError, "住院号"):
            self.reg.infer(req, actor="医生甲")

    def test_inference_binds_hash_params_prompt_and_carries_model_version(self):
        inf, reused = self.reg.infer(infer_request(self.model.model_id), actor="医生甲")
        self.assertFalse(reused)
        self.assertEqual(inf.model_version, "1.0.0")
        self.assertEqual(inf.package_hash, HASH_A)
        self.assertEqual(inf.image_hash, HASH_B)
        self.assertIn("parameters", inf.to_dict())

    def test_retry_and_duplicate_upload_yield_single_case_result(self):
        key = "upload-7788"
        first, reused1 = self.reg.infer(infer_request(self.model.model_id),
                                        actor="医生甲", idempotency_key=key)
        second, reused2 = self.reg.infer(infer_request(self.model.model_id, image_hash=HASH_B),
                                         actor="医生甲", idempotency_key=key)
        third, reused3 = self.reg.infer(infer_request(self.model.model_id),
                                        actor="医生甲", idempotency_key=key)
        self.assertTrue(reused2 and reused3)
        self.assertEqual({first.inference_id, second.inference_id, third.inference_id},
                         {first.inference_id})
        self.assertEqual(first.case_id, second.case_id)

    def test_gray_scope_enforced_by_scenario_and_device(self):
        reg2 = ModelRegistry()
        m = reg2.register_model(
            registration(name="急诊专用模型", version="9.0.0", package_hash=HASH_C),
            actor="管理员")
        reg2.approve_model(m.model_id, actor="主任")
        reg2.enter_gray(m.model_id, actor="主任", scenarios=["急腹症"], devices=["CT-B"])
        with self.assertRaisesRegex(StateError, "灰度范围"):
            reg2.infer(infer_request(m.model_id, scenario="常规检查"), actor="医生")
        with self.assertRaisesRegex(StateError, "设备"):
            reg2.infer(infer_request(m.model_id, scenario="急腹症", device="CT-A"), actor="医生")
        inf, _ = reg2.infer(infer_request(m.model_id, scenario="急腹症", device="CT-B"),
                            actor="医生")
        self.assertEqual(inf.model_id, m.model_id)

    def test_contraindication_hit_blocks_inference(self):
        with self.assertRaisesRegex(StateError, "禁用条件"):
            self.reg.infer(infer_request(self.model.model_id, hits=["孕妇"]), actor="医生甲")

    def test_freeze_blocks_new_requests_but_keeps_issued_reports(self):
        inf, _ = self.reg.infer(infer_request(self.model.model_id), actor="医生甲")
        self.reg.submit_review(inf.inference_id, actor="医生甲",
                               conclusion="采纳", report_id="RPT-1")
        self.reg.freeze_model(self.model.model_id, actor="质量负责人",
                              reason="急腹症敏感性周环比下降 6%")
        with self.assertRaisesRegex(StateError, "冻结"):
            self.reg.infer(infer_request(self.model.model_id, image_hash=HASH_C), actor="医生乙")
        # 已签发报告/病例结果仍然可读、可追溯。
        fetched = self.reg.get_inference(inf.inference_id)
        self.assertTrue(fetched.issued)
        self.assertEqual(fetched.review["report_id"], "RPT-1")
        # 恢复后新请求正常。
        self.reg.resume_model(self.model.model_id, actor="质量负责人")
        self.reg.infer(infer_request(self.model.model_id, image_hash=HASH_C), actor="医生乙")


class ReviewAndAuditTest(unittest.TestCase):
    def setUp(self):
        self.reg = ModelRegistry()
        self.model = approved_gray(self.reg)

    def test_review_is_final_and_binds_human_decision(self):
        inf, _ = self.reg.infer(infer_request(self.model.model_id), actor="医生甲")
        done = self.reg.submit_review(inf.inference_id, actor="放射科王医生",
                                      conclusion="修改后采纳", note="降为建议复查",
                                      report_id="RPT-2")
        self.assertTrue(done.issued)
        with self.assertRaises(ConflictError):
            self.reg.submit_review(inf.inference_id, actor="放射科王医生", conclusion="采纳")

    def test_trace_from_cue_to_model_evidence_approver_and_decision(self):
        inf, _ = self.reg.infer(infer_request(self.model.model_id, scenario="急腹症"),
                                actor="医生甲")
        self.reg.submit_review(inf.inference_id, actor="王医生", conclusion="不采纳",
                               note="临床不支持", report_id="RPT-3")
        trace = self.reg.trace_case(inf.case_id)
        self.assertEqual(trace["model"]["version"], "1.0.0")
        self.assertEqual(trace["model"]["package_hash"], HASH_A)
        self.assertIn("sensitivity", trace["validation_evidence"])
        self.assertEqual(trace["approval"]["approved_by"], "科室主任")
        self.assertEqual(trace["human_decision"]["conclusion"], "不采纳")
        self.assertEqual(trace["human_decision"]["reviewed_by"], "王医生")
        actions = [e.action for e in self.reg.audit.for_case(inf.case_id)]
        self.assertEqual(actions, ["inference.run", "review.submit"])

    def test_trace_unknown_case_404(self):
        with self.assertRaises(NotFoundError):
            self.reg.trace_case("case_nope")


class OpsRedactionTest(unittest.TestCase):
    def test_ops_view_has_no_identity_prompt_or_image(self):
        reg = ModelRegistry()
        model = approved_gray(reg)
        inf, _ = reg.infer(infer_request(model.model_id), actor="医生甲")
        reg.submit_review(inf.inference_id, actor="王医生", conclusion="采纳",
                          note="患者张三主诉腹痛", report_id="RPT-9")
        view = redact_for_ops(inf.to_dict())
        blob = repr(view)
        self.assertNotIn("张三", blob)
        self.assertEqual(view["prompt"], "<redacted>")
        self.assertEqual(view["review"]["note"], "<redacted>")
        # 技术元数据仍保留，运维可排障。
        self.assertEqual(view["image_hash"], HASH_B)
        self.assertEqual(view["model_id"], model.model_id)

    def test_nested_identifier_fields_redacted(self):
        clean = redact_for_ops({"meta": {"patient_name": "李四", "device": "CT-A"}})
        self.assertEqual(clean["meta"]["patient_name"], "<redacted>")
        self.assertEqual(clean["meta"]["device"], "CT-A")


if __name__ == "__main__":
    unittest.main()
