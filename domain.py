"""医疗影像模型准入的领域对象与不变量。

只表达领域规则：模型包登记、分场景验证、审批、灰度、病例绑定与脱敏。
业务流程在 admission.py，HTTP 入口在 service.py。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field

HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# 分场景验证必须覆盖常规检查与急腹症，并至少包含一个设备场景，
# 因为同一模型在常规检查、急腹症和不同设备上的表现并不等同。
REQUIRED_SCENARIOS = ("routine", "acute_abdomen")
DEVICE_SCENARIO_PREFIX = "device:"

REQUIRED_METRICS = ("sensitivity", "specificity")

PACKAGE_STATUSES = ("registered", "approved", "grayscale", "frozen")

REVIEW_CONCLUSIONS = ("agree", "modified", "rejected")

# 直接身份字段：推理请求中无论出现在哪一层，都不得进入存储与日志。
IDENTITY_FIELDS = frozenset({
    "patient", "patient_name", "name", "id_number", "mrn",
    "phone", "birth_date", "address", "identity",
})

# 观测指标相对登记验证值的允许下降幅度，超过即视为指标下降。
METRIC_DROP_TOLERANCE = 0.05


class DomainError(Exception):
    """领域规则被违反；HTTP 层按 status 映射为响应码。"""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def canonical_hash(payload):
    """结构化数据的稳定摘要，用于请求指纹与幂等比对。"""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def require_hash256(value, field_name):
    if not isinstance(value, str) or not HASH_PATTERN.match(value):
        raise DomainError(f"{field_name} 必须是 64 位小写十六进制 SHA-256 摘要")


def strip_identity(payload):
    """递归移除直接身份字段，返回 (脱敏副本, 被移除的字段名集合)。

    推理请求的任何绑定、存储与日志动作都必须发生在本函数之后，
    保证直接身份信息不会进入病例记录与审计日志。
    """
    removed = set()

    def _clean(node):
        if isinstance(node, dict):
            cleaned = {}
            for key, value in node.items():
                if key in IDENTITY_FIELDS:
                    removed.add(key)
                    continue
                cleaned[key] = _clean(value)
            return cleaned
        if isinstance(node, list):
            return [_clean(item) for item in node]
        return node

    return _clean(payload), removed


def validate_validation_entries(validation):
    """校验分场景验证结果：场景覆盖、指标区间、样本量与数据集说明。"""
    if not isinstance(validation, dict) or not validation:
        raise DomainError("必须登记分场景验证结果 validation")
    for scenario in REQUIRED_SCENARIOS:
        if scenario not in validation:
            raise DomainError(f"分场景验证缺少必需场景: {scenario}")
    if not any(key.startswith(DEVICE_SCENARIO_PREFIX) for key in validation):
        raise DomainError("分场景验证必须包含至少一个设备场景(device:...)")
    for scenario, entry in validation.items():
        if not isinstance(entry, dict):
            raise DomainError(f"场景 {scenario} 的验证结果结构非法")
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            raise DomainError(f"场景 {scenario} 缺少指标 metrics")
        for metric in REQUIRED_METRICS:
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
                raise DomainError(f"场景 {scenario} 的指标 {metric} 必须落在 [0,1]")
        sample_size = entry.get("sample_size")
        if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size <= 0:
            raise DomainError(f"场景 {scenario} 缺少有效样本量 sample_size")
        dataset = entry.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip():
            raise DomainError(f"场景 {scenario} 缺少验证数据集说明 dataset")


def _require_text(payload, key):
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DomainError(f"缺少必填字段: {key}")
    return value.strip()


def _require_str_list(payload, key):
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise DomainError(f"{key} 必须是非空字符串列表")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise DomainError(f"{key} 必须是非空字符串列表")
    return [item.strip() for item in value]


@dataclass
class Approval:
    """科室审批结论：批准者、灰度场景范围与可选病例上限。"""

    approver: str
    department: str
    scenarios: list
    max_cases: int | None
    note: str
    decided_at: str

    def to_dict(self):
        return {
            "approver": self.approver,
            "department": self.department,
            "scenarios": list(self.scenarios),
            "max_cases": self.max_cases,
            "note": self.note,
            "decided_at": self.decided_at,
        }


@dataclass
class ModelPackage:
    """登记在册的模型包及其准入状态。"""

    package_id: str
    name: str
    version: str
    package_hash: str
    training_data_source: str
    organ_coverage: list
    disease_coverage: list
    validation: dict
    contraindications: list
    registered_by: str
    registered_at: str
    status: str = "registered"
    approval: Approval | None = None

    @classmethod
    def from_payload(cls, payload, actor):
        if not isinstance(payload, dict):
            raise DomainError("请求体必须是 JSON 对象")
        package_hash = payload.get("package_hash")
        require_hash256(package_hash, "package_hash")
        validation = payload.get("validation")
        validate_validation_entries(validation)
        contraindications = payload.get("contraindications")
        if not isinstance(contraindications, list) or not all(
            isinstance(item, str) and item.strip() for item in contraindications
        ):
            raise DomainError("contraindications 必须是字符串列表（无禁用条件时给空列表）")
        return cls(
            package_id=new_id("pkg"),
            name=_require_text(payload, "name"),
            version=_require_text(payload, "version"),
            package_hash=package_hash,
            training_data_source=_require_text(payload, "training_data_source"),
            organ_coverage=_require_str_list(payload, "organ_coverage"),
            disease_coverage=_require_str_list(payload, "disease_coverage"),
            validation=validation,
            contraindications=[item.strip() for item in contraindications],
            registered_by=actor,
            registered_at=now_iso(),
        )

    def to_dict(self):
        return {
            "package_id": self.package_id,
            "name": self.name,
            "version": self.version,
            "package_hash": self.package_hash,
            "training_data_source": self.training_data_source,
            "organ_coverage": list(self.organ_coverage),
            "disease_coverage": list(self.disease_coverage),
            "validation": self.validation,
            "contraindications": list(self.contraindications),
            "status": self.status,
            "registered_by": self.registered_by,
            "registered_at": self.registered_at,
            "approval": self.approval.to_dict() if self.approval else None,
        }


@dataclass
class CaseRecord:
    """一次推理绑定的病例结果：只含脱敏后的哈希、参数、提示与复核结论。"""

    case_id: str
    idempotency_key: str | None
    request_fingerprint: str
    image_hash: str
    model_name: str
    package_id: str
    model_version: str
    scenario: str
    params: dict
    prompt: str
    result: dict
    created_by: str
    created_at: str
    status: str = "issued"
    review: dict | None = None

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "idempotency_key": self.idempotency_key,
            "request_fingerprint": self.request_fingerprint,
            "image_hash": self.image_hash,
            "model": self.model_name,
            "package_id": self.package_id,
            "model_version": self.model_version,
            "scenario": self.scenario,
            "params": self.params,
            "prompt": self.prompt,
            "result": self.result,
            "status": self.status,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "review": self.review,
        }


@dataclass
class Channel:
    """模型通道：activations 栈顶为生效版本，其余为可执行回退点。"""

    name: str
    activations: list = field(default_factory=list)

    @property
    def active_package_id(self):
        return self.activations[-1] if self.activations else None


@dataclass
class AuditEvent:
    """追加式审计事件：只记录操作者、动作与哈希引用，不含患者信息。"""

    event_id: str
    at: str
    actor: str
    role: str
    action: str
    refs: dict
    detail: dict

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "at": self.at,
            "actor": self.actor,
            "role": self.role,
            "action": self.action,
            "refs": self.refs,
            "detail": self.detail,
        }
