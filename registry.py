"""医疗影像模型院内准入的领域核心。

覆盖范围：
- 模型包登记：包哈希、训练数据来源说明、器官与病种覆盖、分场景验证结果、禁用条件；
- 科室审批 → 限定灰度 → 版本切换（保留可执行回退点）；
- 推理链路：先剥离直接身份信息，再绑定影像哈希、参数、提示与医生复核结论，
  重复上传与网络重试只产生一个病例结果（幂等键）；
- 指标下降时冻结新请求，且不影响已签发报告；
- 审计：从诊断提示可追到模型版本、验证证据、批准者与最终人工决定；
- 运维日志只记录技术元数据，不含患者影像或身份。

本模块只依赖标准库，存储为内存实现，便于联调与单元测试。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 常量与错误
# ---------------------------------------------------------------------------

# 入院评估必须分别给出表现的场景；整体指标不能代替分场景证据。
SCENARIOS = ("常规检查", "急腹症", "设备分层")

# 模型生命周期状态。
ST_DRAFT = "draft"          # 已登记，待科室审批
ST_APPROVED = "approved"    # 审批通过，尚未进入灰度
ST_GRAY = "gray"            # 限定灰度中，可承接推理流量
ST_SUSPENDED = "suspended"  # 冻结，拒绝新请求（已签发报告不受影响）
ST_REJECTED = "rejected"    # 审批驳回
ST_ARCHIVED = "archived"    # 被新版本替换，保留只读与回退能力

# 复核结论。
REVIEW_ACCEPT = "采纳"
REVIEW_MODIFY = "修改后采纳"
REVIEW_REJECT = "不采纳"
REVIEW_CONCLUSIONS = (REVIEW_ACCEPT, REVIEW_MODIFY, REVIEW_REJECT)

# 身份字段在进入推理链路前必须剥离（直接标识，命中即拒）。
DIRECT_IDENTIFIERS = (
    "patient_name", "patient_id", "id_card", "id_card_no",
    "name", "姓名", "住院号", "门诊号", "手机号", "电话",
    "birth_date", "出生日期", "address", "地址",
)

# 审计/运维日志中即使误传入也要擦除的字段（按后缀匹配）。
SENSITIVE_SUFFIXES = ("name", "姓名", "id_card", "idcard", "phone", "mobile", "address", "地址")

ALGORITHM = "sha256"


class RegistryError(Exception):
    """领域规则冲突，HTTP 层映射为 4xx。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class NotFoundError(RegistryError):
    def __init__(self, message: str):
        super().__init__(message, 404)


class ConflictError(RegistryError):
    def __init__(self, message: str):
        super().__init__(message, 409)


class StateError(RegistryError):
    def __init__(self, message: str):
        super().__init__(message, 422)


def now_ms() -> int:
    return int(time.time() * 1000)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    event_id: str
    ts: int
    actor: str
    action: str
    target: str                       # model_id / inference_id / case_id
    summary: str                      # 人类可读摘要，不含身份与影像
    linked: dict[str, Any] = field(default_factory=dict)
    prior_state: str | None = None
    new_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ts": self.ts,
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "summary": self.summary,
            "linked": self.linked,
            "prior_state": self.prior_state,
            "new_state": self.new_state,
        }


class AuditLog:
    """只增审计日志；普通运维角色只能看到脱敏视图。"""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        # case_id -> 事件序号列表，支持“从诊断提示/病例结果”反查全链。
        self._case_index: dict[str, list[int]] = {}
        # model_id -> 事件序号列表。
        self._model_index: dict[str, list[int]] = {}

    def append(self, event: AuditEvent) -> None:
        idx = len(self._events)
        self._events.append(event)
        case_id = event.linked.get("case_id")
        if case_id:
            self._case_index.setdefault(case_id, []).append(idx)
        model_id = event.linked.get("model_id") or (
            event.target if event.target.startswith("mdl_") else None
        )
        if model_id:
            self._model_index.setdefault(model_id, []).append(idx)

    def _at(self, indexes: list[int]) -> list[AuditEvent]:
        return [self._events[i] for i in indexes]

    def all(self) -> list[AuditEvent]:
        return list(self._events)

    def for_case(self, case_id: str) -> list[AuditEvent]:
        return self._at(self._case_index.get(case_id, []))

    def for_model(self, model_id: str) -> list[AuditEvent]:
        return self._at(self._model_index.get(model_id, []))


# ---------------------------------------------------------------------------
# 模型登记与生命周期
# ---------------------------------------------------------------------------

@dataclass
class ModelRecord:
    model_id: str
    name: str
    version: str
    package_hash: str                 # 模型包 sha256
    training_data: str                # 训练数据来源说明（人群/时间/机构/许可）
    organs: list[str]
    conditions: list[str]             # 覆盖病种（“一百多种腹部病症”的明确清单）
    scenario_validation: dict[str, dict[str, Any]]  # scenario -> {sensitivity, specificity,...}
    contraindications: list[str]      # 禁用条件
    registered_by: str
    registered_ts: int
    state: str = ST_DRAFT
    approved_by: str | None = None
    approved_ts: int | None = None
    approval_comment: str | None = None
    rejected_by: str | None = None
    rejection_reason: str | None = None
    # 灰度边界：允许的场景与设备；空集合表示不限制（仍受禁用条件约束）。
    gray_scenarios: list[str] = field(default_factory=list)
    gray_devices: list[str] = field(default_factory=list)
    supersedes: str | None = None     # 上线时替换掉的旧版本
    lineage: list[str] = field(default_factory=list)  # 同家族版本链（model_id）
    family: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "state": self.state,
            "package_hash": self.package_hash,
            "package_hash_alg": ALGORITHM,
            "training_data_source": self.training_data,
            "organs": self.organs,
            "conditions": self.conditions,
            "condition_count": len(self.conditions),
            "scenario_validation": self.scenario_validation,
            "contraindications": self.contraindications,
            "registered_by": self.registered_by,
            "registered_ts": self.registered_ts,
            "approved_by": self.approved_by,
            "approved_ts": self.approved_ts,
            "approval_comment": self.approval_comment,
            "rejected_by": self.rejected_by,
            "rejection_reason": self.rejection_reason,
            "gray_scenarios": self.gray_scenarios,
            "gray_devices": self.gray_devices,
            "supersedes": self.supersedes,
            "lineage": self.lineage,
        }


@dataclass
class RollbackPoint:
    """一次版本切换的可执行回退点。"""
    rollback_id: str
    family: str
    from_model_id: str                # 切换前承接流量的版本
    to_model_id: str                  # 切换后承接流量的版本
    actor: str
    ts: int
    executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "family": self.family,
            "from_model_id": self.from_model_id,
            "to_model_id": self.to_model_id,
            "actor": self.actor,
            "ts": self.ts,
            "executed": self.executed,
        }


# ---------------------------------------------------------------------------
# 推理病例
# ---------------------------------------------------------------------------

@dataclass
class InferenceRecord:
    inference_id: str
    idempotency_key: str
    case_id: str
    model_id: str
    model_version: str
    package_hash: str
    image_hash: str
    image_hash_alg: str
    scenario: str
    device: str
    parameters: dict[str, Any]
    prompt: str
    cue: dict[str, Any]              # 模型诊断提示（输出）
    cue_ts: int
    review: dict[str, Any] | None = None
    issued: bool = False             # 医生复核后是否已签发报告
    issued_ts: int | None = None
    request_ts: int = field(default_factory=now_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inference_id": self.inference_id,
            "case_id": self.case_id,
            "idempotent": True,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "package_hash": self.package_hash,
            "image_hash": self.image_hash,
            "image_hash_alg": self.image_hash_alg,
            "scenario": self.scenario,
            "device": self.device,
            "parameters": self.parameters,
            "prompt": self.prompt,
            "cue": self.cue,
            "cue_ts": self.cue_ts,
            "review": self.review,
            "issued": self.issued,
            "issued_ts": self.issued_ts,
        }


# ---------------------------------------------------------------------------
# 登记校验
# ---------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def _validate_registration(payload: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(payload, dict), "请求体必须是对象")

    name = str(payload.get("name", "")).strip()
    version = str(payload.get("version", "")).strip()
    _require(name, "缺少模型名称 name")
    _require(version, "缺少模型版本 version")

    package_hash = str(payload.get("package_hash", "")).strip().lower()
    _require(bool(package_hash), "缺少模型包哈希 package_hash")
    _require(len(package_hash) == 64 and all(c in "0123456789abcdef" for c in package_hash),
             "package_hash 必须为 64 位 sha256 十六进制")

    training = str(payload.get("training_data_source", "")).strip()
    _require(bool(training), "必须提供训练数据来源说明 training_data_source")

    organs = payload.get("organs")
    conditions = payload.get("conditions")
    _require(isinstance(organs, list) and all(isinstance(x, str) and x.strip() for x in organs),
             "organs 必须是非空字符串列表")
    _require(isinstance(conditions, list) and len(conditions) >= 1
             and all(isinstance(x, str) and x.strip() for x in conditions),
             "conditions 必须是病种字符串列表（至少 1 项）")

    sv = payload.get("scenario_validation")
    _require(isinstance(sv, dict), "必须提供 scenario_validation 分场景验证结果")
    missing = [s for s in SCENARIOS if s not in sv or not isinstance(sv[s], dict)]
    _require(not missing, f"分场景验证缺少: {', '.join(missing)}（整体指标不能替代）")
    for scenario, metrics in sv.items():
        _require(scenario in SCENARIOS, f"未知场景: {scenario}")
        for key in ("sensitivity", "specificity"):
            value = metrics.get(key)
            _require(isinstance(value, (int, float)) and 0 <= float(value) <= 1,
                     f"场景「{scenario}」的 {key} 必须是 0~1 的数值")
        sample = metrics.get("sample_size")
        _require(sample is None or (isinstance(sample, int) and sample > 0),
                 f"场景「{scenario}」的 sample_size 必须为正整数")

    contra = payload.get("contraindications", [])
    _require(isinstance(contra, list) and all(isinstance(x, str) and x.strip() for x in contra),
             "contraindications 必须是字符串列表（可为空）")

    return {
        "name": name,
        "version": version,
        "package_hash": package_hash,
        "training_data": training,
        "organs": [x.strip() for x in organs],
        "conditions": [x.strip() for x in conditions],
        "scenario_validation": {s: dict(sv[s]) for s in SCENARIOS},
        "contraindications": [x.strip() for x in contra],
    }


# ---------------------------------------------------------------------------
# 注册表（应用服务）
# ---------------------------------------------------------------------------

class ModelRegistry:
    def __init__(self, *, scorer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                 clock: Callable[[], int] = now_ms):
        self._lock = threading.RLock()
        self._models: dict[str, ModelRecord] = {}
        self._by_family_version: dict[tuple[str, str], str] = {}
        self._active_by_family: dict[str, str] = {}      # 承接灰度流量的版本
        self._rollback_points: list[RollbackPoint] = []
        self._inferences: dict[str, InferenceRecord] = {}
        self._idempotency: dict[str, str] = {}           # key -> inference_id
        self.audit = AuditLog()
        self._scorer = scorer or self._default_scorer
        self._clock = clock

    # -- 工具 --------------------------------------------------------------

    def _audit(self, actor: str, action: str, target: str, summary: str,
               linked: dict[str, Any] | None = None,
               prior: str | None = None, new: str | None = None) -> None:
        self.audit.append(AuditEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            ts=self._clock(), actor=actor, action=action, target=target,
            summary=summary, linked=linked or {}, prior_state=prior, new_state=new,
        ))

    @staticmethod
    def _family_of(name: str) -> str:
        return hashlib.sha256(name.strip().encode("utf-8")).hexdigest()[:16]

    def _get(self, model_id: str) -> ModelRecord:
        record = self._models.get(model_id)
        if record is None:
            raise NotFoundError(f"模型版本不存在: {model_id}")
        return record

    # -- 登记 --------------------------------------------------------------

    def register_model(self, payload: dict[str, Any], *, actor: str) -> ModelRecord:
        data = _validate_registration(payload)
        with self._lock:
            family = self._family_of(data["name"])
            fv = (family, data["version"])
            if fv in self._by_family_version:
                raise ConflictError(f"模型 {data['name']} 版本 {data['version']} 已登记")
            # 同包哈希重复登记一般是误操作（不同版本号也需提示）。
            for existing in self._models.values():
                if existing.package_hash == data["package_hash"]:
                    raise ConflictError(
                        f"模型包哈希与 {existing.model_id}（{existing.name} {existing.version}）重复")

            model_id = f"mdl_{uuid.uuid4().hex[:12]}"
            lineage = [model_id]
            record = ModelRecord(
                model_id=model_id, family=family, lineage=lineage,
                registered_by=actor, registered_ts=self._clock(), **data,
            )
            self._models[model_id] = record
            self._by_family_version[fv] = model_id
            self._audit(actor, "model.register", model_id,
                        f"登记 {data['name']} {data['version']}，覆盖 "
                        f"{len(record.organs)} 个器官、{len(record.conditions)} 个病种，"
                        f"分场景验证 {len(record.scenario_validation)} 组",
                        linked={"model_id": model_id, "package_hash": record.package_hash,
                                "version": record.version},
                        new=ST_DRAFT)
            return record

    # -- 审批 --------------------------------------------------------------

    def approve_model(self, model_id: str, *, actor: str, comment: str = "") -> ModelRecord:
        with self._lock:
            record = self._get(model_id)
            if record.state == ST_REJECTED:
                raise StateError("该版本已被驳回，不能审批通过；请重新登记新版本")
            if record.state != ST_DRAFT:
                raise StateError(f"当前状态 {record.state} 不可审批（仅 draft 可审批）")
            record.state = ST_APPROVED
            record.approved_by = actor
            record.approved_ts = self._clock()
            record.approval_comment = comment or None
            self._audit(actor, "model.approve", model_id,
                        f"批准 {record.name} {record.version} 进入准入流程",
                        linked={"model_id": model_id, "package_hash": record.package_hash,
                                "version": record.version},
                        prior=ST_DRAFT, new=ST_APPROVED)
            return record

    def reject_model(self, model_id: str, *, actor: str, reason: str) -> ModelRecord:
        with self._lock:
            record = self._get(model_id)
            if record.state not in (ST_DRAFT,):
                raise StateError(f"当前状态 {record.state} 不可驳回")
            _require(bool(reason.strip()), "驳回必须填写 reason")
            record.state = ST_REJECTED
            record.rejected_by = actor
            record.rejection_reason = reason.strip()
            self._audit(actor, "model.reject", model_id,
                        f"驳回 {record.name} {record.version}：{reason.strip()}",
                        linked={"model_id": model_id}, prior=ST_DRAFT, new=ST_REJECTED)
            return record

    # -- 灰度 --------------------------------------------------------------

    def enter_gray(self, model_id: str, *, actor: str,
                   scenarios: list[str] | None = None,
                   devices: list[str] | None = None) -> ModelRecord:
        with self._lock:
            record = self._get(model_id)
            if record.state not in (ST_APPROVED, ST_GRAY):
                raise StateError(f"当前状态 {record.state}，仅 approved 可进入限定灰度")
            sc = list(scenarios or [])
            dv = [str(d).strip() for d in (devices or []) if str(d).strip()]
            bad = [s for s in sc if s not in SCENARIOS]
            _require(not bad, f"灰度场景未在验证范围内: {', '.join(bad)}")
            _require(sc, "灰度必须限定至少一个场景 scenarios")
            prior = record.state
            record.gray_scenarios = sc
            record.gray_devices = dv
            record.state = ST_GRAY
            if not self._active_by_family.get(record.family):
                self._active_by_family[record.family] = record.model_id
            self._audit(actor, "gray.enter", model_id,
                        f"进入限定灰度：场景 {sc}" + (f"，设备 {dv}" if dv else "（不限设备）"),
                        linked={"model_id": model_id, "scenarios": sc, "devices": dv},
                        prior=prior, new=ST_GRAY)
            return record

    # -- 版本切换 / 回退 ----------------------------------------------------

    def switch_version(self, model_id: str, *, actor: str) -> RollbackPoint:
        """将灰度流量切换到新版本；旧版本归档但保留，返回可执行回退点。"""
        with self._lock:
            new = self._get(model_id)
            if new.state != ST_GRAY:
                raise StateError("只有灰度中的版本可被切换为当前版本")
            old_id = self._active_by_family.get(new.family)
            if old_id == new.model_id:
                raise ConflictError("该版本已经是当前灰度版本")
            old = self._get(old_id) if old_id else None
            if old:
                old.state = ST_ARCHIVED
            new.state = ST_GRAY
            self._active_by_family[new.family] = new.model_id
            new.supersedes = old.model_id if old else None
            if old:
                merged = list(dict.fromkeys([*old.lineage, *new.lineage, new.model_id]))
            else:
                merged = list(dict.fromkeys([*new.lineage, new.model_id]))
            new.lineage = merged

            point = RollbackPoint(
                rollback_id=f"rb_{uuid.uuid4().hex[:12]}",
                family=new.family,
                from_model_id=old.model_id if old else "",
                to_model_id=new.model_id,
                actor=actor, ts=self._clock(),
            )
            self._rollback_points.append(point)
            self._audit(actor, "version.switch", new.model_id,
                        f"灰度版本切换" + (f"：{old.version} → {new.version}" if old else f"：启用 {new.version}"),
                        linked={"model_id": new.model_id, "rollback_id": point.rollback_id,
                                "from": point.from_model_id, "to": point.to_model_id},
                        prior=old.state if old else None, new=ST_GRAY)
            return point

    def rollback(self, rollback_id: str, *, actor: str) -> ModelRecord:
        """执行回退点：恢复旧版本承接新请求，新版本归档。"""
        with self._lock:
            point = next((p for p in self._rollback_points if p.rollback_id == rollback_id), None)
            if point is None:
                raise NotFoundError(f"回退点不存在: {rollback_id}")
            if point.executed:
                raise ConflictError("该回退点已执行；如需再次回退请使用切换产生的新回退点")
            if not point.from_model_id:
                raise StateError("该回退点没有可回退的旧版本（家族首次启用）")
            old = self._get(point.from_model_id)
            new = self._get(point.to_model_id)

            new.state = ST_ARCHIVED
            old.state = ST_GRAY
            self._active_by_family[point.family] = old.model_id
            point.executed = True

            # 回退本身也再生成一个回退点（可“前滚”回新版本）。
            forward = RollbackPoint(
                rollback_id=f"rb_{uuid.uuid4().hex[:12]}",
                family=point.family, from_model_id=new.model_id, to_model_id=old.model_id,
                actor=actor, ts=self._clock(),
            )
            self._rollback_points.append(forward)
            self._audit(actor, "version.rollback", old.model_id,
                        f"执行回退 {rollback_id}：{new.version} → {old.version}",
                        linked={"model_id": old.model_id, "rollback_id": rollback_id,
                                "from": new.model_id, "to": old.model_id,
                                "new_rollback_id": forward.rollback_id},
                        prior=ST_ARCHIVED, new=ST_GRAY)
            return old

    def list_rollback_points(self, family_name: str | None = None) -> list[RollbackPoint]:
        with self._lock:
            points = list(self._rollback_points)
        if family_name:
            fam = self._family_of(family_name)
            points = [p for p in points if p.family == fam]
        return points

    # -- 冻结 --------------------------------------------------------------

    def freeze_model(self, model_id: str, *, actor: str, reason: str) -> ModelRecord:
        """指标下降等场景冻结新请求；已签发/已存在报告保持可读、不受影响。"""
        with self._lock:
            record = self._get(model_id)
            if record.state not in (ST_GRAY, ST_APPROVED, ST_SUSPENDED):
                raise StateError(f"当前状态 {record.state} 不可冻结")
            _require(bool(reason.strip()), "冻结必须填写 reason")
            prior = record.state
            record.state = ST_SUSPENDED
            self._audit(actor, "model.freeze", model_id,
                        f"冻结新请求：{reason.strip()}（已签发报告不受影响）",
                        linked={"model_id": model_id}, prior=prior, new=ST_SUSPENDED)
            return record

    def resume_model(self, model_id: str, *, actor: str, reason: str = "") -> ModelRecord:
        with self._lock:
            record = self._get(model_id)
            if record.state != ST_SUSPENDED:
                raise StateError("仅 suspended 状态可恢复")
            record.state = ST_GRAY
            self._audit(actor, "model.resume", model_id,
                        f"恢复新请求" + (f"：{reason.strip()}" if reason.strip() else ""),
                        linked={"model_id": model_id}, prior=ST_SUSPENDED, new=ST_GRAY)
            return record

    # -- 推理 --------------------------------------------------------------

    @staticmethod
    def _strip_identities(payload: dict[str, Any]) -> None:
        """进入推理链路前剥离直接身份信息；命中即拒绝，杜绝带身份入库。"""
        for key in DIRECT_IDENTIFIERS:
            if key in payload:
                raise RegistryError(f"检测到直接身份字段 {key}，必须在调用前剥离")
        for key, value in payload.items():
            lk = key.lower()
            if lk in DIRECT_IDENTIFIERS:
                raise RegistryError(f"检测到直接身份字段 {key}，必须在调用前剥离")
            if isinstance(value, dict):
                ModelRegistry._strip_identities(value)

    @staticmethod
    def _default_scorer(request: dict[str, Any]) -> dict[str, Any]:
        """占位推理：实际部署由模型适配器替换；输出必须带版本无关的稳定结构。"""
        digest = hashlib.sha256(
            (request["image_hash"] + request["model_id"] + json.dumps(request["parameters"], sort_keys=True)
             + request["prompt"]).encode("utf-8")
        ).hexdigest()
        rank = 0.5 + (int(digest[:8], 16) % 500) / 1000.0
        return {"finding": "疑似腹部异常（占位输出）", "confidence": round(rank, 4),
                "differentials": ["待人工判读"]}

    def infer(self, request: dict[str, Any], *, actor: str,
              idempotency_key: str | None = None) -> tuple[InferenceRecord, bool]:
        """发起一次推理。

        返回 (记录, 是否复用既有结果)。相同 idempotency_key 的重复上传/网络重试
        只产生一个病例结果。
        """
        with self._lock:
            self._strip_identities(request)

            key = (idempotency_key or "").strip() or None
            if key and key in self._idempotency:
                return self._inferences[self._idempotency[key]], True

            model_id = str(request.get("model_id", "")).strip()
            image_hash = str(request.get("image_hash", "")).strip().lower()
            scenario = str(request.get("scenario", "")).strip()
            device = str(request.get("device", "")).strip()
            prompt = str(request.get("prompt", "")).strip()
            parameters = request.get("parameters", {}) or {}

            _require(model_id, "缺少 model_id")
            _require(len(image_hash) == 64 and all(c in "0123456789abcdef" for c in image_hash),
                     "image_hash 必须为 64 位 sha256（影像须先在本地去身份后取哈希）")
            _require(scenario in SCENARIOS, f"scenario 必须是 {SCENARIOS} 之一")
            _require(prompt, "缺少 prompt（给模型的提示词必须留痕）")
            _require(isinstance(parameters, dict), "parameters 必须是对象")

            record_model = self._get(model_id)
            if record_model.state == ST_SUSPENDED:
                raise StateError(f"模型 {record_model.version} 已冻结，暂不接受新请求（已签发报告不受影响）")
            if record_model.state != ST_GRAY:
                raise StateError(f"模型状态 {record_model.state}，仅灰度版本可承接推理")
            active_id = self._active_by_family.get(record_model.family)
            if active_id != model_id:
                active = self._models.get(active_id) if active_id else None
                raise StateError(
                    "该版本不是当前承接流量的灰度版本"
                    + (f"（当前为 {active.version}）" if active else "（家族尚无生效版本）"))
            if record_model.gray_scenarios and scenario not in record_model.gray_scenarios:
                raise StateError(f"场景 {scenario} 不在该版本灰度范围 {record_model.gray_scenarios}")
            if record_model.gray_devices and device not in record_model.gray_devices:
                raise StateError(f"设备 {device or '(未标注)'} 不在该版本灰度设备范围")
            # 禁用条件由调用方（网关/工作站）按登记清单显式判定后上报命中项。
            hits = request.get("contraindication_hits", []) or []
            _require(isinstance(hits, list) and all(isinstance(h, str) for h in hits),
                     "contraindication_hits 必须是字符串列表")
            blocked = [h for h in hits if h in record_model.contraindications]
            if blocked:
                raise StateError(f"命中禁用条件，禁止推理: {', '.join(blocked)}")

            cue = self._scorer({"image_hash": image_hash, "model_id": model_id,
                                "parameters": parameters, "prompt": prompt})

            inference_id = f"inf_{uuid.uuid4().hex[:12]}"
            case_id = f"case_{uuid.uuid4().hex[:12]}"
            inf = InferenceRecord(
                inference_id=inference_id, idempotency_key=key or inference_id,
                case_id=case_id, model_id=model_id, model_version=record_model.version,
                package_hash=record_model.package_hash, image_hash=image_hash,
                image_hash_alg=ALGORITHM, scenario=scenario, device=device,
                parameters=parameters, prompt=prompt, cue=cue, cue_ts=self._clock(),
            )
            self._inferences[inference_id] = inf
            self._idempotency[inf.idempotency_key] = inference_id
            self._audit(actor, "inference.run", inference_id,
                        f"{scenario} 场景推理完成（{record_model.name} {record_model.version}）",
                        linked={"case_id": case_id, "model_id": model_id,
                                "inference_id": inference_id,
                                "package_hash": record_model.package_hash,
                                "image_hash": image_hash})
            return inf, False

    def submit_review(self, inference_id: str, *, actor: str, conclusion: str,
                      note: str = "", report_id: str | None = None) -> InferenceRecord:
        """绑定医生复核结论（最终人工决定），并可签发报告。"""
        with self._lock:
            inf = self._inferences.get(inference_id)
            if inf is None:
                raise NotFoundError(f"推理记录不存在: {inference_id}")
            _require(conclusion in REVIEW_CONCLUSIONS,
                     f"conclusion 必须是 {REVIEW_CONCLUSIONS} 之一")
            if inf.review is not None:
                raise ConflictError("该病例已有最终复核结论，不可更改（如需更正请重新发起病例）")
            inf.review = {
                "reviewed_by": actor,
                "reviewed_ts": self._clock(),
                "conclusion": conclusion,
                "note": note.strip(),
                "report_id": report_id,
            }
            inf.issued = True
            inf.issued_ts = self._clock()
            self._audit(actor, "review.submit", inference_id,
                        f"医生复核：{conclusion}" + (f"，报告 {report_id}" if report_id else ""),
                        linked={"case_id": inf.case_id, "model_id": inf.model_id,
                                "inference_id": inference_id, "conclusion": conclusion},
                        new="issued" if inf.issued else None)
            return inf

    # -- 查询 / 审计链 ------------------------------------------------------

    def get_model(self, model_id: str) -> ModelRecord:
        with self._lock:
            return self._get(model_id)

    def list_models(self) -> list[ModelRecord]:
        with self._lock:
            return sorted(self._models.values(), key=lambda r: r.registered_ts)

    def active_model(self, family_name: str) -> ModelRecord | None:
        with self._lock:
            mid = self._active_by_family.get(self._family_of(family_name))
            return self._models.get(mid) if mid else None

    def get_inference(self, inference_id: str) -> InferenceRecord:
        with self._lock:
            inf = self._inferences.get(inference_id)
            if inf is None:
                raise NotFoundError(f"推理记录不存在: {inference_id}")
            return inf

    def trace_case(self, case_id: str) -> dict[str, Any]:
        """审计追溯：从病例（诊断提示）→ 模型版本/验证证据/批准者 → 人工决定。"""
        with self._lock:
            events = self.audit.for_case(case_id)
            inf = next((i for i in self._inferences.values() if i.case_id == case_id), None)
            if inf is None and not events:
                raise NotFoundError(f"病例不存在: {case_id}")
            chain: dict[str, Any] = {"case_id": case_id, "events": [e.to_dict() for e in events]}
            if inf is not None:
                model = self._models[inf.model_id]
                chain["inference"] = inf.to_dict()
                chain["model"] = {
                    "model_id": model.model_id,
                    "name": model.name,
                    "version": model.version,
                    "state": model.state,
                    "package_hash": model.package_hash,
                }
                chain["validation_evidence"] = model.scenario_validation.get(inf.scenario)
                chain["approval"] = {
                    "approved_by": model.approved_by,
                    "approved_ts": model.approved_ts,
                    "comment": model.approval_comment,
                }
                chain["human_decision"] = inf.review
            return chain

    def audit_events(self) -> list[AuditEvent]:
        with self._lock:
            return self.audit.all()


# ---------------------------------------------------------------------------
# 运维日志脱敏
# ---------------------------------------------------------------------------

def redact_for_ops(obj: Any) -> Any:
    """生成普通运维可见的日志视图：擦除身份与影像内容，仅保留技术元数据。

    - 任何疑似身份键（name/id_card/phone/address/姓名…）一律擦除；
    - 影像本体（image 字节/base64）不落日志，只保留 image_hash；
    - prompt/note 等自由文本可能携带身份，运维视图中隐藏。
    """
    REDACTED_KEYS = {"prompt", "note", "image", "image_b64", "image_data", "dicom",
                     "training_data_source", "finding", "differentials"}
    if isinstance(obj, dict):
        clean: dict[str, Any] = {}
        for key, value in obj.items():
            lk = str(key).lower()
            if lk in REDACTED_KEYS or any(lk.endswith(s) for s in SENSITIVE_SUFFIXES):
                clean[key] = "<redacted>"
            else:
                clean[key] = redact_for_ops(value)
        return clean
    if isinstance(obj, list):
        return [redact_for_ops(x) for x in obj]
    return obj
