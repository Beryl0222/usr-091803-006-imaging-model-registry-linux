"""准入业务流程：登记、审批、灰度、推理绑定、复核、冻结与回退。

所有状态变更都追加审计事件；推理请求先脱敏再绑定；
幂等键与请求指纹保证重复上传与网络重试只产生一个病例。
"""

from __future__ import annotations

import hashlib
import threading

from domain import (
    METRIC_DROP_TOLERANCE,
    REQUIRED_METRICS,
    REVIEW_CONCLUSIONS,
    Approval,
    AuditEvent,
    CaseRecord,
    Channel,
    DomainError,
    ModelPackage,
    canonical_hash,
    new_id,
    now_iso,
    require_hash256,
    strip_identity,
)


def _stub_result(package, image_hash, params):
    """占位推理引擎：由影像哈希与参数确定性地生成结果。

    真实引擎接入后替换本函数；结果结构保持 {"engine", "findings"}。
    请求方也可直接携带引擎输出（findings 字段）登记绑定。
    """
    seed = hashlib.sha256((image_hash + canonical_hash(params)).encode("utf-8")).digest()
    pool = package.disease_coverage or ["未见明确异常"]
    findings = [
        {
            "finding": pool[seed[index] % len(pool)],
            "confidence": round(0.5 + seed[16 + index] / 510.0, 3),
        }
        for index in range(min(2, len(pool)))
    ]
    return {"engine": "stub-v0", "findings": findings}


class AdmissionService:
    """内存版准入服务：登记、审批、灰度、推理、复核、冻结与回退。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._packages = {}
        self._channels = {}
        self._cases = {}
        self._audit = []
        self._idempotency = {}
        self._fingerprints = {}

    # ---- 内部工具 ----

    def _record(self, actor, role, action, refs, detail=None):
        event = AuditEvent(new_id("evt"), now_iso(), actor, role, action, refs, detail or {})
        self._audit.append(event)
        return event

    def _must_package(self, package_id):
        package = self._packages.get(package_id)
        if package is None:
            raise DomainError("模型包不存在", 404)
        return package

    def _must_case(self, case_id):
        case = self._cases.get(case_id)
        if case is None:
            raise DomainError("病例不存在", 404)
        return case

    # ---- 登记与审批 ----

    def register_package(self, actor, role, payload):
        """登记模型包：哈希、训练来源、覆盖范围、分场景验证与禁用条件缺一不可。"""
        with self._lock:
            package = ModelPackage.from_payload(payload, actor)
            for existing in self._packages.values():
                if existing.package_hash == package.package_hash:
                    raise DomainError("相同哈希的模型包已登记", 409)
                if existing.name == package.name and existing.version == package.version:
                    raise DomainError("同名同版本的模型包已登记", 409)
            self._packages[package.package_id] = package
            self._record(
                actor, role, "package.registered",
                {"package_id": package.package_id},
                {
                    "name": package.name,
                    "version": package.version,
                    "package_hash": package.package_hash,
                    "scenarios": sorted(package.validation),
                },
            )
            return package

    def approve_package(self, actor, role, package_id, payload):
        """科室审批：灰度场景必须已有分场景验证证据，可附加病例上限。"""
        with self._lock:
            package = self._must_package(package_id)
            if package.status != "registered":
                raise DomainError("仅登记状态的模型包可以审批", 409)
            department = payload.get("department")
            if not isinstance(department, str) or not department.strip():
                raise DomainError("缺少审批科室 department")
            scope = payload.get("scope")
            if not isinstance(scope, dict):
                raise DomainError("缺少灰度范围 scope")
            scenarios = scope.get("scenarios")
            if not isinstance(scenarios, list) or not scenarios:
                raise DomainError("灰度范围必须给出场景列表 scenarios")
            unknown = sorted({item for item in scenarios if item not in package.validation})
            if unknown:
                raise DomainError("灰度场景缺少分场景验证证据: " + ", ".join(unknown))
            max_cases = scope.get("max_cases")
            if max_cases is not None and (
                not isinstance(max_cases, int) or isinstance(max_cases, bool) or max_cases <= 0
            ):
                raise DomainError("max_cases 必须是正整数")
            package.approval = Approval(
                approver=actor,
                department=department.strip(),
                scenarios=list(scenarios),
                max_cases=max_cases,
                note=payload.get("note", ""),
                decided_at=now_iso(),
            )
            package.status = "approved"
            self._record(
                actor, role, "package.approved",
                {"package_id": package_id},
                {
                    "department": department.strip(),
                    "scenarios": list(scenarios),
                    "max_cases": max_cases,
                },
            )
            return package

    # ---- 灰度与版本切换 ----

    def activate_package(self, actor, role, package_id):
        """审批通过后激活进入限定灰度，并在通道上留下回退点。"""
        with self._lock:
            package = self._must_package(package_id)
            if package.status != "approved":
                raise DomainError("模型包未审批或当前状态不允许激活", 409)
            channel = self._channels.setdefault(package.name, Channel(package.name))
            if channel.active_package_id == package_id:
                raise DomainError("该版本已在灰度中生效", 409)
            previous_id = channel.active_package_id
            if previous_id is not None:
                previous = self._packages[previous_id]
                if previous.status == "grayscale":
                    previous.status = "approved"
            channel.activations.append(package_id)
            package.status = "grayscale"
            self._record(
                actor, role, "package.activated",
                {"package_id": package_id},
                {"channel": package.name, "previous_package_id": previous_id},
            )
            return package, channel

    def rollback_channel(self, actor, role, name, reason):
        """执行回退：通道恢复到上一生效版本，回退本身也留下记录。"""
        with self._lock:
            channel = self._channels.get(name)
            if channel is None:
                raise DomainError("模型通道不存在", 404)
            if len(channel.activations) < 2:
                raise DomainError("没有可执行的回退点", 409)
            current_id = channel.activations.pop()
            restored_id = channel.activations[-1]
            current = self._packages[current_id]
            if current.status == "grayscale":
                current.status = "approved"
            restored = self._packages[restored_id]
            if restored.status == "approved":
                restored.status = "grayscale"
            self._record(
                actor, role, "channel.rollback",
                {"package_id": restored_id},
                {
                    "channel": name,
                    "from_package_id": current_id,
                    "to_package_id": restored_id,
                    "reason": reason,
                },
            )
            return channel, current_id, restored_id

    def channel_state(self, name):
        with self._lock:
            channel = self._channels.get(name)
            if channel is None:
                raise DomainError("模型通道不存在", 404)
            return {
                "name": name,
                "active_package_id": channel.active_package_id,
                "rollback_points": [
                    {
                        "package_id": package_id,
                        "version": self._packages[package_id].version,
                        "status": self._packages[package_id].status,
                    }
                    for package_id in channel.activations
                ],
            }

    # ---- 冻结与指标 ----

    def freeze_package(self, actor, role, package_id, reason, automatic=False):
        """冻结只拦截新请求，已签发报告与复核流程不受影响。"""
        with self._lock:
            package = self._must_package(package_id)
            if package.status == "frozen":
                raise DomainError("模型包已处于冻结状态", 409)
            if package.status != "grayscale":
                raise DomainError("仅灰度中的模型包可以冻结", 409)
            package.status = "frozen"
            action = "package.auto_frozen" if automatic else "package.frozen"
            self._record(actor, role, action, {"package_id": package_id}, {"reason": reason})
            return package

    def unfreeze_package(self, actor, role, package_id, reason):
        with self._lock:
            package = self._must_package(package_id)
            if package.status != "frozen":
                raise DomainError("模型包未处于冻结状态", 409)
            channel = self._channels.get(package.name)
            if channel is not None and channel.active_package_id == package_id:
                package.status = "grayscale"
            else:
                package.status = "approved"
            self._record(actor, role, "package.unfrozen", {"package_id": package_id}, {"reason": reason})
            return package

    def record_metrics(self, actor, role, package_id, payload):
        """登记观测指标；相对验证值下降超过容忍度时自动冻结新请求。"""
        with self._lock:
            package = self._must_package(package_id)
            scenario = payload.get("scenario")
            entry = package.validation.get(scenario) if isinstance(scenario, str) else None
            if entry is None:
                raise DomainError("观测指标的场景未在登记的分场景验证中")
            observed = payload.get("metrics")
            if not isinstance(observed, dict):
                raise DomainError("缺少观测指标 metrics")
            drops = {}
            for metric in REQUIRED_METRICS:
                value = observed.get(metric)
                if value is None:
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
                    raise DomainError(f"观测指标 {metric} 必须落在 [0,1]")
                drop = round(entry["metrics"][metric] - value, 4)
                if drop > METRIC_DROP_TOLERANCE:
                    drops[metric] = drop
            degraded = bool(drops)
            self._record(
                actor, role, "metrics.observed",
                {"package_id": package_id},
                {"scenario": scenario, "degraded": degraded, "drops": drops},
            )
            if degraded and package.status == "grayscale":
                package.status = "frozen"
                self._record(
                    actor, role, "package.auto_frozen",
                    {"package_id": package_id},
                    {"reason": "观测指标下降超过容忍度", "scenario": scenario, "drops": drops},
                )
            return {
                "package_id": package_id,
                "scenario": scenario,
                "degraded": degraded,
                "drops": drops,
                "package_status": package.status,
            }

    # ---- 推理与复核 ----

    def submit_inference(self, actor, role, payload, idempotency_key=None):
        """推理入口：先剥离直接身份，再绑定影像哈希、参数、提示与模型版本。

        幂等键或相同请求指纹的重复上传与网络重试返回同一病例；
        幂等判定先于冻结检查，进行中的重试不受冻结影响。
        """
        with self._lock:
            if not isinstance(payload, dict):
                raise DomainError("请求体必须是 JSON 对象")
            if idempotency_key is not None and not str(idempotency_key).strip():
                idempotency_key = None
            clean, removed = strip_identity(payload)
            image_hash = clean.get("image_hash")
            require_hash256(image_hash, "image_hash")
            model_name = clean.get("model")
            if not isinstance(model_name, str) or not model_name.strip():
                raise DomainError("缺少必填字段: model")
            model_name = model_name.strip()
            scenario = clean.get("scenario")
            if not isinstance(scenario, str) or not scenario.strip():
                raise DomainError("缺少必填字段: scenario")
            scenario = scenario.strip()
            prompt = clean.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise DomainError("缺少必填字段: prompt")
            params = clean.get("params") or {}
            if not isinstance(params, dict):
                raise DomainError("params 必须是 JSON 对象")
            fingerprint = canonical_hash({
                "image_hash": image_hash,
                "model": model_name,
                "scenario": scenario,
                "params": params,
                "prompt": prompt,
            })
            if idempotency_key is not None:
                seen = self._idempotency.get(idempotency_key)
                if seen is not None:
                    if seen[0] != fingerprint:
                        raise DomainError("幂等键已被不同请求占用", 409)
                    return self._cases[seen[1]], True
            existing = self._fingerprints.get(fingerprint)
            if existing is not None:
                return self._cases[existing], True
            channel = self._channels.get(model_name)
            package = self._packages.get(channel.active_package_id) if channel else None
            if package is None:
                raise DomainError("该模型没有生效中的灰度版本", 409)
            if package.status == "frozen":
                raise DomainError("模型已冻结，新请求暂停；已签发报告不受影响", 409)
            if package.status != "grayscale":
                raise DomainError("模型未处于灰度服务状态", 409)
            if scenario not in package.approval.scenarios:
                raise DomainError("场景不在批准的灰度范围内", 403)
            if package.approval.max_cases is not None:
                used = sum(1 for case in self._cases.values() if case.package_id == package.package_id)
                if used >= package.approval.max_cases:
                    raise DomainError("灰度病例数已达上限", 409)
            result = clean.get("findings")
            if not isinstance(result, dict) or not result:
                result = _stub_result(package, image_hash, params)
            case = CaseRecord(
                case_id=new_id("case"),
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                image_hash=image_hash,
                model_name=model_name,
                package_id=package.package_id,
                model_version=package.version,
                scenario=scenario,
                params=params,
                prompt=prompt.strip(),
                result=result,
                created_by=actor,
                created_at=now_iso(),
            )
            self._cases[case.case_id] = case
            self._fingerprints[fingerprint] = case.case_id
            if idempotency_key is not None:
                self._idempotency[idempotency_key] = (fingerprint, case.case_id)
            self._record(
                actor, role, "inference.completed",
                {"case_id": case.case_id, "package_id": package.package_id},
                {
                    "image_hash": image_hash,
                    "model": model_name,
                    "scenario": scenario,
                    "deidentified_fields": sorted(removed),
                },
            )
            return case, False

    def review_case(self, actor, role, case_id, payload):
        """医生复核：最终人工决定只允许签发一次。"""
        with self._lock:
            case = self._must_case(case_id)
            if case.review is not None:
                raise DomainError("病例已有最终复核结论", 409)
            conclusion = payload.get("conclusion")
            if conclusion not in REVIEW_CONCLUSIONS:
                raise DomainError("复核结论必须是 agree/modified/rejected 之一")
            case.review = {
                "reviewer": actor,
                "conclusion": conclusion,
                "notes": payload.get("notes", ""),
                "at": now_iso(),
            }
            case.status = "reviewed"
            self._record(
                actor, role, "case.reviewed",
                {"case_id": case_id, "package_id": case.package_id},
                {"conclusion": conclusion},
            )
            return case

    # ---- 查询与追溯 ----

    def get_package(self, package_id):
        with self._lock:
            return self._must_package(package_id).to_dict()

    def list_packages(self):
        with self._lock:
            return [
                {
                    "package_id": package.package_id,
                    "name": package.name,
                    "version": package.version,
                    "package_hash": package.package_hash,
                    "status": package.status,
                }
                for package in self._packages.values()
            ]

    def get_case(self, case_id):
        with self._lock:
            return self._must_case(case_id).to_dict()

    def list_cases(self):
        with self._lock:
            return [
                {
                    "case_id": case.case_id,
                    "model": case.model_name,
                    "model_version": case.model_version,
                    "scenario": case.scenario,
                    "image_hash": case.image_hash,
                    "status": case.status,
                    "created_at": case.created_at,
                }
                for case in self._cases.values()
            ]

    def trace_case(self, case_id):
        """从诊断提示追到模型、验证证据、批准者与最终人工决定。"""
        with self._lock:
            case = self._must_case(case_id)
            package = self._packages[case.package_id]
            events = [
                event.to_dict()
                for event in self._audit
                if case_id in event.refs.values() or package.package_id in event.refs.values()
            ]
            return {
                "case": case.to_dict(),
                "model": package.to_dict(),
                "validation": package.validation,
                "approval": package.approval.to_dict() if package.approval else None,
                "review": case.review,
                "audit_events": events,
            }

    def audit_events(self):
        with self._lock:
            return [event.to_dict() for event in self._audit]
