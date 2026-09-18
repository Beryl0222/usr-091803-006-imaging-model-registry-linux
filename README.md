# 医疗影像模型准入

本项目服务于医疗影像辅助模型的院内准入。模型包、脱敏影像、验证指标、灰度范围和医生复核必须可以相互追溯。 系统应支持清晰的领域对象、事件记录和责任追溯，运行入口提供稳定的健康检查，便于本地联调和运维巡检。

## 运行与自检

- `python3 service.py --check`：基础配置检查，并用一次性实例跑通 登记→审批→灰度→推理→复核→追溯 的最小链路。
- `python3 service.py --port 8000`：启动服务，`GET /health` 返回服务身份。
- `npm test`：运行契约测试（service_contract）与领域测试（test_admission）。

## 模块

- `domain.py`：领域对象与不变量（登记必填项、分场景验证结构、直接身份剥离、状态机）。
- `admission.py`：准入业务流程与内存存储；所有状态变更追加审计事件。
- `service.py`：HTTP 入口、路由与角色鉴权。

## 角色

| 角色 | 职责 |
| --- | --- |
| `registrar` | 准入登记、激活、冻结/解冻、指标观测、回退 |
| `approver` | 科室审批，给出灰度场景范围与可选病例上限 |
| `doctor` | 发起推理、签发复核结论 |
| `auditor` | 查询病例追溯与审计事件 |
| `ops` | 仅 `/health`；临床数据与审计日志不开放 |

写操作需同时携带 `X-Role` 与 `X-Actor`（操作人）请求头，否则返回 401；角色不符返回 403。

## 接口

| 方法 | 路径 | 角色 | 说明 |
| --- | --- | --- | --- |
| GET | `/health` | 任意 | 服务身份 |
| POST | `/models` | registrar | 登记模型包 |
| GET | `/models` | 读角色 | 模型包列表 |
| GET | `/models/{id}` | 读角色 | 详情（分场景验证、禁用条件、审批） |
| POST | `/models/{id}/approve` | approver | 科室审批（灰度范围、病例上限） |
| POST | `/models/{id}/activate` | registrar | 激活进入限定灰度 |
| POST | `/models/{id}/freeze` | registrar | 人工冻结新请求 |
| POST | `/models/{id}/unfreeze` | registrar | 解冻恢复 |
| POST | `/models/{id}/metrics` | registrar | 登记观测指标，下降超阈值自动冻结 |
| GET | `/channels/{name}` | registrar/approver/auditor | 通道状态与可执行回退点 |
| POST | `/channels/{name}/rollback` | registrar | 回退到上一生效版本 |
| POST | `/inferences` | doctor | 推理（`Idempotency-Key` 支持重试去重） |
| GET | `/cases` | auditor | 病例列表（脱敏摘要） |
| GET | `/cases/{id}` | doctor/auditor | 病例详情 |
| POST | `/cases/{id}/review` | doctor | 医生复核（最终结论仅一次） |
| GET | `/cases/{id}/trace` | auditor | 追溯：模型、验证证据、批准者、人工决定 |
| GET | `/audit/events` | auditor | 审计事件流 |

## 关键规则

- **登记**：包哈希（SHA-256）、训练数据来源说明、器官与病种覆盖、分场景验证结果、禁用条件缺一不可；分场景验证必须覆盖 `routine`（常规检查）、`acute_abdomen`（急腹症）与至少一个 `device:*` 设备场景，且每个场景含灵敏度/特异度、样本量与数据集说明。相同哈希或同名同版本拒绝重复登记。
- **审批与灰度**：灰度场景必须已有分场景验证证据；科室审批后准入方激活才进入限定灰度，可用 `max_cases` 限制灰度规模。
- **版本切换**：通道保留激活栈作为可执行回退点；回退恢复上一生效版本并记录审计事件。
- **推理**：先剥离直接身份字段（patient/姓名/证件号/病历号等），再绑定影像哈希、参数、提示与模型版本；幂等键或相同请求指纹的重复上传与网络重试只产生一个病例。
- **冻结**：观测指标相对登记验证值下降超过 0.05 自动冻结；冻结只拦截新请求，已签发报告与复核不受影响；解冻需人工执行。
- **审计**：事件只记录操作者与哈希引用；审计角色可从任一诊断提示追到模型包、分场景验证证据、批准者与最终人工决定；ops 角色无法经接口或日志接触患者影像与身份。
