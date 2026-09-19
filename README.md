# 医疗影像模型准入

放射科通用腹部影像模型（可识别一百多种腹部病症）的**院内准入服务**。模型包、脱敏影像、
分场景验证指标、灰度范围和医生复核结论必须可以相互追溯；运行入口提供稳定的健康检查，
便于本地联调和运维巡检。

## 解决什么问题

论文里的整体指标不足以决定临床使用范围——常规检查、急腹症、不同设备上的表现并不等同；
医生也必须知道屏幕上的提示来自哪一版模型。本服务把准入要求固化为领域规则：

1. **登记必须完整**：模型包 sha256、训练数据来源说明、器官与病种覆盖清单、
   分场景验证结果（常规检查 / 急腹症 / 设备分层，各需敏感性与特异性）、禁用条件，
   缺一不可；整体指标不能替代分场景证据，哈希重复登记被拒绝。
2. **科室审批后才能进入限定灰度**：状态机为 `draft → approved → gray`（另有
   `rejected / suspended / archived`），灰度可限定场景与设备，灰度场景必须在已验证范围内。
3. **版本切换留有可执行回退点**：切换归档旧版本并生成一次性回退点；执行回退即恢复旧版本
   承接新流量，回退同时生成反向回退点。
4. **推理链路全程绑定**：每次推理先拒绝任何直接身份字段（`patient_name/姓名/住院号` 等，
   含嵌套），再绑定影像哈希、模型版本与包哈希、参数、提示词（prompt）和模型输出的诊断提示；
   每条病例结果带模型版本，医生随时可知提示来源。
5. **重复上传与网络重试只产生一个病例结果**：通过 `Idempotency-Key`（或请求体
   `idempotency_key`）去重，重试返回同一 `case_id`/`inference_id`。
6. **指标下降可冻结新请求**：`freeze` 使版本转为 `suspended`，拒绝新推理，但**已签发报告和
   既有病例结果保持可读、可追溯**（冻结不是删除）；`resume` 后恢复。
7. **审计可从诊断提示追到最终人工决定**：`GET /v1/cases/{case_id}/trace` 一次给出
   模型版本与包哈希、该场景的验证证据、批准者与批准时间、医生复核结论（采纳 / 修改后采纳 /
   不采纳）及报告号。
8. **普通运维不能借日志看到患者影像或身份**：`X-Role: ops` 的审计/病例/推理视图经
   `redact_for_ops` 擦除身份字段、prompt、复核备注、影像内容等，仅保留 image_hash、
   设备、状态、参数结构等技术元数据。

## 运行

```bash
python3 service.py --check           # 领域自检（登记→审批→灰度→推理→幂等→切换→回退）
python3 service.py --port 8000       # 启动 HTTP 服务
curl localhost:8000/health           # 稳定服务身份
npm test                             # 运行全部 28 个测试（3 个模块）
```

## HTTP 接口

变更类接口需带 `X-Actor`（操作者工号；HTTP 头只能用 ASCII），可选 `X-Role`
（`doctor`/`auditor`/`ops`，审计与运维视图不同）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 服务身份 |
| POST | `/v1/models` | 登记模型包（哈希/来源/覆盖/分场景验证/禁用条件） |
| GET | `/v1/models` · `/v1/models/{id}` | 模型清单/详情 |
| POST | `/v1/models/{id}/approve` | 科室审批（comment） |
| POST | `/v1/models/{id}/reject` | 驳回（必须填 reason） |
| POST | `/v1/models/{id}/gray` | 进入限定灰度（scenarios 必填，devices 可选） |
| POST | `/v1/models/{id}/switch` | 切换为当前灰度版本，返回回退点 |
| POST | `/v1/rollbacks/{id}` | 执行回退点 |
| GET | `/v1/rollback-points?name=` | 回退点清单 |
| POST | `/v1/models/{id}/freeze` · `/resume` | 冻结/恢复新请求 |
| POST | `/v1/inferences` | 推理（支持 `Idempotency-Key`；直接身份字段拒绝入链） |
| POST | `/v1/inferences/{id}/review` | 医生复核并签发（结论一次性、不可改） |
| GET | `/v1/inferences/{id}` | 病例结果（ops 视图脱敏） |
| GET | `/v1/cases/{id}/trace` | 审计追溯链（ops 视图脱敏） |
| GET | `/v1/audit/events` | 只增审计事件流（ops 视图脱敏） |

## 登记请求示例

```json
{
  "name": "腹部通用影像模型",
  "version": "1.0.0",
  "package_hash": "64 位 sha256 …",
  "training_data_source": "合作医院 2019-2024 脱敏 CT 12 万例，含许可与人群说明",
  "organs": ["肝脏", "胆囊", "胰腺", "阑尾", "肠道"],
  "conditions": ["急性阑尾炎", "胆囊炎", "肠梗阻", "肝占位"],
  "scenario_validation": {
    "常规检查": {"sensitivity": 0.91, "specificity": 0.93, "sample_size": 5000},
    "急腹症":   {"sensitivity": 0.84, "specificity": 0.88, "sample_size": 2200},
    "设备分层": {"sensitivity": 0.80, "specificity": 0.85, "sample_size": 1800}
  },
  "contraindications": ["孕妇", "非腹部扫描部位"]
}
```

## 关键约束与设计取舍

- **去身份在入库前**：推理请求命中直接身份字段即 400，而不是“先存再删”；影像本体不进服务，
  链路上只保存影像 sha256（影像在科室侧完成脱敏与取哈希）。
- **禁用条件**登记为明确清单；网关/工作站按清单判定后通过 `contraindication_hits` 上报命中项，
  命中即拒绝推理，避免用设备名做脆弱的子串猜测。
- **冻结语义**：只阻断新推理；既有记录不改动，保证已签发报告的证据链稳定。
- **复核一次性**：医生结论提交即终态（防止事后篡改），更正需重新发起病例并重新留痕。
- **回退点一次性**：防止重复回退造成状态摆动；每次回退生成新的反向回退点。
- 当前存储为线程安全的**内存实现**（标准库，无外部依赖），持久化/模型推理适配器可在
  `registry.ModelRegistry` 构造处替换（`scorer` 即推理适配点）。
