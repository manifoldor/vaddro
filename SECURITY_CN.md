# Vaddro 安全策略

[English](SECURITY.md)

## 安全模型

Vaddro 将 Seed 保管与 GPU 计算分离：

- 离线机保管 Seed 私有标量 `k_seed`。
- GPU Worker 只接收对应公钥点、搜索规则和分片配置。
- 命中后，Worker 返回 Search Share、Variant、地址和公开验证元数据。
- 最终私钥只在离线机上合成并验证。

~~~text
离线可信机                                        在线 GPU Worker
---------                                        ---------------
seed.py
  -> Seed 私有标量 k_seed（始终留在离线端）
  -> 公钥点 P_seed  ----------------------------> P_seed + 搜索规则 + 分片
                                                   CUDA/PTX 搜索
verify_result.py  <------------------------------ Share + Variant + 公开元数据
combine.py
  -> k_seed + 已验证 Share
  -> 最终私钥与独立验证的 EVM 地址
~~~

在 secp256k1 椭圆曲线离散对数安全假设成立时，Worker 无法从公钥点恢复 Seed 私有标量。Split-Key 不能保护已被入侵的离线机、泄露的 Seed 或钱包、恶意构建以及实现缺陷。该模型与 CVE-2022-40769（归档版 Profanity 的 32 位 RNG 种子）披露后用于替代该工具的 Split-Key 方案同类。

CUDA 启动自检、运行时 canary、Host 命中重建和 `verify_result.py` 用于发现计算错误，但不能替代发布包校验和离线独立验证。

### 分片安全边界

确定性分片把随机基础 Search Share 的 48 位高 limb 命名空间划分为连续区间：

~~~text
width       = 2^48 / SHARD_COUNT
shard_begin = width × SHARD_INDEX
shard_end   = shard_begin + width
~~~

- 同一任务的 Worker 使用相同且为 2 的幂、范围为 1 到 65536 的 `SHARD_COUNT`，并分别使用唯一的零基 `SHARD_INDEX`。
- 唯一的活动索引可以避免基础 Search Share 区间重叠；Vaddro 不负责跨主机集中分配或协调索引。
- 分片信息会写入状态、JSONL 事件和结果文件；公开验证会拒绝超出声明区间的带分片标签 Share。
- Worker 重启时会在所属分片内重新选择随机起点。分片用于协调并发 Worker，不是持久化断点，也不记录历史覆盖范围。
- GLV/取负变体在基础空间分片之后生成。不同分片理论上仍可能得到等价的有效候选，但概率在密码学意义上可忽略。

## 必须遵守的安全规则

- 只在隔离、可信的机器上运行 `seed.py` 和 `combine.py`。
- 不得把 Seed、最终私钥或钱包文件复制到 GPU Worker、代码仓库、共享文件系统、日志或支持报告中。
- 校验发布包摘要，并确认源码来自预期的正式标签。
- 结果进入离线环境前，先运行 `verify_result.py`。
- 正式使用前，独立确认最终地址，并完成小额发送和返回测试。
- 妥善保护离线备份和私密文件；系统支持时使用 `0600` 等严格权限。
- 通过独立控制的官方渠道公布正式地址。靓号特征本身不能证明地址归属。

## 支持版本

只有最新正式标签版本接收安全修复。开发快照和无标签二进制不在支持范围内。

## 报告安全漏洞

请通过私密渠道报告安全漏洞或密码学缺陷：

- [GitHub 私密安全公告](https://github.com/manifoldor/vaddro/security/advisories/new)
- 邮箱：yishan（[linyishan@gmail.com](mailto:linyishan@gmail.com)）

报告应包含清晰的问题描述、复现步骤和可选的修复建议。请使用合成测试材料，不得提交生产 Seed、私钥、钱包或未经脱敏的秘密信息。

在修复和披露完成协调前，不要通过公开 Issue 报告严重漏洞。
