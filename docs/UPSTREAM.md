# MiniOneRec 检查与适配

官方来源：https://github.com/AkaliKong/MiniOneRec
固定提交：`526306229326013a7ca46f5b2d8f55ea669aa159`
原始检出：`/home/zhuzy/projects/MiniOneRec`，保持未修改。

## 原流程

1. `data/` 清洗 Amazon 序列，`rq/text2emb` 编码商品标题/描述。
2. `rq/` 提供 RQ-VAE、RQ-Kmeans 等离散 SID 构造方法。
3. `convert_dataset.py` 输出带 `history_item_sid` 和 `item_sid` 的 CSV。
4. `sft.py` 联合序列推荐、标题与 SID 对齐任务，默认脚本为 8 卡。
5. `rl.py` / `minionerec_trainer.py` 使用 GRPO；本项目第一版不运行 RL。
6. `evaluate.py` 调用约束 beam search，`calc.py` 统计 HR/NDCG。

## 环境实测与处理

Ubuntu 20.04，48 逻辑 CPU，250GiB RAM，RTX 3090 Ti 24GB，驱动 535.161.08。宿主默认 Python 3.13，另建 Python 3.11.16 专用环境。PyTorch 2.6.0+cu118 可用 CUDA，不修改宿主驱动。

上游 requirements 同时列出 CUDA 11/12 runtime 与未使用的 TorchRec 等依赖。SignalFeed 按实际模块整理依赖并保存传递依赖锁定文件。安装中实际遇到 PyArrow 新版要求 NumPy 2，修复为 PyArrow 21；httpx 在本机 SOCKS 代理下需要 socksio。

## 复用与修改边界

`vendor/minionerec/data.py` 来源原始 `data.py`，直接用于 Qwen 上游验证；将 `eval` 改为 `ast.literal_eval`，不执行数据文本。保留原 `LogitProcessor.py` 副本作来源记录。

新 `TrieProcessor` 接收实际 prompt 长度，取消原处理器依赖模型名的前缀长度和跨调用计数器。同一处理器可重复调用，多个 beam 的合法集合逐条计算。

SignalFeed 的残差量化、紧凑 SID Transformer、MIND 数据适配、传统召回、排序、评测和网站是独立新增代码。SID Transformer 沿用序列→SID 自回归监督范式，明确不等同于上游预训练 LLM。Qwen 全量 MIND 训练、文本对齐扩展和 GRPO 留作下一阶段。

## 最小验证含义

真实 Amazon 随机子集：256 个训练样本，32 个测试样本；32 步训练实际消费 64 个样本，不代表完整训练过 256 个样本。Qwen 0.5B + LoRA + 可训练输入/输出 embeddings；checkpoint 保存后重新载入生成。

生成合法率、HR/NDCG、训练 loss 和耗时由 `upstream-smoke/report.json` 记录。HR 为零也原样报告，这一步只证明真实训练、持久化推理和评测路径可运行，不能引用为论文复现或模型有效性结论。
