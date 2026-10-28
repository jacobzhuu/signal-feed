# 本机运行、测试与维护

## 环境

Python 3.11、Node 22、单张 CUDA GPU 用于训练；线上服务默认 CPU。首次安装使用 `bash scripts/install.sh`。本机环境已安装在 `/data1/zhuzy/signalfeed/venv`，`scripts/env.sh` 配置路径与缓存。不要向宿主 conda base 环境安装依赖。

原工作分区只有约 12GB 可用，大数据、模型和专用环境均放在 `/data1`。原始数据和检查点不纳入 Git；`requirements.lock.txt` 与 `frontend/package-lock.json` 固定依赖。

## 应用

```bash
source scripts/env.sh
signalfeed sync
npm --prefix frontend run build
bash scripts/start.sh
curl http://127.0.0.1:8765/api/health
# 停止、更新代码后重启
bash scripts/stop.sh
bash scripts/start.sh
```

服务每 15 分钟同步 HN，启动时可用上次内容；同步命令可手动重复执行。SQLite WAL 与幂等 upsert 支持重复抓取，删除/失效状态从 API 反映到内容库。抓取失败写日志并保留上次目录。当前只跟踪最新/热门列表及部分旧条目，长期内容归档与删除全量扫描尚未实现。

配置 `SIGNALFEED_DB` 可隔离数据库。模型和原始数据默认共用 `SIGNALFEED_HOME`。运行入口：`signalfeed --help`。

默认后台 PID 文件在运行目录，日志 `logs/server.log`。这不是系统服务托管；机器重启后手动启动。公网部署需另加 TLS、登录/管理权限、限流、备份与进程托管，本版本没有默认暴露公网。

## 测试数据库与浏览器

```bash
source scripts/env.sh
pytest -q
python - <<'PY'
import sqlite3
source=sqlite3.connect('/data1/zhuzy/signalfeed/online.sqlite')
target=sqlite3.connect('/data1/zhuzy/signalfeed/browser-test.sqlite')
source.backup(target)
target.close();source.close()
PY
SIGNALFEED_DB=/data1/zhuzy/signalfeed/browser-test.sqlite \
SIGNALFEED_TEST_EVENTS=1 signalfeed serve --port 8766
# 在另一终端：
cd frontend
npx playwright install chromium
npx playwright test
```

测试事件标记由服务端环境决定，客户端无法提交“真实/测试”标记。测试执行前应使用专用数据库，不针对实际用户库进行点击测试。`scripts/test_research.py` 连接 8766，验证生成进程冷启动超时回退、完成后命中真实模型缓存。

## 特征、模型与产物

MIND pipeline 先 prepare→encode→features，再训练；输入、码本、模型保持同一数据版本。需要重建码本时，应重新训练生成器，不能只替换线上索引。码本冻结后新文章可增量编码，不改变旧 SID。

检查点、原始预测和指标均在外部数据目录。单次新实验会重写同一 seed 的默认目录；要保留不同方案，使用独立 `SIGNALFEED_HOME` 或先复制产物。不要同时运行相同 seed/输出目录的任务。

反馈重训只使用真实且成熟的曝光，训练与验证按请求时间分离。验证通过后写 `online-models/active.json` 原子指针。回滚可把指针替换成已验证历史版本的 manifest；移除指针会恢复内容推荐。备份数据库时使用 SQLite backup API，而非只复制正在写入的主文件。

## 当前限制

- 默认 SID 配置是 63 万参数模型，完整 MIND Qwen SFT 与 GRPO 尚未运行；不引用上游论文结果。
- 3×32 SID 码本碰撞较多，原始 SID 基线采用桶映射；需要研究更大码本和训练规模，而非将碰撞当作非法文章。
- 在线匿名身份绑定浏览器 Cookie，不跨设备同步。兴趣画像立即更新，但权重训练是显式批任务。
- 在线训练样本不足时不训练、不发布，不生成模拟指标。现有自动验证门槛只是工程保护，不能替代有真实流量的 A/B 实验。
- 生成研究入口有单进程、队列保护和 2 秒等待预算；持续卡住的模型进程需重启服务。默认内容推荐不依赖该进程。
- 性能实测来自共享主机，应同时记录正在运行的训练/评测负载。缓存与未缓存结果要分别解释。
- 未提供广告竞价/转化业务；当前能力覆盖搜索和推荐，CTR 标签来自真实内容曝光。

性能策略：应用保留一个空闲连接避免每次请求关闭最后连接都触发 WAL checkpoint。可重建的返回列表日志采用 NORMAL，曝光、点击、收藏事务仍采用 FULL，后者提交会同时刷出此前的请求关联记录。断电可能丢失尚未关联真实反馈的返回日志；正常重启不会丢失。

## 本轮修复后的安装、迁移与验证

`requirements.lock.txt` 使用 `-e .`；`scripts/install.sh` 会先切换到脚本所属仓库根目录，再安装当前 checkout。新机器可显式选择有写权限的数据目录，例如：

```bash
SIGNALFEED_HOME="$HOME/.local/share/signalfeed" bash scripts/install.sh
export SIGNALFEED_HOME="$HOME/.local/share/signalfeed"
source scripts/env.sh
```

原机器的 `/data1/zhuzy/signalfeed` 默认配置保留；`SIGNALFEED_HOME` 必须在安装、服务、训练中保持一致。安装回归测试会复制源码到另一条含空格的路径，在临时虚拟环境中实际执行 editable 安装并校验导入位置；不重复下载全部 GPU 依赖。

升级后重启服务，或执行 `signalfeed retrain-online`，会自动创建 `request_snapshots` 表，并在事务中将旧 `once_event` 索引迁移为仅约束 impression/click 的部分唯一索引，原有请求、事件和收藏不删除。收藏/取消收藏使用 `event_id` 幂等：**网络重试必须复用原 UUID，新的用户操作必须使用新 UUID**。服务只对成功插入的新事件修改收藏，且插入与状态修改在同一个事务中提交。

旧请求缺少快照，不能补写为训练样本。旧在线模型若缺少 `data_until` 证明，自动晋升会返回 `incumbent_provenance_unavailable`；应先审计其来源，必要时按原回滚流程移除活动指针恢复内容策略，再收集新样本。不要并行执行多个在线重训/人工切换发布任务。

浏览器测试可以指定独立服务地址和输出位置：

```bash
SIGNALFEED_BASE_URL=http://127.0.0.1:8767 \
SIGNALFEED_BROWSER_REPORT=/tmp/signalfeed-browser-tests.json \
npm --prefix frontend test
```

`benchmark.py`、`test_research.py` 同样支持 `SIGNALFEED_BASE_URL`；产物目录由 `SIGNALFEED_HOME` 决定。浏览器截图默认在 Playwright 的测试输出目录。全部点击测试仍要求独立数据库和 `SIGNALFEED_TEST_EVENTS=1`。

快照保留完整浮点特征会增加请求日志体积和写入延迟。本轮未增加压缩、自动清理或新的存储系统，备份和容量管理需计入 `request_snapshots`。训练仅从不可变快照读取，不能以节省容量为由重新从当前行为重建旧特征。

延迟曝光对照的可复现命令（只生成隔离的合成回归数据，不代表真实在线指标）：

```bash
source scripts/env.sh
review_run_dir="$(mktemp -d)"
git show 6552342:src/signalfeed/online_learning.py > "$review_run_dir/legacy.py"
python scripts/review_online.py --output "$review_run_dir/before" \
  --legacy-source "$review_run_dir/legacy.py"
python scripts/review_online.py --output "$review_run_dir/after"
# 查看两个输出目录中的 comparison.json 和模型 manifest
```

本轮 MIND 重跑使用独立的 `SIGNALFEED_HOME`：只读复用原 `mind/`、`hf/`、`models/sid-{42,43,44}`、`models/encoder`、`models/fusion.json`，将新 ranker 与 reports 写入新目录。执行三个种子的 `signalfeed train --model ranker --seed N`，无个性化 `train(personal=False)`，然后每种子 `signalfeed evaluate --seed N --retrieval-limit 1000`（默认全量排序），最后 `python scripts/summarize.py`。原产物保持不动。


## 审计 v2 操作约定

- 新离线训练会生成 `bundle.json`，加载时校验数据、ID 排序、向量、码本、特征和权重。旧检查点必须重新训练，或在人工/代码审计完成后显式 `signalfeed bind-offline --attestation '<审计记录>'`。该命令记录复用来源，不声称重训。`fusion.json` 必须按 retrieval-v2 在至少 2,000 条内部验证请求上重新选择。
- 编码器优先使用固定 revision 的本地缓存。`signalfeed serve` 启动预热，`/api/health` 区分 warming/ready，`/api/ready` 未就绪返回 503。
- 在线特征版本升级为 online-text-v2；旧版本模型不会静默加载。保留旧快照但训练排除不兼容版本，不用新特征重建旧样本。
- 模型回滚/激活：`signalfeed activate-online --version <版本>`；不传版本恢复内容策略。命令与训练共享发布锁；不要手工改 active.json 绕过协调。
- 快照存储为压缩 float32 二进制，调试请调用 `signalfeed.snapshots.unpack(payload)`。后台每轮清理 90 天之前的快照，保留真实反馈及其请求元数据。训练默认最多读取最近 50,000 个曝光，完整保留请求组并报告被容量限制丢弃的样本数。
- 测试与报告：在独立 SIGNALFEED_HOME 下启动 `SIGNALFEED_TEST_EVENTS=1` 服务；设置 SIGNALFEED_BASE_URL 后执行 `python scripts/verify.py --browser`、`python scripts/benchmark.py`。性能脚本会写请求和测试曝光，拒绝运行在真实事件服务上。
- `python scripts/write_report.py --run <独立运行目录>` 要求同一源码/运行的指标、验证及成本证据。旧的无参数报告生成方式已移除。不要用旧 9 项测试或旧延迟覆盖新结果。

## ColdFuse 独立实验与服务

- `GET /api/feed?cold=true` 对应网页“兴趣加速（实验）”。缺少研究权重时仍可提供先验/内容/CF/热门推荐；研究推理有 2 秒等待与 30 秒硬期限，不能把所有成功 HTTP 请求等同于成功生成推理。
- 独立实验：设置 `SIGNALFEED_BASELINE_HOME` 为经过绑定校验的基线目录、`SIGNALFEED_HOME` 为新的输出目录，然后 `bash scripts/run_cold_experiments.sh`。默认 30,000 训练 impressions、三个种子、27 组内部验证、每种子 1,000 个配对主样本及 300 自然冷样本。已有部分预测不被覆盖，失败应保留并另建运行目录。
- 新功能只读取 `models/coldfuse-42`、已绑定的 SID、码本及验证选择 `models/coldfuse.json`；MIND 跨域模型不自动晋升为真实线上模型。
- `signalfeed retrain-online --cold` 从 `cold-semantic-v1` 请求快照读取成熟真实反馈；`signalfeed activate-online --cold --version <版本>` 经过合同及哈希校验回滚，空版本停用 ColdFuse 在线 DNN。原 `retrain-online` 仍只处理旧语义契约。
- `SIGNALFEED_BENCH_COLD=1 python scripts/benchmark.py` 对隔离测试服务测试 ColdFuse/搜索混合负载，并分别统计研究完成、缓存及各类回退。

### 本机已验证部署

真实入口为 http://127.0.0.1:8765，现有 tmux 会话 `3` 的 `signalfeed-live` 窗口管理。运行目录 `coldfuse-live` 的 mind/models/hf 链接到已冻结的 `coldfuse-v2` 产物，`cold-online-models` 与测试服务独立。真实数据库仍为 `/data1/zhuzy/signalfeed/online.sqlite`。

```bash
export SIGNALFEED_HOME=/data1/zhuzy/signalfeed/coldfuse-live
export SIGNALFEED_DB=/data1/zhuzy/signalfeed/online.sqlite
export SIGNALFEED_TEST_EVENTS=0
bash scripts/start.sh
# 停止同一实例：bash scripts/stop.sh
# 真实反馈重训：.venv/bin/python -m signalfeed.cli retrain-online --cold
```

自动化浏览器和压测使用 8768、`coldfuse-v2/browser.sqlite`、`SIGNALFEED_TEST_EVENTS=1`。性能工具会拒绝真实服务；不向真实入口发送合成点击。通过自动化执行器启动持久服务时应在 tmux 中运行，避免执行器结束时回收子进程。
