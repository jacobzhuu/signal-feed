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
