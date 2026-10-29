# ColdFuse 受控发布记录（2026-10-30）

用户已授权受控部署、必要重启与自动回滚；禁止训练、晋升及删除/重置生产数据。
本文件封存发布设计和部署前证据。实际切换、进程、备份、线上验收及失败记录保存于
`/data1/zhuzy/signalfeed/deploy-20261030/`，由最终交付报告引用。

## 发布范围与关闭流程

包含校园网前端兼容、测试用户隔离、ColdFuse 完整研究调用预算与画像编码后台化。
`BoundedRunner.close` 关闭准入并有限等待；未完成任务仍占用槽位。
API 先关闭全部准入，再设置研究 worker 最终关闭标记；哈希完成较晚的任务在锁内检查标记，不能重新启动子进程。
目录预热/周期刷新使用有界 daemon 任务，避免默认 executor 的长任务阻塞进程退出。

`start.sh` 使用封存目录的绝对 PYTHONPATH、独立进程会话及 RELEASE 标识，启动前检查端口；
`stop_server.py` 核对 PID 身份、保留超时 PID 文件、检查已记录子进程；失败不得直接启动第二实例。
健康探测不再创建匿名用户。Server-Timing 日志携带 request_id、实验状态、HTTP 状态及各阶段时间，不输出用户凭据。

## 配置

生产 HOME：`/data1/zhuzy/signalfeed/coldfuse-live`；数据库：`/data1/zhuzy/signalfeed/online.sqlite`。
`deploy.env` 位于 HOME，权限0600，由 scripts/env.sh 读取；明确配置 DB、TEST_USERS_FILE、HOST、PORT。
`SIGNALFEED_RETENTION_CLEANUP=0` 在本次发布中关闭原有自动90天快照清理，确保不删除生产数据。
`SIGNALFEED_TEST_EVENTS=0` 保留普通用户反馈；已确认的测试身份和验收身份列入私有 JSON。

额外支持数据库同名旁置规则文件 `<resolved-db-path>.feedback-policy.json`：未显式设置 TEST_USERS_FILE 时自动读取。
生产旁置文件链接到同一私有名单，避免独立 CLI/审计忘记继承 Web 环境而漏掉排除规则。
显式路径优先；存在但无效或断开的旁置链接报错，不静默使用空名单。

## 部署前证据与迭代

- 生产备份通过 SQLite backup API 创建，包含 WAL，完整性检查ok：217用户、858请求、245事件、6收藏、42快照。
- 首轮生命周期回归4项失败：测试在API关闭后复用已关闭全局worker；调整独立测试的显式startup/runner后50项通过。
- 另外10项部署及在线筛选测试通过。训练和晋升测试明确排除。
- 前端构建通过；9项Playwright竞态、加载、LAN事件重试测试通过。
- 真实模型+浏览器隔离验收20个首次身份：p95 234.5ms，最大268.5ms；research与cache均出现。
- 首次验收曾因冷worker尚未完成、连续立即重试而未观测cache失败；增加有记录的重试间隔后通过，不丢弃首次结果。
- 编码锁20秒：已有缓存81.5ms；新兴趣190.8ms并标记profile-warming-fallback。
- 身份校验I/O增加8秒：2037.4ms timeout-fallback。
- 回滚包独立服务验证：正常推荐、真实research、cache、兴趣/收藏隔离通过。
- 两隔离服务停止后监听端口与相关worker均退出。
- 初次构建误从仓库根目录执行npm导致缺少package.json；改到隔离frontend目录后构建通过，生产dist未改动。

## 切换和回滚

版本包位于 `/data1/zhuzy/signalfeed/releases/`，包含完整源码、前端、启动脚本、Git RELEASE及文件哈希；不使用可变editable目录。
先停止旧8765进程并确认退出，再取最终备份，启动封存版本，检查ready、release、源码路径、实际HTTP及独立浏览器身份。
初次预热约14秒，计入计划停机，不计为服务ready后的新用户首次请求。
保留旧哈希静态资源兼容已打开的页面。

启动或验收失败时先确认候选进程退出，再启动已验证回滚包，继续使用同一生产数据库和隔离配置。
普通代码回滚绝不恢复旧数据库，不覆盖发布后反馈；恢复数据库需要单独确认损坏证据与可恢复性。
`rollback-pre-latency-20261030` 来自此前冻结源码（已包含隔离修复）及原生产前端，不声称精确重建原进程全部内存导入版本。

## 验收口径与边界

scripts/accept_release.mjs 使用独立Chromium上下文；生产匿名身份先登记排除规则再浏览和反馈。
记录点击至新列表绘制的端到端时间、HTTP响应时间、Server-Timing、实验状态与请求ID。
保留上次卡片不算新列表已完成。至少20个独立身份首次个性化请求测量p95。
统计research/cache、各类fallback及无画像请求，正常串行与并发分开；后台后来完成不改写原响应的回退状态。

故障注入只针对隔离进程；线上冷worker自然timeout如出现则直接记录，不通过破坏线上worker制造超时。
本机Chromium通过网站HTTP验证不等同于每台校园网客户端的Chrome扩展/网络环境。
任务容量有界但不能强制取消Python线程；SQLite15秒锁等待及极端主机/网络负载仍可能超出3秒预算。
当前没有训练或晋升操作；保留已有模型文件，发布前后验证模型文件哈希。
