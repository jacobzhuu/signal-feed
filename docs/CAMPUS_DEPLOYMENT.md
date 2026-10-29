# 校园网部署

前端是 `frontend/` 中的 React + TypeScript 网站。构建产物由 FastAPI 同时提供，页面和 `/api` 共用 8765 端口，无需额外运行 Vite。

2026-10-30，本机部署切换为监听 `0.0.0.0:8765`，校园网 IPv4 地址为 **http://10.21.169.127:8765**，本机仍可使用 http://127.0.0.1:8765。仅启用 IPv4 监听；未启用公网 IPv6 入口。地址来自 DHCP，可能变化，可用 `ip -brief address` 查看。

## 启停

沿用真实数据库和 ColdFuse 模型目录。在仓库目录执行：

```bash
export SIGNALFEED_HOME=/data1/zhuzy/signalfeed/coldfuse-live
export SIGNALFEED_DB=/data1/zhuzy/signalfeed/online.sqlite
export SIGNALFEED_TEST_EVENTS=0
export SIGNALFEED_HOST=0.0.0.0
export SIGNALFEED_PORT=8765
bash scripts/start.sh
# 更改监听地址或更新后端时，先停止再启动：
# bash scripts/stop.sh
# bash scripts/start.sh
```

本次在独立 tmux 会话 `signalfeed-campus` 中启动，日志仍为 `$SIGNALFEED_HOME/logs/server.log`，PID 文件仍为 `$SIGNALFEED_HOME/server.pid`。不依赖 SSH 连接持续在线；机器重启后需要重新执行启动命令，当前没有配置开机自启。

通用启动脚本默认仍为 `127.0.0.1:8765`；`SIGNALFEED_HOST` 和 `SIGNALFEED_PORT` 用于显式覆盖。若服务已运行，脚本不会自动替换其监听地址。恢复仅本机访问：停止后设置 `SIGNALFEED_HOST=127.0.0.1` 再启动。

更新前端使用 `npm --prefix frontend run build`。页面不再请求 Google 字体；普通内网 HTTP 使用 `crypto.getRandomValues` 生成 UUID，支持曝光和收藏反馈。前端资源无需外网，但服务器刷新 HN 内容及用户打开文章原站仍需要相应外网连接；抓取失败时保留已有内容。

## 验证与访问边界

服务器本机可以验证首页、JS/CSS 与就绪接口：

```bash
ss -ltnp '( sport = :8765 )'
curl --noproxy '*' http://10.21.169.127:8765/api/ready
```

本机请求内网 IP 成功不能证明跨设备连通。请在另一台连接校园网的电脑或手机浏览器打开上述地址；无需 SSH 转发。若无法访问，在该设备检查：

```bash
curl --noproxy '*' --connect-timeout 5 http://10.21.169.127:8765/api/ready
```

连接超时需检查校园网的访客隔离、无线客户端隔离、楼宇/VLAN 路由和主机防火墙；连接被拒绝时先检查服务和 IP。全校可访问需要学校允许目标网段到该服务器 TCP 8765 的连接，应用本身无法解除网络隔离。必要时由管理员按学校实际网段放行该端口。本次未修改主机防火墙；无法使用免密 sudo 查询有效规则，`/etc/ufw/ufw.conf` 中为 `ENABLED=no`，不据此断言全部过滤规则均关闭。

浏览器回归 `cd frontend && npx playwright test e2e/lan.spec.ts` 使用模拟 HTTP 域名、本地构建和模拟 API，验证非安全上下文下的曝光、收藏、重试 UUID 一致性和无外部资源请求，不向生产写入测试行为。不要对生产执行自动点击或压测。

此前 [本机入口审计](LOCAL_ACCESS_V7.md) 记录的是切换前状态，其“只监听本机”描述不再代表当前部署。
