> 后续发布增加数据库旁置规则，配置与实际部署证据见 [发布记录](RELEASE_COLDFUSE_20261030.md)。下文保留最初代码验收时点的记录。

# 测试用户反馈隔离

本次修复仅完成代码和隔离测试，未配置或部署到线上。没有修改生产数据库、重写历史事件或快照、启动训练或重启服务。未配置排除文件的进程仍保留原有行为。

## 规则与配置

服务、在线学习命令和反馈审计进程都读取环境变量 `SIGNALFEED_TEST_USERS_FILE` 指定的 JSON 文件。以完整的32位小写十六进制匿名用户 ID 精确匹配，不接受前缀、通配符、重复 ID、空原因或未知字段；目前最多256个测试身份。

```json
{
  "version": 1,
  "users": [
    {
      "id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "reason": "人工功能自测；排除该身份的全部历史及后续反馈"
    }
  ]
}
```

上面是合成 ID，不能用于隔离实际用户。ID 同时用于匿名会话 Cookie，应将实际文件放在仓库外，限制为服务账号可读，不提交 Git、不打印到公开日志。后续部署时为所有消费者设置相同路径，例如：

```bash
export SIGNALFEED_TEST_USERS_FILE=/absolute/private/path/test-users.json
```

本次只读核对的 `4ab93890` 前缀唯一匹配一个完整 ID，其 SHA-256 为 `aef51018929e1b3579245794f3bd7731657a66250fcb5b201718881d92263e7d`。该身份有18次请求、112条事件、4条收藏和13份快照，事件与请求的用户关联错误为0。以上是核对时的数量，不应当作未来固定值。

下面是**供后续获准配置时使用、此次未执行**的文件准备方式。它只读查询生产数据库并新建私有配置文件，不改数据库，不启动任何服务；请先将 `output` 改为实际私有路径，父目录须已存在。完整 ID 不输出到终端，也不放入业务代码：

```python
import hashlib
import json
import os
import sqlite3

output = "/absolute/private/path/test-users.json"
with sqlite3.connect(
    "file:/data1/zhuzy/signalfeed/online.sqlite?mode=ro", uri=True
) as conn:
    conn.execute("PRAGMA query_only=ON")
    rows = conn.execute("SELECT id FROM users WHERE id LIKE ?", ("4ab93890%",)).fetchall()
if len(rows) != 1:
    raise RuntimeError("用户前缀未唯一匹配；停止配置")
uid = rows[0][0]
if hashlib.sha256(uid.encode()).hexdigest() != "aef51018929e1b3579245794f3bd7731657a66250fcb5b201718881d92263e7d":
    raise RuntimeError("完整用户 ID 与核对记录不符；停止配置")
config = {"version": 1, "users": [{"id": uid, "reason": "2026-10-30 人工功能自测；全身份排除"}]}
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as stream:
    json.dump(config, stream, ensure_ascii=False, indent=2)
```

配置后可仅检查解析结果和指纹，不接触数据库或模型：

```bash
.venv/bin/python -B -c 'from signalfeed.feedback_policy import load_feedback_policy; print(load_feedback_policy().metadata())'
```

未设置变量表示空排除集合。设置了路径但文件丢失、不可读或格式错误时会报错，**不会静默降级为空名单**；API 启动时也校验配置。每次操作读取一份配置，因此已配置路径的进程能够读取后续文件更新；更新须使用同目录临时文件和原子替换，避免半写文件。首次向已有进程添加环境变量仍需单独部署，不会因修改当前终端环境自动生效。

## 各路径的行为

| 路径 | 隔离方式 |
|---|---|
| 普通 / ColdFuse 学习 | 在读取请求快照、构造样本及时间切分前跳过配置中的用户；历史 `is_test=0` 同样被排除 |
| 学习报告 | 保存 `feedback_policy` 的版本、SHA-256、人数；`excluded_exposures.test_user_requests` 单独记录排除请求数（该键单位为请求） |
| 新事件 | `SIGNALFEED_TEST_EVENTS=1` 或用户在名单中即写 `is_test=1`；客户端提交的同名字段不决定测试属性 |
| 两条协同召回 | 对提供共同点击/收藏证据的其他用户执行排除，继续保留已有事件级 `is_test=0` 条件 |
| `/api/stats` | 排除测试用户的成熟曝光、点击、CTR及全站收藏数；原始表记录不变 |
| 只读反馈审计 | 配置中的身份并入测试用户集合；配置指纹写入结果，原始 inventory 仍报告物理记录数 |
| 本人浏览 | 搜索、兴趣设置、收藏、取消收藏、查看个人收藏和自己的阅读历史照常工作；推荐快照照常保存 |

一次学习调用固定使用同一配置对象及指纹，避免采样与报告使用不同规则。配置不会追溯修复已有模型，也不会重写已有快照。不要在训练或晋升过程中更改排除规则；应在下一次学习开始前完成隔离配置。当前修复不改变模型激活接口的历史合同校验。

## 回滚与边界

- 恢复之前的 JSON 文件即可回滚排除规则；历史行从未修改。移除身份会让它原有 `is_test=0` 数据重新具备入选资格，不能把清空名单视作安全的永久隔离。
- 已通过新规则写为 `is_test=1` 的事件在移除名单后仍保持测试属性，不自动改回真实事件。
- 同一身份同时用于自然阅读和测试时，会整体排除。切换浏览器/Cookie会产生新身份，新身份需要重新登记；未来自测仍推荐独立测试数据库与浏览器配置。
- 不能仅为服务配置文件而漏掉独立训练、统计审计进程。环境变量缺失时采用兼容旧版本的空名单，不会自动发现生产测试用户。
- 本人的测试历史有意保留用于自测。配置排除阻止该身份的反馈进入学习和跨用户协同证据，但无法撤销隔离前已对其他用户产生的推荐或已保存特征；当前会话核对时未发现它已进入在线模型。
- 单独设置实例级 `SIGNALFEED_TEST_EVENTS=1` 仍不等于按用户隔离历史数据；共享数据库上的全站收藏统计按配置用户排除。测试实例应使用独立数据库。

## 验证

`tests/test_feedback_policy.py` 使用合成身份和临时数据库，覆盖两路成熟样本及训练/验证切分、协同召回输入与结果、历史不可变性、新事件与重试、客户端伪造标记、普通用户、测试用户自身功能以及无效配置拒绝。该文件禁止调用训练入口。

`tests/test_temporal_audit.py` 覆盖配置身份的只读审计排除。已有 API、快照和学习筛选回归另行选取不执行训练的测试；不运行训练、模型晋升或生产浏览器测试。

2026-10-30 验证结果：新增及已有 API/审计测试26项通过；选取的普通学习筛选、时间切分、快照与 ColdFuse API 回归6项通过，共32项。审计人数统计修正后再次复核对应2项通过。Ruff 和 `git diff --check` 通过；只有既有 Starlette/AnyIO 弃用告警。

另将实际 `feedback_records` 的数据库入口替换为强制 `mode=ro`、`query_only=ON` 连接，以内存中的未来时间作成熟投影，并分别传入空规则和目标完整 ID 的规则：普通路径该身份样本 **58 → 0**，ColdFuse 路径 **24 → 0**。该验证没有调用训练，没有变更系统时间或配置线上环境，也没有写入生产数据库。两路都在进一步筛选前跳过了15个带有非测试曝光/点击的该身份请求，其中包括随后本就会被策略过滤的搜索和收藏列表请求；这不是15个训练样本。
