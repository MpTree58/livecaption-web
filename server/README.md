# livecaption 中央服务器（多用户会议库）

一个**独立的 Web 服务**：团队里每个人在自己 Mac 上用 `livecaption-web` 采集会议，录完自动同步到这台服务器；大家用浏览器登录各自账户，浏览 / 搜索 / 加笔记 / 导出自己的和被分享的会议。

**这台服务器不做语音识别、不用 GPU、不装 MLX**——它只管账户 + 存储 + 网页。可以跑在任意一台常开的 Linux（或 Mac）机器上。

- 权限模型：会议**默认私有**，主人可以分享给指定同事，或设为**全员可见**；只有主人能改 / 删。
- 数据库：**SQLite**——就是一个文件（`livecaption.db`），**Python 自带，不需要单独安装数据库**（不像 MySQL/Postgres 要装服务、配置）。零运维。

---

## 一、Linux 环境要求（Ubuntu 22.04 为例）

只需要两样，**没有数据库要装**：

1. **Python ≥ 3.10**（Ubuntu 22.04 自带 3.10，够用）
2. **[uv](https://docs.astral.sh/uv/)**（Python 包管理器，一条命令装依赖）

装 uv：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 装完重开终端，或： source $HOME/.local/bin/env
```

> 就这些。SQLite 随 Python 自带,fastapi/uvicorn 等依赖下一步用 uv 一键装好。

## 二、安装 & 首次启动

```bash
# 1) 拿到代码（从你的仓库 clone，或拷贝 server/ 目录到服务器）
git clone https://github.com/MpTree58/livecaption-web.git
cd livecaption-web/server

# 2) 装依赖（自动建 .venv，装 fastapi / uvicorn / itsdangerous 等）
uv sync

# 3) 建用户（内部工具，没有自助注册，管理员在这里开账户）
uv run livecaption-server-admin adduser alice
#   → 会提示设密码，并打印出 alice 的「agent API token」，记下来给 alice

# 4) 启动服务器
uv run livecaption-server
#   默认监听 0.0.0.0:8080（局域网可访问）
```

浏览器打开 `http://<服务器IP>:8080`，用刚建的账户登录即可。

**环境变量（可选）：**
| 变量 | 默认 | 说明 |
|---|---|---|
| `LC_SERVER_HOST` | `0.0.0.0` | 监听地址（`0.0.0.0`=局域网可达） |
| `LC_SERVER_PORT` | `8080` | 端口 |
| `LC_SERVER_DATA` | `server/data` | 数据目录（存 `livecaption.db` 和会话密钥）——**记得备份这个目录** |

## 三、管理员命令

```bash
uv run livecaption-server-admin adduser <名字>          # 建用户（打印其 agent token）
uv run livecaption-server-admin adduser <名字> --admin  # 建管理员
uv run livecaption-server-admin listusers               # 列出所有用户 + 他们的 token
uv run livecaption-server-admin token <名字>            # 查某人的 token
uv run livecaption-server-admin token <名字> --regen    # 重置某人的 token
uv run livecaption-server-admin passwd <名字>           # 管理员帮某人改密码
# 删用户（离职移除）。若此人名下还有会议，必须二选一：
uv run livecaption-server-admin deluser <名字> --transfer <另一个人>   # 把 TA 的会议转给别人后删
uv run livecaption-server-admin deluser <名字> --purge                # 连 TA 的会议一起删
```

> **注册是白名单制**：没有网页自助注册，账户只能管理员用上面的 `adduser` 开。
> **用户可自己改密码**：登录后点右上角「Password」即可改自己的密码（不用找管理员）。

## 四、每个人的 Mac 怎么连上来（采集端）

在**每个人的 Mac** 上（已装好 livecaption 的机器），配置两个环境变量，把「服务器地址 + 自己的 token」告诉采集端。加到各自的 `~/.zshrc`：

```bash
export LC_SYNC_URL="http://<服务器IP>:8080"
export LC_SYNC_TOKEN="lc_xxxxxxxx…"     # 管理员给他的那个 agent token
```

之后照常 `caption-web` 开会，**每场录完会自动推到服务器**（终端会打印 "Central sync ON"）。本地仍留一份，网络断了也不丢，恢复后可重开重推（幂等，不会重复）。

> 也可以不写进 zshrc，用命令行传：`caption-web --sync-url http://... --sync-token lc_...`

## 五、做成开机自启的常驻服务（systemd）

让服务器一直开着、崩了自动重启、开机自动拉起。新建 `/etc/systemd/system/livecaption-server.service`：

```ini
[Unit]
Description=livecaption central meeting-records server
After=network.target

[Service]
Type=simple
User=livecaption
WorkingDirectory=/opt/livecaption/server
Environment=LC_SERVER_HOST=0.0.0.0
Environment=LC_SERVER_PORT=8080
Environment=LC_SERVER_DATA=/var/lib/livecaption
ExecStart=/home/livecaption/.local/bin/uv run livecaption-server
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

然后：
```bash
sudo mkdir -p /var/lib/livecaption          # 数据目录
sudo systemctl daemon-reload
sudo systemctl enable --now livecaption-server
sudo systemctl status livecaption-server    # 看是否 running
journalctl -u livecaption-server -f         # 看日志
```

> 把 `WorkingDirectory` / `User` / uv 路径按你的实际情况改。`ExecStart` 里 uv 的绝对路径用 `which uv` 查。

## 六、安全提醒

- 服务器默认监听局域网（`0.0.0.0`）。**只在公司内网 / VPN 内使用**;不要直接暴露到公网。
- 登录用签名 Cookie。若要经公网访问,强烈建议前面套一层 nginx/caddy 做 **HTTPS**。
- `LC_SERVER_DATA` 目录里的 `livecaption.db` 是全部数据（用户 + 所有会议 + 笔记）——**定期备份**这个文件即可（复制走就行,SQLite 单文件）。
- agent token 等于该用户的上传凭证,泄露了用 `token <名字> --regen` 重置。

---

## 数据流一览

```
每个人的 Mac                                这台 Linux 服务器
┌──────────────────┐                      ┌────────────────────────────┐
│ caption-web 采集   │  录完 POST /api/ingest │ FastAPI + SQLite            │
│ (模型本地跑)       │  ──(带 token)──────▶  │ 账户 · 会议库 · 分享 · 搜索  │
│ 本地也留一份        │                      │ 浏览器登录 → 浏览/笔记/导出   │
└──────────────────┘                      └────────────────────────────┘
```
