# GPN-Map（河西走廊诗词地图）

包含 1230 首诗词、本地地图底图、诗词检索、语义分析、地图定位、现实影像检索、意境图/视频模块及游客/管理员权限。原始开发项目保持独立；本目录作为独立 Git 仓库维护，并同步至 GitHub 私有仓库 syf876/GPN-Map。当前未公开或部署。

## 本地启动

需要 Python 3.11 或更新版本。已在 Python 3.14 的独立虚拟环境验证。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

浏览器打开 http://127.0.0.1:8080 。按 Ctrl+C 停止。也可以分别运行 `python -m uvicorn api_round2:app --host 127.0.0.1 --port 8000` 和 `python frontend/server.py`。启动依赖当前目录文件，不读取原项目配置。

## 功能与配置

- 1230 首诗词可检索、筛选与定位；原地图底图保留，本地图无需 API Key。地图布局为示意展示，不用于导航或测绘。
- 默认语义模式按本地已有推理结果匹配库内诗词；未知文本仅进行规则地名识别，未知类别明确标记。原 Transformer 推理代码保留，模型权重及外部模型缓存不捆绑；自行取得授权模型后安装 `requirements-optional.txt` 并设置 `HEXI_SEMANTIC_BACKEND=transformer`。BGE-M3 不可用时回退到本地关键词检索。
- 三组本地演示图片、视频和解析结果可离线读取：酒泉子·花映柳条、使至塞上、凉州词。播放演示不会调用生成 API。
- 现实影像检索使用 Bilibili 免费公开搜索，返回原站链接；依赖网络和上游接口，不捆绑转载现实影像。访问受限时明确提示失败，不改用付费服务。
- 游客可使用检索、分析、地图和本地演示。实时意境图和视频生成需要管理员会话、CSRF 校验及显式启用付费开关，默认关闭。

在自己的交互终端运行 `python scripts/setup_local_admin.py`，输入至少 12 位密码；本地生成 `.env` 中的密码哈希与随机会话密钥。没有内置管理员密码。重启后登录。本地 HTTP 使用 `AUTH_COOKIE_SECURE=false`。若后续自行部署 HTTPS，应改为 true 并准确配置允许来源。

`.env.example` 只含空凭据模板。外部付费 API 默认 `ENABLE_PAID_APIS=false`；仅在使用者自行配置供应商、真实凭据及该开关后可调用。任何生成和语音合成均未在整理过程中调用或计费。视频处理需自行安装 FFmpeg 或可选依赖。

## 目录

```text
backend/                 API、鉴权、离线语义与服务编排
  service/               智能体、检索、现实影像、图像/视频模块
frontend/                页面及本地代理
  static/demo/           三组本地演示图片、视频、结果
  static/vendor/leaflet/ 本地地图组件及许可
  static/                原地图底图与地图适配器
  data/                  演示配置
dataset/                 精简语义推理结果表（1230条）
labels/                  朝代、地名、意象、情绪标签
transformer/             原神经网络模型定义与配置
scripts/                 本地管理员配置、安全扫描
tests/                   离线功能与权限验证
licenses/                第三方许可
run.py                   双服务启动入口
hexi_poems_clean_simplified.json 诗词库
poems_master.csv         诗词地理坐标表
```

检查方式和实测结果见 `VALIDATION.md`；清理范围与本地媒体权属说明见 `CLEANUP_REPORT.md`。未保留旧日志、私人配置、会话数据库、开发环境依赖副本、旧 Git 历史、论文或实验产物。
