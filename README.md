# 默启未来 —— 唇语识别后端

基于多模态深度学习的实时唇语识别服务。接收视频输入，输出唇语识别文本。

> 隶属"默启未来"项目 | 眼镜 APP：[lip-reading-glasses-app](https://github.com/Prorroues/lip-reading-glasses-app) | ESP 家居：[lip-reading-esp32-home](https://github.com/Prorroues/lip-reading-esp32-home)

## 功能

- **唇语识别（核心）**：上传视频 → MediaPipe 唇部定位 → CMLR 中文唇语模型 → 输出文字
- **年龄性别估计（可选）**：MiVOLO 模型，视频抽帧推理
- **大模型语义纠错（可选）**：Dify 工作流，修正同音字/语义歧义
- **编辑距离计算**：拼音相似度评分，用于识别质量反馈

## 快速开始

### 1. 环境要求

- Python 3.12+
- CUDA 11.8（GPU 可选，CPU 也能跑但慢）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型权重

```powershell
powershell -File scripts/download_models.ps1
```

权重说明见 [docs/models.md](docs/models.md)。

### 4. 配置环境变量（可选）

不配置也能跑唇语识别主功能。如需可选模块：

```bash
cp .env.example .env
# 编辑 .env 填入你的配置
```

### 5. 启动服务

```bash
python app2.py
# 或使用 Linux 脚本
bash run.sh
```

服务监听 `0.0.0.0:24060`。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/upload/videos` | 上传视频（`?wait=false&fast=true` 异步快速模式） |
| GET | `/upload/videos/status/{task_id}` | 轮询异步识别结果 |
| GET | `/recognition/latest` | 获取最近一次唇语纯文本（供家居服务拉取） |
| GET | `/edit_distance` | 拼音编辑距离得分 |
| POST | `/llm_interaction` | Dify 大模型纠错（需配置 .env） |
| GET | `/test` | 健康检查 |
| GET | `/image/{filename}` | 获取年龄性别标注图 |
| GET | `/video/{filename}` | 获取唇部关键点视频 |

## 项目结构

```
├── app2.py                  # 主入口（FastAPI 服务）
├── middlewares/             # 启动时模型装载
├── pydantic_models/         # 请求体定义
├── utils/                   # 工具（存文件、旋转视频、编辑距离等）
├── MiVOLO/                  # 年龄性别估计（可选模块）
├── Visual_Speech_Recognition_for_Multiple_Languages/  # CMLR 唇语识别核心
├── scripts/                 # 工具脚本
└── docs/                    # 文档
```

## 许可证

Apache License 2.0 — 详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

## 致谢

- [Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)（Imperial College London, Apache-2.0）
- [MiVOLO](https://github.com/WildChlamydia/MiVOLO)（年龄性别估计）
