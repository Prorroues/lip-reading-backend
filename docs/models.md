# 模型权重说明

本仓库不含模型权重（文件过大，不适合 git 管理）。权重均为**官方原版**，未做 fine-tune。

## 快速获取（推荐）

```powershell
powershell -File scripts/download_models.ps1
```

脚本会从本仓库的 [Release v1.0.0](https://github.com/Prorroues/lip-reading-backend/releases) 下载 `models-v1.0.zip` 并解压到正确位置。

## 包含的权重

| 文件 | 用途 | 大小 |
|---|---|---|
| `Visual_Speech_Recognition_for_Multiple_Languages/benchmarks/CMLR/models/CMLR_V_WER8.0/model.pth` | CMLR 中文唇语识别模型 | ~210MB |
| `Visual_Speech_Recognition_for_Multiple_Languages/benchmarks/CMLR/models/CMLR_V_WER8.0/model.json` | 模型配置 | 极小 |
| `Visual_Speech_Recognition_for_Multiple_Languages/benchmarks/CMLR/language_models/lm_zh/model.pth` | 中文语言模型 | ~201MB |
| `Visual_Speech_Recognition_for_Multiple_Languages/benchmarks/CMLR/language_models/lm_zh/model.json` | 语言模型配置 | 极小 |
| `MiVOLO/models/mivolo_imbd.pth.tar` | 年龄性别估计 | ~105MB |
| `MiVOLO/models/yolov8x_person_face.pt` | 人脸/人体检测器 | ~130MB |

## 官方出处（兜底）

如果 Release 下载失败，可从官方渠道获取：

- **CMLR 唇语模型 + 中文语言模型**：[VSR 项目 Model Zoo](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages#model-zoo)（Google Drive 链接）
- **MiVOLO 年龄性别模型**：[HuggingFace iitolstykh/mivolo_v2](https://huggingface.co/iitolstykh/mivolo_v2)
- **YOLO 人脸/人体检测器**：[HuggingFace iitolstykh/YOLO-Face-Person-Detector](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector)

## 许可证注意

- VSR 代码与模型：Apache-2.0
- MiVOLO：仅供研究/学习用途，商用前请查阅 `MiVOLO/license/` 中的许可证文本

