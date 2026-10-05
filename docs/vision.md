# 视觉：让 Agent 看一眼（摄像头 + 人脸检测）

这是 **#18 Phase 1**：只回答「**画面里有没有人、脸在哪**」。
「这是谁」属于下一步（识别 + 身份库），本阶段**不做身份识别、不做人脸比对**。

## 装

```bash
pip install "handcraft-agent[vision]"
```

`vision` extra = `opencv-python-headless` **+** `numpy`。

> **为什么封顶 `opencv<5`**：OpenCV **5.x 移除了 Haar 级联** —— `cv2.CascadeClassifier`、`cv2.objdetect`
> 都没了，连随包的模型文件也不再提供（只剩要另外下载模型的 `FaceDetectorYN`）。而 Haar 正是
> 「**零下载、离线、CI 里也能真跑**」的关键，所以这里显式封顶 4.x。装了 5.x 时会得到一句人话提示，
> 而不是一个 `AttributeError`。

不装也能跑：**这两个工具根本不会注册**（工具列表里不出现 → 模型就不会规划它做不到的动作）。

## 两个工具

| 工具 | 干什么 | 只读分级 |
|---|---|---|
| `look(index=0)` | 取一帧 → 回**尺寸 / 人脸数 / 每张脸的框**（只回数字，**不落盘**） | ✅ 只读 |
| `look_image(path, index=0)` | 取一帧 → 画框 → **存成文件**，返回路径 | ⚠️ 写操作（会留图） |

`look` 的输出长这样：

```
画面里有 2 张人脸：#1 在 (312,180) 大小 96x96；#2 在 (88,204) 大小 72x72（画面 1280x720，来自 camera:0）
```

要拿图像本身给视觉模型看（DeepSeek 无视觉，可以走 DashScope/Qwen-VL 那条路），用 `look_image` 落盘后再喂 —— 
**留图是显式动作**，不会被悄悄做掉。

## 直接用（不用 Agent）

```python
from forge.camera import OpenCvFrameSource, HaarFaceDetector, capture, observe, format_observation, probe_camera

print(probe_camera())                       # 先如实探测：摄像头有没有、能不能开
with OpenCvFrameSource(index=0) as src:
    frame, faces = capture(src, HaarFaceDetector())
    print(format_observation(observe(frame, faces)))
```

## 隐私（请当硬约束读）

人脸属于**敏感个人信息**（PIPL / GDPR 都按特殊类别处理）：

- 本模块**默认只在内存里处理**帧，`look` 不回身份、不落盘；
- **不存原图**、**不做人脸比对**（Phase 1 完全没有）；
- 要留图必须显式调 `look_image` —— 属于写操作，走审批那一档；
- 将来做「认识人」（Phase 2）也要遵守：**只存特征向量、可删除、需本人同意**。

## 真人脸检测精度：本阶段**未验证**

本机当前**没有摄像头**（`ffmpeg` 枚举不到；Windows「设置 → 隐私和安全性 → 相机」被关掉是经典坑）。
所以自动化测试覆盖的是**链路逻辑**（取帧 / 重试 / 检测结果流转 / 文案 / 只读分级 / 能力探测 /
5.x 退化路径），**不含真实人脸的检出率**。

想确认它对真人脸的效果：装好 `[vision]` 后跑

```python
from forge.camera import probe_camera, OpenCvFrameSource, HaarFaceDetector, capture
print(probe_camera())                        # 先确认摄像头可用
```

或者用 `look_image` 存一张带框的图自己看一眼。检出率不理想时，Phase 2 可以换成 ONNX 检测器
（YuNet / RetinaFace）——**只要实现同一个 `FaceDetector` 接口**，上层不用动。
