# 人脸识别：让 Agent「认识」人（#18 Phase 2）

Phase 1 只回答「**有没有人、脸在哪**」；这一层回答「**这是谁**」——前提是你先把人**登记**过。

```
camera.capture()  →  Frame + [Face]        # 有没有人
faces.crop_face() →  一张脸的裁剪
embedder.embed()  →  128 维特征向量
store.match()     →  名字 / 「未知」
```

**核心观念：「认识」= 你自己的特征库，不是模型记忆。** 视觉模型每看一帧都只是在描述画面，
不会跨会话记得谁是谁；让它认识人的，是这里登记下来的向量库。所以登记这一步是核心，模型只是可替换的零件。

## 四个工具

| 工具 | 干什么 | 只读分级 |
|---|---|---|
| `face_people()` | 列出已登记的人（名字 + 各几张向量） | ✅ 只读 |
| `face_who(index=0)` | 看一眼并回答「这是谁」，多人时逐个给 | ✅ 只读 |
| `face_enroll(name, samples=3)` | 抓几帧、取向量、登记到该名字下 | ⚠️ 写操作 |
| `face_forget(name)` | 彻底删除该人的全部向量 | ⚠️ 写操作 |

## 装

```bash
pip install "handcraft-agent[vision]"      # numpy + opencv（识别不需要额外 pip 包）
```

还差**一份模型文件**（约 37MB，Apache-2.0）。**程序不会自己下载** ✗，请手动放：

```bash
# 从 OpenCV Zoo 取 SFace 模型
curl -L -o data/models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

放在 `data/models/` 或 `~/.forge/models/`，或用环境变量 `FORGE_FACE_MODEL=/path/to.onnx` 指定。

实测记录：`opencv-python-headless` **4.14 自带 `cv2.FaceRecognizerSF`** ✓（不需要 contrib 包 ✗，也不必装 insightface ✗）。
opencv 仍封顶 `<5`（5.x 移除了 Haar 检测，见 `docs/vision.md`）。

## 怎么用（不用 Agent）

```python
from forge.camera import OpenCvFrameSource, HaarFaceDetector, capture
from forge.faces import crop_face, make_embedder, FaceStore

det, emb, store = HaarFaceDetector(), make_embedder(), FaceStore()
with OpenCvFrameSource(0) as src:
    frame, faces = capture(src, det)

# 登记
vecs = [emb.embed(crop_face(frame.image, faces[0])) for _ in range(3)]
store.enroll("老王", vecs)

# 识别
m = store.match(emb.embed(crop_face(frame.image, faces[0])))
print(m.name or "未知", m.score, m.reason)
```

## 判定规则：两条闸门，宁可不认

| 参数 | 默认 | 含义 |
|---|---|---|
| `threshold` | **0.36** | 余弦相似度低于它就回「未知」（SFace 官方参考值 0.363） |
| `margin` | **0.03** | 最佳与次佳至少差这么多，否则也回「未知」 |

第二条是防「认错人」的关键：两个人长得像时，最佳分可能刚过阈值但优势不明显 ——
这时候说「未知」比认错人便宜得多。**代码从不硬猜** ✓。

> **阈值与维度强相关**（实现时踩到的真坑 ✗✓）：随机单位向量的 |cos| 期望约 `1/√d`。
> 8 维时 ≈ 0.35（正好贴着 0.36 阈值，导致不相干的人也过闸 ✗）；128 维 ≈ 0.09（安全 ✓）。
> 所以换提取器（维度变了）**必须重新校准阈值，并重新登记**（维度不一致会明确报错）。

## 隐私（硬约束，请当设计前提读）

1. **只存特征向量，绝不存原图** —— 库里 `faces` 表只有 `vec` 一个 BLOB 字段；删一个人就是删几行向量，
   **无法从库里还原出人脸** ✓；
2. **`data/` 与 `*.db` 已在 `.gitignore`** ✓ —— 结构上进不了 git、上不了 GitHub；
3. **`face_forget(name)` 彻底删除**（含全部向量，不可恢复）；
4. **登记需本人同意** —— 代码只能提供删除能力，「同意」这件事由使用者负责。

> ⚠️ **注意同步范围**：`data/faces.db` 在 `D:\知识库` 里 → 会**被 Syncthing 同步到你的其它设备** ✗。
> 生物特征数据如果你不希望跨设备同步，把库指到库外：
>
> ```yaml
> # config/models.yaml
> face:
>   db: "~/.forge/faces.db"     # 库外、不参与 Syncthing
> ```

## 尚未验证的部分（如实标注）

- **真人识别精度未测** ✗：本机没有摄像头，也没放模型 → 自动化测试只覆盖**判定链路**
  （相似度、登记/删除/持久化、阈值与 margin 两条闸门、维度不一致、能力探测、工具分级）。
- **对齐问题**：SFace 官方建议先做人脸对齐（`alignCrop`，需要关键点）。现在用 Haar 框直接裁再缩放，
  **不做对齐**，精度会低一些。想更准就把检测器换成能出关键点的（如 YuNet），
  再走 `cv2.FaceRecognizerSF.alignCrop` —— `FaceDetector` 接口已经留好位置。
- 登记质量比算法更重要：**正脸、光线足、别太远**，每人 3~5 张不同角度最稳。
