"""人脸识别（#18 Phase 2）—— 让 Agent「认识」人：登记 + 识别 + 身份库。

与 `camera.py` 的分工：那边只回答「**有没有人、脸在哪**」；本模块回答「**这是谁**」（登记过的话）。

    camera.capture()  →  Frame + [Face]
    faces.crop_face() →  一张脸的裁剪
    embedder.embed()  →  特征向量（128 维，SFace）
    store.match()     →  名字 / 「未知」

**「认识」= 你自己的特征库，不是模型记忆。** 视觉模型每看一帧都只是「描述画面」，不会跨会话记住谁是谁；
真正让它认识人的，是这里登记下来的向量库。所以登记（enroll）这一步是核心，模型只是可替换的零件。

隐私（硬约束，读代码前先读这四条）：

1. **只存特征向量，绝不存原图** —— 库文件在 `data/` 下，`.gitignore` 已排除（结构上进不了仓库）；
2. **低于阈值一律回「未知」，绝不硬猜** —— 还要求最佳与次佳有明显差距（margin），防两个人被认混；
3. **可一键删除** —— `face_forget(name)` 删该人的全部向量；
4. **登记需本人同意** —— 代码只能提供删除能力，同意这件事得由使用者保证。

识别用的模型（SFace / InsightFace 之类）需要另外下载模型文件，**本模块不偷偷联网下载** ✗：
找不到模型就给一句人话 + 去哪拿 + 放哪里。
"""
from __future__ import annotations

import datetime
import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:                          # 核心安装不含 numpy/cv2（只在 [vision]/[face] extra 里）
    import numpy as np
except ModuleNotFoundError:   # pragma: no cover
    np = None

try:
    import cv2
except ModuleNotFoundError:   # pragma: no cover
    cv2 = None

__all__ = [
    "FaceError",
    "Embedder",
    "StubEmbedder",
    "SFaceEmbedder",
    "crop_face",
    "cosine",
    "FaceStore",
    "face_defaults",
    "sface_available",
    "available",
    "MODEL_URL_HINT",
]

SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"

# SFace 官方用的余弦阈值是 0.363（OpenCV Zoo 给的参考值）——低于它就该说「未知」，别硬认。
DEFAULT_THRESHOLD = 0.36
# 最佳与次佳至少要差这么多，否则说明「这个人跟两个人都很像」→ 宁可回未知
DEFAULT_MARGIN = 0.03

MODEL_URL_HINT = (
    "SFace 模型（约 37MB，Apache-2.0）从 OpenCV Zoo 取：\n"
    "  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx\n"
    "放到 ~/.forge/models/ 或 data/models/，或用环境变量 FORGE_FACE_MODEL 指定路径。\n"
    "（本程序**不会**自己下载模型；也不要求联网识别人脸。）"
)


class FaceError(RuntimeError):
    """人脸识别链路不可用 —— 带「怎么办」，不是裸异常。"""


def _require_numpy():
    if np is None:
        raise FaceError("人脸识别需要 numpy：pip install \"handcraft-agent[vision]\"")


# ══════════════════════════════════════════════════════════════════
# 特征提取器（可插拔：SFace / 替身 / 以后接别的）


class Embedder:
    """特征提取接口：吃「一张脸的裁剪」，吐定长向量。"""

    name = "base"
    dim = 0

    def embed(self, face_image) -> List[float]:      # pragma: no cover - 接口
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}（{self.dim} 维）"


class StubEmbedder(Embedder):
    """替身提取器：**按图像内容**决定性地生成向量（测试用，不需要模型/摄像头）。

    为什么不用随机数：测试要验「同一个人同一张图 → 匹配；不同的图 → 不匹配」，
    所以替身必须**对内容敏感**：同一张图恒定，不同图差别足够大。

    为什么默认 128 维（和 SFace 一致）：**余弦阈值的含义与维度强相关** ——
    随机单位向量之间的 |cos| 期望约 1/√d：8 维 ≈ 0.35（贴着 0.36 的阈值，会导致
    「不相干的人」也能过闸），128 维 ≈ 0.09（安全）。替身维度对齐真实提取器，
    测出来的阈值行为才有参考价值。
    """

    name = "stub"

    def __init__(self, dim: int = 128):
        self.dim = int(dim)

    def embed(self, face_image) -> List[float]:
        import hashlib
        raw = getattr(face_image, "tobytes", lambda: bytes(str(face_image), "utf-8"))()
        out: List[float] = []
        seed = raw
        while len(out) < self.dim:
            seed = hashlib.sha256(seed).digest()
            # **必须居中**（减 127.5）：直接用 b/255 全是正数 → 所有向量挤在正卦限 →
            # 随便两个的余弦都高达 ~0.8（实测 0.779），阈值就完全失效了。居中后才是各向同性。
            out.extend((b - 127.5) / 255.0 for b in seed)
        v = out[: self.dim]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]                     # 归一化：余弦相似度才有意义


class SFaceEmbedder(Embedder):
    """SFace（OpenCV 自带 `cv2.FaceRecognizerSF`，128 维，本地、离线）。

    2026-10-03 实测：`opencv-python-headless` 4.14 就带 `cv2.FaceRecognizerSF` ✓
    （不需要 contrib 包）。**模型文件不随 pip 包发布** ✗，得自己放一份（见 `MODEL_URL_HINT`）。

    **对齐到底值不值？实测说话**（2026-10-03，本机摄像头，同批画面 10 帧，同一人）：

        对齐(YuNet+alignCrop) 中位 0.892 · 零边距裁剪 中位 0.862  → **+0.030**
        但「裁剪时外扩 25% 边距」也有 0.894 —— **同样的 +0.03 便宜的拿得到** ✗✓

    结论：对齐**确实**比"紧贴框裁剪"好约 +0.03 ✓，但**不比默认的（带边距）裁剪更好** ✗，
    而且两条路都是 10/10 命中、远高于 0.36 阈值 → **功能上无差别** ✓。
    所以 YuNet 的真正收益在**检测更稳**（逆光 12/12 vs Haar 0/12 ✓）+ **关键点可作他用**，
    而不是"识别更准" ✗。别把对齐说成精度银弹 ✓。
    """

    name = "sface"
    dim = 128

    def __init__(self, model_path: Optional[str] = None):
        _require_numpy()
        path = model_path or os.environ.get("FORGE_FACE_MODEL", "")
        if not path:
            for cand in (os.path.join("data", "models", SFACE_FILENAME),
                         os.path.expanduser(os.path.join("~", ".forge", "models", SFACE_FILENAME))):
                if os.path.isfile(cand):
                    path = cand
                    break
        if not path:
            raise FaceError("找不到 SFace 模型文件。\n" + MODEL_URL_HINT)
        if not os.path.isfile(path):
            raise FaceError(f"SFace 模型路径不存在：{path}\n" + MODEL_URL_HINT)
        if cv2 is None or not hasattr(cv2, "FaceRecognizerSF"):
            raise FaceError(
                "当前 opencv 没有 FaceRecognizerSF（人脸特征提取）。"
                "装 4.x 版：pip install \"opencv-python-headless>=4.9,<5\""
            )
        try:
            self._rec = cv2.FaceRecognizerSF.create(path, "")
        except Exception as e:                        # pragma: no cover - 模型损坏等
            raise FaceError(f"SFace 模型加载失败：{e}")
        self.model_path = path

    def embed(self, face_image) -> List[float]:
        # SFace 要求 112x112 输入。这条路**不做对齐**（对齐需要关键点，Haar 给不出）——
        # 有 YuNet 时请走 embed_aligned ✓。
        feat = self._rec.feature(cv2.resize(face_image, (112, 112)))
        return [float(x) for x in np.asarray(feat).reshape(-1)]

    def embed_aligned(self, image, row) -> List[float]:
        """用检测器的**原始输出行**（含 5 个关键点）先做人脸对齐，再提特征。

        `row` 就是 `camera.YuNetFaceDetector.detect_with_landmarks()` 返回的那一行 ✓。

        实测（10 帧同一人）：对齐 中位 0.892 vs 零边距裁剪 0.862 → **+0.030** ✓，
        但带 25% 边距的普通裁剪也有 0.894 ✗ → 对齐**不比默认路径更好**，别指望它提精度 ✓。
        """
        aligned = self._rec.alignCrop(image, row)
        feat = self._rec.feature(aligned)
        return [float(x) for x in np.asarray(feat).reshape(-1)]

    def describe(self) -> str:
        return f"sface（128 维 · {os.path.basename(self.model_path)}）"


def sface_available(model_path: Optional[str] = None) -> bool:
    """SFace 到底能不能用（**能力探测**，不是只看 import）——同 camera.haar_available 的教训。"""
    if np is None or cv2 is None or not hasattr(cv2, "FaceRecognizerSF"):
        return False
    p = model_path or os.environ.get("FORGE_FACE_MODEL", "")
    if p:
        return os.path.isfile(p)
    return any(os.path.isfile(c) for c in (
        os.path.join("data", "models", SFACE_FILENAME),
        os.path.expanduser(os.path.join("~", ".forge", "models", SFACE_FILENAME)),
    ))


# ══════════════════════════════════════════════════════════════════
# 裁剪与相似度


def crop_face(image, box, margin: float = 0.25):
    """按检测框裁一张脸（略微外扩，识别更稳）。返回裁剪后的图（BGR）。"""
    _require_numpy()
    h, w = image.shape[:2]
    x, y, bw, bh = int(box.x), int(box.y), int(box.w), int(box.h)
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    if x1 <= x0 or y1 <= y0:
        raise FaceError("人脸框无效（裁出来是空图）")
    return image[y0:y1, x0:x1]


def embed_face(embedder, image, detector=None, index: int = 0) -> Tuple[List[float], str]:
    """把画面里第 `index` 张脸变成特征向量，返回 `(向量, 走的那条路)`。

    有 `YuNet + SFace` 时走 `alignCrop` 对齐 ✓，否则退回「裁脸 → 缩放」✓（Haar、或替身提取器）。

    对齐的实测收益很小（中位相似度 +0.030），**不如它带来的"检测更稳"重要** ——
    详见 `SFaceEmbedder` 的类文档，那里有完整的三腿对照数字 ✓。
    """
    from .camera import make_detector                   # 延迟导入，避免顶层循环引用
    det = detector if detector is not None else make_detector()
    if hasattr(det, "detect_with_landmarks") and hasattr(embedder, "embed_aligned"):
        found = det.detect_with_landmarks(image)
        if not found:
            raise FaceError("画面里没检测到人脸")
        _face, row = found[min(max(0, index), len(found) - 1)]
        return embedder.embed_aligned(image, row), "对齐"
    faces = det.detect(image)
    if not faces:
        raise FaceError("画面里没检测到人脸")
    return embedder.embed(crop_face(image, faces[min(max(0, index), len(faces) - 1)])), "裁剪"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度（两个向量都该是归一化的；这里仍做一次归一化防手抖）。"""
    _require_numpy()
    va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# ══════════════════════════════════════════════════════════════════
# 身份库（只存向量，SQLite）


@dataclass
class Match:
    """匹配结果。`unknown=True` 时 `name` 为空 —— **绝不硬猜**。"""

    name: str = ""
    score: float = 0.0
    runner_up: float = 0.0
    unknown: bool = True
    reason: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "score": round(self.score, 4),
                "runner_up": round(self.runner_up, 4), "unknown": self.unknown, "reason": self.reason}


class FaceStore:
    """人脸身份库：`people` + `faces(向量 BLOB)`。

    **只存向量，不存图** —— 库里没有原图，删掉一个人就是删掉几行向量，无法从库里还原出脸。
    """

    def __init__(self, path: Optional[str] = None):
        _require_numpy()
        self.path = path or face_defaults()["db"]
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS people (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT UNIQUE NOT NULL,
                created TEXT NOT NULL,
                notes   TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS faces (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                dim       INTEGER NOT NULL,
                vec       BLOB NOT NULL,
                created   TEXT NOT NULL,
                source    TEXT DEFAULT '',
                FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
            """
        )
        self._conn.commit()

    # ── 登记 / 删除 / 查询 ────────────────────────────────────────
    def enroll(self, name: str, vectors: Sequence[Sequence[float]], source: str = "") -> dict:
        """登记（或追加）一个人的若干张脸。返回 {name, added, total}. """
        name = (name or "").strip()
        if not name:
            raise FaceError("登记需要名字")
        vecs = [list(map(float, v)) for v in vectors]
        if not vecs:
            raise FaceError("至少要有一个特征向量才能登记")
        now = datetime.datetime.now().isoformat(timespec="seconds")
        cur = self._conn.execute("SELECT id FROM people WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is None:
            cur = self._conn.execute("INSERT INTO people(name, created) VALUES (?, ?)", (name, now))
            pid = int(cur.lastrowid)
            created = True
        else:
            pid = int(row["id"])
            created = False
        for v in vecs:
            self._conn.execute(
                "INSERT INTO faces(person_id, dim, vec, created, source) VALUES (?, ?, ?, ?, ?)",
                (pid, len(v), np.asarray(v, dtype="float32").tobytes(), now, source),
            )
        self._conn.commit()
        return {"name": name, "added": len(vecs), "total": self.count_faces(name), "new_person": created}

    def set_note(self, name: str, note: str) -> dict:
        """给某人写一行备注（#18 按人记忆的最小版 ✓）。`face_forget` 会连人带备注一起删 ✓。"""
        cur = self._conn.execute("UPDATE people SET notes = ? WHERE name = ?", (str(note or ""), str(name)))
        if cur.rowcount == 0:
            raise FaceError("没登记过这个人（备注无处可放）：%s" % name)
        self._conn.commit()
        return {"name": str(name), "note": str(note or "")}

    def get_note(self, name: str) -> str:
        row = self._conn.execute("SELECT notes FROM people WHERE name = ?", (str(name),)).fetchone()
        if row is None:
            raise FaceError("没登记过这个人：%s" % name)
        return row["notes"] or ""

    def forget(self, name: str) -> dict:
        """删除某人（连同其全部向量）。库里没有原图，删了就真的没了。"""
        cur = self._conn.execute("SELECT id FROM people WHERE name = ?", ((name or "").strip(),))
        row = cur.fetchone()
        if row is None:
            return {"name": name, "removed": 0, "found": False}
        pid = int(row["id"])
        n = self._conn.execute("SELECT COUNT(*) c FROM faces WHERE person_id = ?", (pid,)).fetchone()["c"]
        self._conn.execute("DELETE FROM faces WHERE person_id = ?", (pid,))
        self._conn.execute("DELETE FROM people WHERE id = ?", (pid,))
        self._conn.commit()
        return {"name": name, "removed": int(n), "found": True}

    def people(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT p.name, p.created, p.notes, COUNT(f.id) AS faces FROM people p "
            "LEFT JOIN faces f ON f.person_id = p.id GROUP BY p.id ORDER BY p.name"
        ).fetchall()
        return [{"name": r["name"], "created": r["created"], "note": r["notes"] or "",
                 "faces": int(r["faces"])} for r in rows]

    def count_faces(self, name: Optional[str] = None) -> int:
        if name is None:
            return int(self._conn.execute("SELECT COUNT(*) c FROM faces").fetchone()["c"])
        return int(self._conn.execute(
            "SELECT COUNT(*) c FROM faces f JOIN people p ON p.id = f.person_id WHERE p.name = ?",
            (name,)).fetchone()["c"])

    def vectors(self) -> Tuple[List[str], "object"]:
        """取出全部向量（人少时直接全量比 —— 几百人以内没必要上向量索引）。"""
        rows = self._conn.execute(
            "SELECT p.name AS name, f.dim AS dim, f.vec AS vec FROM faces f JOIN people p ON p.id = f.person_id"
        ).fetchall()
        names = [r["name"] for r in rows]
        vecs = [np.frombuffer(r["vec"], dtype="float32") for r in rows]
        if not vecs:
            return [], np.zeros((0, 0), dtype="float32")
        dims = {int(v.shape[0]) for v in vecs}
        if len(dims) > 1:
            # 换过特征提取器就会出现这种情况。**必须明确报错**：零填充会让不同人比出假相似度，
            # 静默地把某个人「认成别人」或「永远认不出」—— 比直接报错危险得多。
            raise FaceError(
                f"人脸库里存在 {len(dims)} 种维度的向量 {sorted(dims)} —— 通常是换过特征提取器导致的。"
                "请按新提取器重新登记（face_forget 之后 face_enroll）。"
            )
        return names, np.vstack(vecs)

    def match(self, vec: Sequence[float], threshold: float = DEFAULT_THRESHOLD,
              margin: float = DEFAULT_MARGIN) -> Match:
        """最近邻 + 阈值 + **最佳/次佳差距**。三条都过才给名字，否则「未知」。

        为什么要 margin：两个人长相接近时，最佳分数可能略高于阈值但优势不明显 ——
        这时候认错人的代价比说「未知」大得多，所以宁可回未知。

        **关键：比较单元是「人」，不是「样本」** ✗✓ —— 一个人常常登记了 3~5 张向量，
        若拿"每张样本"排序，同一个人自己的第 2 张就会变成"次佳"，于是 margin 永远不过、
        谁都认不出来（2026-10-03 真机实测命中率只有 2/10，原因就是这个 ✗）。
        所以先按人取最高分，再在**人**之间比。
        """
        names, mat = self.vectors()
        if not names:
            return Match(unknown=True, reason="库是空的（还没登记过人）")
        q = np.asarray(list(map(float, vec)), dtype="float32")
        if q.shape[0] != mat.shape[1]:
            raise FaceError(f"向量维度不一致：查询 {q.shape[0]} 维，库里 {mat.shape[1]} 维"
                            f"（换了提取器就会这样，需要重新登记）")
        nq = float(np.linalg.norm(q)) or 1.0
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        sims = (mat @ q) / (norms * nq)
        per_person: Dict[str, float] = {}
        for nm, s in zip(names, sims):
            s = float(s)
            if s > per_person.get(nm, -1.0):
                per_person[nm] = s                      # 一个人取他自己最高的那张
        ranked = sorted(per_person.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        if best < threshold:
            return Match(score=best, runner_up=runner, unknown=True,
                         reason=f"最佳只有 {best:.3f}，低于阈值 {threshold:.2f}")
        if best - runner < margin:
            return Match(score=best, runner_up=runner, unknown=True,
                         reason=f"与两个人都像（{best:.3f} vs {runner:.3f}，差 {best - runner:.3f} < {margin}）")
        return Match(name=best_name, score=best, runner_up=runner, unknown=False, reason="命中")

    def stats(self) -> dict:
        return {"db": self.path, "people": len(self.people()), "vectors": self.count_faces()}

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:                             # pragma: no cover
            pass


# ══════════════════════════════════════════════════════════════════
# 配置（config 的 face: 段；命令行/环境变量优先）


def face_defaults() -> dict:
    """读取人脸相关配置。优先级：环境变量 > config 的 face: 段 > 默认值。"""
    out = {
        "db": os.path.join("data", "faces.db"),
        "embedder": "auto",              # auto | sface | stub | none
        "model": os.environ.get("FORGE_FACE_MODEL", ""),
        "threshold": DEFAULT_THRESHOLD,
        "margin": DEFAULT_MARGIN,
        "samples": 12,                   # 登记时默认抓几张（依据见下 ✓）
        # 登记默认抓几帧：**跨姿态泛化实测**（2026-10-06，同一人 4 分钟不同姿态、留出法）——
        #   3 帧 68% · 5 帧仅 80% ✗ · 8 帧 91% · **12 帧 94%** ✓ · 20 帧 98% ✓；
        #   **任何帧数下认错都是 0** ✓✓（错的形态只有「说不清」✓，正是设计要的）。
        #   20fps 下 12 帧 ≈0.6 秒 → 没理由省 ✓，故默认从 3 提到 12。
        "detector": "auto",              # auto | yunet | haar（见 camera.make_detector）
    }
    try:
        from .config import load_config
        cfg = load_config() or {}
        seg = cfg.get("face") or {}
        if isinstance(seg, dict):
            for k in ("db", "embedder", "model", "threshold", "margin", "samples", "detector"):
                if k in seg and seg[k] not in (None, ""):
                    out[k] = seg[k]
        if not out["model"]:
            out["model"] = os.environ.get("FORGE_FACE_MODEL", "")
    except Exception:                                 # pragma: no cover - 配置坏了不该拖垮识别
        pass
    # 路径统一展开 `~` ✓ —— 配置里写 `~/.forge/faces.db` 是**人类可读**的写法，
# 但不展开的话 sqlite 会去建一个名字就叫 `~` 的目录 ✗（模型路径同理）。
    for k in ("db", "model"):
        if out.get(k):
            out[k] = os.path.expanduser(str(out[k]))
    try:
        out["threshold"] = float(out["threshold"])
        out["margin"] = float(out["margin"])
        out["samples"] = max(1, int(out["samples"]))
    except (TypeError, ValueError):                   # pragma: no cover
        out["threshold"], out["margin"], out["samples"] = DEFAULT_THRESHOLD, DEFAULT_MARGIN, 12
    return out


def make_embedder(kind: Optional[str] = None, model_path: Optional[str] = None) -> Embedder:
    """按配置造提取器。`auto`：有 SFace 就用 SFace，否则**明确报不可用**（不假装有）。"""
    d = face_defaults()
    kind = (kind or d["embedder"] or "auto").lower()
    if kind == "none":
        raise FaceError("配置里人脸提取器是 none")
    if kind == "stub":
        return StubEmbedder()
    if kind == "sface":
        return SFaceEmbedder(model_path or d["model"])
    # auto
    if sface_available(model_path or d["model"]):
        return SFaceEmbedder(model_path or d["model"])
    raise FaceError(
        "没有可用的特征提取器（identify 需要它）。\n"
        "要么放一份 SFace 模型（见下），要么在 config 的 face.embedder 写 stub（仅供测试）。\n"
        + MODEL_URL_HINT
    )


def available() -> dict:
    """如实盘点人脸能力（同 camera.available 的「不假装有」）。"""
    from .camera import available as _camera_available
    d = face_defaults()
    return {
        "detector": _camera_available().get("detector", ""),   # 检测器名（yunet/haar/空）
        "sface": sface_available(d["model"]),
        "embedder": "sface" if sface_available(d["model"]) else "",
        "db": d["db"],
        "threshold": d["threshold"],
        "margin": d["margin"],
    }


# ══════════════════════════════════════════════════════════════════
# 工具（接进工具系统；同样是「没依赖就不挂」）


def register_face_tools() -> bool:
    """注册 face_* 工具。没装 numpy/cv2 就不挂（不假装有）。"""
    if np is None or cv2 is None:
        return False
    from .tools import tool

    def _store() -> FaceStore:
        return FaceStore(face_defaults()["db"])

    def _embedder() -> Embedder:
        return make_embedder()

    @tool(
        name="face_people",
        description="列出人脸库里已登记的人（名字 + 各有几张人脸向量）。只是名单，不含任何图像数据。",
        parameters={"type": "object", "properties": {}},
        read_only=True,
    )
    def face_people() -> str:
        st = _store()
        try:
            ppl = st.people()
            if not ppl:
                return "人脸库是空的（还没登记过人）。"
            return "已登记 " + str(len(ppl)) + " 人：" + "；".join(
                f"{p['name']}（{p['faces']} 张向量）" + (f"，备注：{p['note']}" if p.get("note") else "")
                for p in ppl)
        finally:
            st.close()

    @tool(
        name="face_who",
        description=(
            "看一眼摄像头并回答「这是谁」（需先登记过）。返回登记的名字与相似度；"
            "低于阈值或跟两个人都像时**如实回「未知」**，不会硬猜。"
        ),
        parameters={"type": "object",
                    "properties": {"index": {"type": "integer", "description": "摄像头编号，默认 0"}}},
        read_only=True,
    )
    def face_who(index: int = 0) -> str:
        from .camera import OpenCvFrameSource, capture, make_detector
        d = face_defaults()
        emb = _embedder()
        det = make_detector(d["detector"])          # 有 YuNet 就用 YuNet（更稳 + 能对齐）✓
        src = OpenCvFrameSource(index=int(index))
        try:
            src.open()
            frame, faces = capture(src, det)
        finally:
            src.close()
        if not faces:
            return "画面里没有人（或没检测到正脸）。"
        lines = []
        st = _store()
        try:
            for i in range(len(faces)):
                v, _how = embed_face(emb, frame.image, det, i)     # 走对齐那条更准的路 ✓
                m = st.match(v, d["threshold"], d["margin"])
                who = m.name if not m.unknown else "未知"
                lines.append(f"#{i} {who}（相似度 {m.score:.3f}"
                             + ("" if not m.unknown else f"，原因：{m.reason}") + "）")
        finally:
            st.close()
        return f"画面里 {len(faces)} 张脸：" + "；".join(lines)

    @tool(
        name="face_enroll",
        description=(
            "给画面里的人登记名字（抓若干帧取特征向量存进本地人脸库）。"
            "**只存向量、不存图**。这是写操作，且应当事先取得对方同意。"
        ),
        parameters={"type": "object",
                    "properties": {"name": {"type": "string", "description": "要登记的名字"},
                                   "samples": {"type": "integer", "description": "抓几帧（默认 12；跨姿态实测 12 帧≈94%、3 帧仅 68% ✓）"},
                                   "index": {"type": "integer", "description": "摄像头编号，默认 0"}},
                    "required": ["name"]},
        read_only=False,
    )
    def face_enroll(name: str, samples: int = 0, index: int = 0) -> str:
        from .camera import OpenCvFrameSource, make_detector
        d = face_defaults()
        n = max(1, int(samples or d["samples"]))
        emb = _embedder()
        det = make_detector(d["detector"])
        vecs, has_align, misses = [], 0, 0
        src = OpenCvFrameSource(index=int(index))
        try:
            src.open()
            for _ in range(n):
                frame = src.read()
                if frame is None:
                    misses += 1
                    continue
                try:
                    vec, how = embed_face(emb, frame.image, det, 0)
                except FaceError:                      # 这一帧没检测到脸
                    misses += 1
                    continue
                vecs.append(vec)
                has_align += 1 if how == "对齐" else 0
        finally:
            src.close()
        if not vecs:
            raise FaceError(f"登记失败：{n} 帧里一帧都没检测到人脸（请正对摄像头、光线足一些）")
        st = _store()
        try:
            res = st.enroll(name, vecs, source=emb.name)
        finally:
            st.close()
        how = f"（{has_align}/{len(vecs)} 张走了关键点对齐）" if has_align else "（未做关键点对齐）"
        return (f"已登记 {res['name']}：新增 {res['added']} 张向量，共 {res['total']} 张"
                + ("（新名字）" if res["new_person"] else "（追加到已有名字）") + how
                + (f"；有 {misses} 帧没检测到人脸" if misses else ""))

    @tool(
        name="face_note",
        description=(
            "给**已登记**的人写一行备注（按人记忆的最小版）。例如「喜欢冰美式」「在带孩子」"
            "「上次问过保养流程」。备注会随 face_forget 一起删除；传空串即清除。"
            "读回来用 face_people（它会一起列出备注）。这是写操作。"
        ),
        parameters={"type": "object",
                    "properties": {"name": {"type": "string", "description": "已登记的名字"},
                                   "note": {"type": "string", "description": "备注内容；空串 = 清除"}},
                    "required": ["name", "note"]},
        read_only=False,
    )
    def face_note(name: str, note: str) -> str:
        st = _store()
        try:
            st.set_note(name, note)
            who = name
            return ("已记住 " + who + " 的备注：" + note) if note else ("已清除 " + who + " 的备注。")
        except FaceError:
            return "人脸库里没有「" + str(name) + "」这个人 —— 先用 face_enroll 登记，再写备注。"
        finally:
            st.close()

    @tool(
        name="face_forget",
        description="从人脸库彻底删除某人的全部特征向量（不可恢复；库里本来也没有原图）。",
        parameters={"type": "object",
                    "properties": {"name": {"type": "string", "description": "要删除的名字"}},
                    "required": ["name"]},
        read_only=False,
    )
    def face_forget(name: str) -> str:
        st = _store()
        try:
            res = st.forget(name)
        finally:
            st.close()
        if not res["found"]:
            return f"人脸库里没有「{res['name']}」这个人（无需删除）。"
        return f"已删除「{res['name']}」及其 {res['removed']} 张向量。"

    return True
