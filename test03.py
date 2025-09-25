from __future__ import annotations
import argparse
import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

# ====== MediaPipe Tasks: PoseLandmarker（多人）======
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.framework.formats import landmark_pb2

# for landmark indices (same as Solutions Pose 33)
mp_pose_enum = mp.solutions.pose.PoseLandmark
mp_draw = mp.solutions.drawing_utils
POSE_CONN = mp.solutions.pose.POSE_CONNECTIONS
LM_SPEC = mp_draw.DrawingSpec(thickness=2, circle_radius=2)
CONN_SPEC = mp_draw.DrawingSpec(thickness=2)


# ---------------- 參數（沿用 test02.py） ----------------
@dataclass
class Params:
    fall_vdrop: float = 0.40  # 每影像高度/秒
    fall_angle_deg: float = 55.0  # 軀幹相對垂直角度（≥ 視為水平）
    fall_aspect: float = 1.20  # bbox 寬/高（≥ 視為水平）
    fall_dwell_s: float = 0.5  # 水平姿勢需持續秒數
    candidate_window_s: float = 1.0  # 快速下降後的驗證窗口
    cooldown_s: float = 3.0  # 抑制短時間重複觸發
    min_vis: float = 0.3  # Tasks 無 visibility，此值僅保留語意
    recover_dwell_s: float = 0.7  # 站回直立需持續多久才解除 FALLEN


# ---------------- 跌倒狀態機（沿用 test02.py） ----------------
class FallDetector:
    def __init__(self, img_h: int, params: Params):
        self.h = img_h
        self.params = params
        self.prev_t: Optional[float] = None
        self.prev_center_y: Optional[float] = None
        self.vel_y: float = 0.0  # px/sec
        self.center_hist = deque(maxlen=15)

        self.state = "IDLE"  # IDLE -> CANDIDATE -> FALLEN
        self.candidate_t0: Optional[float] = None
        self.last_fall_time: float = -1e9
        self.recover_t0: Optional[float] = None

    def update(
        self, t: float, center_y_px: float, horizontal_ok: bool
    ) -> Tuple[str, Dict[str, float]]:
        # 速度估計
        if self.prev_t is not None:
            dt = max(1e-3, t - self.prev_t)
            self.vel_y = (center_y_px - (self.prev_center_y or center_y_px)) / dt
        self.prev_t = t
        self.prev_center_y = center_y_px
        self.center_hist.append(center_y_px)

        now = t
        state_info: Dict[str, float] = {}
        cooldown_ok = (now - self.last_fall_time) >= self.params.cooldown_s

        # 垂直下降速度（影像高度比例/秒）
        vdrop_ratio = (self.vel_y / self.h) if self.h > 0 else 0.0
        fast_drop = vdrop_ratio > self.params.fall_vdrop
        state_info.update(
            {
                "vel_y_px_s": self.vel_y,
                "vdrop_ratio": vdrop_ratio,
                "fast_drop": float(fast_drop),
            }
        )

        if self.state == "IDLE":
            if cooldown_ok and fast_drop:
                self.state = "CANDIDATE"
                self.candidate_t0 = now
                self.recover_t0 = None

        elif self.state == "CANDIDATE":
            elapsed = now - (self.candidate_t0 or now)
            state_info["candidate_elapsed"] = elapsed
            if not cooldown_ok:
                self.state = "IDLE"
                self.candidate_t0 = None
                self.recover_t0 = None
            else:
                if horizontal_ok and elapsed >= self.params.fall_dwell_s:
                    self.state = "FALLEN"
                    self.last_fall_time = now
                    self.recover_t0 = None
                elif elapsed > self.params.candidate_window_s:
                    self.state = "IDLE"
                    self.candidate_t0 = None
                    self.recover_t0 = None

        elif self.state == "FALLEN":
            # 姿勢驅動解除（不以 cooldown 自動解除）
            if horizontal_ok:
                self.recover_t0 = None
            else:
                if self.recover_t0 is None:
                    self.recover_t0 = now
                recover_elapsed = now - self.recover_t0
                state_info["recover_elapsed"] = recover_elapsed
                if recover_elapsed >= self.params.recover_dwell_s:
                    self.state = "IDLE"
                    self.candidate_t0 = None
                    self.recover_t0 = None

        return self.state, state_info


# ---------------- 工具函式 ----------------
def _to_px_normed(lm, w: int, h: int) -> Tuple[float, float, float]:
    # Tasks 的 landmark 沒有 visibility，統一給 1.0 以保留 test02.py 的 API 介面
    return float(lm.x * w), float(lm.y * h), 1.0


def _valid(p: Tuple[float, float, float], min_vis: float) -> bool:
    return p[2] >= min_vis


def _center(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _torso_angle_deg(
    shoulder_c: Tuple[float, float], hip_c: Tuple[float, float]
) -> float:
    dx = shoulder_c[0] - hip_c[0]
    dy = shoulder_c[1] - hip_c[1]
    # 與垂直方向夾角：0=垂直，90=水平
    angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
    return float(angle)


def _bbox_from_points(
    pts: List[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _float_ok(x: Optional[float]) -> bool:
    return (x is not None) and (not math.isnan(x)) and (x > 0)


def estimate_input_fps(
    cap: cv2.VideoCapture, src: str, max_probe_frames: int = 120
) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if _float_ok(fps):
        return float(fps)
    if not src.isdigit():
        times: List[float] = []
        cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(max_probe_frames):
            ok = cap.grab()
            if not ok:
                break
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if _float_ok(t_ms):
                times.append(t_ms)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
        if len(times) >= 2:
            dt_ms = (times[-1] - times[0]) / max(1, (len(times) - 1))
            dt_s = dt_ms / 1000.0
            if _float_ok(dt_s):
                return float(1.0 / dt_s)
    return 30.0


def open_capture(src: str) -> cv2.VideoCapture:
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        if not os.path.exists(src):
            raise FileNotFoundError(f"找不到輸入檔案：{src}")
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("無法開啟輸入來源，請確認裝置/路徑/權限。")
    return cap


def init_writer(
    w: int, h: int, fps: float, out_path: Optional[str]
) -> Optional[cv2.VideoWriter]:
    if not out_path:
        return None
    if not _float_ok(fps):
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(
        *("mp4v" if out_path.lower().endswith(".mp4") else "XVID")
    )
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (int(w), int(h)))
    if not writer.isOpened():
        raise RuntimeError("無法建立輸出影片，請確認路徑/副檔名。")
    return writer


# ---------------- 模型資產（避免使用 full） ----------------
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def ensure_model(model_name: str, model_dir: str = "models") -> str:
    os.makedirs(model_dir, exist_ok=True)
    if model_name not in MODEL_URLS:
        raise ValueError("model 必須為 'lite' 或 'heavy'")
    model_path = os.path.join(model_dir, f"pose_landmarker_{model_name}.task")
    if not os.path.exists(model_path):
        import urllib.request

        print(f"[INFO] 下載模型：{MODEL_URLS[model_name]}")
        urllib.request.urlretrieve(MODEL_URLS[model_name], model_path)
    return model_path


# ---------------- 簡易「多人無 ID」追蹤器 ----------------
@dataclass
class Track:
    detector: FallDetector
    last_center: Tuple[float, float]
    last_update_t: float
    state: str = "IDLE"


class MultiTracker:
    def __init__(
        self,
        img_h: int,
        params: Params,
        dist_thresh_px: float = 120.0,
        stale_s: float = 1.2,
    ):
        self.params = params
        self.img_h = img_h
        self.dist_thresh = dist_thresh_px
        self.stale_s = stale_s
        self.tracks: List[Track] = []

    def _dist2(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def update(
        self, t: float, centers: List[Tuple[float, float]], horiz_flags: List[bool]
    ):
        assigned = [False] * len(centers)

        # 先嘗試為既有 track 指派最近的人
        for tr in self.tracks:
            best_j = -1
            best_d2 = 1e18
            for j, c in enumerate(centers):
                if assigned[j]:
                    continue
                d2 = self._dist2(tr.last_center, c)
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j
            if best_j >= 0 and best_d2 <= (self.dist_thresh * self.dist_thresh):
                c = centers[best_j]
                state, _ = tr.detector.update(t, c[1], horiz_flags[best_j])
                tr.state = state
                tr.last_center = c
                tr.last_update_t = t
                assigned[best_j] = True

        # 未指派者 → 新增 track
        for j, c in enumerate(centers):
            if assigned[j]:
                continue
            det = FallDetector(self.img_h, self.params)
            state, _ = det.update(t, c[1], horiz_flags[j])
            self.tracks.append(
                Track(detector=det, last_center=c, last_update_t=t, state=state)
            )

        # 回收過舊 track
        self.tracks = [
            tr for tr in self.tracks if (t - tr.last_update_t) <= self.stale_s
        ]

        # 依 centers 順序回傳對齊的 (state, detector)
        out_states: List[str] = []
        out_dets: List[Optional[FallDetector]] = []
        for c in centers:
            best_tr = None
            best_d2 = 1e18
            for tr in self.tracks:
                d2 = self._dist2(tr.last_center, c)
                if d2 < best_d2:
                    best_d2 = d2
                    best_tr = tr
            out_states.append(best_tr.state if best_tr else "IDLE")
            out_dets.append(best_tr.detector if best_tr else None)
        return out_states, out_dets


# ---------------- CLI ----------------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MediaPipe Tasks 多人骨架 + 跌倒辨識（沿用 test02.py 演算法；不使用 pose_landmarker_full.task）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", type=str, default="0", help="攝影機索引（如 0）或影片路徑"
    )
    p.add_argument(
        "--output", type=str, default="output.mp4", help="輸出標註影片（.mp4/.avi）"
    )
    p.add_argument("--mirror", action="store_true", help="鏡像顯示（自拍視角）")
    p.add_argument("--no-draw", action="store_true", help="不畫圖，只偵測事件")
    p.add_argument("--show-debug", action="store_true", help="疊加偵測指標與狀態")

    # 跌倒參數（與 test02.py 一致）
    p.add_argument(
        "--fall-vdrop", type=float, default=0.10, help="垂直下降門檻（影像高度比例/秒）"
    )
    p.add_argument(
        "--fall-angle", type=float, default=55.0, help="水平姿勢門檻（度，軀幹對垂直）"
    )
    p.add_argument(
        "--fall-aspect", type=float, default=1.00, help="水平姿勢門檻（bbox 寬/高）"
    )
    p.add_argument(
        "--fall-dwell", type=float, default=0.3, help="水平姿勢需持續時間（秒）"
    )
    p.add_argument(
        "--candidate-window",
        type=float,
        default=1.0,
        help="從快速下降起計的驗證窗口（秒）",
    )
    p.add_argument(
        "--min-vis",
        type=float,
        default=0.5,
        help="關鍵點最低可見度（Tasks 無 visibility，此值僅保留）",
    )

    # 模型 & 偵測設定
    p.add_argument(
        "--model",
        type=str,
        default="heavy",
        choices=["lite", "heavy"],
        help="選擇 Tasks 模型（不含 full）",
    )
    p.add_argument("--max-poses", type=int, default=6, help="同幀最大人數")
    p.add_argument("--min-detection", type=float, default=0.5, help="初始偵測信心")
    p.add_argument(
        "--min-presence", type=float, default=0.5, help="存在信心（pose presence）"
    )
    p.add_argument("--min-tracking", type=float, default=0.5, help="追蹤信心")
    return p.parse_args()


# ---------------- 主程式 ----------------
def main():
    args = build_args()

    # 來源 & 時間軸（以輸入真實 FPS 為準）
    cap = open_capture(args.input)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("讀不到任何影格。")
    if args.mirror:
        frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    input_fps = estimate_input_fps(cap, args.input)
    writer = init_writer(W, H, input_fps, args.output)

    params = Params(
        fall_vdrop=args.fall_vdrop,
        fall_angle_deg=args.fall_angle,
        fall_aspect=args.fall_aspect,
        fall_dwell_s=args.fall_dwell,
        candidate_window_s=args.candidate_window,
        min_vis=args.min_vis,
    )

    # 多人追蹤器（不顯示 ID）
    dist_thresh = max(80.0, 0.08 * max(W, H))
    tracker = MultiTracker(
        img_h=H, params=params, dist_thresh_px=dist_thresh, stale_s=1.2
    )

    # 下載/載入模型（不用 full）
    model_path = ensure_model(args.model)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=args.max_poses,
        min_pose_detection_confidence=args.min_detection,
        min_pose_presence_confidence=args.min_presence,
        min_tracking_confidence=args.min_tracking,
        output_segmentation_masks=False,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # 事件 CSV（與 test02.py 欄位一致/擴充 idx）
    csv_fh = None
    csv_writer = None
    if args.output and args.output != "":
        events_csv = os.path.splitext(args.output)[0] + "_events.csv"
    else:
        events_csv = None
    if events_csv:
        csv_fh = open(events_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(
            [
                "timestamp_s",
                "frame",
                "idx_in_frame",
                "state",
                "vel_y_px_s",
                "vdrop_ratio",
                "torso_angle_deg",
                "bbox_aspect",
            ]
        )

    frame_idx = 0
    try:
        while True:
            if frame_idx > 0:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.mirror:
                    frame = cv2.flip(frame, 1)

            ts_s = frame_idx / max(1.0, input_fps)
            ts_ms = int(ts_s * 1000)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_image, ts_ms)

            annotated = frame.copy()

            centers: List[Tuple[float, float]] = []
            horiz_flags: List[bool] = []
            torsos: List[Optional[float]] = []
            aspects: List[Optional[float]] = []
            bboxes: List[Tuple[int, int, int, int]] = []

            # ---- 針對每個人計算 test02.py 的指標 ----
            if res and res.pose_landmarks:
                for lm_list in res.pose_landmarks:
                    lms = lm_list
                    try:
                        l_sh = lms[mp_pose_enum.LEFT_SHOULDER.value]
                        r_sh = lms[mp_pose_enum.RIGHT_SHOULDER.value]
                        l_hp = lms[mp_pose_enum.LEFT_HIP.value]
                        r_hp = lms[mp_pose_enum.RIGHT_HIP.value]
                        nose = lms[mp_pose_enum.NOSE.value]
                        l_ank = lms[mp_pose_enum.LEFT_ANKLE.value]
                        r_ank = lms[mp_pose_enum.RIGHT_ANKLE.value]
                    except Exception:
                        torsos.append(None)
                        aspects.append(None)
                        continue

                    l_sh_px = _to_px_normed(l_sh, W, H)
                    r_sh_px = _to_px_normed(r_sh, W, H)
                    l_hp_px = _to_px_normed(l_hp, W, H)
                    r_hp_px = _to_px_normed(r_hp, W, H)
                    nose_px = _to_px_normed(nose, W, H)
                    l_ank_px = _to_px_normed(l_ank, W, H)
                    r_ank_px = _to_px_normed(r_ank, W, H)

                    if all(
                        _valid(p, params.min_vis)
                        for p in [l_sh_px, r_sh_px, l_hp_px, r_hp_px]
                    ):
                        shoulder_c = _center(l_sh_px, r_sh_px)
                        hip_c = _center(l_hp_px, r_hp_px)
                        torso_angle = _torso_angle_deg(shoulder_c, hip_c)
                        center_y_px = (shoulder_c[1] + hip_c[1]) / 2.0

                        # bbox 估計（含鼻/腳踝，如「視為可見」）
                        lm_xy = [(shoulder_c[0], shoulder_c[1]), (hip_c[0], hip_c[1])]
                        for p in (nose_px, l_ank_px, r_ank_px):
                            if _valid(p, params.min_vis):
                                lm_xy.append((p[0], p[1]))
                        if len(lm_xy) >= 2:
                            x1, y1, x2, y2 = _bbox_from_points(lm_xy)
                            w_box = max(1.0, x2 - x1)
                            h_box = max(1.0, y2 - y1)
                            bbox_aspect = w_box / h_box
                        else:
                            bbox_aspect = None

                        horizontal_ok = False
                        if torso_angle is not None and bbox_aspect is not None:
                            horizontal_ok = (torso_angle >= params.fall_angle_deg) or (
                                bbox_aspect >= params.fall_aspect
                            )

                        centers.append(((shoulder_c[0] + hip_c[0]) / 2.0, center_y_px))
                        horiz_flags.append(horizontal_ok)
                        torsos.append(torso_angle)
                        aspects.append(bbox_aspect)
                        bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                    else:
                        torsos.append(None)
                        aspects.append(None)

                    # 畫骨架（只要偵測到就畫，不顯示 ID）
                    nl = landmark_pb2.NormalizedLandmarkList(
                        landmark=[
                            landmark_pb2.NormalizedLandmark(
                                x=float(lm.x),
                                y=float(lm.y),
                                z=float(lm.z),
                                visibility=1.0,
                            )
                            for lm in lms
                        ]
                    )
                    mp_draw.draw_landmarks(
                        annotated,
                        nl,
                        POSE_CONN,
                        landmark_drawing_spec=LM_SPEC,
                        connection_drawing_spec=CONN_SPEC,
                    )

            # ---- 多人狀態更新 ----
            if centers:
                states, dets = tracker.update(ts_s, centers, horiz_flags)
            else:
                tracker.update(ts_s, [], [])
                states, dets = [], []

            # ---- 視覺化（FALLEN 疊色/提示）----
            if not args.no_draw and centers:
                for i, (state, box) in enumerate(zip(states, bboxes)):
                    if not box:
                        continue
                    x1, y1, x2, y2 = box
                    if state == "FALLEN":
                        overlay = annotated.copy()
                        cv2.rectangle(
                            overlay,
                            (max(0, x1 - 8), max(0, y1 - 8)),
                            (min(W - 1, x2 + 8), min(H - 1, y2 + 8)),
                            (0, 0, 255),
                            -1,
                        )
                        annotated = cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0)
                        cv2.putText(
                            annotated,
                            "FALL DETECTED",
                            (max(5, x1), max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0) if state != "FALLEN" else (0, 0, 255),
                        2,
                    )

                if args.show_debug:
                    y0 = 28
                    dy = 22

                    def put(txt: str):
                        nonlocal y0
                        cv2.putText(
                            annotated,
                            txt,
                            (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                        y0 += dy

                    put(f"FPS_in: {input_fps:.2f}  Persons: {len(centers)}")
                    for i, (ang, asp, st, det) in enumerate(
                        zip(torsos, aspects, states, dets)
                    ):
                        vdrop = (det.vel_y / H) if (det and H > 0) else float("nan")
                        put(
                            f"[{i}] state={st}  torso={ang if ang is not None else float('nan'):5.1f} "
                            f"deg  aspect={asp if asp is not None else float('nan'):4.2f}  vdrop={vdrop:4.2f}"
                        )

            # ---- 事件紀錄 ----
            if csv_writer is not None and centers:
                for i, (st, ang, asp, det) in enumerate(
                    zip(states, torsos, aspects, dets)
                ):
                    v = det.vel_y if det else float("nan")
                    r = (det.vel_y / H) if (det and H > 0) else float("nan")
                    csv_writer.writerow(
                        [
                            f"{ts_s:.3f}",
                            frame_idx,
                            i,
                            st,
                            f"{v:.3f}",
                            f"{r:.3f}",
                            f"{ang:.2f}" if ang is not None else "",
                            f"{asp:.3f}" if asp is not None else "",
                        ]
                    )

            # ---- 顯示/輸出 ----
            if writer is not None:
                writer.write(annotated)
            if not args.no_draw:
                cv2.imshow("Multi-Person Pose + Fall (no ID)", annotated)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    pass  # 需要可中斷就換成 break

            frame_idx += 1

    finally:
        landmarker.close()
        cap.release()
        if writer is not None:
            writer.release()
        if csv_fh is not None:
            csv_fh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
