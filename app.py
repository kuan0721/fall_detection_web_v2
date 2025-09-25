from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import json
import threading
import time
import os
import requests
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

# 導入您的跌倒檢測模組
from test03 import (
    Params,
    FallDetector,
    MultiTracker,
    ensure_model,
    open_capture,
    _to_px_normed,
    _valid,
    _center,
    _torso_angle_deg,
    _bbox_from_points,
    estimate_input_fps,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
socketio = SocketIO(app, cors_allowed_origins="*")

# Telegram Bot 設定
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

# MediaPipe 設定
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.framework.formats import landmark_pb2

mp_pose_enum = mp.solutions.pose.PoseLandmark
mp_draw = mp.solutions.drawing_utils
POSE_CONN = mp.solutions.pose.POSE_CONNECTIONS
LM_SPEC = mp_draw.DrawingSpec(thickness=2, circle_radius=2)
CONN_SPEC = mp_draw.DrawingSpec(thickness=2)


class FallDetectionApp:
    def __init__(self):
        self.is_monitoring = False
        self.cap = None
        self.landmarker = None
        self.tracker = None
        self.params = Params(
            fall_vdrop=0.10,
            fall_angle_deg=55.0,
            fall_aspect=1.00,
            fall_dwell_s=0.3,
            candidate_window_s=1.0,
            min_vis=0.5,
        )
        self.last_notification_time = {}
        self.notification_cooldown = 30  # 30秒內不重複通知

    def send_telegram_notification(self, message, image_data=None):
        """發送Telegram通知"""
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}

            response = requests.post(url, data=data)

            # 如果有圖片，也發送圖片
            if image_data is not None:
                photo_url = (
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                )
                files = {"photo": image_data}
                photo_data = {"chat_id": TELEGRAM_CHAT_ID, "caption": message}
                requests.post(photo_url, data=photo_data, files=files)

            return response.status_code == 200
        except Exception as e:
            print(f"Telegram通知發送失敗: {e}")
            return False

    def process_frame(self, frame, ts_s):
        """處理單一幀並檢測跌倒"""
        H, W = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(ts_s * 1000)
        res = self.landmarker.detect_for_video(mp_image, ts_ms)

        annotated = frame.copy()
        centers = []
        horiz_flags = []
        bboxes = []
        fall_detected = False

        if res and res.pose_landmarks:
            for person_idx, lm_list in enumerate(res.pose_landmarks):
                lms = lm_list
                try:
                    # 取得關鍵點
                    l_sh = lms[mp_pose_enum.LEFT_SHOULDER.value]
                    r_sh = lms[mp_pose_enum.RIGHT_SHOULDER.value]
                    l_hp = lms[mp_pose_enum.LEFT_HIP.value]
                    r_hp = lms[mp_pose_enum.RIGHT_HIP.value]
                    nose = lms[mp_pose_enum.NOSE.value]
                    l_ank = lms[mp_pose_enum.LEFT_ANKLE.value]
                    r_ank = lms[mp_pose_enum.RIGHT_ANKLE.value]

                    # 轉換為像素座標
                    l_sh_px = _to_px_normed(l_sh, W, H)
                    r_sh_px = _to_px_normed(r_sh, W, H)
                    l_hp_px = _to_px_normed(l_hp, W, H)
                    r_hp_px = _to_px_normed(r_hp, W, H)
                    nose_px = _to_px_normed(nose, W, H)
                    l_ank_px = _to_px_normed(l_ank, W, H)
                    r_ank_px = _to_px_normed(r_ank, W, H)

                    if all(
                        _valid(p, self.params.min_vis)
                        for p in [l_sh_px, r_sh_px, l_hp_px, r_hp_px]
                    ):
                        shoulder_c = _center(l_sh_px, r_sh_px)
                        hip_c = _center(l_hp_px, r_hp_px)
                        torso_angle = _torso_angle_deg(shoulder_c, hip_c)
                        center_y_px = (shoulder_c[1] + hip_c[1]) / 2.0

                        # 計算bbox
                        lm_xy = [(shoulder_c[0], shoulder_c[1]), (hip_c[0], hip_c[1])]
                        for p in (nose_px, l_ank_px, r_ank_px):
                            if _valid(p, self.params.min_vis):
                                lm_xy.append((p[0], p[1]))

                        if len(lm_xy) >= 2:
                            x1, y1, x2, y2 = _bbox_from_points(lm_xy)
                            w_box = max(1.0, x2 - x1)
                            h_box = max(1.0, y2 - y1)
                            bbox_aspect = w_box / h_box

                            horizontal_ok = (
                                torso_angle >= self.params.fall_angle_deg
                            ) or (bbox_aspect >= self.params.fall_aspect)

                            centers.append(
                                ((shoulder_c[0] + hip_c[0]) / 2.0, center_y_px)
                            )
                            horiz_flags.append(horizontal_ok)
                            bboxes.append((int(x1), int(y1), int(x2), int(y2)))

                    # 畫骨架
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
                except Exception as e:
                    continue

        # 多人狀態更新
        if centers:
            states, dets = self.tracker.update(ts_s, centers, horiz_flags)
        else:
            self.tracker.update(ts_s, [], [])
            states, dets = [], []

        # 視覺化和通知
        for i, (state, box) in enumerate(zip(states, bboxes)):
            if not box:
                continue

            x1, y1, x2, y2 = box

            if state == "FALLEN":
                fall_detected = True

                # 檢查通知冷卻時間
                now = time.time()
                person_key = f"person_{i}"

                if (
                    person_key not in self.last_notification_time
                    or now - self.last_notification_time[person_key]
                    > self.notification_cooldown
                ):

                    # 準備通知訊息
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    message = f"🚨 <b>跌倒警報!</b>\n\n⏰ 時間: {timestamp}\n👤 檢測到第 {i+1} 人跌倒\n📍 位置: 監控區域"

                    # 準備圖片
                    try:
                        _, buffer = cv2.imencode(".jpg", annotated)
                        img_bytes = BytesIO(buffer)

                        # 發送通知
                        if self.send_telegram_notification(message, img_bytes):
                            self.last_notification_time[person_key] = now

                            # 通過WebSocket發送給前端
                            socketio.emit(
                                "fall_detected",
                                {
                                    "person_id": i,
                                    "timestamp": timestamp,
                                    "bbox": [x1, y1, x2, y2],
                                },
                            )
                    except Exception as e:
                        print(f"處理跌倒通知時發生錯誤: {e}")

                # 視覺標記
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

            # 畫邊框
            color = (0, 0, 255) if state == "FALLEN" else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        return annotated, fall_detected, len(centers)

    def start_monitoring(self, source="0", model_name="heavy"):
        """開始監控"""
        if self.is_monitoring:
            return False

        try:
            # 初始化攝影機
            self.cap = open_capture(source)
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("無法讀取攝影機")

            H, W = frame.shape[:2]
            input_fps = estimate_input_fps(self.cap, source)

            # 初始化追蹤器
            dist_thresh = max(80.0, 0.08 * max(W, H))
            self.tracker = MultiTracker(
                img_h=H, params=self.params, dist_thresh_px=dist_thresh, stale_s=1.2
            )

            # 載入模型
            model_path = ensure_model(model_name)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=6,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

            self.is_monitoring = True

            # 開始監控線程
            thread = threading.Thread(target=self._monitoring_loop)
            thread.daemon = True
            thread.start()

            return True
        except Exception as e:
            print(f"啟動監控失敗: {e}")
            return False

    def _monitoring_loop(self):
        """監控主循環"""
        frame_idx = 0
        start_time = time.time()

        while self.is_monitoring and self.cap and self.cap.isOpened():
            try:
                ok, frame = self.cap.read()
                if not ok:
                    break

                current_time = time.time()
                ts_s = current_time - start_time

                # 處理幀
                annotated, fall_detected, person_count = self.process_frame(frame, ts_s)

                # 編碼為JPEG並發送到前端
                _, buffer = cv2.imencode(".jpg", annotated)
                frame_data = base64.b64encode(buffer).decode("utf-8")

                # 通過WebSocket發送
                socketio.emit(
                    "video_frame",
                    {
                        "image": frame_data,
                        "timestamp": datetime.now().isoformat(),
                        "person_count": person_count,
                        "fall_detected": fall_detected,
                    },
                )

                frame_idx += 1
                time.sleep(0.03)  # 約30 FPS

            except Exception as e:
                print(f"監控循環錯誤: {e}")
                break

        self.stop_monitoring()

    def stop_monitoring(self):
        """停止監控"""
        self.is_monitoring = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
        self.tracker = None


# 全域應用實例
fall_app = FallDetectionApp()


# 路由
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_monitoring():
    data = request.json
    source = data.get("source", "0")
    model = data.get("model", "heavy")

    success = fall_app.start_monitoring(source, model)
    return jsonify({"success": success})


@app.route("/api/stop", methods=["POST"])
def stop_monitoring():
    fall_app.stop_monitoring()
    return jsonify({"success": True})


@app.route("/api/status")
def get_status():
    return jsonify(
        {
            "is_monitoring": fall_app.is_monitoring,
            "telegram_configured": bool(TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE"),
        }
    )


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json

    # 更新參數
    if "fall_vdrop" in data:
        fall_app.params.fall_vdrop = float(data["fall_vdrop"])
    if "fall_angle_deg" in data:
        fall_app.params.fall_angle_deg = float(data["fall_angle_deg"])
    if "fall_aspect" in data:
        fall_app.params.fall_aspect = float(data["fall_aspect"])
    if "fall_dwell_s" in data:
        fall_app.params.fall_dwell_s = float(data["fall_dwell_s"])

    return jsonify({"success": True})


@socketio.on("connect")
def handle_connect():
    print("客戶端已連接")
    emit("connected", {"data": "連接成功"})


@socketio.on("disconnect")
def handle_disconnect():
    print("客戶端已斷開")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
