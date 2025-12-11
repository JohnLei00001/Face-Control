import argparse
import ctypes
import math
import time
import os
import urllib.request
from collections import deque
import cv2
import numpy as np
from pynput.mouse import Controller, Button

class OneEuro:
    def __init__(self, min_cutoff=1.4, beta=0.2, d_cutoff=1.5):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    def filter(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = self.dx_prev + a_d * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = self.x_prev + a * (x - self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def get_screen_size():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--sensitivity", type=float, default=3.0)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--deadzone", type=float, default=0.02)
    parser.add_argument("--dwell", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dwell_ms", type=int, default=900)
    parser.add_argument("--dwell_radius", type=int, default=10)
    parser.add_argument("--invert_x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--invert_y", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gamma_x", type=float, default=0.6)
    parser.add_argument("--gamma_y", type=float, default=0.6)
    parser.add_argument("--min_cutoff", type=float, default=1.4)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--d_cutoff", type=float, default=1.6)
    parser.add_argument("--dead_enter", type=float, default=0.03)
    parser.add_argument("--dead_exit", type=float, default=0.02)
    parser.add_argument("--median_window", type=int, default=5)
    parser.add_argument("--stable_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alpha_gain", type=float, default=4.0)
    parser.add_argument("--alpha_min", type=float, default=0.08)
    parser.add_argument("--alpha_max", type=float, default=0.6)
    parser.add_argument("--stick_guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stick_enter", type=float, default=0.010)
    parser.add_argument("--stick_exit", type=float, default=0.006)
    parser.add_argument("--snap_px", type=float, default=2.0)
    parser.add_argument("--blink_click", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blink_when_moving", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ear_thresh", type=float, default=0.21)
    parser.add_argument("--ear_high_delta", type=float, default=0.04)
    parser.add_argument("--auto_ear", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ear_alpha", type=float, default=0.3)
    parser.add_argument("--ear_adapt_ms", type=int, default=2000)
    parser.add_argument("--blink_min_ms", type=int, default=60)
    parser.add_argument("--blink_max_ms", type=int, default=600)
    parser.add_argument("--dbl_window_ms", type=int, default=900)
    parser.add_argument("--min_interblink_ms", type=int, default=80)
    parser.add_argument("--blink_cooldown_ms", type=int, default=400)
    parser.add_argument("--blink_freeze", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eye_score_thresh", type=float, default=0.08)
    parser.add_argument("--decimate_n", type=int, default=1)
    parser.add_argument("--pos_avg_n", type=int, default=5)
    parser.add_argument("--click_mode", type=str, default="double")
    parser.add_argument("--ear_adapt_window_ms", type=int, default=5000)
    parser.add_argument("--ear_close_ratio", type=float, default=0.50)
    parser.add_argument("--ear_open_ratio", type=float, default=0.70)
    parser.add_argument("--pipeline", type=str, default="auto")
    parser.add_argument("--max_runtime", type=float, default=0.0)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    mouse = Controller()
    sw, sh = get_screen_size()

    neutral_center = None
    neutral_cursor = (sw // 2, sh // 2)
    mouse.position = neutral_cursor
    last_pos = np.array([float(neutral_cursor[0]), float(neutral_cursor[1])])
    dwell_enabled = args.dwell
    dwell_start = None
    dwell_cooldown_until = 0.0
    controlling = True

    t0 = time.time()
    autocalib_done = False
    invert_x = args.invert_x
    invert_y = args.invert_y
    gamma_x = args.gamma_x
    gamma_y = args.gamma_y
    fx = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)
    fy = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)
    dx_buf = deque(maxlen=max(3, args.median_window))
    dy_buf = deque(maxlen=max(3, args.median_window))
    pos_avg_buf = deque(maxlen=max(1, args.pos_avg_n))
    moving = False
    dead_enter = args.dead_enter
    dead_exit = args.dead_exit
    cur_smoothing = args.smoothing
    stable_mode = args.stable_mode
    last_face = None
    blink_click = args.blink_click
    ear_thresh = args.ear_thresh
    blink_when_moving = args.blink_when_moving
    auto_ear = args.auto_ear
    ear_alpha = args.ear_alpha
    blink_min = args.blink_min_ms / 1000.0
    blink_max = args.blink_max_ms / 1000.0
    dbl_win = args.dbl_window_ms / 1000.0
    blink_cooldown = args.blink_cooldown_ms / 1000.0
    ear_high_delta = args.ear_high_delta
    min_interblink = args.min_interblink_ms / 1000.0
    blink_freeze = args.blink_freeze
    eye_score_thresh = args.eye_score_thresh
    alpha_gain = args.alpha_gain
    alpha_min = args.alpha_min
    alpha_max = args.alpha_max
    stick_guard = args.stick_guard
    stick_enter = args.stick_enter
    stick_exit = args.stick_exit
    snap_px = args.snap_px
    fe = OneEuro(2.0, 0.0, 2.0)
    ear_open_sum = 0.0
    ear_open_cnt = 0
    ear_calib_until = time.time() + args.ear_adapt_ms / 1000.0
    prev_target = np.array([float(neutral_cursor[0]), float(neutral_cursor[1])])
    last_t = time.time()
    moving_stick = True
    blink_active = False
    ear_f_last = -1.0
    ear_hist = deque(maxlen=int(max(30, args.ear_adapt_window_ms / 1000.0 * 30)))
    ear_lo_dyn = ear_thresh
    ear_hi_dyn = ear_thresh + ear_high_delta
    ear_dyn_ready = False
    click_mode = (args.click_mode or "double").strip().lower()
    ear_close_ratio = clamp(args.ear_close_ratio, 0.05, 0.9)
    ear_open_ratio = clamp(args.ear_open_ratio, 0.1, 0.95)
    mp_mesh = None
    hud_pipe = "CV"
    if args.pipeline in ("auto", "mediapipe"):
        try:
            import mediapipe as mp
            mp_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
            hud_pipe = "MP"
        except Exception:
            mp_mesh = None
            hud_pipe = "CV"
    def ensure_lbf():
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        models_dir = os.path.abspath(models_dir)
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, "lbfmodel.yaml")
        if not os.path.exists(path):
            url = "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/modules/face/data/lbfmodel.yaml"
            try:
                urllib.request.urlretrieve(url, path)
            except Exception:
                return None
        return path
    facemark = None
    if hasattr(cv2, "face"):
        try:
            model_path = ensure_lbf()
            if model_path:
                facemark = cv2.face.createFacemarkLBF()
                facemark.loadModel(model_path)
        except Exception:
            facemark = None
    class Blink:
        def __init__(self):
            self.closed_t0 = None
            self.last_blink = None
            self.cool_until = 0.0
            self.state_closed = False
            self.last_release_t = 0.0
        def update(self, ear, t, ear_lo, ear_hi, single):
            closed_low = ear > 0.0 and ear < ear_lo
            open_high = ear > ear_hi
            clicked = False
            if not self.state_closed and closed_low:
                self.closed_t0 = t
                self.state_closed = True
            elif self.state_closed and open_high:
                dur = t - self.closed_t0
                self.closed_t0 = None
                self.state_closed = False
                self.last_release_t = t
                if dur >= blink_min and dur <= blink_max:
                    if single:
                        if t >= self.cool_until:
                            clicked = True
                            self.cool_until = t + blink_cooldown
                            self.last_blink = None
                    else:
                        allow_interval = (self.last_blink is None) or ((t - self.last_blink) >= min_interblink)
                        if self.last_blink is not None and (t - self.last_blink) <= dbl_win and allow_interval and t >= self.cool_until:
                            clicked = True
                            self.cool_until = t + blink_cooldown
                            self.last_blink = None
                        else:
                            self.last_blink = t
            return clicked
    blink = Blink()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        used_mp = False
        cx, cy = None, None
        ear = -1.0
        ear_for_blink = -1.0
        eye_open_score = -1.0
        if mp_mesh is not None:
            res = mp_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if getattr(res, "multi_face_landmarks", None):
                lm = res.multi_face_landmarks[0].landmark
                pts_xy = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)
                mn = pts_xy.min(axis=0)
                mx = pts_xy.max(axis=0)
                x = int(mn[0]); y = int(mn[1])
                fw = int(mx[0] - mn[0]); fh = int(mx[1] - mn[1])
                cx = x + fw // 2
                cy = y + fh // 2
                last_face = (x, y, fw, fh)
                def ear_from_xy(p):
                    p = np.asarray(p, dtype=np.float32)
                    A = np.linalg.norm(p[1] - p[5])
                    B = np.linalg.norm(p[2] - p[4])
                    C = np.linalg.norm(p[0] - p[3])
                    return float((A + B) / (2.0 * C + 1e-6))
                idx_l = [33,160,158,133,153,144]
                idx_r = [362,385,387,263,373,380]
                left = [pts_xy[i] for i in idx_l]
                right = [pts_xy[i] for i in idx_r]
                ear_l = ear_from_xy(left)
                ear_r = ear_from_xy(right)
                ear = (ear_l + ear_r) * 0.5
                ear_for_blink = ear
                used_mp = True
                if args.show:
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        if last_face is not None:
            lx, ly, lw_, lh_ = last_face
            rx0 = max(0, int(lx - lw_ * 0.2))
            ry0 = max(0, int(ly - lh_ * 0.2))
            rx1 = min(w, int(lx + lw_ * 1.2))
            ry1 = min(h, int(ly + lh_ * 1.2))
            roi = gray[ry0:ry1, rx0:rx1]
            if roi.size > 0:
                rfaces = face_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                for (ix, iy, iw_, ih_) in rfaces:
                    faces.append((ix + rx0, iy + ry0, iw_, ih_))
        if not faces:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if (not used_mp) and len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            cx = x + fw // 2
            cy = y + fh // 2
            last_face = (x, y, fw, fh)
            if facemark is not None:
                rects = np.array([[x, y, fw, fh]])
                ok_l, shapes = facemark.fit(gray, rects)
                if ok_l and len(shapes) > 0:
                    pts = shapes[0][0]
                    def ear_from_pts(p):
                        p = np.asarray(p, dtype=np.float32)
                        A = np.linalg.norm(p[1] - p[5])
                        B = np.linalg.norm(p[2] - p[4])
                        C = np.linalg.norm(p[0] - p[3])
                        v = (A + B) / (2.0 * C + 1e-6)
                        return float(v)
                    left = [pts[i] for i in [36,37,38,39,40,41]]
                    right = [pts[i] for i in [42,43,44,45,46,47]]
                    ear_l = ear_from_pts(left)
                    ear_r = ear_from_pts(right)
                    ear = (ear_l + ear_r) * 0.5
            ryh = int(fh * 0.6)
            ryh = max(1, ryh)
            roi_eye = gray[y:y+ryh, x:x+fw]
            eyes = eye_cascade.detectMultiScale(roi_eye, scaleFactor=1.1, minNeighbors=4, minSize=(max(12, int(fw*0.12)), max(10, int(fh*0.08))))
            if len(eyes) > 0:
                eye_open_score = sum(hh for (_,_,_,hh) in eyes) / float(fh)
            else:
                eye_open_score = 0.0
            eye_open_bool = eye_open_score >= (eye_score_thresh + 0.02)
            eye_closed_bool = (len(eyes) == 0) or (eye_open_score < eye_score_thresh)
            if ear <= 0.0:
                ear_for_blink = 0.5 if eye_open_bool else 0.05
            else:
                ear_for_blink = ear
            if args.show:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        elif (not used_mp) and last_face is not None:
            x, y, fw, fh = last_face
            cx = x + fw // 2
            cy = y + fh // 2

        if cx is not None and cy is not None:
            if neutral_center is None:
                if not autocalib_done and time.time() - t0 > 1.5:
                    neutral_center = (cx, cy)
                    autocalib_done = True
            else:
                dx = (cx - neutral_center[0]) / float(w)
                dy = (cy - neutral_center[1]) / float(h)
                dx_buf.append(dx)
                dy_buf.append(dy)
                mdx = float(np.median(np.array(dx_buf)))
                mdy = float(np.median(np.array(dy_buf)))
                now = time.time()
                fdx = fx.filter(mdx, now)
                fdy = fy.filter(mdy, now)
                if ear_for_blink > 0.0:
                    closed_low = ear_for_blink < ear_lo_dyn
                    open_high = ear_for_blink > ear_hi_dyn
                    if blink_freeze:
                        if (not blink_active) and closed_low:
                            blink_active = True
                        elif blink_active and open_high:
                            blink_active = False
                if invert_x:
                    fdx = -fdx
                if invert_y:
                    fdy = -fdy
                if stable_mode:
                    r = math.hypot(fdx, fdy)
                    if moving:
                        if r <= dead_exit:
                            moving = False
                    else:
                        if r >= dead_enter:
                            moving = True
                fdx = math.copysign(abs(fdx) ** gamma_x, fdx)
                fdy = math.copysign(abs(fdy) ** gamma_y, fdy)
                target_x = neutral_cursor[0] + fdx * args.sensitivity * sw
                target_y = neutral_cursor[1] + fdy * args.sensitivity * sh
                target_x = clamp(target_x, 0, sw - 1)
                target_y = clamp(target_y, 0, sh - 1)
                raw_target = np.array([target_x, target_y], dtype=float)
                pos_avg_buf.append(raw_target)
                target = np.mean(np.array(pos_avg_buf), axis=0)
                if blink_freeze and blink_active:
                    target = last_pos.copy()
                if stick_guard:
                    dnx = abs(target[0] - last_pos[0]) / float(sw)
                    dny = abs(target[1] - last_pos[1]) / float(sh)
                    dr = math.hypot(dnx, dny)
                    if moving_stick:
                        if dr <= stick_exit:
                            moving_stick = False
                    else:
                        if dr >= stick_enter:
                            moving_stick = True
                    if not moving_stick:
                        target = last_pos.copy()
                if stable_mode and not moving:
                    target = last_pos.copy()
                now2 = now
                dt = max(now2 - last_t, 1e-6)
                vt = np.linalg.norm(target - prev_target) / float(sw)
                cur_smoothing = args.smoothing / (1.0 + alpha_gain * vt)
                cur_smoothing = clamp(cur_smoothing, alpha_min, alpha_max)
                pos = last_pos * (1.0 - cur_smoothing) + target * cur_smoothing
                if controlling:
                    mouse.position = (int(pos[0]), int(pos[1]))
                moved = np.linalg.norm(pos - last_pos)
                if dwell_enabled and controlling and not blink_click:
                    if moved < args.dwell_radius:
                        if dwell_start is None:
                            dwell_start = now
                        if now - dwell_start >= args.dwell_ms / 1000.0 and now >= dwell_cooldown_until:
                            mouse.click(Button.left, 1)
                            dwell_cooldown_until = now + 0.6
                            dwell_start = None
                    else:
                        dwell_start = None
                if ear_for_blink > 0.0:
                    ear_f = fe.filter(ear_for_blink, now)
                    ear_f_last = ear_f
                    if auto_ear and now <= ear_calib_until:
                        ear_open_sum += max(ear_f, 0.0)
                        ear_open_cnt += 1
                        if ear_open_cnt > 15:
                            ear_thresh = (ear_open_sum / max(1, ear_open_cnt)) * 0.75
                    if auto_ear:
                        ear_hist.append(max(ear_f, 0.0))
                        if len(ear_hist) >= 20:
                            hist = np.array(list(ear_hist), dtype=np.float32)
                            open_est = float(np.percentile(hist, 75))
                            close_est = float(np.percentile(hist, 10))
                            gap = max(1e-6, open_est - close_est)
                            ear_lo_dyn = close_est + ear_close_ratio * gap
                            ear_hi_dyn = close_est + max(ear_close_ratio + 0.05, ear_open_ratio) * gap
                            ear_dyn_ready = True
                    if blink_freeze:
                        closed_low_f = ear_f > 0.0 and ear_f < (ear_lo_dyn if ear_dyn_ready else ear_thresh)
                        open_high_f = ear_f > (ear_hi_dyn if ear_dyn_ready else (ear_thresh + ear_high_delta))
                        if (not blink_active) and closed_low_f:
                            blink_active = True
                        elif blink_active and open_high_f:
                            blink_active = False
                    if blink_click and controlling:
                        allow = blink_when_moving or (not moving_stick)
                        if allow:
                            ear_lo_cur = ear_lo_dyn if ear_dyn_ready else ear_thresh
                            ear_hi_cur = ear_hi_dyn if ear_dyn_ready else (ear_thresh + ear_high_delta)
                            single = (click_mode == "single")
                            if blink.update(ear_f, now, ear_lo_cur, ear_hi_cur, single):
                                mouse.click(Button.left, 1)
                last_pos = pos
                prev_target = target
                last_t = now

        if args.show:
            status = []
            status.append(f"calib={'OK' if neutral_center else '...'}")
            status.append(f"control={'ON' if controlling else 'OFF'}")
            status.append(f"dwell={'ON' if dwell_enabled and not blink_click else 'OFF'}")
            status.append(f"sens={args.sensitivity:.2f}")
            status.append(f"invX={'Y' if invert_x else 'N'}")
            status.append(f"invY={'Y' if invert_y else 'N'}")
            status.append(f"stab={'ON' if stable_mode else 'OFF'}")
            status.append(f"stick={'ON' if stick_guard else 'OFF'}")
            status.append(f"blink={'ON' if blink_click else 'OFF'}")
            status.append(f"freeze={'ON' if blink_active else 'OFF'}")
            status.append(f"clk={'1' if click_mode=='single' else '2'}")
            status.append(f"pipe={'MP' if used_mp else 'CV'}")
            cv2.putText(frame, " ".join(status), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Q退出 C校准 D驻留 S开关 X/Y反向 R/F灵敏度 H稳态 J黏附 B眨眼", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            ear_text = "n/a" if ear_f_last < 0.0 else f"{ear_f_last:.2f}"
            cv2.putText(frame, f"EAR={ear_text}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            now_hud = time.time()
            dbl_left = 0.0
            if blink.last_blink is not None:
                dbl_left = max(0.0, dbl_win - (now_hud - blink.last_blink))
            cool_left = max(0.0, blink.cool_until - now_hud)
            cv2.putText(frame, f"DBL剩余={dbl_left:.2f}s 冷却={cool_left:.2f}s", (120, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            bar_w = 200
            bar_h = 8
            bx = 10
            by = 90
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), 1)
            if dbl_win > 0:
                fill_w = int(bar_w * (dbl_left / dbl_win))
                cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), (0, 200, 255), -1)
            by2 = by + 18
            cv2.rectangle(frame, (bx, by2), (bx + bar_w, by2 + bar_h), (60, 60, 60), 1)
            if cool_left > 0:
                cw = min(cool_left, blink_cooldown)
                fill_w2 = int(bar_w * (cw / blink_cooldown))
                cv2.rectangle(frame, (bx, by2), (bx + fill_w2, by2 + bar_h), (0, 128, 255), -1)
            cv2.imshow("Head Mouse", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('c'):
                neutral_center = None
                t0 = time.time()
                autocalib_done = False
            if k == ord('d'):
                dwell_enabled = not dwell_enabled
                dwell_start = None
            if k == ord('s'):
                controlling = not controlling
            if k == ord('x'):
                invert_x = not invert_x
            if k == ord('y'):
                invert_y = not invert_y
            if k == ord('r'):
                args.sensitivity = clamp(args.sensitivity + 0.3, 0.1, 8.0)
            if k == ord('f'):
                args.sensitivity = clamp(args.sensitivity - 0.3, 0.1, 8.0)
            if k == ord('h'):
                stable_mode = not stable_mode
            if k == ord('b'):
                blink_click = not blink_click
            if k == ord('j'):
                stick_guard = not stick_guard

        if args.max_runtime > 0.0 and (time.time() - t0) >= args.max_runtime:
            break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    if mp_mesh is not None:
        try:
            mp_mesh.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
