import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
from collections import deque
from datetime import datetime

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window

# -------- FULL SCREEN --------
Window.fullscreen = 'auto'

# ---------------- MediaPipe ----------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ---------------- Utilities ----------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 180
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def lm_xy(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

# ---------------- Exercise Logic ----------------
def squat(lm, w, h):
    ang = calculate_angle(
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_HIP], w, h),
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_KNEE], w, h),
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE], w, h)
    )
    pct = np.clip((170 - ang), 0, 100)
    return pct, "Good Squat" if pct > 40 else "Go Lower"

def pushup(lm, w, h):
    ang = calculate_angle(
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h),
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_ELBOW], w, h),
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST], w, h)
    )
    pct = np.clip((180 - ang), 0, 100)
    return pct, "Good Push-up" if pct > 40 else "Lower Body"

def plank(lm, w, h):
    ang = calculate_angle(
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h),
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_HIP], w, h),
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE], w, h)
    )
    pct = np.clip((ang - 140) * 2.5, 0, 100)
    return pct, "Hold Plank" if ang > 165 else "Straighten Body"

def overhead(lm, w, h):
    ang = calculate_angle(
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h),
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_ELBOW], w, h),
        lm_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST], w, h)
    )
    pct = np.clip((180 - ang), 0, 100)
    return pct, "Good Press" if pct > 50 else "Push Higher"

class FitnessApp(App):
    def build(self):
        self.exercise = "squat"
        self.reps = 0
        self.stage = "up"
        self.history = deque(maxlen=7)
        self.plank_start = None
        self.current_pct = 0

        today = datetime.now().strftime("%Y-%m-%d")
        self.csv_file = f"fitness_log_{today}.csv"

        if not os.path.exists(self.csv_file):
            pd.DataFrame(columns=[
                "timestamp", "exercise", "rep",
                "accuracy_pct", "feedback", "duration_sec"
            ]).to_csv(self.csv_file, index=False)

        self.cap = cv2.VideoCapture(0)
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        root = BoxLayout(orientation="vertical", padding=12, spacing=12)

        root.add_widget(Label(
            text="[b]AI FITNESS TRAINER[/b]",
            markup=True,
            font_size=32,
            size_hint_y=0.08
        ))

        main = BoxLayout(spacing=12)
        self.img = Image(allow_stretch=True, keep_ratio=True)
        main.add_widget(self.img)

        info = BoxLayout(orientation="vertical", size_hint_x=0.32, spacing=18)

        self.exercise_lbl = Label(font_size=24)
        self.reps_lbl = Label(font_size=40)
        self.percent_lbl = Label(font_size=26)
        self.feedback_lbl = Label(font_size=22, color=(1, 0, 0, 1))

        log_btn = Button(text="SHOW WORKOUT LOG", size_hint_y=0.7)

        log_btn.bind(on_press=self.open_log)

        for w in [
            self.exercise_lbl,
            self.reps_lbl,
            self.percent_lbl,
            self.feedback_lbl,
            log_btn
        ]:
            info.add_widget(w)

        main.add_widget(info)
        root.add_widget(main)

        controls = BoxLayout(size_hint_y=0.12, spacing=8)

        for ex in ["squat", "pushup", "plank", "overhead"]:
            btn = Button(text=ex.upper(), font_size=20)
            btn.bind(on_press=self.change_exercise)
            controls.add_widget(btn)

        quit_btn = Button(
            text="EXIT",
            font_size=20,
            background_color=(1, 0, 0, 1)
        )
        quit_btn.bind(on_press=self.quit_app)
        controls.add_widget(quit_btn)

        root.add_widget(controls)

        Clock.schedule_interval(self.update, 1 / 30)
        return root

    def save_csv(self, exercise, rep, pct, feedback, duration=None):
        pd.DataFrame([{
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "exercise": exercise,
            "rep": rep,
            "accuracy_pct": round(pct, 2) if pct is not None else None,
            "feedback": feedback,
            "duration_sec": duration
        }]).to_csv(self.csv_file, mode="a", header=False, index=False)

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        feedback, pct = "Detecting...", 0

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            if self.exercise == "squat":
                pct, feedback = squat(lm, w, h)
            elif self.exercise == "pushup":
                pct, feedback = pushup(lm, w, h)
            elif self.exercise == "plank":
                pct, feedback = plank(lm, w, h)
            elif self.exercise == "overhead":
                pct, feedback = overhead(lm, w, h)

            self.history.append(pct)
            smooth = np.mean(self.history)
            self.current_pct = smooth

            if self.exercise == "plank":
                if smooth > 80:
                    self.plank_start = self.plank_start or time.time()
                    if time.time() - self.plank_start >= 20:
                        self.reps += 1
                        self.save_csv("plank", self.reps, None, feedback, 20)
                        self.plank_start = None
                else:
                    self.plank_start = None
            else:
                if self.stage == "up" and smooth > 60:
                    self.stage = "down"
                elif self.stage == "down" and smooth < 20:
                    self.stage = "up"
                    self.reps += 1
                    self.save_csv(self.exercise, self.reps, smooth, feedback)

            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        self.exercise_lbl.text = f"Exercise: {self.exercise.title()}"
        self.reps_lbl.text = f"Reps: {self.reps}"
        self.percent_lbl.text = f"Accuracy: {int(self.current_pct)}%"
        self.feedback_lbl.text = feedback

        tex = Texture.create(size=(w, h), colorfmt="bgr")
        tex.blit_buffer(
            cv2.flip(frame, 0).tobytes(),
            colorfmt="bgr",
            bufferfmt="ubyte"
        )
        self.img.texture = tex

    def change_exercise(self, instance):
        self.exercise = instance.text.lower()
        self.reps = 0
        self.stage = "up"
        self.history.clear()
        self.plank_start = None

    def open_log(self, instance):
        box = BoxLayout(orientation="vertical", spacing=12)
        scroll = ScrollView()
        grid = GridLayout(cols=1, size_hint_y=None, padding=12, spacing=12)
        grid.bind(minimum_height=grid.setter("height"))

        df = pd.read_csv(self.csv_file)
        for _, row in df.tail(100).iterrows():
            grid.add_widget(Label(
                text=f"{row['timestamp']} | {row['exercise']} | Rep {row['rep']} | {row['accuracy_pct']}%",
                font_size=18,
                size_hint_y=None,
                height=50
            ))

        scroll.add_widget(grid)
        box.add_widget(scroll)

        close = Button(text="CLOSE", size_hint_y=0.12)
        popup = Popup(
            title="Workout Log",
            content=box,
            size_hint=(0.95, 0.95)
        )
        close.bind(on_press=popup.dismiss)
        box.add_widget(close)
        popup.open()

    def quit_app(self, instance):
        if self.cap.isOpened():
            self.cap.release()
        App.get_running_app().stop()

if __name__ == "__main__":
    FitnessApp().run()
