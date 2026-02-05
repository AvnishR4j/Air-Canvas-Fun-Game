# 🎨 AI Air Canvas & Circle Tracing Game
### *Touchless Drawing Experience | Built for URJA Fest*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-teal?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)]()

> **"Turn your finger into a digital brush."** > An interactive Computer Vision project that tracks your index finger in real-time to draw on a virtual canvas. Features a competitive "Circle Tracing Game" with live accuracy scoring—perfect for tech fests and exhibitions.

---

## 📸 Demo Preview
<img width="1481" height="798" alt="demo" src="https://github.com/user-attachments/assets/54db146e-a095-440b-a72f-e5fb8e270305" />

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **✋ Real-time Tracking** | High-speed hand detection using Google's **MediaPipe**. |
| **☝️ Smart Drawing** | Draws only when the **Index Finger** is up. Stops when you make a fist. |
| **🎯 Circle Challenge** | A gamified mode where users trace a generated circle to test precision. |
| **📊 Live Scoring** | Algorithms calculate accuracy based on **Radius, Smoothness, & Consistency**. |
| **🏆 Winner Effects** | Celebratory animations trigger for high scores (>90%). |
| **💾 Save & Share** | Instantly save your art + camera frame to the local drive. |
| **🖥️ Public Ready** | Includes **Fullscreen Mode** for kiosks and projector displays. |

---

## 🛠️ Tech Stack

* **Core Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **AI Model:** MediaPipe (Hand Landmarker Task)
* **Maths/Logic:** NumPy

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/AvnishR4j/Air-Canvas-Fun-Game.git](https://github.com/AvnishR4j/Air-Canvas-Fun-Game.git)
cd Air-Canvas-Fun-Game

2. Install DependenciesBashpip install opencv-python mediapipe numpy
3. Run the AppBashpython app.py
Note: The system will automatically download the required hand_landmarker.task model on the first run.🎮 Controls & UsageKey / GestureActionIndex Finger Up ☝️Start DrawingFist / 2+ Fingers ✊Stop Drawing (Hover Mode)GStart New Game (Circle Mode)NGenerate New TargetCClear CanvasSSave Artwork (to /saves folder)FToggle FullscreenQQuit Application🧪 How Accuracy is CalculatedThe system uses a weighted algorithm to score the user's circle drawing:Radius Accuracy (30%): How close is your radius to the target radius?Consistency (40%): Standard Deviation check—is your circle shaky or stable?Smoothness (20%): Measures jagged edges vs. smooth curves.Coverage (10%): Did you complete the full 360° loop?$$ \text{Final Score} = (\text{Radius} \times 0.3) + (\text{Consistency} \times 0.4) + (\text{Smooth} \times 0.2) + (\text{Coverage} \times 0.1) $$📂 Project StructureBashair-canvas/
├── 📄 app.py                 # Main logic (CV pipeline + Game loop)
├── 🧠 hand_landmarker.task   # MediaPipe AI Model (Auto-downloaded)
├── 📂 saves/                 # Output folder for user drawings
├── 📄 requirements.txt       # Dependency list
└── 📄 README.md              # Documentation
🎓 Use CasesCollege Fests: Set it up as a "Precision Challenge" booth.Interactive Kiosks: Touchless displays for museums or malls.Skill Development: A fun way to practice motor skills.Resume Project: Demonstrates proficiency in Applied AI and Linear Algebra.🙌 AuthorAvnish RajThapar University – URJA SocietyGitHub Profile • LinkedInBuilt with ❤️ for the love of Computer Vision.
