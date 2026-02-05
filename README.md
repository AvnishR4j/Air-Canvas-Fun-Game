🚀 Features

✋ Real-time hand tracking using MediaPipe

☝️ Index-finger drawing (draw only when one finger is raised)

🎯 Circle tracing game with random target circles

📊 Accuracy calculation based on:

Radius correctness

Consistency

Smoothness

Coverage

🏆 Winner animation for high accuracy scores

💾 Save drawings and camera frames

🖥️ Fullscreen mode for public displays

🖱️ Mouse fallback mode if hand tracking fails

🧠 How It Works

The webcam captures live video.

MediaPipe detects hand landmarks.

When only the index finger is raised, drawing starts.

The user traces a yellow target circle in the air.

Once the circle is completed, the system:

Compares drawn points with the target circle

Computes an accuracy percentage

High scores trigger a winner celebration 🎉

🛠️ Tech Stack

Python

OpenCV

MediaPipe (Hand Landmarker – Tasks API)

NumPy

📦 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/air-canvas.git
cd air-canvas

2️⃣ Install dependencies
pip install opencv-python mediapipe numpy

▶️ Usage

Run the application:

python app.py


Make sure:

A webcam is connected

Lighting is good for hand detection

🎮 Controls
Key	Action
☝️ Index finger only	Start drawing
✊ Fist / multiple fingers	Stop drawing
g	Start new game
n	Generate new circle
s	Save drawing & camera image
c	Clear canvas
f	Toggle fullscreen
q	Quit application
📂 Project Structure
├── app.py                  # Main Air Canvas application
├── hand_landmarker.task    # MediaPipe hand tracking model
├── saves/                  # Saved drawings & camera images
└── README.md               # Project documentation


ℹ️ The hand_landmarker.task model is auto-downloaded if not found.

🧪 Accuracy Scoring Logic

The final accuracy score is a weighted combination of:

Radius Accuracy (30%)

Consistency (Standard Deviation) (40%)

Smoothness of Drawing (20%)

Angular Coverage (10%)

Final score is capped between 0–100%.

🎓 Use Cases

College sports fests & tech events

Interactive exhibition booths

Computer Vision project demos

Human-Computer Interaction (HCI) experiments

Resume / internship projects

⚠️ Requirements

Python 3.7+

Webcam

Good lighting

Decent CPU (no GPU required)

🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

📄 License

This project is for educational and demonstration purposes.
You may reuse or modify it with proper credit.

🙌 Author

Built with ❤️ by Avnish Raj
(Thapar University – URJA Society)
