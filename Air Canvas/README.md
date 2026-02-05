# Air Canvas - Hand Tracking Drawing App

Draw in the air using your index finger!

## Installation
```bash
pip install opencv-python mediapipe numpy
```

## Usage
```bash
python app.py
```

## Controls
- **Show hand to camera** - Hand tracking starts automatically
- **Move index finger** - Draws green lines
- **'s'** - Save drawing as 'drawing.png'
- **'c'** - Clear canvas
- **'q'** - Quit application

## Requirements
- Python 3.7+
- Webcam
- Good lighting for hand detection

## Files
- `app.py` - Main application
- `hand_landmarker.task` - Hand tracking model (auto-downloaded)
