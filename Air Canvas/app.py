import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
import time

class AirCanvas:
    def __init__(self):
        # Try to use the available MediaPipe API
        try:
            # Try the new Tasks API first
            self.use_tasks_api = True
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            # Download model if needed
            model_path = "hand_landmarker.task"
            if not os.path.exists(model_path):
                self.download_model()
            
            # Create hand landmarker
            base_options = BaseOptions(model_asset_path=model_path)
            options = HandLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionRunningMode.VIDEO,
                num_hands=1
            )
            
            self.hand_landmarker = HandLandmarker.create_from_options(options)
            self.timestamp_ms = int(time.time() * 1000)
            print("Hand tracking initialized successfully!")
            
        except Exception as e:
            print(f"Tasks API failed: {e}")
            # Fall back to a simpler approach without hand tracking
            self.use_tasks_api = False
            print("Using mouse tracking instead - move your mouse to draw")
        
        # Initialize canvas
        self.drawing_color = (0, 255, 0)  # Green color for drawing
        self.thickness = 5
        
        # Previous position for drawing lines
        self.prev_x = None
        self.prev_y = None
        
        # Drawing state
        self.is_drawing = False
        self.last_hand_state = None
        
        # Fullscreen state
        self.is_fullscreen = False
        
        # Game state
        self.target_circle = None
        self.drawing_points = []
        self.circle_completed = False
        self.game_started = False
        self.winner_animation = 0
        self.show_winner = False
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
            
        # Canvas dimensions - start with reasonable size
        self.width = 1280
        self.height = 720
        
        # Create drawing mask
        self.drawing_mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def download_model(self):
        """Download the hand landmarker model"""
        import urllib.request
        
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        model_path = "hand_landmarker.task"
        
        print("Downloading hand landmarker model...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please download the model manually from:")
            print(model_url)
            raise
    
    def is_only_index_finger(self, hand_landmarks):
        """Check if only index finger is extended - simplified and more reliable"""
        if not hand_landmarks:
            return False
            
        try:
            # Get key landmarks
            index_tip = hand_landmarks[8]
            index_pip = hand_landmarks[6]
            middle_tip = hand_landmarks[12]
            middle_pip = hand_landmarks[10]
            ring_tip = hand_landmarks[16]
            ring_pip = hand_landmarks[14]
            pinky_tip = hand_landmarks[20]
            pinky_pip = hand_landmarks[18]
            
            # Check if index finger is extended (tip above pip)
            index_extended = index_tip.y < index_pip.y - 0.02  # Add small threshold
            
            # Check if other fingers are NOT extended (tip below or close to pip)
            middle_down = middle_tip.y > middle_pip.y - 0.01
            ring_down = ring_tip.y > ring_pip.y - 0.01
            pinky_down = pinky_tip.y > pinky_pip.y - 0.01
            
            # Return true if index is up and others are down
            return index_extended and middle_down and ring_down and pinky_down
            
        except:
            return False
    
    def generate_target_circle(self):
        """Generate a target circle in the upper half of the screen"""
        import random
        margin = 100
        
        # Keep circles in upper half for better finger tracking
        max_y = self.height // 2 - 50  # Upper half with some margin
        
        radius = random.randint(80, 150)
        center_x = random.randint(margin + radius, self.width - margin - radius)
        center_y = random.randint(margin + radius, max_y)
        
        self.target_circle = {
            'center': (center_x, center_y),
            'radius': radius
        }
        print(f"New circle: Center ({center_x}, {center_y}), Radius: {radius}")
    
    def calculate_circle_accuracy(self):
        """Calculate how accurate the drawn circle is compared to target"""
        if not self.target_circle or len(self.drawing_points) < 10:
            return 0
        
        center_x, center_y = self.target_circle['center']
        target_radius = self.target_circle['radius']
        
        # Calculate average distance from center for all drawn points
        distances = []
        for point in self.drawing_points:
            dist = np.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
            distances.append(dist)
        
        avg_radius = np.mean(distances)
        radius_std = np.std(distances)
        
        # Calculate accuracy based on:
        # 1. How close average radius is to target radius (40% weight)
        # 2. How consistent the radius is (lower std = better) (60% weight)
        radius_accuracy = max(0, 100 - abs(avg_radius - target_radius) / target_radius * 100)
        consistency_accuracy = max(0, 100 - radius_std / target_radius * 100)
        
        total_accuracy = radius_accuracy * 0.4 + consistency_accuracy * 0.6
        
        return min(100, total_accuracy)
    
    def check_circle_completion(self):
        """Check if the circle has been completed (drawn around and returned near start)"""
        if len(self.drawing_points) < 30:
            return False
        
        # Check if the last point is close to the first point
        first_point = self.drawing_points[0]
        last_point = self.drawing_points[-1]
        
        distance = np.sqrt((last_point[0] - first_point[0])**2 + (last_point[1] - first_point[1])**2)
        
        # Consider circle complete if we've drawn enough points and returned near start
        if distance < 30 and len(self.drawing_points) > 50:
            return True
        
        return False
    
    def calculate_circle_accuracy(self):
        """Calculate highly accurate circle comparison"""
        if not self.target_circle or len(self.drawing_points) < 30:
            return 0
        
        center_x, center_y = self.target_circle['center']
        target_radius = self.target_circle['radius']
        
        # Calculate distances from center for all drawn points
        distances = []
        for point in self.drawing_points:
            dist = np.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
            distances.append(dist)
        
        # Filter outliers (points that are too far from expected radius)
        filtered_distances = [d for d in distances if abs(d - target_radius) < target_radius * 0.5]
        
        if len(filtered_distances) < len(distances) * 0.7:  # If too many outliers
            return max(0, 50 - len(distances) * 0.5)  # Penalty for bad drawing
        
        avg_radius = np.mean(filtered_distances)
        radius_std = np.std(filtered_distances)
        
        # 1. Radius accuracy (30% weight) - how close to target radius
        radius_error = abs(avg_radius - target_radius)
        radius_accuracy = max(0, 100 - (radius_error / target_radius) * 200)  # Stricter penalty
        
        # 2. Consistency accuracy (40% weight) - how uniform the circle is
        consistency_error = radius_std / target_radius
        consistency_accuracy = max(0, 100 - consistency_error * 300)  # Much stricter
        
        # 3. Smoothness accuracy (20% weight) - how smooth the drawing is
        smoothness_penalty = 0
        for i in range(1, len(self.drawing_points) - 1):
            p1, p2, p3 = self.drawing_points[i-1], self.drawing_points[i], self.drawing_points[i+1]
            
            # Calculate angle between consecutive segments
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            if np.sqrt(v1[0]**2 + v1[1]**2) > 0 and np.sqrt(v2[0]**2 + v2[1]**2) > 0:
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
                angle = np.arccos(cos_angle)
                
                # Penalty for sharp angles (not smooth)
                if angle > np.pi/4:  # More than 45 degrees
                    smoothness_penalty += (angle - np.pi/4) * 10
        
        smoothness_accuracy = max(0, 100 - smoothness_penalty / len(self.drawing_points))
        
        # 4. Coverage accuracy (10% weight) - how much of the circle was drawn
        angles = []
        for point in self.drawing_points:
            angle = np.arctan2(point[1] - center_y, point[0] - center_x)
            angles.append(angle)
        
        # Normalize angles to [0, 2π]
        angles = [(a + 2*np.pi) % (2*np.pi) for a in angles]
        angles.sort()
        
        # Calculate angular coverage
        max_gap = 0
        for i in range(len(angles)):
            next_i = (i + 1) % len(angles)
            gap = angles[next_i] - angles[i] if next_i > i else (2*np.pi - angles[i] + angles[next_i])
            max_gap = max(max_gap, gap)
        
        coverage = ((2*np.pi - max_gap) / (2*np.pi)) * 100
        
        # Weighted accuracy calculation with stricter weights
        total_accuracy = (radius_accuracy * 0.3 + consistency_accuracy * 0.4 + 
                         smoothness_accuracy * 0.2 + coverage * 0.1)
        
        return min(100, max(0, total_accuracy))
    
    def draw_winner_celebration(self, frame):
        """Draw winner celebration animation"""
        if self.show_winner:
            # Animated text size based on animation frame
            self.winner_animation += 1
            size = 2 + np.sin(self.winner_animation * 0.1) * 0.5
            
            # Draw winner text with animation
            text = "🏆 WINNER! 🏆"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 3)[0]
            text_x = (self.width - text_size[0]) // 2
            text_y = self.height // 2
            
            # Pulsing color
            color_val = int(128 + 127 * np.sin(self.winner_animation * 0.05))
            color = (0, color_val, 0)  # Green pulsing
            
            # Draw text shadow
            cv2.putText(frame, text, (text_x + 3, text_y + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 5)
            # Draw main text
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, size, color, 3)
            
            # Draw confetti effect
            for i in range(20):
                x = int((np.sin(self.winner_animation * 0.1 + i) + 1) * self.width / 2)
                y = int((self.winner_animation * 2 + i * 30) % self.height)
                colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 0, 0)]
                color = colors[i % len(colors)]
                cv2.circle(frame, (x, y), 5, color, -1)
            
            # Subtitle
            subtitle = f"Perfect Score: {self.calculate_circle_accuracy():.1f}%"
            subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            subtitle_x = (self.width - subtitle_size[0]) // 2
            cv2.putText(frame, subtitle, (subtitle_x, text_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def draw_target_circle(self, frame):
        """Draw the target circle on the frame"""
        if self.target_circle:
            center = self.target_circle['center']
            radius = self.target_circle['radius']
            
            # Draw target circle
            cv2.circle(frame, center, radius, (0, 255, 255), 3)  # Yellow circle
            cv2.circle(frame, center, 5, (0, 255, 255), -1)  # Center point
            
            # Draw "START" text
            cv2.putText(frame, "START", (center[0] - 30, center[1] - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def draw_game_info(self, frame):
        """Draw game information on the frame"""
        # Title
        cv2.putText(frame, "CIRCLE DRAWING GAME", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if not self.game_started:
            cv2.putText(frame, "Press 'g' to start game", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.circle_completed:
            accuracy = self.calculate_circle_accuracy()
            color = (0, 255, 0) if accuracy > 80 else (0, 255, 255) if accuracy > 60 else (0, 0, 255)
            cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Grade
            if accuracy > 95:
                grade = "PERFECT!"
            elif accuracy > 85:
                grade = "EXCELLENT!"
            elif accuracy > 75:
                grade = "GOOD!"
            elif accuracy > 65:
                grade = "OKAY"
            else:
                grade = "TRY AGAIN"
            
            cv2.putText(frame, grade, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame, "Press 'n' for new circle", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Draw the yellow circle!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Points: {len(self.drawing_points)}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show progress towards completion
            if len(self.drawing_points) > 10:
                progress = min(100, (len(self.drawing_points) / 50) * 100)
                cv2.putText(frame, f"Progress: {progress:.0f}%", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show real-time accuracy estimate
                if len(self.drawing_points) > 20:
                    temp_accuracy = self.calculate_circle_accuracy()
                    color = (0, 255, 0) if temp_accuracy > 80 else (0, 255, 255) if temp_accuracy > 60 else (0, 0, 255)
                    cv2.putText(frame, f"Live: {temp_accuracy:.0f}%", (10, 155), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def get_finger_count_simple(self, hand_landmarks):
        """Simple finger count - just count fingertips above wrist"""
        if not hand_landmarks:
            return 0
            
        try:
            wrist = hand_landmarks[0]
            count = 0
            
            # Count fingertips significantly above wrist
            fingertips = [4, 8, 12, 16, 20]
            for tip_idx in fingertips:
                tip = hand_landmarks[tip_idx]
                if tip.y < wrist.y - 0.1:  # Finger is raised
                    count += 1
            
            return count
            
        except:
            return 0
    
    def get_index_finger_tip(self, hand_landmarks):
        """Get the coordinates of index finger tip (landmark 8)"""
        if hand_landmarks:
            if hasattr(hand_landmarks[0], 'x'):  # New API format
                index_finger_tip = hand_landmarks[8]
                x = int(index_finger_tip.x * self.width)
                y = int(index_finger_tip.y * self.height)
            else:  # Old API format
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * self.width)
                y = int(index_finger_tip.y * self.height)
            return x, y
        return None, None
    
    def draw_on_canvas(self, x, y):
        """Draw a line from previous position to current position"""
        if self.is_drawing and self.prev_x is not None and self.prev_y is not None:
            cv2.line(self.drawing_mask, (self.prev_x, self.prev_y), (x, y), 
                    self.drawing_color, self.thickness)
            
            # Add point to drawing points for game
            if self.game_started and not self.circle_completed:
                self.drawing_points.append((x, y))
                
        self.prev_x = x
        self.prev_y = y
    
    def reset_drawing(self):
        """Reset the previous position to stop drawing"""
        self.prev_x = None
        self.prev_y = None
    
    def save_canvas(self):
        """Save both camera view and drawing to a folder"""
        # Create saves folder if it doesn't exist
        if not os.path.exists('saves'):
            os.makedirs('saves')
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save drawing mask
        drawing_filename = f'saves/drawing_{timestamp}.png'
        cv2.imwrite(drawing_filename, self.drawing_mask)
        
        # Save camera frame (current frame)
        if hasattr(self, 'current_frame'):
            camera_filename = f'saves/camera_{timestamp}.png'
            cv2.imwrite(camera_filename, self.current_frame)
            print(f"✅ Saved both images to 'saves' folder:")
            print(f"   📝 Drawing: {drawing_filename}")
            print(f"   📸 Camera: {camera_filename}")
        else:
            print(f"✅ Drawing saved as '{drawing_filename}'")
    
    def update_current_frame(self, frame):
        """Store current frame for saving"""
        self.current_frame = frame.copy()
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks on frame"""
        if hand_landmarks:
            # Simple landmark drawing
            for i, landmark in enumerate(hand_landmarks):
                if hasattr(landmark, 'x'):  # New API format
                    x = int(landmark.x * self.width)
                    y = int(landmark.y * self.height)
                else:  # Old API format
                    x = int(landmark.landmark[i].x * self.width)
                    y = int(landmark.landmark[i].y * self.height)
                
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
    
    def run(self):
        """Main loop for the Air Canvas application"""
        print("Air Canvas started!")
        print("=== CIRCLE DRAWING GAME ===")
        print("Controls:")
        print("  'g' = Start new game")
        print("  'n' = New circle (after completing one)")
        print("  's' = Save drawing as 'drawing.png'")
        print("  ☝️  Only index finger = Start drawing")
        print("  ✊ Fist or 2+ fingers = Stop drawing")
        print("  'f' = Toggle fullscreen mode")
        print("  'c' = Clear canvas")
        print("  'q' = Quit application")
        
        if not self.use_tasks_api:
            print("Note: Using mouse tracking - move your mouse to draw on the canvas")
        
        try:
            # Create resizable window
            cv2.namedWindow('Air Canvas', cv2.WINDOW_NORMAL)
            
            mouse_drawing = False
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Resize frame
                frame = cv2.resize(frame, (self.width, self.height))
                
                if self.use_tasks_api:
                    try:
                        # Convert BGR to RGB for MediaPipe
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Create MediaPipe image
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        
                        # Process the frame and detect hands
                        self.timestamp_ms = int(time.time() * 1000)
                        result = self.hand_landmarker.detect_for_video(mp_image, self.timestamp_ms)
                        
                        # Draw hand landmarks on frame
                        if result.hand_landmarks:
                            for hand_landmarks in result.hand_landmarks:
                                self.draw_landmarks(frame, hand_landmarks)
                                
                                # Check hand state with improved detection
                                only_index = self.is_only_index_finger(hand_landmarks)
                                finger_count = self.get_finger_count_simple(hand_landmarks)
                                
                                # Determine drawing state based on hand gesture
                                if only_index and not self.is_drawing:
                                    # Start drawing when only index finger is extended
                                    self.is_drawing = True
                                    self.reset_drawing()  # Reset position for new line
                                    print("✏️ Drawing started - Only index finger detected")
                                elif not only_index and self.is_drawing:
                                    # Stop drawing when not just index finger
                                    self.is_drawing = False
                                    if finger_count == 0:
                                        print("✋ Drawing stopped - Fist detected")
                                    else:
                                        print(f"✋ Drawing stopped - {finger_count} fingers detected")
                                
                                # Get index finger tip position
                                x, y = self.get_index_finger_tip(hand_landmarks)
                                
                                if x is not None and y is not None:
                                    # Draw circle at index finger tip
                                    if self.is_drawing:
                                        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # Green when drawing
                                    else:
                                        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)  # Blue when not drawing
                                    
                                    # Draw on canvas (only if drawing is enabled)
                                    self.draw_on_canvas(x, y)
                                    
                                    # Check if circle is completed
                                    if self.game_started and not self.circle_completed:
                                        if self.check_circle_completion():
                                            self.circle_completed = True
                                            accuracy = self.calculate_circle_accuracy()
                                            print(f"🎯 Circle completed! Accuracy: {accuracy:.1f}%")
                                            
                                            if accuracy >= 90:
                                                self.show_winner = True
                                                self.winner_animation = 0
                                                print("🏆🏆🏆 WINNER! PERFECT SCORE! 🏆🏆🏆")
                                            
                                            if accuracy > 95:
                                                print("🏆 PERFECT!")
                                            elif accuracy > 85:
                                                print("⭐ EXCELLENT!")
                                            elif accuracy > 75:
                                                print("👍 GOOD!")
                                            elif accuracy > 65:
                                                print("👌 OKAY")
                                            else:
                                                print("💪 TRY AGAIN!")
                                
                                # Display current state and finger count
                                status_text = f"DRAWING ({finger_count} finger)" if self.is_drawing else f"STOPPED ({finger_count} fingers)"
                                status_color = (0, 255, 0) if self.is_drawing else (0, 0, 255)
                                cv2.putText(frame, status_text, (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                                
                                # Show gesture hint
                                if only_index:
                                    cv2.putText(frame, "Perfect! Only index finger", (10, 60), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            # Reset drawing when hand is not detected
                            self.is_drawing = False
                            self.reset_drawing()
                    except Exception as e:
                        print(f"Hand tracking error: {e}")
                        self.reset_drawing()
                else:
                    # Mouse-based drawing as fallback
                    cv2.putText(frame, "Move mouse to draw", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Get mouse position
                    mouse_x, mouse_y = None, None
                    if cv2.waitKey(1) & 0xFF == ord('m'):  # Toggle drawing with 'm'
                        mouse_drawing = not mouse_drawing
                        print(f"Mouse drawing: {'ON' if mouse_drawing else 'OFF'}")
                    
                    if mouse_drawing:
                        # Use a fixed position for demo (you can replace with actual mouse tracking)
                        demo_x, demo_y = self.width // 2, self.height // 2
                        cv2.circle(frame, (demo_x, demo_y), 10, (255, 0, 0), -1)
                        self.draw_on_canvas(demo_x, demo_y)
                
                # Store current frame for saving
                self.update_current_frame(frame)
                
                # Draw game elements
                self.draw_target_circle(frame)
                self.draw_game_info(frame)
                self.draw_winner_celebration(frame)
                
                # Combine frame with drawing mask
                combined_frame = cv2.addWeighted(frame, 0.7, self.drawing_mask, 0.3, 0)
                
                # Display the combined frame in fullscreen
                cv2.imshow('Air Canvas', combined_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('g'):
                    # Start new game
                    self.game_started = True
                    self.circle_completed = False
                    self.drawing_points = []
                    self.drawing_mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    self.show_winner = False
                    self.winner_animation = 0
                    self.generate_target_circle()
                    print("🎮 New game started! Draw the yellow circle!")
                elif key == ord('n') and self.circle_completed:
                    # New circle
                    self.circle_completed = False
                    self.drawing_points = []
                    self.drawing_mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    self.generate_target_circle()
                    print("🔄 New circle generated!")
                elif key == ord('f'):
                    # Toggle fullscreen
                    self.is_fullscreen = not self.is_fullscreen
                    if self.is_fullscreen:
                        cv2.setWindowProperty('Air Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print("Fullscreen mode ON")
                    else:
                        cv2.setWindowProperty('Air Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print("Fullscreen mode OFF")
                elif key == ord('s'):
                    # Save the current drawing
                    self.save_canvas()
                elif key == ord('c'):
                    self.drawing_mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    self.reset_drawing()
                    self.drawing_points = []
                    print("Canvas cleared")
        
        except KeyboardInterrupt:
            print("\nApplication interrupted")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'hand_landmarker'):
                self.hand_landmarker.close()

if __name__ == "__main__":
    try:
        app = AirCanvas()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
