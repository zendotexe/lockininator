import cv2
import mediapipe as mp
import os
import pygame

# 1. Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. Setup Audio
pygame.mixer.init()
video_path = "1.mov"
audio_path = "audio.mp3" # Ensure you have this file!

audio_loaded = False
if os.path.exists(audio_path):
    try:
        pygame.mixer.music.load(audio_path)
        audio_loaded = True
    except Exception as e:
        print(f"Audio Error: {e}")

# 3. Setup Video
video_cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

video_window_name = "GET BACK TO WORK"
window_open = False
audio_playing = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    looking_away = True 

    if results.multi_face_landmarks:
        nose = results.multi_face_landmarks[0].landmark[4]
        # Detection box
        if 0.30 < nose.x < 0.70 and 0.30 < nose.y < 0.70:
            looking_away = False

    if looking_away:
        # --- RESTART LOGIC ---
        # If the window wasn't already open, this is the FIRST frame of distraction
        if not window_open:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to start
            if audio_loaded:
                pygame.mixer.music.play(-1) # Restart audio from beginning
                audio_playing = True
            window_open = True

        ret, v_frame = video_cap.read()
        
        # Loop video if it hits the end while you're still looking away
        if not ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, v_frame = video_cap.read()

        if ret and v_frame is not None:
            cv2.imshow(video_window_name, v_frame)
            # Keep the "Shame" window on top
            cv2.setWindowProperty(video_window_name, cv2.WND_PROP_TOPMOST, 1)
    else:
        # If you are focused, shut everything down
        if window_open:
            if audio_playing:
                pygame.mixer.music.stop()
                audio_playing = False
            try:
                cv2.destroyWindow(video_window_name)
            except:
                pass
            window_open = False

    # Preview Window
    status_text = "DISTRACTED!" if looking_away else "FOCUSED"
    status_color = (0, 0, 255) if looking_away else (0, 255, 0)
    cv2.putText(frame, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.imshow("AI Face Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()