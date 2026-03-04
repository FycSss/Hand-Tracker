import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

def main():
    # Path to the model file
    model_path = '/home/FycSss/camera/hand_landmarker.task'

    # Create HandLandmarker options
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize the HandLandmarker
    landmarker = HandLandmarker.create_from_options(options)

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set resolution to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Get actual resolution to confirm
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera Resolution: {width}x{height}")

    window_name = "Hand Tracking"
    # Combine WINDOW_NORMAL (resizable) with WINDOW_GUI_NORMAL (no toolbars/status bar)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    
    is_fullscreen = False

    print("Tracking started.")
    print("- Press 'q' to exit")
    print("- Press 'f' to toggle Fullscreen")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Force resize to 1920x1080 to fill the screen and remove white bars
        frame = cv2.resize(frame, (1920, 1080))

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Get timestamp in milliseconds
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        # If timestamp is 0 (can happen on some cameras), use current time
        if frame_timestamp_ms == 0:
            frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        # Detect hands
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw results
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # MediaPipe tasks use a slightly different drawing structure
                # We can draw manually for total control
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw connections
                # Hand connections: (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), ...
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4), # thumb
                    (0, 5), (5, 6), (6, 7), (7, 8), # index
                    (5, 9), (9, 10), (10, 11), (11, 12), # middle
                    (9, 13), (13, 14), (14, 15), (15, 16), # ring
                    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # pinky
                ]
                for start_idx, end_idx in connections:
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                    end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
