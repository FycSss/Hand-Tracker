import cv2

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    window_name = "Camera"
    
    # WINDOW_GUI_NORMAL removes the status bar and toolbar found in some OpenCV builds
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    
    # Ensure it starts with a clean look
    cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

    print("Camera started. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Show the frame
        cv2.imshow(window_name, frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
