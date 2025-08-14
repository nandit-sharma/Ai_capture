import cv2

def open_camera(source=0, width=640, height=480, target_fps=30):
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise Exception("Could not open camera")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    return cap

def get_frame(cap):
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        raise Exception("Failed to capture frame")
    return frame

def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()