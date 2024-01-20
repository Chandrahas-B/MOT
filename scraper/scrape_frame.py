import cv2
from PIL import Image

class FrameExtractor:
    
    def __call__(self,video_path, frame_number):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            return

        cap.release()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = Image.fromarray(frame)

        return frame

if __name__ == "__main__":
    
    video_path = 'Sample Videos/Cars, Busy Streets, City Traffic.mp4'
    frame_number = 50
    extract_frame = FrameExtractor()
    
    print(extract_frame(video_path, frame_number))