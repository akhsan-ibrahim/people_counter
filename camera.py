import cv2
from ultralytics import YOLO

class Camera():
   def __init__(self):
      self.cap = cv2.VideoCapture(0)
      self.model = YOLO("yolov8n.pt")
      
   def get_frame(self):
      """Get video frame"""
      ret, frame = self.cap.read()
      
      results = self.model.predict(frame, classes=0, stream_buffer=True, stream=True)
      for result in results:
         detections = []
         for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            class_id = int(class_id)
            detections.append([x1, y1, x2, y2, score])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
      
      ret, jpeg = cv2.imencode(".jpg", frame) # Convert frame to jpeg image
      return jpeg.tobytes() # Convert jpeg to byte for send to web browser
   
   def stream(self):
      """Stream video frames"""
      while True:
         frame = self.get_frame()
         yield( # Package for displayed on web browser
            b"--frame\r\n" # Deliver frame to web browser
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
         )