import cv2
from ultralytics import YOLO
from tracker import Tracker
import random as rd
import numpy as np

class Camera():
   def __init__(self):
      # self.cap = cv2.VideoCapture(0)
      # self.cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.195.167:554/Streaming/Channels/501")
      self.cap = cv2.VideoCapture("student.mp4")
      self.model = YOLO("yolov8n.pt")
      self.tracker = Tracker()
      self.colors = [(rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)) for i in range(10)]
      # self.polygon_points = [[500,400], [500+300,400], [500+300,400+100], [500,400+100]]
      self.polygon_points = [(494+20,289+120), (505+20,499+120), (578+20,496+120), (530+20,292+120)]
      
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
            
         self.tracker.update(frame, detections)
         for track in self.tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            person_id = track.track_id
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), self.colors[person_id % len(self.colors)], 3)
            cv2.putText(frame, str(person_id), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
      
      cv2.polylines(frame, [np.array(self.polygon_points,np.int32)], True, (0,0,0), 1)
      cv2.line(frame, (self.polygon_points[0]), (self.polygon_points[1]), (255,0,0), 3)
      cv2.line(frame, (self.polygon_points[2]), (self.polygon_points[3]), (0,255,0), 3)
      
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