import cv2
from ultralytics import YOLO
from tracker import Tracker
import random as rd
import numpy as np
import math

class Camera():
   def __init__(self):
      # self.cap = cv2.VideoCapture(0)
      # self.cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.195.167:554/Streaming/Channels/501")
      self.cap = cv2.VideoCapture("student.mp4")
      self.model = YOLO("yolov8n.pt")
      self.tracker = Tracker()
      self.colors = [(rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)) for i in range(10)]
      # self.polygon_points = [[500,400], [500+300,400], [500+300,400+100], [500,400+100]]
      self.polygon_points = [
         (520,420), # Top left
         (520,700), # Bottom left
         (600,700), # Bottom right
         (600,420) # Top right
      ]
      self.people = {}
      self.people_in = {}
      self.people_out = {}
      
   def distance(self, x1, y1, x2, y2):
      num_a = math.pow((x2-x1), 2)
      num_b = math.pow((y2-y1), 2)
      result = math.sqrt(num_a + num_b)
      return result
      
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
            
            xa = self.polygon_points[0][0] # top
            ya = self.polygon_points[0][1] # left
            xb = self.polygon_points[1][0] # bottom
            yb = self.polygon_points[1][1] # left
            xc = self.polygon_points[2][0] # bottom
            yc = self.polygon_points[2][1] # right
            xd = self.polygon_points[3][0] # top
            yd = self.polygon_points[3][1] # right
            
            result_a = self.distance(xa, ya, x2, y2) + self.distance(xb, yb, x2, y2)
            result_b = self.distance(xa, ya, xb, yb)
            result_1 = round(result_a, 3) - round(result_b, 3)
            
            result_c = self.distance(xc, yc, x2, y2) + self.distance(xd, yd, x2, y2)
            result_d = self.distance(xc, yc, xd, yd)
            result_2 = round(result_c, 3) - round(result_d, 3)
            
            if result_1 <= 1 and result_1 >= -1: # Left line
               if person_id not in self.people.keys():
                  self.people[person_id] = "in"
               if self.people[person_id] == "out":
                  self.people_in[person_id] = [x2,y2]
                  self.people.pop(person_id)
                  
            if result_2 <= 1 and result_2 >= -1: # Right line
               if person_id not in self.people.keys():
                  self.people[person_id] = "out"
               if self.people[person_id] == "in":
                  self.people_out[person_id] = [x2,y2]
                  self.people.pop(person_id)
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), self.colors[person_id % len(self.colors)], 3)
            cv2.putText(frame, str(person_id), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
      
      print(self.people, self.people_in, self.people_out)
      cv2.polylines(frame, [np.array(self.polygon_points,np.int32)], True, (0,0,0), 1)
      cv2.line(frame, (self.polygon_points[0]), (self.polygon_points[1]), (255,0,0), 3) # Left line
      cv2.line(frame, (self.polygon_points[2]), (self.polygon_points[3]), (0,255,0), 3) # Right line
      
      cv2.putText(frame, f"Enter : {len(self.people_in)}", (10,60), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
      cv2.putText(frame, f"Exit : {len(self.people_out)}", (10,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
      
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