import cv2
from ultralytics import YOLO
from tracker import Tracker
import random as rd
import numpy as np
import math
import time

class Camera():
   def __init__(self):
      self.live = True
      self.cap = self.source(self.live)
      self.frame_save = []
      # self.model_y = YOLO("yolov8n.pt")
      # self.model_y.export(format="openvino")
      self.model = YOLO("yolov8n_openvino_model/")
      self.tracker = Tracker()
      self.colors = [(rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)) for i in range(10)]
      self.polygon_points = [
         [900,610-60], # Blue Left
         [605,610], # Blue Right
         [605+5,680], # Green Left
         [900+40,680-65] # Green Right
      ]
      self.people = {}
      self.enter = 3
      self.exit = 0
      
      self.frame_count = 0
      self.count_skip = 0
      
   def distance(self, x1, y1, x2, y2):
      """Find distance between two points with Pythagorean theorem"""
      num_a = math.pow((x2-x1), 2)
      num_b = math.pow((y2-y1), 2)
      c = math.sqrt(num_a + num_b) # Longest side
      return c
   
   def source(self, live):
      cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.195.167:554/Streaming/Channels/501")
      if live == False:
         cap = cv2.VideoCapture("rekaman300424.mp4")
      return cap
      
   def get_frame(self):
      """Get video frame"""
      start = time.time()
      ret, frame = self.cap.read()
      
      self.frame_save.append(frame)
      # frame = cv2.resize(frame,(880,440))
      self.frame_count += 1
      
      if self.frame_count%2 == 0:
         
         if self.live:
            self.frame_count = self.count_skip = 0
            
         results = self.model.predict(frame, classes=0, stream_buffer=True, stream=True)
         for result in results:
            detections = []
            for r in result.boxes.data.tolist():
               x1, y1, x2, y2, score, class_id = r
               if score <= 0.4:
                  continue
               x1 = int(x1)
               y1 = int(y1)
               x2 = int(x2)
               y2 = int(y2)
               class_id = int(class_id)
               detections.append([x1, y1, x2, y2, score])
            
            if not detections:
               self.count_skip += 10
               self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.count_skip+self.frame_count)
               continue
            else:
               self.tracker.update(frame, detections)
               for track in self.tracker.tracks:
                  bbox = track.bbox
                  x1, y1, x2, y2 = bbox
                  x1 = int(x1)
                  y1 = int(y1)
                  x2 = int(x2)
                  y2 = int(y2)
                  person_id = track.track_id
                  
                  px1 = self.polygon_points[0][0] # top
                  py1 = self.polygon_points[0][1] # left
                  px2 = self.polygon_points[1][0] # bottom
                  py2 = self.polygon_points[1][1] # left
                  px3 = self.polygon_points[2][0] # bottom
                  py3 = self.polygon_points[2][1] # right
                  px4 = self.polygon_points[3][0] # top
                  py4 = self.polygon_points[3][1] # right
                  
                  d1_person = self.distance(px1, py1, x2, y2) + self.distance(px2, py2, x2, y2)
                  d1_line = self.distance(px1, py1, px2, py2)
                  person_to_line_1 = round(d1_person, 3) - round(d1_line, 3)
                  
                  d2_person = self.distance(px3, py3, x2, y2) + self.distance(px4, py4, x2, y2)
                  d2_line = self.distance(px3, py3, px4, py4)
                  person_to_line_2 = round(d2_person, 3) - round(d2_line, 3)
                  
                  if person_to_line_2 <= 1: # Green line
                     if person_id not in self.people.keys():
                        self.people[person_id] = "in"
                        
                     if self.people[person_id] == "out":
                        # self.enter[person_id] = [x2,y2]
                        self.enter += 1
                        self.people.pop(person_id)
                        
                  if person_to_line_1 <= 1: # Blue line
                     if person_id not in self.people.keys():
                        self.people[person_id] = "out"
                        
                     if self.people[person_id] == "in":
                        # self.exit[person_id] = [x2,y2]
                        self.exit += 1
                        self.people.pop(person_id)
                  
                  cv2.rectangle(frame, (x1,y1), (x2,y2), self.colors[person_id % len(self.colors)], 3)
                  # cv2.putText(frame, str(person_id), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                  # cv2.line(frame, (px1,py1), (x2,y2), (0,255,255), 1, cv2.LINE_8)
                  # cv2.putText(frame, f"{round(person_to_line_1,2)}", (x2,y2+5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
      
      cv2.polylines(frame, [np.array(self.polygon_points,np.int32)], True, (0,0,0), 1)
      cv2.line(frame, (self.polygon_points[0]), (self.polygon_points[1]), (255,0,0), 3) # Left line
      cv2.line(frame, (self.polygon_points[2]), (self.polygon_points[3]), (0,255,0), 3) # Right line
      
      cv2.putText(frame, f"{self.enter} Enter", (60,160), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 4)
      cv2.putText(frame, f"{self.exit} Exit", (60,220), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 4)
      cv2.putText(frame, f"{self.enter - self.exit} Inside", (60,280), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 4)
      
      end = time.time()
      print("Duration :", end-start)
      
      if not frame.any():
         frame = self.frame_save[0]
      
      ret, jpeg = cv2.imencode(".jpg", frame) # Convert frame to jpeg image
      
      if self.frame_save:
         self.frame_save.pop(0)
         
      return jpeg.tobytes() # Convert jpeg to byte for send to web browser
   
   def stream(self):
      """Stream video frames"""
      while True:
         frame = self.get_frame()
         yield( # Package for displayed on web browser
            b"--frame\r\n" # Deliver frame to web browser
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
         )
