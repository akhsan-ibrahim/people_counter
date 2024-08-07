import cv2
from ultralytics import YOLO
from tracker import Tracker
import random as rd
import numpy as np
import math
import time

class Camera():
   def __init__(self):
      self.live = False
      self.cap = self.source(self.live)
      
      self.frame_save = []
      self.frame_count = 0
      self.count_skip = 0
      self.model = YOLO("yolov8n.pt")
      # self.model_y.export(format="openvino")
      # self.model = YOLO("yolov8n_openvino_model/")
      self.tracker = Tracker()
      self.colors = [(rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)) for i in range(10)]
      # self.polygon_points = [
      #    [900,610-60], # Blue Left
      #    [605,610], # Blue Right
      #    [605+5,680], # Green Left
      #    [900+40,680-65] # Green Right
      # ]
      self.polygon_points = [
         [900+40,680-65],
         [610, 680],
         [610+70, 1080],
         [900+300, 1080]
      ]
      '''
      self.area_out = [
         [900-130,340-30],
         [610-30, 340],
         [610, 680],
         [900+40, 680-65]
      ]
      self.area_in = [
         [900+40,680-65],
         [610, 680],
         [610+60, 1080],
         [900+300, 1080]
      ]
      '''
      self.area_out = [
         [530, 70],
         [825-1, 40],
         [825-1, 200],
         [720-1, 210-1],
         [630, 260-1],
         [490, 270-1]
      ]
      self.area_in = [
         [825, 120],
         [880, 125],
         [1920, 1080],
         [400, 1080],
         [490, 270],
         [630, 260],
         [720, 210],
         [825, 200]
      ]
            
      self.px1 = self.polygon_points[0][0] # top
      self.py1 = self.polygon_points[0][1] # left
      self.px2 = self.polygon_points[1][0] # bottom
      self.py2 = self.polygon_points[1][1] # left
      self.px3 = self.polygon_points[2][0] # bottom
      self.py3 = self.polygon_points[2][1] # right
      self.px4 = self.polygon_points[3][0] # top
      self.py4 = self.polygon_points[3][1] # right
      
      self.locations = {}
      
      self.people = {}
      self.enter = 0
      self.exit = 0
      
      # self.people = []
      self.person_enter = [i for i in range(0,8)]
      self.person_exit = [i for i in range(0,3)]
      
   def distance(self, x1, y1, x2, y2):
      """Find distance between two points with Pythagorean theorem"""
      num_a = math.pow((x2-x1), 2)
      num_b = math.pow((y2-y1), 2)
      c = math.sqrt(num_a + num_b) # Longest side
      return c
   
   def source(self, live):
      cap = cv2.VideoCapture("resources/Demo.mp4")
      if live == True:
         cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.195.167:554/Streaming/Channels/501")
      return cap
      
   def get_frame(self):
      """Get video frame"""
      start = time.time()
      ret, frame = self.cap.read()
      # print(f"RET : {ret}")
               
      roi = frame[0:1080, 0:860]
      
      self.frame_save.append(frame)
      # frame = cv2.resize(frame,(880,440))
      self.frame_count += 1
      
      if self.frame_count%2 == 0:
         
         if self.live:
            self.frame_count = self.count_skip = 0
            
         results = self.model.predict(roi, classes=0, stream_buffer=True, stream=True)
         for result in results:
            detections = []
            for r in result.boxes.data.tolist():
               x1, y1, x2, y2, score, class_id = r
               
               if score <= 0.4:
                  # print("LESS ACCURACY")
                  continue
               
               x1 = int(x1)
               y1 = int(y1)
               x2 = int(x2)
               y2 = int(y2)
               class_id = int(class_id)
               detections.append([x1, y1, x2, y2, score])
               print(f"SCORE : {score}")
            
            if not detections:
               self.count_skip += 10
               self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.count_skip+self.frame_count)
               continue
            else:
               self.tracker.update(roi, detections)
               for track in self.tracker.tracks:
                  bbox = track.bbox
                  x1, y1, x2, y2 = bbox
                  x1 = int(x1)
                  y1 = int(y1)
                  x2 = int(x2)
                  y2 = int(y2)
                  person_id = track.track_id
                  
                  check_out = cv2.pointPolygonTest(np.array(self.area_out,np.int32), (x1,y1), False)
                  check_in = cv2.pointPolygonTest(np.array(self.area_in,np.int32), (x1,y1), False)
                  
                  if check_out == 1:
                     if person_id not in self.people.keys():
                        # self.exit += 1
                        self.people[person_id] = "out"
                        # self.people.pop(person_id)
                     if self.people[person_id] == "in":
                        self.person_exit.append(person_id)
                        self.people.clear()
                  
                  if check_in == 1:
                     if person_id not in self.people.keys():
                        # self.enter += 1
                        self.people[person_id] = "in"
                        # self.people.pop(person_id)
                     if self.people[person_id] == "out":
                        self.person_enter.append(person_id)
                        self.people.clear()
                  
                  cv2.rectangle(roi, (x1,y1), (x2,y2), self.colors[person_id % len(self.colors)], 3)
                  cv2.putText(frame, f"id {person_id}", (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                  # cv2.line(frame, (self.px1,self.py1), (x2,y2), (0,255,255), 1, cv2.LINE_8)
                  # cv2.putText(frame, f"{round(person_to_line_1,2)}", (x2,y2+5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                  # cv2.putText(frame, f"Cek +{tes}", (x2,y2+5), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
      
      # print("ALL :",self.people)
      # print("IN",self.person_enter)
      # print("OUT",self.person_exit)
      
      # cv2.polylines(frame, [np.array(self.polygon_points,np.int32)], True, (0,0,0), 1)
      cv2.polylines(frame, [np.array(self.area_out,np.int32)], True, (255,0,0), 2)
      cv2.polylines(frame, [np.array(self.area_in,np.int32)], True, (0,255,0), 2)
      # cv2.line(frame, (self.polygon_points[0]), (self.polygon_points[1]), (255,0,0), 3) # Left line
      # cv2.line(frame, (self.polygon_points[2]), (self.polygon_points[3]), (0,255,0), 3) # Right line
      cv2.rectangle(frame, (0,0), (860, 1080), (255,255,255), 2)
      
      # cv2.putText(frame, f"{self.enter} Enter", (60,160), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 4)
      # cv2.putText(frame, f"{self.exit} Exit", (60,220), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 4)
      # cv2.putText(frame, f"{self.enter - self.exit} Inside", (60,280), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 4)
      cv2.putText(frame, f"{len(self.person_enter)} Enter", (60,160), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 4)
      cv2.putText(frame, f"{len(self.person_exit)} Exit", (60,220), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 4)
      cv2.putText(frame, f"{len(self.person_enter) - len(self.person_exit)} Inside", (60,280), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 4)
      
      end = time.time()
      print("Duration :", end-start)
      
      if frame is None:
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
