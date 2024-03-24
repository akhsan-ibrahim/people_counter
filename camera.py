import cv2

class Camera():
   def __init__(self):
      self.cap = cv2.VideoCapture(0)
      
   def get_frame(self):
      """Get video frame"""
      ret, frame = self.cap.read()
      
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