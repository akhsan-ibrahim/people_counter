from flask import Flask, render_template, Response
from camera import Camera

app = Flask(__name__)
cam = Camera()

@app.route("/")
def index():
   return render_template("index.html")

@app.route("/video_feed")
def video_feed():
   return Response(
      cam.stream(),
      mimetype="multipart/x-mixed-replace; boundary=frame" # Set html receiving response -- Send to web browser for display the frame within any object (multipart)
   )

if __name__ == "__main__":
   app.run(debug=True)