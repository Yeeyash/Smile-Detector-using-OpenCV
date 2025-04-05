from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
from fastapi.templating import Jinja2Templates
from mesh import Smile_Detector

app = FastAPI()

vid = cv2.VideoCapture(0)
web = Jinja2Templates(directory="/")


# def generate_frames():
#     while True:
#         success, img = vid.read()
#         _, buffer = cv2.imencode('.jpg', img)
#         yield (b'--img\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
#                )

@app.get("/videofeed")
async def videofeed():
    return StreamingResponse(Smile_Detector().main(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def reedroot():
    with open("index.html", "r") as f:
        html_content = f.read()
    return Response(content=html_content, media_type="text/html")
