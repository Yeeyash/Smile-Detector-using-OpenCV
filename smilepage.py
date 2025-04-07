from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from mesh import Smile_Detector

app = FastAPI()

app.mount("/static", StaticFiles(directory="."), name="static") #changed the position of bracket to end of line instead of after the comma.
temples = Jinja2Templates(directory=".")

# @app.get("/")
# async def readitem(request: Request):
#     return temples.TemplateResponse("index.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def countitem(request: Request):
    return temples.TemplateResponse("index.html", {"request":request})

@app.get("/videofeed")
async def videofeed():
    return StreamingResponse(Smile_Detector().main(), media_type="multipart/x-mixed-replace; boundary=frame")


# @app.get("/")
# async def reedroot():
#     with open("index.html", "r") as f:
#         html_content = f.read()
#     return Response(content=html_content, media_type="text/html")
