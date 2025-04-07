from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from mesh import Smile_Detector

app = FastAPI()

app.mount("/static", StaticFiles(directory="."), name="static") #changed the position of bracket to end of line instead of after the comma.
temples = Jinja2Templates(directory=".")

@app.get("/", response_class=HTMLResponse)
async def countitem(request: Request):
    return temples.TemplateResponse("index.html", {"request":request})

@app.get("/videofeed")
async def videofeed():
    return StreamingResponse(Smile_Detector().main(), media_type="multipart/x-mixed-replace; boundary=frame")
