import os
from uuid import uuid4

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.stable_diffusion import Leo

with open("README.md", "r") as file:
    next(file)
    description = file.read()

VERSION = "0.0.1"
API = FastAPI(
    title="StableDiffusion API",
    version=VERSION,
    docs_url="/",
    description=description,
)
API.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
leo = Leo()


@API.get("/version", tags=["General Operations"])
async def version():
    return VERSION


@API.get("/text-to-image", tags=["StableDiffusion"])
async def text_to_image(queue: BackgroundTasks, prompt: str):
    image_id = str(uuid4())
    queue.add_task(leo, image_id=image_id, prompt=prompt, epochs=20)
    return image_id


@API.get("/image/{image_id}")
async def image_by_id(image_id: str):
    image_path = os.path.join("app", "images", f"{image_id}.png")
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    else:
        return "Work in progress"
