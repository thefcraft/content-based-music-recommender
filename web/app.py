import os
from pathlib import Path

import chromadb
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from src.config import Config
from src.encoder import AudioEncoder
from src import utils
from base64 import urlsafe_b64decode, urlsafe_b64encode
from inference import (
    InferenceAudioProcessor,
    aggregate_query_results,
)


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=BASE_DIR.joinpath("static")), name="static")
templates = Jinja2Templates(directory=BASE_DIR.joinpath("templates"))


config = Config.from_dotenv(dotenv_path=BASE_DIR.joinpath("..", ".env"))


# =========================
# LOAD MODEL
# =========================
checkpoint = torch.load(
    config.train_checkpoint_dir.joinpath(config.inference_model_checkpoint_name)
)

model = AudioEncoder(embedding_dim=config.embedding_dim).to(config.device)

model.load_state_dict(checkpoint["encoder"])
model.eval()


# =========================
# LOAD CHROMA
# =========================
client = chromadb.PersistentClient(path=config.vector_db_dir)
collection = client.get_or_create_collection(name=config.vector_db_collection_name)


processor = InferenceAudioProcessor(config=config)


songs = sorted(os.listdir(config.music_dir))


# =========================
# HELPERS
# =========================
@torch.no_grad()
def get_song_embeddings(song_path: Path):
    embeddings: list[np.ndarray] = []

    for batch in processor.process_batch(song_path):
        output = model(batch)

        output = torch.nn.functional.normalize(
            output,
            dim=-1,
        )

        embeddings.extend([x.cpu().numpy() for x in output])

    return embeddings


@torch.no_grad()
def recommend(song_name: str, top_k: int = 10):
    song_path = config.music_dir.joinpath(song_name)

    embeddings = get_song_embeddings(song_path)

    results = collection.query(
        query_embeddings=[
            *embeddings,
            np.mean(embeddings, axis=0),
        ],
        n_results=top_k,
        where={
            "filename": {
                "$ne": song_name,
            }
        },
    )

    aggregated = aggregate_query_results(
        results,
        aggregate_key="num_matches",
    )

    cleaned: list[dict[str, str | float | int | list[int]]] = []

    seen: set[str] = set()

    for item in aggregated:
        filename = utils.cast(item["filename"], str)

        if filename in seen:
            continue

        seen.add(filename)

        cleaned.append(
            {
                "filename": filename,
                "score": item["num_matches"],
            }
        )

    return cleaned[:top_k]


# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "songs": [
                {
                    "songname": song,
                    "songid": urlsafe_b64encode(song.encode()).decode(),
                }
                for song in songs
            ],
        },
    )


@app.get("/api/recommend/{songid}")
async def get_recommendations(songid: str):  # pyright: ignore[reportUnknownParameterType]
    song_name = urlsafe_b64decode(songid.encode()).decode()
    results = recommend(song_name)

    return {
        "query": song_name,
        "recommendations": [
            {
                "id": urlsafe_b64encode(result["filename"].encode()).decode(),  # type: ignore
                **result,
            }
            for result in results
        ],
    }


@app.get("/music/{songid}")
async def stream_music(songid: str):
    song_name = urlsafe_b64decode(songid.encode()).decode()
    file_path = config.music_dir.joinpath(song_name)

    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
