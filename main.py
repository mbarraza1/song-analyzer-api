from typing import List

import io

import librosa
import numpy as np
import requests
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="Song Analyzer API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    preview_url: str


class AnalyzeResponse(BaseModel):
    tempo: float
    beat_strength: float
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    zero_crossing_rate_mean: float
    mfcc_means: List[float]
    chroma_means: List[float]
    tonnetz_means: List[float]


@app.get("/")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_track(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Download a short audio preview (e.g. from Spotify), extract core rhythmic features,
    and return a compact feature representation that the frontend can use for similarity.
    """
    if not req.preview_url:
        raise HTTPException(status_code=400, detail="preview_url is required")

    try:
        resp = requests.get(req.preview_url, timeout=15)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Failed to download audio: {exc}") from exc

    audio_bytes = io.BytesIO(resp.content)

    try:
        # soundfile can read from file-like objects; librosa then works on the array
        data, sr = sf.read(audio_bytes, always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to decode audio: {exc}") from exc

    # Convert to mono if stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    y = data.astype(np.float32)

    # Core rhythmic, spectral, and harmonic features
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = float(onset_env.mean())

        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = float(spec_centroid.mean())

        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = float(spec_bandwidth.mean())

        zcr = librosa.feature.zero_crossing_rate(y=y)
        zero_crossing_rate_mean = float(zcr.mean())

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = mfcc.mean(axis=1).astype(float).tolist()

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = chroma.mean(axis=1).astype(float).tolist()

        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_means = tonnetz.mean(axis=1).astype(float).tolist()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to extract features: {exc}") from exc

    return AnalyzeResponse(
        tempo=float(tempo),
        beat_strength=beat_strength,
        spectral_centroid_mean=spectral_centroid_mean,
        spectral_bandwidth_mean=spectral_bandwidth_mean,
        zero_crossing_rate_mean=zero_crossing_rate_mean,
        mfcc_means=mfcc_means,
        chroma_means=chroma_means,
        tonnetz_means=tonnetz_means,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

