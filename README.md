# Song Analyzer API

Python FastAPI service that analyzes short audio previews (for example, Spotify track previews) and returns a compact set of audio features that the `song-matcher-web` app can use for similarity matching.

## Endpoints

- `GET /`  
  Healthcheck. Returns `{"status": "ok"}`.

- `POST /analyze`  
  Request body:
  ```json
  {
    "preview_url": "https://example.com/audio-preview.mp3"
  }
  ```

  Response body:
  ```json
  {
    "tempo": 120.5,
    "beat_strength": 0.23,
    "spectral_centroid_mean": 2450.1,
    "mfcc_means": [/* 13 numbers */]
  }
  ```

## Installation

It is recommended to use a virtual environment.

```bash
cd song-analyzer-api
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the server

You can run the API with:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Or simply:

```bash
python main.py
```

The service will be available at `http://127.0.0.1:8001`.

## Using with `song-matcher-web`

In the `song-matcher-web` project, configure the Python analyzer URL (if needed) in `.env.local`:

```bash
PYTHON_ANALYZER_URL=http://127.0.0.1:8001
```

Then start the Next.js app (usually `npm install` then `npm run dev`). From the web UI you can:

1. Search for tracks on Spotify.
2. Click **Analyze** on tracks that have a preview.
3. Click **Similar** to see rhythmically similar tracks based on the extracted features.

