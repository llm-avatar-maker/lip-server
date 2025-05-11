# MuseTalk Server API (AI-Generated Documentation)

A Python server using AIOHTTP that provides API endpoints for preprocessing and inference of talking avatar videos.

## Features

- Preprocess videos or image sequences to create avatars for talking head generation.
- Real-time and batch inference for lip-syncing avatars to audio.
- Supports multiple MuseTalk model versions.
- Handles large model files and temporary data efficiently.

## Prerequisites

- Python 3.10+
- FFmpeg installed and available in PATH
- Required Python packages (see `requirements.txt`):
  - aiohttp
  - torch
  - numpy
  - opencv-python
  - transformers
  - tqdm

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Directory Structure

- `uploads/`: Temporary storage for uploaded files
- `results/avatars/`: Storage for processed avatars
- `temp/`: Temporary files during processing
- `models/`: Model weights and caches
- `src/`: Source code for services and models
- `reference/`: Reference scripts for inference

## API Endpoints

### 1. Preprocess Avatar
`POST /preprocess`

Preprocesses a video or image sequence to create an avatar.

**Required Parameters:**
- `avatar_id`: Unique identifier for the avatar
- `video`: Video file or directory of images

**Optional Parameters:**
- `version`: MuseTalk version ("v1" or "v15", default: "v15")
- `bbox_shift`, `extra_margin`, `parsing_mode`, `left_cheek_width`, `right_cheek_width`, `vae_type`: Advanced options for face cropping and model selection

**Example:**
```bash
curl -X POST http://localhost:8080/preprocess \
  -F "avatar_id=test_avatar" \
  -F "video=@input_video.mp4"
```

### 2. Real-time Inference
`POST /realtime-inference`

Performs real-time inference with streaming response.

**Required Parameters:**
- `avatar_id`: ID of the preprocessed avatar
- `audio`: Audio file for lip-sync

**Optional Parameters:** `version`, `fps`, `batch_size`, `audio_padding_length_left`, `audio_padding_length_right`, `skip_save_images`, `gpu_id`, `vae_type`

**Example:**
```bash
curl -X POST http://localhost:8080/realtime-inference \
  -F "avatar_id=test_avatar" \
  -F "audio=@input_audio.wav"
```

### 3. Standard Inference
`POST /inference`

Performs non-realtime inference and returns the complete video file.

**Parameters:** Same as realtime-inference endpoint.

**Example:**
```bash
curl -X POST http://localhost:8080/inference \
  -F "avatar_id=test_avatar" \
  -F "audio=@input_audio.wav"
```

## Response Formats

- **Success:** JSON message or video file (Content-Type: video/mp4)
- **Error:** JSON with `{ "error": "Error message description" }`

## Running the Server

```bash
python server.py
```

The server will start on `http://localhost:8080`.

## Error Handling

The server includes comprehensive error handling for:
- Missing required parameters
- File upload issues
- Processing errors
- FFmpeg availability
- Directory creation failures

All errors are logged and returned with appropriate HTTP status codes.