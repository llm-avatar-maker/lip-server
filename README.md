# MuseTalk Server API

A Python server using AIOHTTP that provides API endpoints for preprocessing and inference of talking avatar videos.

## Features

- Preprocess videos or image sequences to create avatars for talking head generation.
- Real-time and batch inference for lip-syncing avatars to audio.
- Supports multiple MuseTalk model versions.
- Handles large model files and temporary data efficiently.

## Installation
To prepare the Python environment and install additional packages such as opencv, diffusers, mmcv, etc., please follow the steps below:

### Build environment
We recommend Python 3.10 and CUDA 11.7. Set up your environment as follows:

```shell
conda create -n MuseTalk python==3.10
conda activate MuseTalk
```

### Install PyTorch 2.0.1
Choose one of the following installation methods:

```shell
# Option 1: Using pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Option 2: Using conda
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Install Dependencies
Install the remaining required packages:

```shell
pip install -r requirements.txt
```

### Install MMLab Packages
Install the MMLab ecosystem packages:

```bash
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
```
### Setup FFmpeg
1. [Download](https://github.com/BtbN/FFmpeg-Builds/releases) the ffmpeg-static package

2. Configure FFmpeg based on your operating system:

For Linux:
```bash
export FFMPEG_PATH=/path/to/ffmpeg
# Example:
export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static
```

For Windows:
Add the `ffmpeg-xxx\bin` directory to your system's PATH environment variable. Verify the installation by running `ffmpeg -version` in the command prompt - it should display the ffmpeg version information.

### Download weights
You can download weights in two ways:

#### Option 1: Using Download Scripts
We provide two scripts for automatic downloading:

For Linux:
```bash
sh ./download_weights.sh
```

For Windows:
```batch
# Run the script
download_weights.bat


#### Option 2: Manual Download
You can also download the weights manually from the following links:

1. Download our trained [weights](https://huggingface.co/TMElyralab/MuseTalk/tree/main)
2. Download the weights of other components:
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)
   - [whisper](https://huggingface.co/openai/whisper-tiny/tree/main)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [syncnet](https://huggingface.co/ByteDance/LatentSync/tree/main)
   - [face-parse-bisent](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?pli=1)
   - [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)

Finally, these weights should be organized in `models` as follows:
```
./models/
├── musetalk
│   └── musetalk.json
│   └── pytorch_model.bin
├── musetalkV15
│   └── musetalk.json
│   └── unet.pth
├── syncnet
│   └── latentsync_syncnet.pt
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    ├── config.json
    ├── pytorch_model.bin
    └── preprocessor_config.json
    

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