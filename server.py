import os
import json
from pathlib import Path
import logging
import asyncio
from datetime import datetime, timezone
import uuid
import re
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web
from motor.motor_asyncio import AsyncIOMotorClient
import cv2
from src.models.model_manager import ModelManager
from src.services.preprocessor import Preprocessor
from src.services.inference_service import InferenceService
from src.services.queue_processor import QueueProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
UPLOAD_DIR = Path("uploads")
AVATAR_DIR = Path("results/avatars")
TEMP_DIR = Path("temp")

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
AVATAR_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# MongoDB setup
MONGO_URI = "mongodb://172.25.0.1:27017"
DB_NAME = "avatarcreator"
COLLECTION_STATUS = "processing_status"
COLLECTION_AVATARS = "avatars"

# Create thread pool for CPU-intensive tasks
MAX_WORKERS = 1
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Task queue for preprocessing
preprocess_queue = asyncio.Queue()
inference_queue = asyncio.Queue()
realtime_inference_queue = asyncio.Queue()

# Initialize services
model_manager = ModelManager()
preprocessor = Preprocessor(model_manager)
inference_service = InferenceService(model_manager)
queue_processor = QueueProcessor(
    preprocess_queue,
    inference_queue,
    realtime_inference_queue,
    MAX_WORKERS,
    model_manager,
)


def generate_avatar_id() -> str:
    """Generate a unique avatar ID"""
    return f"{uuid.uuid4().hex[:8]}"


def sanitize_avatar_id(avatar_id: str) -> str:
    """Sanitize avatar ID to ensure it's valid"""
    # Remove any non-alphanumeric characters except underscore and hyphen
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", avatar_id)
    return sanitized if sanitized else generate_avatar_id()


async def init_mongodb():
    """Initialize MongoDB connection with error handling"""
    try:
        client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Test the connection
        await client.admin.command("ping")
        logger.info("Successfully connected to MongoDB")
        db = client[DB_NAME]
        status_collection = db[COLLECTION_STATUS]
        avatars_collection = db[COLLECTION_AVATARS]
        return status_collection, avatars_collection
    except Exception as e:
        logger.error("Failed to connect to MongoDB: %s", str(e))
        raise web.HTTPInternalServerError(
            text=json.dumps({"error": "Database connection failed", "details": str(e)})
        )


async def check_avatar_exists(avatars_collection, avatar_id: str) -> bool:
    """Check if avatar ID already exists"""
    return await avatars_collection.find_one({"avatar_id": avatar_id}) is not None


async def store_avatar_info(avatars_collection, avatar_id: str, params: dict) -> dict:
    """Store avatar information in MongoDB"""
    # Set path based on version
    if params["version"] == "v15":
        processed_path = f"./results/{params['version']}/avatars/{avatar_id}"
    else:  # v1
        processed_path = f"./results/avatars/{avatar_id}"

    avatar_info = {
        "avatar_id": avatar_id,
        "created_at": datetime.now(timezone.utc),
        "parameters": params,
        "processed_path": processed_path,
        "version": params["version"],  # Store version explicitly
    }
    await avatars_collection.insert_one(avatar_info)
    return avatar_info


async def update_processing_status(
    collection, avatar_id: str, status: str, start_time=None, finish_time=None
) -> None:
    """Update processing status in MongoDB"""
    update_data = {"status": status}
    if start_time:
        update_data["start_time"] = start_time
    if finish_time:
        update_data["finish_time"] = finish_time

    await collection.update_one(
        {"avatar_id": avatar_id}, {"$set": update_data}, upsert=True
    )


async def process_avatar_task(
    status_collection, avatars_collection, avatar_id: str, file_path: str, params: dict
) -> None:
    """Process avatar in a separate thread"""
    try:
        start_time = datetime.now(timezone.utc)
        # Update status to processing with start time
        await update_processing_status(
            status_collection, avatar_id, "processing", start_time=start_time
        )

        # Run preprocessing in thread pool
        loop = asyncio.get_event_loop()
        logger.info("Starting preprocessing for avatar_id: %s", avatar_id)
        success = await loop.run_in_executor(
            thread_pool,
            preprocessor.preprocess,
            avatar_id,
            str(file_path),
            params["version"],
            params["bbox_shift"],
            params["extra_margin"],
            params["parsing_mode"],
            params["left_cheek_width"],
            params["right_cheek_width"],
        )
        logger.info("Completed preprocessing for avatar_id: %s", avatar_id)

        finish_time = datetime.now(timezone.utc)
        # Update final status with finish time
        if success:
            await update_processing_status(
                status_collection,
                avatar_id,
                "completed",
                start_time=start_time,
                finish_time=finish_time,
            )
            # Update avatar info with completion status
            await avatars_collection.update_one(
                {"avatar_id": avatar_id},
                {
                    "$set": {
                        "preprocessed": True,
                    }
                },
            )
        else:
            await update_processing_status(
                status_collection,
                avatar_id,
                "failed",
                start_time=start_time,
                finish_time=finish_time,
            )
            # Update avatar info with failure status
            await avatars_collection.update_one(
                {"avatar_id": avatar_id},
                {
                    "$set": {
                        "preprocessed": False,
                        "status": "failed",
                        "failed_at": finish_time,
                        "error": "Processing failed",
                    }
                },
            )

    except Exception as e:
        logger.error("Error in processing: %s", str(e))
        finish_time = datetime.now(timezone.utc)
        await update_processing_status(
            status_collection,
            avatar_id,
            "failed",
            start_time=start_time,
            finish_time=finish_time,
        )
        # Update avatar info with error
        await avatars_collection.update_one(
            {"avatar_id": avatar_id},
            {"$set": {"status": "failed", "failed_at": finish_time, "error": str(e)}},
        )
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


async def process_inference_task(
    future, avatar_id: str, audio_path: str, params: dict
) -> None:
    """Process inference in a separate thread"""
    try:
        # Run inference
        output_vid = inference_service.inference(
            avatar_id=avatar_id,
            audio_path=audio_path,
            out_vid_name=avatar_id,
            version=params["version"],
            fps=params["fps"],
            audio_padding_length_left=params["audio_padding_length_left"],
            audio_padding_length_right=params["audio_padding_length_right"],
        )

        if output_vid and os.path.exists(output_vid):
            future.set_result(output_vid)
        else:
            future.set_exception(Exception("Failed to generate video"))
    except Exception as e:
        future.set_exception(e)
    finally:
        # Cleanup audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)


async def process_inference_task_in_realtime(
    response, avatar_id: str, audio_path: str, params: dict, stream_callback
) -> None:
    """Process inference in a separate thread"""
    try:
        output_vid = inference_service.realtime_inference(
            avatar_id=avatar_id,
            audio_path=audio_path,
            out_vid_name=avatar_id,
            version=params["version"],
            fps=params["fps"],
            batch_size=params["batch_size"],
            audio_padding_length_left=params["audio_padding_length_left"],
            audio_padding_length_right=params["audio_padding_length_right"],
            skip_save_images=params["skip_save_images"],
            stream_callback=stream_callback,
        )

        if output_vid:
            # Stream the output video
            with open(output_vid, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    await response.write(chunk)

            # Cleanup
            os.remove(output_vid)

        # Cleanup
        os.remove(audio_path)

    except Exception as e:
        logger.error("Error in streaming: %s", str(e))
        raise
    finally:
        await response.write_eof()


async def preprocess_handler(request):
    """Handle preprocessing of images/videos to create avatars"""
    try:
        # Get form data
        data = await request.post()

        # Get or generate avatar_id
        avatar_id = data.get("avatar_id")
        if avatar_id:
            avatar_id = sanitize_avatar_id(avatar_id)
        else:
            avatar_id = generate_avatar_id()
        logger.info("Processing request for avatar_id: %s", avatar_id)

        video_file = data["video"]

        # Check required parameters
        if video_file is None:
            logger.warning("No video file uploaded")
            return web.Response(
                status=400, text=json.dumps({"error": "No video/image file uploaded"})
            )

        # Get MongoDB collections
        try:
            status_collection, avatars_collection = await init_mongodb()
        except Exception as e:
            logger.error("MongoDB connection failed: %s", str(e))
            return web.Response(
                status=500,
                text=json.dumps(
                    {"error": "Database connection failed", "details": str(e)}
                ),
            )

        # Check if avatar_id already exists
        if await check_avatar_exists(avatars_collection, avatar_id):
            logger.warning("Avatar ID %s already exists", avatar_id)
            return web.Response(
                status=400,
                text=json.dumps({"error": f"Avatar ID {avatar_id} already exists"}),
            )

        # Get parameters with defaults
        params = {
            "version": data.get("version", "v15"),
            "bbox_shift": int(data.get("bbox_shift", "0")),
            "extra_margin": int(data.get("extra_margin", "10")),
            "parsing_mode": data.get("parsing_mode", "jaw"),
            "left_cheek_width": int(data.get("left_cheek_width", "90")),
            "right_cheek_width": int(data.get("right_cheek_width", "90")),
        }

        # Save uploaded file with random ID
        file_extension = os.path.splitext(video_file.filename)[1]
        random_id = uuid.uuid4().hex[:8]
        file_path = UPLOAD_DIR / f"{random_id}{file_extension}"
        with open(file_path, "wb") as f:
            f.write(video_file.file.read())

        # Store avatar information
        try:
            avatar_info = await store_avatar_info(avatars_collection, avatar_id, params)
        except Exception as e:
            logger.error("Failed to store avatar info: %s", str(e))
            return web.Response(
                status=500,
                text=json.dumps(
                    {"error": "Failed to store avatar information", "details": str(e)}
                ),
            )

        # Create initial status record
        try:
            await update_processing_status(status_collection, avatar_id, "queued")
        except Exception as e:
            logger.error("Failed to update processing status: %s", str(e))
            return web.Response(
                status=500,
                text=json.dumps(
                    {"error": "Failed to update processing status", "details": str(e)}
                ),
            )

        # Add task to queue
        queue_processor.add_task(
            "preprocess",
            (status_collection, avatars_collection, avatar_id, file_path, params),
        )

        # Return immediately with success response
        return web.Response(
            status=200,
            text=json.dumps(
                {
                    "message": "Processing started",
                    "avatar_id": avatar_id,
                    "status": "queued",
                    "created_at": avatar_info["created_at"].isoformat(),
                }
            ),
        )

    except Exception as e:
        logger.error("Error in preprocessing: %s", str(e))
        return web.Response(
            status=500,
            text=json.dumps({"error": "Internal server error", "details": str(e)}),
        )


async def get_status_handler(request):
    """Get processing status for an avatar"""
    try:
        avatar_id = request.query.get("avatar_id")
        if not avatar_id:
            return web.Response(
                status=400, text=json.dumps({"error": "avatar_id is required"})
            )

        status_collection, avatars_collection = await init_mongodb()
        status = await status_collection.find_one({"avatar_id": avatar_id})
        avatar_info = await avatars_collection.find_one({"avatar_id": avatar_id})

        if not status or not avatar_info:
            return web.Response(
                status=404, text=json.dumps({"error": "Avatar not found"})
            )

        response_data = {
            "avatar_id": avatar_id,
            "status": status["status"],
            "created_at": avatar_info["created_at"].isoformat(),
            "parameters": avatar_info["parameters"],
            "preprocessed": avatar_info.get("preprocessed", False),
        }

        # Add timing information if available
        if "start_time" in status:
            response_data["start_time"] = status["start_time"].isoformat()
        if "finish_time" in status:
            response_data["finish_time"] = status["finish_time"].isoformat()
            duration = status["finish_time"] - status["start_time"]
            response_data["duration_seconds"] = duration.total_seconds()

        # Add error information if failed
        if status["status"] == "failed" and "error" in avatar_info:
            response_data["error"] = avatar_info["error"]

        return web.Response(status=200, text=json.dumps(response_data))

    except Exception as e:
        logger.error("Error getting status: %s", str(e))
        return web.Response(status=500, text=json.dumps({"error": str(e)}))


async def realtime_inference_handler(request):
    """Handle real-time inference with streaming response"""
    try:
        # Get form data
        data = await request.post()

        # Get or generate avatar_id
        avatar_id = data.get("avatar_id")
        if not avatar_id:
            return web.Response(
                status=400, text=json.dumps({"error": "avatar_id is required"})
            )
        avatar_id = sanitize_avatar_id(avatar_id)

        # Get avatar info from MongoDB to get the version
        _, avatars_collection = await init_mongodb()
        avatar_info = await avatars_collection.find_one({"avatar_id": avatar_id})
        if not avatar_info:
            return web.Response(
                status=404, text=json.dumps({"error": "Avatar not found"})
            )

        # Get audio file
        audio_file = data.get("audio")
        if not audio_file:
            return web.Response(
                status=400, text=json.dumps({"error": "No audio file uploaded"})
            )

        # Get parameters with defaults, using stored version
        params = {
            "version": avatar_info["version"],  # Use stored version
            "fps": int(data.get("fps", "25")),
            "batch_size": int(data.get("batch_size", "20")),
            "audio_padding_length_left": int(
                data.get("audio_padding_length_left", "2")
            ),
            "audio_padding_length_right": int(
                data.get("audio_padding_length_right", "2")
            ),
            "skip_save_images": data.get("skip_save_images", "false").lower() == "true",
        }

        # Save audio file temporarily with original extension
        file_extension = os.path.splitext(audio_file.filename)[1]
        audio_path = TEMP_DIR / f"{avatar_id}_audio{file_extension}"
        with open(audio_path, "wb") as f:
            f.write(audio_file.file.read())

        # Set up streaming response
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            },
        )
        await response.prepare(request)

        # Define frame streaming callback
        async def stream_frame(frame):
            # Convert frame to JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            # Write frame to response
            await response.write(
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(frame_bytes)).encode()
                + b"\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        # Add task to realtime inference queue
        queue_processor.add_task(
            "realtime_inference",
            (response, avatar_id, str(audio_path), params, stream_frame),
        )

        return response

    except Exception as e:
        logger.error("Error in realtime inference: %s", str(e))
        return web.Response(status=500, text=json.dumps({"error": str(e)}))


async def inference_handler(request):
    try:
        # Get form data
        data = await request.post()

        # Get or generate avatar_id
        avatar_id = data.get("avatar_id")
        if not avatar_id:
            return web.Response(
                status=400, text=json.dumps({"error": "avatar_id is required"})
            )
        avatar_id = sanitize_avatar_id(avatar_id)

        # Get avatar info from MongoDB to get the version
        _, avatars_collection = await init_mongodb()
        avatar_info = await avatars_collection.find_one({"avatar_id": avatar_id})
        if not avatar_info:
            return web.Response(
                status=404, text=json.dumps({"error": "Avatar not found"})
            )

        # Get audio file
        audio_file = data.get("audio")
        if not audio_file:
            return web.Response(
                status=400, text=json.dumps({"error": "No audio file uploaded"})
            )

        # Get parameters with defaults, using stored version
        params = {
            "version": avatar_info["version"],  # Use stored version
            "fps": int(data.get("fps", "25")),
            "audio_padding_length_left": int(
                data.get("audio_padding_length_left", "2")
            ),
            "audio_padding_length_right": int(
                data.get("audio_padding_length_right", "2")
            ),
        }

        # Save audio file temporarily with original extension
        file_extension = os.path.splitext(audio_file.filename)[1]
        audio_path = TEMP_DIR / f"{avatar_id}_audio{file_extension}"
        with open(audio_path, "wb") as f:
            f.write(audio_file.file.read())

        # Create a future to wait for the result
        future = asyncio.Future()

        # Add task to queue
        queue_processor.add_task(
            "inference",
            (future, avatar_id, str(audio_path), params),
        )

        # Wait for the result
        try:
            output_vid = await future
            # Return the video file
            return web.FileResponse(
                output_vid,
                headers={
                    "Content-Disposition": f'attachment; filename="{avatar_id}.mp4"',
                    "Content-Type": "video/mp4",
                },
            )
        except Exception as e:
            logger.error("Error in inference: %s", str(e))
            return web.Response(
                status=500,
                text=json.dumps({"error": f"Inference failed: {str(e)}"}),
            )

    except Exception as e:
        logger.error("Error in inference: %s", str(e))
        return web.Response(status=500, text=json.dumps({"error": str(e)}))


async def get_thread_status_handler(_request):
    """Get thread pool status"""
    try:
        response_data = {
            "max_workers": thread_pool._max_workers,
            "active_threads": len(thread_pool._threads),
            "queue_size": preprocess_queue.qsize(),
            "processing_tasks": len(queue_processor.processing_tasks),
            "queue_status": queue_processor.get_status(),
        }
        return web.Response(status=200, text=json.dumps(response_data))
    except Exception as e:
        logger.error("Error getting thread status: %s", str(e))
        return web.Response(status=500, text=json.dumps({"error": str(e)}))


async def init_app():
    app = web.Application(client_max_size=1024**3)  # 1GB max upload size

    # Initialize models
    model_manager.init_models(
        gpu_id=0,
        vae_type="sd-vae",
        unet_config="./models/musetalk/musetalk.json",
        unet_model_path="./models/musetalk/pytorch_model.bin",
        whisper_dir="./models/whisper",
        ffmpeg_path="./ffmpeg-4.4-x64-static/",
    )

    # Add routes
    app.router.add_post("/preprocess", preprocess_handler)
    app.router.add_post("/realtime-inference", realtime_inference_handler)
    app.router.add_post("/inference", inference_handler)
    app.router.add_get("/status", get_status_handler)
    app.router.add_get("/thread-status", get_thread_status_handler)
    logger.info("Routes configured")

    # Start queue processor
    asyncio.create_task(
        queue_processor.process_queue(
            process_avatar_task,
            process_inference_task,
            process_inference_task_in_realtime,
        )
    )
    logger.info("Queue processor started")

    return app


if __name__ == "__main__":
    myApp = init_app()
    logger.info("Server started on port 8080")
    web.run_app(myApp, host="0.0.0.0", port=8080, access_log=logger)
