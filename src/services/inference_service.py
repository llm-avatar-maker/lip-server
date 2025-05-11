import os
import glob
import pickle
import shutil
import copy
import queue
import threading
import time
from typing import Optional, Callable
import cv2
import torch
import numpy as np
from tqdm import tqdm
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.blending import get_image_blending


class InferenceService:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._current_task: Optional[threading.Thread] = None

    def realtime_inference(
        self,
        avatar_id: str,
        audio_path: str,
        out_vid_name: str,
        version: str = "v15",
        fps: int = 25,
        batch_size: int = 1,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
        skip_save_images: bool = False,
        stream_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> Optional[str]:
        """
        Perform inference to generate talking avatar video.

        Args:
            avatar_id: Unique identifier for the avatar
            audio_path: Path to input audio file
            out_vid_name: Name for the output video file
            version: Version of MuseTalk ("v1" or "v15")
            fps: Video frames per second
            batch_size: Batch size for inference
            audio_padding_length_left: Left padding length for audio
            audio_padding_length_right: Right padding length for audio
            skip_save_images: Whether to skip saving intermediate images
            stream_callback: Optional callback function to stream frames

        Returns:
            Optional[str]: Path to output video if successful, None otherwise
        """
        try:
            # Setup paths
            if version == "v15":
                base_path = f"./results/{version}/avatars/{avatar_id}"
            else:  # v1
                base_path = f"./results/avatars/{avatar_id}"

            avatar_path = base_path
            video_out_path = f"{avatar_path}/vid_output"
            coords_path = f"{avatar_path}/coords.pkl"
            latents_out_path = f"{avatar_path}/latents.pt"
            mask_out_path = f"{avatar_path}/mask"
            mask_coords_path = f"{avatar_path}/mask_coords.pkl"

            # Check if avatar exists
            if not os.path.exists(avatar_path):
                raise FileNotFoundError(
                    f"Avatar {avatar_id} not found. Please preprocess first."
                )

            # Load preprocessed data
            with open(coords_path, "rb") as f:
                coord_list_cycle = pickle.load(f)
            input_latent_list_cycle = torch.load(latents_out_path)

            input_img_list = sorted(
                glob.glob(
                    os.path.join(f"{avatar_path}/full_imgs", "*.[jpJP][pnPN]*[gG]")
                )
            )
            frame_list_cycle = read_imgs(input_img_list)

            with open(mask_coords_path, "rb") as f:
                mask_coords_list_cycle = pickle.load(f)
            input_mask_list = sorted(
                glob.glob(os.path.join(mask_out_path, "*.[jpJP][pnPN]*[gG]"))
            )
            mask_list_cycle = read_imgs(input_mask_list)

            # Create output directory
            os.makedirs(f"{avatar_path}/tmp", exist_ok=True)
            print("start inference")

            # Extract audio features
            start_time = time.time()
            whisper_input_features, librosa_length = (
                self.model_manager.audio_processor.get_audio_feature(
                    audio_path, weight_dtype=self.model_manager.weight_dtype
                )
            )
            whisper_chunks = self.model_manager.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.model_manager.device,
                self.model_manager.weight_dtype,
                self.model_manager.whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=audio_padding_length_left,
                audio_padding_length_right=audio_padding_length_right,
            )
            print(
                f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms"
            )

            # Process frames
            video_num = len(whisper_chunks)
            res_frame_queue = queue.Queue()
            idx = 0

            def process_frames(res_frame_queue, video_len, skip_save_images):
                nonlocal idx
                while True:
                    if idx >= video_len - 1:
                        break
                    try:
                        res_frame = res_frame_queue.get(block=True, timeout=1)
                    except queue.Empty:
                        continue

                    bbox = coord_list_cycle[idx % len(coord_list_cycle)]
                    ori_frame = copy.deepcopy(
                        frame_list_cycle[idx % len(frame_list_cycle)]
                    )
                    x1, y1, x2, y2 = bbox
                    try:
                        res_frame = cv2.resize(
                            res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                        )
                    except:
                        continue
                    mask = mask_list_cycle[idx % len(mask_list_cycle)]
                    mask_crop_box = mask_coords_list_cycle[
                        idx % len(mask_coords_list_cycle)
                    ]
                    combine_frame = get_image_blending(
                        ori_frame, res_frame, bbox, mask, mask_crop_box
                    )

                    if stream_callback:
                        stream_callback(combine_frame)
                    elif not skip_save_images:
                        cv2.imwrite(
                            f"{avatar_path}/tmp/{str(idx).zfill(8)}.png", combine_frame
                        )
                    idx += 1

            # Create and start processing thread
            process_thread = threading.Thread(
                target=process_frames,
                args=(res_frame_queue, video_num, skip_save_images),
            )
            process_thread.start()

            # Generate frames
            gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
            start_time = time.time()

            for whisper_batch, latent_batch in tqdm(
                gen, total=int(np.ceil(float(video_num) / batch_size))
            ):

                audio_feature_batch = self.model_manager.pe(
                    whisper_batch.to(self.model_manager.device)
                )
                latent_batch = latent_batch.to(
                    device=self.model_manager.device,
                    dtype=self.model_manager.unet.model.dtype,
                )

                pred_latents = self.model_manager.unet.model(
                    latent_batch,
                    self.model_manager.timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
                pred_latents = pred_latents.to(
                    device=self.model_manager.device,
                    dtype=self.model_manager.vae.vae.dtype,
                )
                recon = self.model_manager.vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_queue.put(res_frame)

            # Wait for processing thread to finish
            process_thread.join()

            if skip_save_images:
                print(
                    f"Total process time of {video_num} frames without saving images = {time.time() - start_time}s"
                )
            else:
                print(
                    f"Total process time of {video_num} frames including saving images = {time.time() - start_time}s"
                )

            # Create final video if not streaming
            if (
                out_vid_name is not None
                and not skip_save_images
                and not stream_callback
            ):
                os.makedirs(video_out_path, exist_ok=True)

                # Convert frames to video
                cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {avatar_path}/temp.mp4"
                print(cmd_img2video)
                os.system(cmd_img2video)

                # Combine with audio
                output_vid = os.path.join(video_out_path, f"{out_vid_name}.mp4")
                cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {avatar_path}/temp.mp4 {output_vid}"
                print(cmd_combine_audio)
                os.system(cmd_combine_audio)

                # Cleanup
                os.remove(f"{avatar_path}/temp.mp4")
                shutil.rmtree(f"{avatar_path}/tmp")
                print(f"result is saved to {output_vid}")

                return output_vid

            return None

        except Exception as e:
            print(f"Error in inference: {str(e)}")
            return None

    def inference(
        self,
        avatar_id: str,
        audio_path: str,
        out_vid_name: str,
        version: str = "v15",
        fps: int = 25,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
    ) -> Optional[str]:
        """
        Perform inference to generate talking avatar video.

        Args:
            avatar_id: Unique identifier for the avatar
            audio_path: Path to input audio file
            out_vid_name: Name for the output video file
            version: Version of MuseTalk ("v1" or "v15")
            fps: Video frames per second
            audio_padding_length_left: Left padding length for audio
            audio_padding_length_right: Right padding length for audio

        Returns:
            Optional[str]: Path to output video if successful, None otherwise
        """
        try:
            # Setup paths
            if version == "v15":
                base_path = f"./results/{version}/avatars/{avatar_id}"
            else:  # v1
                base_path = f"./results/avatars/{avatar_id}"

            avatar_path = base_path
            video_out_path = f"{avatar_path}/vid_output"
            coords_path = f"{avatar_path}/coords.pkl"
            latents_out_path = f"{avatar_path}/latents.pt"
            mask_out_path = f"{avatar_path}/mask"
            mask_coords_path = f"{avatar_path}/mask_coords.pkl"

            # Check if avatar exists
            if not os.path.exists(avatar_path):
                raise FileNotFoundError(
                    f"Avatar {avatar_id} not found. Please preprocess first."
                )

            # Load preprocessed data
            print("Loading preprocessed data...")
            with open(coords_path, "rb") as f:
                coord_list_cycle = pickle.load(f)
            input_latent_list_cycle = torch.load(latents_out_path)

            input_img_list = sorted(
                glob.glob(
                    os.path.join(f"{avatar_path}/full_imgs", "*.[jpJP][pnPN]*[gG]")
                )
            )
            frame_list_cycle = read_imgs(input_img_list)

            with open(mask_coords_path, "rb") as f:
                mask_coords_list_cycle = pickle.load(f)
            input_mask_list = sorted(
                glob.glob(os.path.join(mask_out_path, "*.[jpJP][pnPN]*[gG]"))
            )
            mask_list_cycle = read_imgs(input_mask_list)

            # Create output directory
            os.makedirs(f"{avatar_path}/tmp", exist_ok=True)
            print("Starting inference...")

            # Extract audio features
            print("Processing audio...")
            start_time = time.time()
            whisper_input_features, librosa_length = (
                self.model_manager.audio_processor.get_audio_feature(
                    audio_path, weight_dtype=self.model_manager.weight_dtype
                )
            )
            whisper_chunks = self.model_manager.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.model_manager.device,
                self.model_manager.weight_dtype,
                self.model_manager.whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=audio_padding_length_left,
                audio_padding_length_right=audio_padding_length_right,
            )
            print(
                f"Processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms"
            )

            # Batch inference
            print("Starting frame generation...")
            video_num = len(whisper_chunks)
            batch_size = 1
            res_frame_list = []

            # Generate frames
            gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
            start_time = time.time()

            for whisper_batch, latent_batch in tqdm(
                gen, total=int(np.ceil(float(video_num) / batch_size))
            ):
                audio_feature_batch = self.model_manager.pe(
                    whisper_batch.to(self.model_manager.device)
                )
                latent_batch = latent_batch.to(
                    device=self.model_manager.device,
                    dtype=self.model_manager.unet.model.dtype,
                )

                pred_latents = self.model_manager.unet.model(
                    latent_batch,
                    self.model_manager.timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
                pred_latents = pred_latents.to(
                    device=self.model_manager.device,
                    dtype=self.model_manager.vae.vae.dtype,
                )
                recon = self.model_manager.vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)

            print(
                f"Total process time of {video_num} frames = {time.time() - start_time}s"
            )

            # Process each frame and save
            print("Processing and saving frames...")
            for i, res_frame in enumerate(tqdm(res_frame_list)):
                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(
                        res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                    )
                except:
                    continue

                mask = mask_list_cycle[i % len(mask_list_cycle)]
                mask_crop_box = mask_coords_list_cycle[i % len(mask_coords_list_cycle)]
                combine_frame = get_image_blending(
                    ori_frame, res_frame, bbox, mask, mask_crop_box
                )
                cv2.imwrite(f"{avatar_path}/tmp/{str(i).zfill(8)}.png", combine_frame)

            # Create final video
            print("Creating final video...")
            os.makedirs(video_out_path, exist_ok=True)

            # Convert frames to video
            temp_vid_path = f"{avatar_path}/temp.mp4"
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
            print("Video generation command:", cmd_img2video)
            os.system(cmd_img2video)

            # Combine with audio
            output_vid = os.path.join(video_out_path, f"{out_vid_name}.mp4")
            cmd_combine_audio = (
                f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid}"
            )
            print("Audio combination command:", cmd_combine_audio)
            os.system(cmd_combine_audio)

            # Cleanup
            os.remove(temp_vid_path)
            shutil.rmtree(f"{avatar_path}/tmp")
            print(f"Result saved to {output_vid}")

            return output_vid

        except Exception as e:
            print(f"Error in inference: {str(e)}")
            return None
