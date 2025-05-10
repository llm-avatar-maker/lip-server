import os
import glob
import pickle
import json
import shutil
from typing import Callable, Optional, Dict, Any
import cv2
import torch
from tqdm import tqdm
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.blending import get_image_prepare_material


class Preprocessor:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._current_progress = 0
        self._total_steps = 0
        self._progress_callback: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set callback for progress updates"""
        self._progress_callback = callback

    def _update_progress(self, step: int, message: str) -> None:
        """Update progress and call callback if set"""
        self._current_progress = step
        if self._progress_callback:
            progress = (step / self._total_steps) * 100 if self._total_steps > 0 else 0
            self._progress_callback(progress, message)

    def preprocess(
        self,
        avatar_id: str,
        video_path: str,
        version: str = "v15",
        bbox_shift: int = 0,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
    ) -> bool:
        """
        Preprocess video/images to create avatar materials.

        Args:
            avatar_id: Unique identifier for the avatar
            video_path: Path to input video or directory of images
            version: Version of MuseTalk ("v1" or "v15")
            bbox_shift: Bounding box shift value
            extra_margin: Extra margin for face cropping
            parsing_mode: Face blending parsing mode
            left_cheek_width: Width of left cheek region
            right_cheek_width: Width of right cheek region

        Returns:
            bool: True if preprocessing was successful
        """
        try:
            # Setup paths
            if version == "v15":
                base_path = f"./results/{version}/avatars/{avatar_id}"
            else:  # v1
                base_path = f"./results/avatars/{avatar_id}"

            avatar_path = base_path
            full_imgs_path = f"{avatar_path}/full_imgs"
            coords_path = f"{avatar_path}/coords.pkl"
            latents_out_path = f"{avatar_path}/latents.pt"
            mask_out_path = f"{avatar_path}/mask"
            mask_coords_path = f"{avatar_path}/mask_coords.pkl"
            avatar_info_path = f"{avatar_path}/avator_info.json"

            # Create directories
            self._osmakedirs([avatar_path, full_imgs_path, mask_out_path])

            # Save avatar info
            avatar_info = {
                "avatar_id": avatar_id,
                "video_path": video_path,
                "bbox_shift": bbox_shift,
                "version": version,
            }
            with open(avatar_info_path, "w", encoding="utf-8") as f:
                json.dump(avatar_info, f)

            # Process input video/images
            self._update_progress(1, "Processing input video/images")
            if os.path.isfile(video_path):
                self._video2imgs(video_path, full_imgs_path)
            else:
                print(f"copy files in {video_path}")
                files = os.listdir(video_path)
                files.sort()
                files = [file for file in files if file.split(".")[-1] == "png"]
                for filename in files:
                    shutil.copyfile(
                        f"{video_path}/{filename}", f"{full_imgs_path}/{filename}"
                    )

            # Extract landmarks and prepare materials
            input_img_list = sorted(
                glob.glob(os.path.join(full_imgs_path, "*.[jpJP][pnPN]*[gG]"))
            )
            self._update_progress(2, "Extracting landmarks")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)

            # Process frames and create latents
            self._update_progress(3, "Creating latents")
            input_latent_list = []
            idx = -1
            coord_placeholder = (0.0, 0.0, 0.0, 0.0)
            for bbox, frame in zip(coord_list, frame_list):
                idx = idx + 1
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                if version == "v15":
                    y2 = y2 + extra_margin
                    y2 = min(y2, frame.shape[0])
                    coord_list[idx] = [x1, y1, x2, y2]
                crop_frame = frame[y1:y2, x1:x2]
                resized_crop_frame = cv2.resize(
                    crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
                )
                latents = self.model_manager.vae.get_latents_for_unet(
                    resized_crop_frame
                )
                input_latent_list.append(latents)

            # Create cycle lists
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            # Process masks
            self._update_progress(4, "Processing masks")
            mask_coords_list_cycle = []
            mask_list_cycle = []

            for i, frame in enumerate(tqdm(frame_list_cycle)):
                cv2.imwrite(f"{full_imgs_path}/{str(i).zfill(8)}.png", frame)

                x1, y1, x2, y2 = coord_list_cycle[i]
                if version == "v15":
                    mode = parsing_mode
                else:
                    mode = "raw"
                mask, crop_box = get_image_prepare_material(
                    frame,
                    [x1, y1, x2, y2],
                    fp=self.model_manager.create_face_parser(
                        version=version,
                        left_cheek_width=left_cheek_width,
                        right_cheek_width=right_cheek_width,
                    ),
                    mode=mode,
                )

                cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)
                mask_coords_list_cycle.append(crop_box)
                mask_list_cycle.append(mask)

            # Save processed data
            self._update_progress(5, "Saving processed data")
            with open(mask_coords_path, "wb") as f:
                pickle.dump(mask_coords_list_cycle, f)
            with open(coords_path, "wb") as f:
                pickle.dump(coord_list_cycle, f)
            torch.save(input_latent_list_cycle, latents_out_path)

            self._update_progress(6, "Preprocessing completed")
            return True

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return False

    def _video2imgs(
        self, vid_path: str, save_path: str, cut_frame: int = 10000000
    ) -> None:
        """Convert video to images"""
        cap = cv2.VideoCapture(vid_path)
        count = 0
        while True:
            if count > cut_frame:
                break
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
                count += 1
            else:
                break

    def _osmakedirs(self, path_list: list) -> None:
        """Create directories if they don't exist"""
        for path in path_list:
            if not os.path.exists(path):
                os.makedirs(path)

    def get_status(self) -> Dict[str, Any]:
        """Get current preprocessing status"""
        return {
            "current_progress": self._current_progress,
            "total_steps": self._total_steps,
            "progress_percentage": (
                (self._current_progress / self._total_steps * 100)
                if self._total_steps > 0
                else 0
            ),
        }
