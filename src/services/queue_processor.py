import asyncio
import logging
from typing import Set, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time
from src.services.inference_service import InferenceService


@dataclass
class QueueTask:
    timestamp: float
    data: Tuple
    task_type: str


logger = logging.getLogger(__name__)


class QueueProcessor:
    def __init__(
        self,
        preprocess_queue: asyncio.Queue,
        inference_queue: asyncio.Queue,
        realtime_inference_queue: asyncio.Queue,
        max_workers: int = 1,
        model_manager=None,
    ):
        """
        Initialize QueueProcessor with queues and worker configuration.

        Args:
            preprocess_queue: Queue for preprocessing tasks
            inference_queue: Queue for normal inference tasks
            realtime_inference_queue: Queue for realtime inference tasks
            max_workers: Maximum number of concurrent workers
            model_manager: Model manager instance
        """
        self.preprocess_queue = preprocess_queue
        self.inference_queue = inference_queue
        self.realtime_inference_queue = realtime_inference_queue
        self.max_workers = max_workers
        self.model_manager = model_manager
        self.processing_tasks: Set[asyncio.Task] = set()
        self.preprocess_counter = 0
        self.last_task_type: Optional[str] = None
        self._task_stats: Dict[str, Dict[str, Any]] = {
            "preprocess": {"total": 0, "completed": 0, "failed": 0},
            "inference": {"total": 0, "completed": 0, "failed": 0},
            "realtime_inference": {"total": 0, "completed": 0, "failed": 0},
        }
        self._start_time = time.time()

    def process_preprocess_task(self, process_avatar_task) -> bool:
        """
        Process a task from preprocess queue.

        Args:
            process_avatar_task: Coroutine function to process avatar tasks

        Returns:
            bool: True if task was processed, False otherwise
        """
        if self.preprocess_queue.empty():
            return False

        try:
            task = self.preprocess_queue.get_nowait()
            if isinstance(task, QueueTask):
                task_data = task.data
            else:
                task_data = task

            processing_task = asyncio.create_task(process_avatar_task(*task_data))
            self.processing_tasks.add(processing_task)
            self.last_task_type = "preprocess"
            self._task_stats["preprocess"]["total"] += 1

            def cleanup_preprocess(task):
                self.processing_tasks.discard(task)
                self.preprocess_queue.task_done()
                self.preprocess_counter = 0
                if task.exception():
                    self._task_stats["preprocess"]["failed"] += 1
                    logger.error("Preprocess task failed: %s", task.exception())
                else:
                    self._task_stats["preprocess"]["completed"] += 1

            processing_task.add_done_callback(cleanup_preprocess)
            return True
        except Exception as e:
            logger.error("Error creating preprocess task: %s", str(e))
            self.preprocess_queue.task_done()
            self._task_stats["preprocess"]["failed"] += 1
            return False

    def process_inference_task(self, process_inference_task) -> bool:
        """
        Process a task from normal inference queue.

        Args:
            process_inference_task: Coroutine function to process inference tasks

        Returns:
            bool: True if task was processed, False otherwise
        """
        if self.inference_queue.empty():
            return False

        try:
            task = self.inference_queue.get_nowait()
            if isinstance(task, QueueTask):
                task_data = task.data
            else:
                task_data = task

            processing_task = asyncio.create_task(process_inference_task(*task_data))
            self.processing_tasks.add(processing_task)
            self.last_task_type = "inference"
            self._task_stats["inference"]["total"] += 1

            def cleanup_inference(task):
                self.processing_tasks.discard(task)
                self.inference_queue.task_done()
                if task.exception():
                    self._task_stats["inference"]["failed"] += 1
                    logger.error("Inference task failed: %s", task.exception())
                else:
                    self._task_stats["inference"]["completed"] += 1

            processing_task.add_done_callback(cleanup_inference)
            return True
        except Exception as e:
            logger.error("Error creating inference task: %s", str(e))
            self.inference_queue.task_done()
            self._task_stats["inference"]["failed"] += 1
            return False

    def process_realtime_inference_task(
        self, process_inference_task_in_realtime
    ) -> bool:
        """
        Process a task from realtime inference queue.

        Args:
            process_inference_task_in_realtime: Coroutine function to process realtime inference tasks

        Returns:
            bool: True if task was processed, False otherwise
        """
        if self.realtime_inference_queue.empty():
            return False

        try:
            task = self.realtime_inference_queue.get_nowait()
            if isinstance(task, QueueTask):
                task_data = task.data
            else:
                task_data = task

            processing_task = asyncio.create_task(
                process_inference_task_in_realtime(*task_data)
            )
            self.processing_tasks.add(processing_task)
            self.last_task_type = "realtime_inference"
            self._task_stats["realtime_inference"]["total"] += 1

            def cleanup_realtime_inference(task):
                self.processing_tasks.discard(task)
                self.realtime_inference_queue.task_done()
                if task.exception():
                    self._task_stats["realtime_inference"]["failed"] += 1
                    logger.error("Realtime inference task failed: %s", task.exception())
                else:
                    self._task_stats["realtime_inference"]["completed"] += 1

            processing_task.add_done_callback(cleanup_realtime_inference)
            return True
        except Exception as e:
            logger.error("Error creating realtime inference task: %s", str(e))
            self.realtime_inference_queue.task_done()
            self._task_stats["realtime_inference"]["failed"] += 1
            return False

    async def process_queue(
        self,
        process_avatar_task,
        process_inference_task,
        process_inference_task_in_realtime,
    ) -> None:
        """
        Background task to process the queue.

        Args:
            process_avatar_task: Coroutine function to process avatar tasks
            process_inference_task: Coroutine function to process normal inference tasks
            process_inference_task_in_realtime: Coroutine function to process realtime inference tasks
        """
        while True:
            try:
                # Check if we've reached max workers by counting running tasks
                running_tasks = sum(
                    1 for task in self.processing_tasks if not task.done()
                )
                if running_tasks >= self.max_workers:
                    await asyncio.sleep(0.1)
                    continue

                # If last task was inference and we have preprocess tasks waiting, give preprocess a chance
                if (
                    self.last_task_type in ["inference", "realtime_inference"]
                    and not self.preprocess_queue.empty()
                    and (
                        self.preprocess_counter >= 3
                        or (
                            self.inference_queue.empty()
                            and self.realtime_inference_queue.empty()
                        )
                    )
                ):
                    if self.process_preprocess_task(process_avatar_task):
                        continue

                # Check realtime inference queue first (higher priority)
                if not self.realtime_inference_queue.empty():
                    if self.process_realtime_inference_task(
                        process_inference_task_in_realtime
                    ):
                        continue

                # Check normal inference queue
                if not self.inference_queue.empty():
                    if self.process_inference_task(process_inference_task):
                        continue

                # If no inference tasks, process preprocess queue
                if not self.preprocess_queue.empty():
                    self.process_preprocess_task(process_avatar_task)
                else:
                    self.preprocess_counter += 1

            except Exception as e:
                logger.error("Error processing queue: %s", str(e))
            await asyncio.sleep(0.1)

    def add_task(
        self,
        task_type: str,
        task_data: Tuple,
    ) -> None:
        """
        Add a task to the appropriate queue.

        Args:
            task_type: Type of task ("preprocess", "inference", or "realtime_inference")
            task_data: Task data tuple
        """
        task = QueueTask(
            timestamp=time.time(),
            data=task_data,
            task_type=task_type,
        )

        if task_type == "preprocess":
            self.preprocess_queue.put_nowait(task)
        elif task_type == "inference":
            self.inference_queue.put_nowait(task)
        elif task_type == "realtime_inference":
            self.realtime_inference_queue.put_nowait(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the queue processor.

        Returns:
            dict: Status information including queue sizes and active tasks
        """
        uptime = time.time() - self._start_time
        return {
            "max_workers": self.max_workers,
            "active_tasks": len(self.processing_tasks),
            "preprocess_queue_size": self.preprocess_queue.qsize(),
            "inference_queue_size": self.inference_queue.qsize(),
            "realtime_inference_queue_size": self.realtime_inference_queue.qsize(),
            "last_task_type": self.last_task_type,
            "preprocess_counter": self.preprocess_counter,
            "task_stats": self._task_stats,
            "uptime_seconds": uptime,
            "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
        }
