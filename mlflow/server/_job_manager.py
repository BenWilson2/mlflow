import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from mlflow.environment_variables import MLFLOW_MAX_CONCURRENT_PROMPT_OPTIMIZATION_JOBS
from mlflow.genai.scorers.builtin_scorers import get_builtin_scorer_by_name
from mlflow.genai.optimize.types import LLMParams, OptimizerConfig
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.service_pb2 import GetOptimizePromptsJob

_logger = logging.getLogger(__name__)


# Prompt Optimization Job Management
class PromptOptimizationJobManager:
    """Manages prompt optimization jobs using a thread pool for controlled concurrency."""

    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}
        self._next_job_id = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=MLFLOW_MAX_CONCURRENT_PROMPT_OPTIMIZATION_JOBS.get(),
            thread_name_prefix="prompt_opt"
        )

    def create_job(self, *, train_dataset_id: str, eval_dataset_id: str | None,
                   prompt_url: str, scorers: list[str], target_llm: str,
                   algorithm: str | None) -> str:
        """Create a new prompt optimization job."""
        with self._lock:
            job_id = str(self._next_job_id)
            self._next_job_id += 1

            self._jobs[job_id] = {
                "status": GetOptimizePromptsJob.PromptOptimizationJobStatus.PENDING,
                "train_dataset_id": train_dataset_id,
                "eval_dataset_id": eval_dataset_id,
                "prompt_url": prompt_url,
                "scorers": scorers,
                "target_llm": target_llm,
                "algorithm": algorithm,
                "result": None,
                "error": None,
                "created_time": time.time()
            }

            # Submit the job to the thread pool
            self._executor.submit(self._run_job, job_id)

            return job_id

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status and result."""
        return self._jobs.get(job_id).copy()

    def _run_job(self, job_id: str):
        """Run the prompt optimization job."""
        from mlflow.genai.datasets import get_dataset
        from mlflow.genai.optimize import optimize_prompt
        from mlflow.genai.prompts import load_prompt

        try:
            job = self._jobs[job_id]
            job["status"] = GetOptimizePromptsJob.PromptOptimizationJobStatus.RUNNING

            # Load datasets
            train_dataset = get_dataset(job["train_dataset_id"])
            eval_dataset = get_dataset(job["eval_dataset_id"])

            # Convert to pandas DataFrames
            train_df = train_dataset.to_df()
            eval_df = eval_dataset.to_df() if eval_dataset else None

            # Validate dataset structure
            required_columns = ["inputs", "expectations"]
            missing_columns = []
            for col in required_columns:
                if col not in train_df.columns:
                    missing_columns.append(col)

            if missing_columns:
                raise MlflowException(
                    f"Training dataset missing required columns: {missing_columns}. "
                    f"Available columns: {list(train_df.columns)}",
                    INVALID_PARAMETER_VALUE,
                )

            # Get scorers
            scorer_instances = []
            for scorer_name in job["scorers"]:
                try:
                    scorer = get_builtin_scorer_by_name(scorer_name)
                    scorer_instances.append(scorer)
                except Exception as e:
                    raise MlflowException(
                        f"Failed to create scorer '{scorer_name}': {e}",
                        INVALID_PARAMETER_VALUE,
                    )

            if not scorer_instances:
                raise MlflowException(
                    "No valid scorers provided for optimization",
                    INVALID_PARAMETER_VALUE,
                )

            # Log optimization parameters
            _logger.info(f"Starting prompt optimization job: {job_id}")

            # Set up LLM parameters

            # Parse target_llm to create LLMParams
            target_llm = job['target_llm']
            llm_params = LLMParams(model_name=target_llm)

            # Set up optimizer config
            algorithm = job['algorithm']

            algorithm_kwarg = {"algorithm": algorithm} if algorithm else {}

            optimizer_config = OptimizerConfig(
                algorithm=algorithm,
                **algorithm_kwarg,
            )

            prompt_input_url = job["prompt_url"]

            # Call the actual optimize_prompt function
            result = optimize_prompt(
                target_llm_params=llm_params,
                prompt=prompt_input_url,
                train_data=train_df,
                scorers=scorer_instances,
                eval_data=eval_df,
                optimizer_config=optimizer_config,
            )

            _logger.info(f"Prompt optimization job {job_id} completed.")

            # Save optimization result
            job["result"] = {
                "prompt_url": result.prompt.uri,
                "evaluation_score": result.final_eval_score
            }
            job["status"] = GetOptimizePromptsJob.PromptOptimizationJobStatus.COMPLETED

        except Exception as e:
            job["status"] = GetOptimizePromptsJob.PromptOptimizationJobStatus.FAILED
            job["error"] = str(e)
            _logger.error(f"Prompt optimization job {job_id} failed: {e}")


# Global job manager instance
_prompt_optimization_job_manager = PromptOptimizationJobManager()
