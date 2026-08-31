from huggingface_hub import HfApi
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

TARGET_RESULTS_FILE = "evaluation_results.jsonl"


def load_official_gaia_questions_strictly():
   
    try:
        print("Fetching official GAIA benchmark questions...")
        
     
        raise RuntimeError("Official question fetching logic needs to be connected here.")

    except Exception as e:
        logger.error(f"Critical error fetching official questions: {e}")
        print(f"\n[ERROR]: Stopping execution because official questions could not be fetched. No fallback allowed! Details: {e}")
        sys.exit(1)


def save_result_line(result_data: dict, file_path: str = TARGET_RESULTS_FILE):
    
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] Could not save result to {file_path}: {e}")


def upload_results_to_huggingface(local_file_path: str = TARGET_RESULTS_FILE):
   
    api = HfApi()
    repo_id = "Nadoura/gaia-agent-results"
    
    if not os.path.exists(local_file_path):
        print(f"[WARNING]: Results file '{local_file_path}' not found yet. Nothing to upload.")
        return

    try:
        print(f"Uploading evaluation results from '{local_file_path}' to Hugging Face dataset ({repo_id})...")
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo="results.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("Results successfully uploaded to Hugging Face! 🚀")
    except Exception as e:
        print(f"[ERROR] Failed to upload results to Hugging Face: {e}")