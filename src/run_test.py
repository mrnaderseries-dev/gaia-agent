from huggingface_hub import HfApi
import json
import os
import sys

TARGET_RESULTS_FILE = "evaluation_results.jsonl"

def save_result_line(result_data: dict, file_path: str = TARGET_RESULTS_FILE):
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
        print(f"Successfully saved test result to {file_path}")
    except Exception as e:
        print(f"[ERROR] Could not save result: {e}")

def upload_results_to_huggingface(local_file_path: str = TARGET_RESULTS_FILE):
    api = HfApi()
    repo_id = "Nadoura/gaia-agent-results"
    
    if not os.path.exists(local_file_path):
        print(f"[WARNING]: Results file '{local_file_path}' not found.")
        return

    try:
        print(f"Uploading to Hugging Face dataset ({repo_id})...")
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo="results.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("Results successfully uploaded to Hugging Face! 🚀")
    except Exception as e:
        print(f"[ERROR] Failed to upload: {e}")

if __name__ == "__main__":
    print("--- STARTING FULL EVALUATION RUNNER ---")
    
    # 1. حفظ النتيجة التجريبية بالملف
    sample_result = {"question_id": "test_1", "agent_answer": "test_output", "correct": True}
    save_result_line(sample_result)
    
    # 2. رفع الملف إلى هักينغ فايس
    upload_results_to_huggingface(TARGET_RESULTS_FILE)
    print("--- SCRIPT FINISHED SUCCESSFULLY ---")
