from huggingface_hub import HfApi
import os

HF_USERNAME = "Nadoura"
SPACE_NAME = "Gaia-Agent-Submission"
space_id = f"{HF_USERNAME}/{SPACE_NAME}"

print(f"Your Space ID is: {space_id}")

api = HfApi()

try:
    api.upload_folder(
        folder_path=".",
        repo_id=space_id,
        repo_type="space"
    )
    print("Successfully uploaded files to Hugging Face Space!")
except Exception as e:
    print(f"Upload error: {e}")

