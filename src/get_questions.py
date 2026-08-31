import json
from datasets import load_dataset

print("Downloading GAIA evaluation subset from Hugging Face...")
dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")

subset_questions = []
for i, item in enumerate(dataset):
    if i >= 20:
        break
    subset_questions.append({
        "id": item.get("task_id"),
        "text": item.get("Question")
    })

with open("questions.jsonl", "w", encoding="utf-8") as f:
    for q in subset_questions:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"Successfully downloaded and saved {len(subset_questions)} questions to 'questions.jsonl'! 🚀")
