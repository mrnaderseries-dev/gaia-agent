import json
import os

if os.path.exists("results.json"):
    with open("results.json", "r") as f:
        completed_answers = json.load(f)
else:
    completed_answers = {}

# مثال افتراضي للـ dataset (قم بتعديلها لتطابق مشروعك)
# for item in dataset:
#     task_id = item.get("task_id")
#     if task_id in completed_answers:
#         print(f"تم تخطي السؤال {task_id} لأنه حل مسبقاً ✅")
#         continue
