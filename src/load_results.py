import os
import json

if os.path.exists("results.json"):
    with open("results.json", "r") as f:
        completed_answers = json.load(f)
else:
    completed_answers = {}
