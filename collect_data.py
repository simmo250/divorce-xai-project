from datasets import load_dataset
from itertools import islice
import pandas as pd

dataset = load_dataset("overthelex/indian-court-decisions",
                       "high_courts", split="train", streaming=True)
small_data = list(islice(dataset, 100000))
divorce_cases = []
for case in small_data:
    text = case.get("full_text", "")
    if not text:
        continue
    t = text.lower()
    if ("divorce" in t or "matrimonial" in t or "dissolution of marriage" in t):
        outcome = case.get("disposal_nature_normalized", "unknown")
        if "allowed" in outcome.lower():
            label = "Accepted"
        elif "dismissed" in outcome.lower() or "disposed" in outcome.lower():
            label = "Rejected"
        else:
            continue
        divorce_cases.append({"text": text, "outcome": label,
                              "title": case.get("title", ""),
                              "year": case.get("year", "")})
    if len(divorce_cases) >= 1000:
        break
pd.DataFrame(divorce_cases).to_csv("data/divorce_data_large.csv", index=False)
print("Saved data/divorce_data_large.csv")
