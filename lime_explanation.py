import pandas as pd, numpy as np, torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

mp = "./model_90_accuracy"
tokenizer = AutoTokenizer.from_pretrained(mp)
model = AutoModelForSequenceClassification.from_pretrained(mp)
model.eval()
data = pd.read_csv("data/divorce_data_balanced.csv")

def predict_proba(texts):
    bs = 16
    out = []
    for i in range(0, len(texts), bs):
        b = texts[i:i+bs]
        inp = tokenizer(list(b), padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model(**inp)
            out.append(torch.softmax(o.logits, dim=1).cpu().numpy())
    return np.vstack(out)

explainer = LimeTextExplainer(class_names=["Accepted","Rejected"])
sample = data.iloc[0]
print(f"Case: {sample.get('title','Unknown')}")
print(f"Actual: {sample['outcome']}")
exp = explainer.explain_instance(sample["text"], predict_proba,
                                  num_features=10, num_samples=500)
print("\n--- Explainable AI Output ---")
for w, wt in exp.as_list():
    print(f"  {w} --> {'ACCEPTED' if wt>0 else 'REJECTED'}")
