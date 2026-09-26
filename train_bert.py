import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset

data = pd.read_csv("data/divorce_data_balanced.csv")
data["label"] = data["outcome"].map({"Accepted": 0, "Rejected": 1})
train_df, test_df = train_test_split(data, test_size=0.2,
                                      random_state=42, stratify=data["label"])
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length",
                     truncation=True, max_length=512)

train_ds = Dataset.from_pandas(train_df[["text","label"]]).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df[["text","label"]]).map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "law-ai/InLegalBERT", num_labels=2)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="macro")}

args = TrainingArguments(output_dir="./model_90_accuracy", num_train_epochs=4,
                         per_device_train_batch_size=4, gradient_accumulation_steps=2,
                         per_device_eval_batch_size=4, learning_rate=2e-5,
                         eval_strategy="epoch", save_strategy="epoch",
                         load_best_model_at_end=True, metric_for_best_model="f1",
                         fp16=True, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=test_ds, compute_metrics=compute_metrics)
trainer.train()
trainer.save_model("./model_90_accuracy")
tokenizer.save_pretrained("./model_90_accuracy")
print("Training complete!")
