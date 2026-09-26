# Explainable AI-Based Judgment Prediction for Divorce Cases

## Overview
An NLP + Explainable AI system that predicts whether an Indian divorce case 
will be Accepted or Rejected, and explains the prediction using LIME.

## Dataset
- Source: Indian High Court judgments (Hugging Face: overthelex/indian-court-decisions)
- Size: 1,000 divorce cases (balanced to 1,430 after oversampling)
- Labels: Accepted / Rejected

## Results
| Model | Accuracy | F1 Macro |
|---|---|---|
| TF-IDF + Logistic Regression (Baseline) | 75.00 percent | 72.00 percent |
| InLegalBERT (Balanced + 512 tokens) | 90.21 percent | 90.21 percent |

## Explainable AI
LIME identifies words in the judgment that influenced prediction.

## How to Run
    pip install -r requirements.txt
    python src/collect_data.py
    python src/train_baseline.py
    python src/train_bert.py
    python src/lime_explanation.py

## Author
[Your Name]

## License
For academic and research purposes only.
