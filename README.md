 SmartLeadAnalyzer – AI-Powered Lead Scoring
 
Problem Statement
Build an AI-based tool to enhance lead generation and pre-screening with business intent understanding. Caprae seeks insights to prioritize "hot" leads vs low-quality or research-stage contacts.

Solution Summary
SmartLeadAnalyzer classifies leads into:
- Hot – highly qualified for outreach
- Research – promising, needs monitoring
- Ignore – irrelevant or generic leads

The model uses both **NLP (BERT embeddings)** and **manual feature engineering** to simulate a sales rep’s decision-making process.

 Technical Architecture

- **NLP Engine**: Sentence-BERT (MiniLM-L6-v2)
- **Manual Features**: 
  - Description length  
  - Keyword indicators (AI, SaaS, CRM, etc.)  
  - Generic app detection (e.g., "notes", "habit tracker")
- **Model**: GradientBoostingClassifier (Scikit-learn)
- **Accuracy**: **0.86** on 900+ GPT-labeled leads
- **Frontend**: Streamlit app with:
  -  CSV upload
  -  Real-time predictions
  -  Downloadable results

---

## Why It Matters

- Mimics human sales judgment
- Handles both structured (features) and unstructured (text) data
- Easy to deploy and extend
- Aligns with Caprae's focus on intelligent, founder-centric deal sourcing

---

##  Value-Add Features

- Hybrid model: deep NLP + lightweight scoring
- Smart UX: one-click prediction, drag-drop CSV
- Modular pipeline: easy to extend with CRM integration or GPT API
- Built in < 5 hours

---

## Files Included

- `app.py` – Streamlit frontend  
- `ml_model_bert.py` – model training  
- `gpt_labeled_leads.csv` – 900+ labeled data points  
- `lead_classifier_bert.joblib` – trained model  
- `scaler_bert.joblib` – manual feature scaler  
- `encoder_bert.joblib` – label encoder  

---

> Built with intent.  
> Designed for scale.  
> Ready to deploy. 
