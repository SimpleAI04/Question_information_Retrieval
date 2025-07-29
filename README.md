# 📚 Question Information Retrieval System

An experimental system for retrieving semantically similar questions using both **Vector Space Model (VSM)** and **pre-trained BERT embeddings**. The goal is to compare the retrieval performance of classical and deep learning-based approaches.

---

## 🚀 Features

- ✅ Support for two retrieval methods:
  - **Vector Space Model** using BOW + Cosine Similarity
  - **Pre-trained BERT embeddings** + Cosine Similarity
- 🧠 Leverage semantic understanding of BERT for contextual similarity
- 📊 Compare ranking effectiveness between VSM and BERT
- 📺 Streamlit-powered interactive demo

---

## 🗂️ Project Structure
```
Question_information_Retrieval/
│
├── Dataset/
│ └── Data/ # Dataset containing question texts
│
├── Demo/
│ └── app.py # Streamlit app for interactive retrieval
│
├── Model/
│ ├── textretrival.py # Vector Space Model logic (TF-IDF)
│ └── textretrival_bert.py # BERT-based embedding retrieval
│
└── README.md # Project documentation
```

## ▶️ How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
2. **Run the Streamlit demo**
    ```bash
    streamlit run Demo/app.py
