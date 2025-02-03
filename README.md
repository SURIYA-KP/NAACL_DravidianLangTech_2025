# 🧠 Synapse@DravidianLangTech 2025: Political Sentiment Analysis in Tamil X (Twitter) Comments 

![GitHub repo](https://img.shields.io/badge/GitHub-Synapse-blue?logo=github)![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)![Transformers](https://img.shields.io/badge/🤗_Transformers-HuggingFace-yellow?logo=huggingface)![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?logo=pytorch)![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-lightblue?logo=pandas)![NAACL](https://img.shields.io/badge/NAACL-2025-ff69b4?logo=academia) 

🚀 **Rank 1 of DravidianLangTech@NAACL 2025 Shared Task-**  

## 📌 Overview
This repository contains the implementation of our approach for **Political Multiclass Sentiment Analysis of Tamil X (Twitter) Comments** in Tamil Twitter (X) comments. Our method leverages **IndicBERTv2-MLM-Back-Translation** along with **TF-IDF features** to effectively classify tweets into seven sentiment categories.

🔹 **Key Features**  
✅ Hybrid deep-learning model using Transformer embeddings & Lexical features(TF-IDF)
✅ Smart hashtag expansion using an **Agentic System (LLaMA 3.1)**  
✅ **Rank 1 with Macro F1 Score: 0.38** in the shared task  
✅ Advanced training techniques like **weighted loss** & **incremental batch sizing**  

---

## 🏛 Methodology  
### ✨ Preprocessing Steps  
🔹 **Emoji Text Conversion** – Emojis replaced with meaningful text using `emoji` library  
🔹 **Agentic Hashtag Expansion** – Uses **LLaMA 3.1** to add context to hashtags  
🔹 **Stopword Removal** – Removing non-informative words to improve text representation  

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e04ef85-2bc4-49be-a9e2-44ead374a544" alt="Workflow Diagram">
</p>

### 🔥 Feature Extraction  
📌 **TF-IDF Representation** – Weighted word importance for effective classification  
📌 **Principal Component Analysis (PCA)** – Dimensionality reduction for efficiency  

### ⚙️ Model Architecture  
Our approach fuses **IndicBERTv2 embeddings** with **TF-IDF features** before classification.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/dee737ab-4956-4de0-8ae7-6468fc2ed21a" alt="Model Architecture">
</p>


 

🔹 Uses **AdamW optimizer** with **weighted cross-entropy loss**  
🔹 **Incremental batch training** – batch size grows across epochs  
🔹 **Dropout regularization (0.1)** to prevent overfitting  

---

## 📊 Results  

### 📈 **Performance of IndicBERT-V2 Model Variants**
| Model | Macro-F1 |
|---|---|
| IndicBERTv2-MLM-Back-TLM | **0.383** |
| IndicBERTv2-MLM-Sam-TLM | 0.376 |
| IndicBERTv2-SS | 0.338 |
| IndicBERTv2-MLM-only | 0.290 |

### 🏆 **Leaderboard Ranking**
| Rank | Team Name | Macro-F1 |
|---|---|---|
| **1** | **Synapse** | **0.3773** |
| 2 | KCRL | 0.3710 |
| 3 | byteSizedLLM | 0.3497 |
| 4 | Eureka-CIOL | 0.3187 |
| 5 | Wictory | 0.3115 |

---

## 📦 Installation & Usage  
```bash
git clone https://github.com/SURIYA-KP/NAACL_DravidianLangTech_2025.git
pip install -r requirements.txt
```

### 💻 **Running the Agent System**
```bash
python agent.py
```

---

## 🛠 Dependencies
📌 **Python 3.8+**  
📌 `transformers`, `emoji`, `scikit-learn`, `pandas`, `torch`  

Install dependencies using:  
```bash
pip install -r requirements.txt
```

---

## 🤝 Paper Link
📖 **Synapse@DravidianLangTech 2025: Multiclass Political Sentiment
 Analysis in Tamil X (twitter) Comments: Leveraging Feature Fusion of
 IndicBERTv2andLexicalRepresentations**(https://openreview.net/forum?id=3emk1VVVKQ&referrer=%5Bthe%20profile%20of%20Suriya%20KP%5D(%2Fprofile%3Fid%3D~Suriya_KP1))


