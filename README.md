# Deployment of Models

This repository contains code to deploy trained models using Django for product category classification.

### Models Deployed for Product Classification:
1. Linear Regression
2. Support Vector Machine (SVM)
3. Naive Bayes
4. XGBoost
5. LSTM with word-level embeddings

---

### Models Deployed for Product Cluster Labeling:
1. String Matching
2. Cosine Similarity

### Steps to Deploy:

1. **Clone the Repository**: `git clone <repository-url>`
2. **Navigate to Project Directory**: `cd project-directory`
3. `python manage.py collectstatic`
4. `django compose build`
5.  `django compose up`
6. **Go to localhost:8000 on your browser.**

## Product Structure

### 
    ./
    ├── app
    │   ├── django files
    ├── db.sqlite3
    ├── docker-compose.yml
    ├── dockerfile
    ├── manage.py
    ├── model_deployment
    │   ├── models.py
    │   ├── utils
    │   │   ├─ helper_function.py
    │   ├─ views.py
    ├── model_files
    │   ├── model chekpoints 
    ├── README.md
    ├── requirements.txt
    └── result.txt

### In the model files directory, model checkpoints and other files such as vocab, count_vec are stored.
### In the model_deployment directory models are defined and in view.py get/post methods are implemented.