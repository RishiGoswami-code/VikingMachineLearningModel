# Viking Machine Learning Ensemble Model

This repository contains the backend for the document authenticity verification system. It employs an **Ensemble Learning** approach, combining predictions from **six different machine learning models** to achieve highly robust and reliable classification of certificates as "Genuine" or "Fraud."

The API accepts an image path, runs a simulated Computer Vision (CV) pipeline to extract key similarity scores, and feeds those scores to the ensemble for the final prediction.

***

## Key Features

* **6-Model Ensemble:** Combines the predictive power of **CNN, Random Forest, SVM, Logistic Regression, K-Means, and Naive Bayes**.
* **Deep Learning Feature Integration:** Uses simulated outputs from an image-based feature extractor (signature similarity, seal similarity, layout alignment).
* **FastAPI Deployment:** Robust and fast API server built on FastAPI for easy prediction serving.
* **Tie-Breaker Priority:** Uses a structured voting system with a clear priority (CNN > RF > SVM > LR > K-Means > NB) to resolve 3-3 ties.
* 
---

## **Setup & Installation**

### 1. **Clone Repository**

```bash
git clone [https://github.com/RishiGoswami-code/VikingMachineLearningModel.git]
cd VikingMachineLearningModel
```

---

### 2. **Create & Activate Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: .\venv\Scripts\activate
```

---

### **3. Install Dependencies**

```bash
pip install fastapi uvicorn numpy scikit-learn tensorflow pydantic
```

---


# **Running the API Server**

  ### Start the Server
    ```bash
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    ```
When the system loads and trains all models, you will see:

```cpp
All 6 Models Trained and Ready.
```

### Our API will run at:

```bash
http://127.0.0.1:8000
```

---


# **Swagger API Documentation**

### Open in your browser:

```bash
http://127.0.0.1:8000/docs
```

### This interactive UI lets you test predictions directly.

---

# **API Usage**

### Prediction Endpoint

|**Method** |	**Path**  | **Description**                        |
|-----------|-------------|----------------------------------------|
|POST	    | /predict    |	Runs the full ML pipeline + ensemble.  |

---

# **Input Format (JSON)**

```bash
{
  "image_path": "certificate_good_sig_good_seal_high_clarity.jpg"
}
```

# **Output Example (JSON)**

```bash
{
  "prediction_status": true,
  "prediction_label": "GENUINE"
}
```

---

# **Response Field Description**

| **Field **        | **Type ** | **Description**                   |
| ------------------|-----------|-----------------------------------|
| prediction_status | boolean   | `true` = Genuine, `false` = Fraud |
| prediction_label  | string    | `"GENUINE"` or `"FRAUD"`          |

---


# **Ensemble Voting Logic**

### The final decision is determined using majority voting among all six models:

  - CNN

  - Random Forest

  - SVM

  - Logistic Regression

  - K-Means

  - Naive Bayes

---

# **Tie-Breaker Priority (Highest → Lowest)**

### If the ensemble has a 3–3 split, the model with higher priority wins:

  - CNN

  - Random Forest

  - SVM

  - Logistic Regression

  - K-Means

  - Naive Bayes

---


# **How the System Works**

### `image_analyzer.py` simulates feature extraction from the provided image path.

### Features are fed into all six ML models.

  - Each model outputs its prediction.

  - The ensemble computes majority vote.

  - If tied, priority order decides the final label.

---

# **Technologies Used:**

  - FastAPI (API framework)

  - TensorFlow (CNN model)

  - scikit-learn (RF, SVM, Logistic Regression, Naive Bayes, K-Means)

  - Uvicorn (ASGI server)

  - Pydantic (input validation)

---

# **Future Enhancements**

  - Replace simulated CV pipeline with real image analysis

  - Model retraining from UI

  - Add confidence scores

  - Deploy to cloud (AWS / GCP / Render)

# **Author:**

`Chandan GIri`
