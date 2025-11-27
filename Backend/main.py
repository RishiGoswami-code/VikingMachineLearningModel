from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import math
from collections import defaultdict


def mean(lst):
    return sum(lst) / len(lst)

def variance(lst):
    u = mean(lst)
    return sum((x - u)**2 for x in lst) / len(lst)

def gaussian_logpdf(x, mu, var):
    var = max(var, 1e-4)
    return -0.5 * math.log(2 * math.pi * var) - ((x - mu)**2) / (2 * var)

def safe_log(x):
    return math.log(max(x, 1e-12))

class NaiveBayesModel:
    def __init__(self, data, weights):
        self.data = data
        self.weights = weights

        self.stats = {"genuine": {}, "fraud": {}}

        for label in ["genuine", "fraud"]:
            for feature, values in data[label].items():

                if all(isinstance(x, (int, float)) for x in values):
                    self.stats[label][feature] = {
                        "mu": mean(values),
                        "var": variance(values)
                    }
                else:
                    freq = defaultdict(int)
                    for v in values:
                        freq[v] += 1
                    self.stats[label][feature] = {
                        "freq": freq,
                        "total": len(values),
                        "unique": len(freq)
                    }

    def log_prior(self, label):
        total = len(self.data["genuine"]) + len(self.data["fraud"])
        prior = len(self.data[label]) / total
        return safe_log(prior)

    def log_like_numeric(self, feature, label, value):
        mu = self.stats[label][feature]["mu"]
        var = self.stats[label][feature]["var"]
        return gaussian_logpdf(value, mu, var)

    def log_like_categorical(self, feature, label, value):
        freq = self.stats[label][feature]["freq"].get(value, 0)
        total = self.stats[label][feature]["total"]
        unique = self.stats[label][feature]["unique"]

        prob = (freq + 1) / (total + unique)
        return safe_log(prob)

    def score(self, label, userData):
        score = self.log_prior(label)

        for feature, value in userData.items():

            weight = self.weights.get(feature, 1.0)

            if feature in self.stats[label] and "mu" in self.stats[label][feature]:
                logp = self.log_like_numeric(feature, label, value)
            else:
                logp = self.log_like_categorical(feature, label, value)

            score += weight * logp

        return score

    def predict(self, userData):
        g = self.score("genuine", userData)
        f = self.score("fraud", userData)

        mx = max(g, f)
        pg = math.exp(g - mx)
        pf = math.exp(f - mx)

        prob_g = pg / (pg + pf)
        prob_f = pf / (pg + pf)

        return {
            "genuine_probability": prob_g,
            "fraud_probability": prob_f,
            "prediction": "genuine" if prob_g > prob_f else "fraud"
        }


training_data = {
    "genuine": {
        "seal_sim": [0.85, 0.90, 0.92, 0.87],
        "sig_sim": [0.88, 0.91, 0.93, 0.86],
        "qr_distance": [1, 0, 2, 1],
        "font_score": [0.9, 0.92, 0.95, 0.88]
    },
    "fraud": {
        "seal_sim": [0.2, 0.3, 0.4, 0.25],
        "sig_sim": [0.3, 0.4, 0.45, 0.35],
        "qr_distance": [12, 15, 13, 18],
        "font_score": [0.3, 0.5, 0.4, 0.45]
    }
}

feature_weights = {
    "seal_sim": 1.2,
    "sig_sim": 1.3,
    "qr_distance": 0.9,
    "font_score": 1.1
}

model = NaiveBayesModel(training_data, feature_weights)

#---------------------------------------------------------------------------------
app = FastAPI()

PREDICTION_DB = {}


class UserData(BaseModel):
    seal_sim: float
    sig_sim: float
    qr_distance: float
    font_score: float

@app.post("/predict")
def predict(user_input: UserData):
    user_dict = user_input.dict()

    result = model.predict(user_dict)

    request_id = str(uuid.uuid4())
    PREDICTION_DB[request_id] = result

    return {
        "request_id": request_id,
        "prediction": result["prediction"],
        "fraud_probability": result["fraud_probability"],
        "genuine_probability": result["genuine_probability"]
    }

@app.get("/result/{request_id}")
def get_result(request_id: str):
    if request_id not in PREDICTION_DB:
        return {"error": "Invalid request ID"}

    return PREDICTION_DB[request_id]


