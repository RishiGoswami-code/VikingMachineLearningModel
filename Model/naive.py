from collections import defaultdict
import math


def mean(lst):
    return sum(lst) / len(lst)

def variance(lst):
    u = mean(lst)
    return sum((x - u)**2 for x in lst) / len(lst)   # population variance

def gaussian_logpdf(x, mu, var):
    """Return log P(x | class) using Gaussian formula"""
    var = max(var, 1e-4)   # avoid zero variance
    return -0.5 * math.log(2 * math.pi * var) - ((x - mu)**2) / (2 * var)

def safe_log(x):
    return math.log(max(x, 1e-12))



class NaiveBayesModel:
    def __init__(self, data, userData, weights):
        """
        data = {
            "genuine": {feature: [values...]},
            "fraud": {feature: [values...]}
        }

        userData = {"seal_sim": 0.85, "sig_sim": 0.91, ...}

        weights = {"seal_sim": 1.0, "sig_sim": 1.3, ...}
        """
        self.data = data
        self.userData = userData
        self.weights = weights

        # Precompute means and variances for all numerical features
        self.stats = { "genuine": {}, "fraud": {} }

        for label in ["genuine", "fraud"]:
            for feature, values in data[label].items():
                # If numeric → compute Gaussian stats
                if all(isinstance(x, (int, float)) for x in values):
                    self.stats[label][feature] = {
                        "mu": mean(values),
                        "var": variance(values)
                    }
                else:
                    # Categorical frequency
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

    def log_likelihood_numeric(self, feature, label):
        x = self.userData[feature]
        mu = self.stats[label][feature]["mu"]
        var = self.stats[label][feature]["var"]
        return gaussian_logpdf(x, mu, var)


    def log_likelihood_categorical(self, feature, label):
        val = self.userData[feature]
        freq = self.stats[label][feature]["freq"].get(val, 0)
        total = self.stats[label][feature]["total"]
        unique = self.stats[label][feature]["unique"]

        prob = (freq + 1) / (total + unique)
        return safe_log(prob)


    def class_score(self, label):
        score = self.log_prior(label)

        for feature in self.userData:

            # Weight (importance of feature)
            w = self.weights.get(feature, 1.0)

            # Numeric?
            if feature in self.stats[label] and "mu" in self.stats[label][feature]:
                logp = self.log_likelihood_numeric(feature, label)

            # Categorical?
            else:
                logp = self.log_likelihood_categorical(feature, label)

            score += w * logp

        return score

    def result(self):
        g = self.class_score("genuine")
        f = self.class_score("fraud")

        mx = max(g, f)
        pg = math.exp(g - mx)
        pf = math.exp(f - mx)

        p_genuine = pg / (pg + pf)
        p_fraud = pf / (pg + pf)

        return {
            "genuine_probability": p_genuine,
            "fraud_probability": p_fraud,
            "prediction": "genuine" if p_genuine > p_fraud else "fraud"
        }




training_data = {
    "genuine": {
        "sig_sim":        [0.91, 0.93, 0.89, 0.95, 0.92, 0.94, 0.88, 0.97, 0.90, 0.91],
        "seal_sim":       [0.88, 0.90, 0.87, 0.89, 0.92, 0.91, 0.85, 0.90, 0.88, 0.89],
        "photo_clarity":  [0.93, 0.95, 0.89, 0.92, 0.94, 0.96, 0.88, 0.91, 0.93, 0.95],
        "text_alignment": [0.90, 0.88, 0.91, 0.92, 0.89, 0.93, 0.87, 0.92, 0.91, 0.90],
        "font_match":     ["yes","yes","yes","yes","yes","yes","yes","yes","yes","yes"],
        "layout_ok":      ["yes","yes","yes","yes","yes","yes","yes","yes","yes","yes"]
    },

    "fraud": {
        "sig_sim":        [0.35, 0.42, 0.50, 0.47, 0.39, 0.55, 0.40, 0.52, 0.49, 0.45],
        "seal_sim":       [0.30, 0.38, 0.42, 0.35, 0.40, 0.45, 0.33, 0.47, 0.43, 0.36],
        "photo_clarity":  [0.40, 0.48, 0.52, 0.50, 0.46, 0.58, 0.44, 0.55, 0.49, 0.41],
        "text_alignment": [0.38, 0.42, 0.45, 0.40, 0.44, 0.48, 0.36, 0.47, 0.41, 0.39],
        "font_match":     ["no","no","no","yes","no","yes","no","yes","no","yes"],
        "layout_ok":      ["no","no","yes","no","no","yes","no","yes","no","yes"]
    }
}
weights = {
    "sig_sim": 1.4,
    "seal_sim": 1.2,
    "photo_clarity": 1.1,
    "text_alignment": 1.0,
    "font_match": 1.0,
    "layout_ok": 1.0
}

test_cases = [

    # 1 — Clearly Genuine
    {"sig_sim": 0.94, "seal_sim": 0.91, "photo_clarity": 0.95, "text_alignment": 0.92,
     "font_match": "yes", "layout_ok": "yes"},

    # 2 — Strong signature, weak seal (borderline)
    {"sig_sim": 0.90, "seal_sim": 0.60, "photo_clarity": 0.88, "text_alignment": 0.84,
     "font_match": "yes", "layout_ok": "no"},

    # 3 — Weak everywhere (fraud)
    {"sig_sim": 0.42, "seal_sim": 0.39, "photo_clarity": 0.44, "text_alignment": 0.40,
     "font_match": "no", "layout_ok": "no"},

    # 4 — Fake with good layout
    {"sig_sim": 0.50, "seal_sim": 0.48, "photo_clarity": 0.53, "text_alignment": 0.47,
     "font_match": "no", "layout_ok": "yes"},

    # 5 — Real signature but fake seal
    {"sig_sim": 0.92, "seal_sim": 0.52, "photo_clarity": 0.90, "text_alignment": 0.89,
     "font_match": "yes", "layout_ok": "yes"},

    # 6 — Real-ish, but font mismatch
    {"sig_sim": 0.88, "seal_sim": 0.86, "photo_clarity": 0.91, "text_alignment": 0.90,
     "font_match": "no", "layout_ok": "yes"},

    # 7 — High clarity but low signature
    {"sig_sim": 0.60, "seal_sim": 0.58, "photo_clarity": 0.92, "text_alignment": 0.85,
     "font_match": "yes", "layout_ok": "no"},

    # 8 — Extreme fraud
    {"sig_sim": 0.18, "seal_sim": 0.22, "photo_clarity": 0.30, "text_alignment": 0.25,
     "font_match": "no", "layout_ok": "no"},

    # 9 — Balanced mid case (hard)
    {"sig_sim": 0.75, "seal_sim": 0.70, "photo_clarity": 0.78, "text_alignment": 0.73,
     "font_match": "yes", "layout_ok": "yes"},

    # 10 — Genuine with slight defects
    {"sig_sim": 0.89, "seal_sim": 0.82, "photo_clarity": 0.90, "text_alignment": 0.86,
     "font_match": "yes", "layout_ok": "no"}
]


for i in range(len(test_cases)):
    model = NaiveBayesModel(training_data, test_cases[i], weights)
    print("test case ", i+1, " --->  ", model.result())