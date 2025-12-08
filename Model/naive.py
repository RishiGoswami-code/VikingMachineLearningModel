from collections import defaultdict
import math


def mean(lst):
    return sum(lst) / len(lst)

def variance(lst):
    u = mean(lst)
    return sum((x - u)**2 for x in lst) / len(lst)

def gaussian_logpdf(x, mu, var):
    """Return log P(x | class) using Gaussian formula"""
    # [Image of Gaussian probability density function curve]
    var = max(var, 1e-4)
    return -0.5 * math.log(2 * math.pi * var) - ((x - mu)**2) / (2 * var)

def safe_log(x):
    return math.log(max(x, 1e-12))


class NaiveBayesModel:
    def __init__(self, data, weights):
        """
        Precomputes the statistics (means/variances for numeric, frequencies for categorical)
        from the training data.
        
        data = {
            "genuine": {feature: [values...]},
            "fraud": {feature: [values...]}
        }
        weights = {"seal_sim": 1.0, "sig_sim": 1.3, ...}
        """
        self.data = data
        self.weights = weights

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

    def log_likelihood_numeric(self, feature, label, userData):
        x = userData[feature]
        mu = self.stats[label][feature]["mu"]
        var = self.stats[label][feature]["var"]
        return gaussian_logpdf(x, mu, var)


    def log_likelihood_categorical(self, feature, label, userData):
        val = userData[feature]
        freq = self.stats[label][feature]["freq"].get(val, 0)
        total = self.stats[label][feature]["total"]
        unique = self.stats[label][feature]["unique"]

        # Laplace smoothing (add-one smoothing)
        prob = (freq + 1) / (total + unique)
        return safe_log(prob)


    def class_score(self, label, userData):
        """Calculates the weighted log-probability score for a given class label."""
        score = self.log_prior(label)

        for feature in userData:

            # Weight (importance of feature)
            w = self.weights.get(feature, 1.0)

            # Numeric? (Check if 'mu' exists in the precomputed stats)
            if feature in self.stats[label] and "mu" in self.stats[label][feature]:
                logp = self.log_likelihood_numeric(feature, label, userData)

            # Categorical?
            else:
                logp = self.log_likelihood_categorical(feature, label, userData)

            # Naive Bayes assumption: features are conditionally independent
            score += w * logp

        return score

    def predict(self, userData):
        """
        Calculates and returns the probability and prediction for the test case (userData).
        """
        g = self.class_score("genuine", userData)
        f = self.class_score("fraud", userData)

        # LogSumExp trick for numerical stability (converts log scores back to probabilities)
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