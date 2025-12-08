import math
import numpy as np


def mean(lst):
    return sum(lst) / len(lst)

def variance(lst):
    u = mean(lst)
    return sum((x - u)**2 for x in lst) / len(lst)

def safe_log(x):
    # Used for categorical likelihoods in other models, kept for completeness/future expansion
    return math.log(max(x, 1e-12))

class LogisticRegressionModel:
    def __init__(self, data, weights, learning_rate=0.01, n_iterations=1000):
        """
        Initializes the model, prepares the data (X, y), and performs training.
        """
        self.weights = weights
        self.lr = learning_rate
        self.n_iterations = n_iterations
        
        # Prepare Data: X (features), y (labels: 1 for genuine, 0 for fraud)
        self.X, self.y, self.feature_names = self._prepare_data(data)
        
        # Initialize parameters (weights and bias)
        self.n_features = self.X.shape[1]
        # Initialize weights randomly, bias to 0
        self.W = np.zeros(self.n_features)
        self.b = 0

        # Scale X using feature-wise means and standard deviations
        self.X, self.mu, self.sigma = self._scale_features(self.X)
        
        # Train the model using Gradient Descent
        self._train()


    def _prepare_data(self, data):
        """Converts the dictionary-based training data into standardized numpy arrays (X, y)."""
        
        X_list = []
        y_list = []
        
        # Get feature names from the first label's data keys
        feature_names = list(data["genuine"].keys())
        
        # Function to ensure all feature lists for a given label have the same length
        def get_data_length(label_data):
            if not label_data:
                return 0
            # Assume all feature lists have the same length
            return len(next(iter(label_data.values())))

        # Process 'genuine' data (Label y=1)
        num_genuine = get_data_length(data["genuine"])
        for i in range(num_genuine):
            row = []
            for feature in feature_names:
                value = data["genuine"][feature][i]
                # Convert categorical features ("yes"/"no") to numeric (1/0)
                if isinstance(value, str):
                    row.append(1.0 if value.lower() == "yes" else 0.0)
                else:
                    row.append(value)
            X_list.append(row)
            y_list.append(1) # 1 for genuine

        # Process 'fraud' data (Label y=0)
        num_fraud = get_data_length(data["fraud"])
        for i in range(num_fraud):
            row = []
            for feature in feature_names:
                value = data["fraud"][feature][i]
                if isinstance(value, str):
                    row.append(1.0 if value.lower() == "yes" else 0.0)
                else:
                    row.append(value)
            X_list.append(row)
            y_list.append(0) # 0 for fraud
            
        return np.array(X_list, dtype=float), np.array(y_list, dtype=int), feature_names


    def _scale_features(self, X):
        """Standardizes the features using Mean and Standard Deviation."""
        mu = np.mean(X, axis=0)
        # Avoid division by zero by setting sigma to 1 where standard deviation is 0
        sigma = np.std(X, axis=0)
        sigma[sigma == 0] = 1.0 
        X_scaled = (X - mu) / sigma
        return X_scaled, mu, sigma


    def _sigmoid(self, z):
        """The logistic function."""
        # [Image of the Sigmoid function curve]
        return 1 / (1 + np.exp(-z))


    def _train(self):
        """Performs Gradient Descent to find optimal weights (W) and bias (b)."""
        X = self.X
        y = self.y
        m = X.shape[0] # Number of samples
        
        # Apply weights to features if provided, otherwise default to 1.0
        feature_weights = np.array([self.weights.get(name, 1.0) for name in self.feature_names])
        # Reshape for element-wise multiplication with X
        W_prime = feature_weights 
        
        for _ in range(self.n_iterations):
            # 1. Linear combination (z)
            z = np.dot(X * W_prime, self.W) + self.b
            
            # 2. Prediction (h_theta)
            h = self._sigmoid(z)
            
            # 3. Calculate Gradient (derivative of the cost function)
            # Apply feature weights to the error term for weighted gradient update
            error = (h - y)
            
            # Weighted X: X scaled by feature importance (W_prime)
            weighted_X = X * W_prime
            
            # Gradient for weights (W)
            dW = (1/m) * np.dot(weighted_X.T, error)
            
            # Gradient for bias (b)
            db = (1/m) * np.sum(error)
            
            # 4. Update parameters
            self.W = self.W - self.lr * dW
            self.b = self.b - self.lr * db


    def predict_proba(self, userData):
        """
        Calculates the probability P(genuine) for a single test case (userData).
        """
        # 1. Convert userData dictionary to a numpy array matching feature order
        test_X = []
        for feature in self.feature_names:
            value = userData[feature]
            # Convert categorical features ("yes"/"no") to numeric (1/0)
            if isinstance(value, str):
                test_X.append(1.0 if value.lower() == "yes" else 0.0)
            else:
                test_X.append(value)
        
        test_X = np.array(test_X, dtype=float)

        # 2. Scale the input data using the mean/sigma from training
        # Avoid division by zero if a feature has zero standard deviation
        scaled_X = (test_X - self.mu) / self.sigma
        
        # 3. Apply feature weights to the input vector
        feature_weights = np.array([self.weights.get(name, 1.0) for name in self.feature_names])
        weighted_scaled_X = scaled_X * feature_weights

        # 4. Calculate Linear combination (z)
        z = np.dot(weighted_scaled_X, self.W) + self.b
        
        # 5. Get probability (P(y=1) = P(genuine))
        p_genuine = self._sigmoid(z)
        
        return p_genuine


    def result(self, userData):
        """
        Returns the final prediction and probability breakdown for the test case.
        """
        p_genuine = self.predict_proba(userData)
        p_fraud = 1 - p_genuine
        
        prediction = "genuine" if p_genuine >= 0.5 else "fraud"

        return {
            "genuine_probability": p_genuine,
            "fraud_probability": p_fraud,
            "prediction": prediction
        }