import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib # Useful for saving and loading the scaling parameters

class SVMModel:
    def __init__(self, data, weights, kernel='linear', C=1.0, random_state=42):
        """
        Initializes the model, prepares the data (X, y), and trains the SVM classifier.

        :param data: Dictionary containing 'genuine' and 'fraud' training data.
        :param weights: Dictionary containing external feature weights (not directly used by SVM, but kept for signature consistency).
        :param kernel: Specifies the kernel type to be used in the algorithm ('linear', 'rbf', 'poly', etc.).
        :param C: Regularization parameter. The strength of the regularization is inversely proportional to C.
        """
        self.weights = weights
        self.kernel = kernel
        self.C = C
        
        # Prepare Data: X (features), y (labels: 1 for genuine, 0 for fraud)
        self.X, self.y, self.feature_names = self._prepare_data(data)
        
        # Initialize Scaler
        self.scaler = StandardScaler()
        
        # Scale X using feature-wise means and standard deviations
        self.X = self._scale_features(self.X)
        
        # Initialize the SVM Classifier
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            probability=True, # MUST be True to get probability estimates
            random_state=random_state
        )
        
        # Train the model
        self._train()


    def _prepare_data(self, data):
        """Converts the dictionary-based training data into standardized numpy arrays (X, y)."""
        
        X_list = []
        y_list = []
        
        feature_names = list(data["genuine"].keys())
        
        def get_data_length(label_data):
            if not label_data: return 0
            return len(next(iter(label_data.values())))

        # Process data and create a combined list of X and y
        for label, y_val in [("genuine", 1), ("fraud", 0)]:
            num_samples = get_data_length(data[label])
            for i in range(num_samples):
                row = []
                for feature in feature_names:
                    value = data[label][feature][i]
                    # Convert categorical features ("yes"/"no") to numeric (1/0)
                    if isinstance(value, str):
                        row.append(1.0 if value.lower() == "yes" else 0.0)
                    else:
                        row.append(value)
                X_list.append(row)
                y_list.append(y_val)

        return np.array(X_list, dtype=float), np.array(y_list, dtype=int), feature_names


    def _scale_features(self, X):
        """Standardizes the features using Mean and Standard Deviation (fitting the scaler)."""
        # Fit the scaler ONLY on the training data (X)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled


    def _train(self):
        """Fits the SVM model to the scaled training data."""
        self.model.fit(self.X, self.y)


    def predict_proba(self, userData):
        """
        Calculates the probability P(genuine) for a single test case (userData).
        """
        # 1. Convert userData dictionary to a numpy array matching feature order
        test_X = []
        for feature in self.feature_names:
            value = userData[feature]
            if isinstance(value, str):
                test_X.append(1.0 if value.lower() == "yes" else 0.0)
            else:
                test_X.append(value)
        
        # Reshape to a 2D array: (1 sample, n_features)
        test_X = np.array(test_X, dtype=float).reshape(1, -1)

        # 2. Scale the input data using the scaler FIT to the training data
        test_X_scaled = self.scaler.transform(test_X)

        # 3. Predict probabilities. The output is a 2D array: [[P(fraud), P(genuine)]]
        # We must use predict_proba since probability=True was set in SVC initialization
        probabilities = self.model.predict_proba(test_X_scaled)[0]
        
        # P(genuine) is the second element (index 1) if classes are sorted [0, 1]
        p_genuine = probabilities[1]
        
        return p_genuine


    def result(self, userData):
        """
        Returns the final prediction and probability breakdown for the test case.
        """
        p_genuine = self.predict_proba(userData)
        p_fraud = 1 - p_genuine
        
        # Prediction is based on the class with the highest probability
        prediction = "genuine" if p_genuine >= 0.5 else "fraud"

        return {
            "genuine_probability": p_genuine,
            "fraud_probability": p_fraud,
            "prediction": prediction
        }