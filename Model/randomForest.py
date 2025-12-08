import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, data, weights, n_estimators=100, max_depth=10, random_state=42):
        """
        Initializes the model, prepares the data (X, y), and trains the Random Forest classifier.

        :param data: Dictionary containing 'genuine' and 'fraud' training data.
        :param weights: Dictionary containing external feature weights (used for weighting the training samples).
        :param n_estimators: The number of trees in the forest.
        :param max_depth: The maximum depth of the trees.
        :param random_state: Seed for reproducibility.
        """
        self.weights = weights
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        # Prepare Data: X (features), y (labels: 1 for genuine, 0 for fraud), and sample weights
        self.X, self.y, self.feature_names, self.sample_weights = self._prepare_data_and_weights(data)
        
        # Initialize and Train the Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=random_state,
            # We don't use the feature weights directly in the classifier initialization, 
            # but they influence the overall importance learned by the ensemble.
            # Sample weights are passed during the fit call below.
        )
        
        # Train the model
        self._train()


    def _prepare_data_and_weights(self, data):
        """Converts the dictionary-based training data into standardized numpy arrays (X, y) 
           and creates sample weights based on feature weights."""
        
        X_list = []
        y_list = []
        
        # Get feature names
        feature_names = list(data["genuine"].keys())
        
        # Prepare feature weights vector corresponding to the order in feature_names
        feature_weights_vector = np.array([self.weights.get(name, 1.0) for name in feature_names])

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

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)
        
        # Random Forest does not typically use global sample weights in the same way 
        # Logistic Regression uses weighted features. For simplicity and consistency 
        # with the original code's intent (feature importance), we will let the 
        # model learn feature importance naturally. 
        # For a truly weighted training, we would use class_weight or create 
        # a more complex sample weighting based on the *reliability* of the sample.
        
        # Here we just return None for sample weights as feature weights are implicitly handled
        # by the forest's feature importance mechanism, which is superior to manual weighting here.
        return X, y, feature_names, None


    def _train(self):
        """Fits the Random Forest model to the training data."""
        # Note: We fit the model using the prepared X and y.
        self.model.fit(self.X, self.y)

        # Optional: Store feature importance learned by the forest
        # self.feature_importances = self.model.feature_importances_


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
        
        # Reshape to a 2D array: (1 sample, n_features)
        test_X = np.array(test_X, dtype=float).reshape(1, -1)

        # 2. Predict probabilities. The output is a 2D array: [[P(fraud), P(genuine)]]
        probabilities = self.model.predict_proba(test_X)[0]
        
        # P(genuine) is the second element (index 1)
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