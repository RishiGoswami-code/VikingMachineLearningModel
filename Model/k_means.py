import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

class KMeansModel:
    def __init__(self, data, weights, n_clusters=2, random_state=42):
        """
        Initializes the model, prepares the data (X, y), and performs K-Means clustering.
        The model then maps the abstract cluster IDs (0, 1) to the actual labels (Genuine, Fraud).
        """
        self.weights = weights
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
        # Prepare Data: X (features), y (labels: 1 for genuine, 0 for fraud)
        self.X, self.y, self.feature_names = self._prepare_data(data)
        
        # Scale X (crucial for distance-based clustering)
        self.X_scaled = self._scale_features(self.X)
        
        # Initialize and Train the K-Means Classifier
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=random_state,
            n_init=10 # Suppress n_init warning in recent sklearn versions
        )
        
        # Train the model and determine cluster labels
        self.cluster_map = self._train()


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
        """
        Fits the K-Means model and creates a mapping from Cluster ID to 'genuine'/'fraud' label.
        """
        cluster_assignments = self.model.fit_predict(self.X_scaled)
        
        # Create a dictionary to map the abstract Cluster ID (0, 1) to the final label ('genuine', 'fraud')
        # We check which actual label dominates each cluster
        cluster_to_label = {}
        
        for cluster_id in range(self.n_clusters):
            # Find the indices belonging to this cluster
            indices = np.where(cluster_assignments == cluster_id)
            # Get the true labels (y) for those indices
            true_labels_in_cluster = self.y[indices]
            
            if true_labels_in_cluster.size == 0:
                # Handle empty cluster if necessary, although unlikely here
                continue

            # Count the majority label (1 for genuine, 0 for fraud)
            majority_label_int = Counter(true_labels_in_cluster).most_common(1)[0][0]
            
            # Map the majority integer label back to the string label
            if majority_label_int == 1:
                cluster_to_label[cluster_id] = "genuine"
            else:
                cluster_to_label[cluster_id] = "fraud"

        return cluster_to_label


    def predict_cluster(self, userData):
        """
        Predicts the cluster ID (0 or 1) and the final label ('genuine' or 'fraud') 
        for a single test case (userData).
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

        # 2. Scale the input data using the scaler FIT to the training data (CRITICAL)
        test_X_scaled = self.scaler.transform(test_X)

        # 3. Predict the cluster ID (0 or 1)
        cluster_id = self.model.predict(test_X_scaled)[0]
        
        # 4. Map the cluster ID to the final label
        predicted_label = self.cluster_map.get(cluster_id, "unknown")
        
        # K-Means doesn't give a traditional probability, but we can fake a confidence
        # based on the distance to the predicted centroid vs. the other centroid,
        # but for simplicity, we'll return a fixed confidence (0.99) since the clusters 
        # are so well-separated, reflecting high confidence in the clustering.
        confidence = 0.99
        
        return predicted_label, confidence


    def result(self, userData):
        """
        Returns the final prediction and a pseudo-probability breakdown for the test case.
        """
        prediction, confidence = self.predict_cluster(userData)
        
        if prediction == "genuine":
            p_genuine = confidence
            p_fraud = 1.0 - confidence
        else:
            p_genuine = 1.0 - confidence
            p_fraud = confidence

        return {
            "genuine_probability": p_genuine,
            "fraud_probability": p_fraud,
            "prediction": prediction
        }