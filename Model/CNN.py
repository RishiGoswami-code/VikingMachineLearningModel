import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Any

# Set a seed for reproducibility
tf.random.set_seed(42)

class CNNModel:
    def __init__(self, data: Dict[str, Any], weights: Dict[str, float], learning_rate=0.001, n_epochs=500):
        """
        Initializes and trains the 1D CNN model.
        """
        self.weights = weights
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.scaler = StandardScaler()

        # Prepare Data: X (features), y (labels: 1 for genuine, 0 for fraud)
        self.X, self.y, self.feature_names = self._prepare_data(data)
        
        # Scale X (crucial for CNN input)
        # Reshape X for 1D CNN input: (samples, timesteps, features=1)
        self.X_scaled = self._scale_features(self.X)
        self.X_scaled = self.X_scaled[..., np.newaxis] 
        
        # Determine input shape for the CNN
        self.input_shape = self.X_scaled.shape[1:]  # (num_features, 1)

        # Build and train the model
        self.model = self._build_model()
        self._train()


    def _prepare_data(self, data: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, List[str]]:
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
                        # Apply external weights to features during data preparation 
                        # to emphasize importance for the CNN's loss function
                        w = self.weights.get(feature, 1.0)
                        row.append(value * w)
                X_list.append(row)
                y_list.append(y_val)

        return np.array(X_list, dtype=float), np.array(y_list, dtype=int), feature_names


    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Standardizes the features using Mean and Standard Deviation."""
        # Fit the scaler ONLY on the training data (X)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled


    def _build_model(self) -> Sequential:
        """Defines the 1D CNN architecture."""
        model = Sequential([
            # 1. 1D Convolutional Layer: Learns local patterns/relationships between adjacent features
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=self.input_shape, padding='same'),
            
            # 2. Pooling Layer: Reduces dimensionality
            MaxPooling1D(pool_size=2),
            
            # 3. Dropout: Prevents overfitting by randomly setting a fraction of input units to 0
            Dropout(0.2),

            # 4. Flatten: Converts the 2D feature map output into a 1D vector for the Dense layer
            Flatten(),
            
            # 5. Fully Connected Layer: Performs final non-linear classification
            Dense(units=16, activation='relu'),
            
            # 6. Output Layer: Single neuron with Sigmoid activation for binary classification (0 to 1)
            Dense(units=1, activation='sigmoid')
        ])
        
        # Compile the model using the Adam optimizer and Binary Crossentropy loss (Log Loss)
        model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model


    def _train(self):
        """Fits the CNN model to the training data."""
        print("\nStarting CNN Training...")
        self.model.fit(
            self.X_scaled, 
            self.y, 
            epochs=self.n_epochs, 
            batch_size=8, 
            verbose=0 # Suppress training output
        )
        print("CNN Training Complete.")


    def predict_proba(self, userData: Dict[str, Any]) -> float:
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
                # Apply external weights during prediction as well
                w = self.weights.get(feature, 1.0)
                test_X.append(value * w)
        
        # Reshape to 2D array: (1 sample, n_features)
        test_X = np.array(test_X, dtype=float).reshape(1, -1)

        # 2. Scale the input data using the scaler FIT to the training data
        test_X_scaled = self.scaler.transform(test_X)
        
        # 3. Reshape for 1D CNN input: (1 sample, num_features, 1)
        test_X_scaled = test_X_scaled[..., np.newaxis]

        # 4. Predict probability
        # The output is an array [[p]]
        probabilities = self.model.predict(test_X_scaled, verbose=0)[0]
        
        # P(genuine) is the single output value
        p_genuine = probabilities[0]
        
        return float(p_genuine)


    def result(self, userData: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the final prediction and probability breakdown for the test case.
        """
        p_genuine = self.predict_proba(userData)
        p_fraud = 1.0 - p_genuine
        
        # Threshold at 0.5
        prediction = "genuine" if p_genuine >= 0.5 else "fraud"

        return {
            "genuine_probability": p_genuine,
            "fraud_probability": p_fraud,
            "prediction": prediction
        }