"""
Burnout Neural Network Model
Implements a simple feedforward neural network for binary classification
"""
import numpy as np
from pathlib import Path


class BurnoutNetwork:
    """
    Binary classification neural network for predicting burnout risk
    
    Architecture:
    - Input Layer: 3 neurons (sleep, work, relax hours)
    - Hidden Layer: 4 neurons with sigmoid activation
    - Output Layer: 1 neuron with sigmoid activation (burnout probability)
    """
    
    def __init__(self, input_size=3, hidden_size=4, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid for backpropagation"""
        return x * (1 - x)
    
    def feedforward(self, inputs):
        """
        Forward pass through the network
        
        Args:
            inputs: numpy array of shape (n_samples, 3)
        
        Returns:
            predictions: numpy array of shape (n_samples, 1)
        """
        self.hidden_output = self.sigmoid(np.dot(inputs, self.W1))
        self.output = self.sigmoid(np.dot(self.hidden_output, self.W2))
        return self.output
    
    def train(self, inputs, targets, learning_rate=0.5, epochs=20000, verbose=True):
        """
        Train the network using backpropagation
        
        Args:
            inputs: Training inputs (n_samples, 3)
            targets: Training targets (n_samples, 1)
            learning_rate: Learning rate for gradient descent
            epochs: Number of training iterations
            verbose: Whether to print training progress
        
        Returns:
            error_history: List of errors over epochs
        """
        error_history = []
        
        for epoch in range(epochs):
            # Feedforward
            self.feedforward(inputs)
            
            # Calculate error
            error = targets - self.output
            mean_squared_error = np.mean(error ** 2)
            error_history.append(mean_squared_error)
            
            # Backpropagation
            d_output = error * self.sigmoid_derivative(self.output)
            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)
            
            # Update weights
            self.W2 += self.hidden_output.T.dot(d_output) * learning_rate
            self.W1 += inputs.T.dot(d_hidden) * learning_rate
            
            # Print progress
            if verbose and (epoch + 1) % 2000 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Error: {mean_squared_error:.6f}")
        
        if verbose:
            print(f"Training complete! Final error: {error_history[-1]:.6f}")
        
        return error_history
    
    def predict(self, sleep_hours, work_hours, relax_hours):
        """
        Predict burnout risk for given lifestyle inputs
        
        Args:
            sleep_hours: Hours of sleep per night (0-12)
            work_hours: Hours of work/study per day (0-16)
            relax_hours: Hours of relaxation per day (0-8)
        
        Returns:
            Burnout probability as percentage (0-100)
        """
        # Normalize inputs
        normalized = np.array([[sleep_hours/24, work_hours/24, relax_hours/24]])
        
        # Feedforward
        prediction = self.feedforward(normalized)
        
        return prediction[0][0] * 100
    
    def save_weights(self, filepath):
        """Save model weights to file"""
        np.savez(filepath, W1=self.W1, W2=self.W2)
        print(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath):
        """Load model weights from file"""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.W2 = data['W2']
        print(f"Model weights loaded from {filepath}")


def create_training_data():
    """
    Generate training data for burnout prediction
    
    Returns:
        inputs: Normalized lifestyle inputs (n_samples, 3)
        targets: Binary labels (n_samples, 1) - 0=Healthy, 1=Burnout
    """
    # [Sleep, Work, Relax] hours per day
    raw_data = [
        # Healthy patterns (Target: 0)
        [8, 8, 4],      # Balanced lifestyle
        [9, 6, 5],      # Well-rested, light work
        [7, 7, 3],      # Decent balance
        [8, 9, 3],      # Productive but rested
        [7.5, 8, 4],    # Good balance
        
        # Burnout patterns (Target: 1)
        [4, 14, 0],     # Severe burnout - overworked
        [5, 13, 1],     # High burnout risk
        [4, 12, 0.5],   # Exhausted worker
        [3, 15, 0],     # Extreme burnout
        [5, 12, 0],     # Sleep deprived overworker
        
        # Warning cases (Target: 1)
        [6, 10, 1],     # Warning - needs more rest
        [5.5, 11, 1.5], # High risk
        [6, 11, 0.5],   # Burnout building
    ]
    
    # Normalize inputs (divide by 24 hours)
    inputs = np.array([[sleep/24, work/24, relax/24] for sleep, work, relax in raw_data])
    
    # Targets: 0 = Healthy, 1 = Burnout
    targets = np.array([[0], [0], [0], [0], [0],  # Healthy
                        [1], [1], [1], [1], [1],  # Burnout
                        [1], [1], [1]])           # Warning
    
    return inputs, targets


if __name__ == "__main__":
    # Train and save model
    print("Creating and training Burnout Neural Network...")
    print("=" * 60)
    
    # Create training data
    inputs, targets = create_training_data()
    print(f"Training samples: {len(inputs)}")
    
    # Create and train model
    model = BurnoutNetwork()
    error_history = model.train(inputs, targets, epochs=20000)
    
    # Save weights
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model.save_weights(models_dir / "burnout_weights.npz")
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Testing predictions:")
    print("=" * 60)
    test_cases = [
        (7, 9, 2, "Moderate worker"),
        (5, 12, 1, "Overworked person"),
        (8.5, 7, 4, "Well-balanced"),
    ]
    
    for sleep, work, relax, desc in test_cases:
        risk = model.predict(sleep, work, relax)
        print(f"{desc:20s} | Sleep: {sleep}h, Work: {work}h, Relax: {relax}h → Risk: {risk:.1f}%")
