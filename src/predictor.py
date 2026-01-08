"""
Prediction utilities for the Burnout AI
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_model_weights(weights_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trained model weights from file
    
    Args:
        weights_path: Path to .npz file containing weights (defaults to models/burnout_weights.npz)
    
    Returns:
        Tuple of (W1, W2) weight matrices
    """
    if weights_path is None:
        weights_path = Path(__file__).parent.parent / "models" / "burnout_weights.npz"
    
    data = np.load(weights_path)
    return data['W1'], data['W2']


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))


def predict_burnout_risk(
    sleep_hours: float,
    work_hours: float,
    relax_hours: float,
    W1: np.ndarray = None,
    W2: np.ndarray = None
) -> Dict[str, any]:
    """
    Predict burnout risk based on lifestyle inputs
    
    Args:
        sleep_hours: Hours of sleep per night (0-12)
        work_hours: Hours of work/study per day (0-16)
        relax_hours: Hours of relaxation per day (0-8)
        W1: Input-to-hidden weights (optional, will load from file if None)
        W2: Hidden-to-output weights (optional, will load from file if None)
    
    Returns:
        Dictionary with:
            - probability: Burnout risk percentage (0-100)
            - risk_level: String classification (healthy/caution/warning/danger)
            - status_emoji: Visual indicator
            - recommendation: Health advice
    """
    # Load weights if not provided
    if W1 is None or W2 is None:
        W1, W2 = load_model_weights()
    
    # Normalize inputs to 0-1 range
    normalized_input = np.array([[sleep_hours/24, work_hours/24, relax_hours/24]])
    
    # Feedforward pass
    hidden_output = sigmoid(np.dot(normalized_input, W1))
    output = sigmoid(np.dot(hidden_output, W2))
    
    probability = float(output[0][0] * 100)
    
    # Determine risk level and recommendation
    if probability < 30:
        risk_level = "healthy"
        status_emoji = "🟢"
        recommendation = "Great job! Your lifestyle is well-balanced. Keep it up!"
    elif probability < 50:
        risk_level = "caution"
        status_emoji = "🟡"
        recommendation = "Warning signs detected. Consider increasing sleep or reducing work hours."
    elif probability < 70:
        risk_level = "warning"
        status_emoji = "🟠"
        recommendation = "You're at risk of burnout. Prioritize rest and relaxation immediately."
    else:
        risk_level = "danger"
        status_emoji = "🔴"
        recommendation = "Critical! You need to make major lifestyle changes. Seek support if needed."
    
    return {
        "probability": round(probability, 2),
        "risk_level": risk_level,
        "status_emoji": status_emoji,
        "recommendation": recommendation,
        "inputs": {
            "sleep_hours": sleep_hours,
            "work_hours": work_hours,
            "relax_hours": relax_hours
        }
    }


def get_health_status(probability: float) -> str:
    """
    Convert probability to human-readable status
    
    Args:
        probability: Burnout probability (0-100)
    
    Returns:
        Status string
    """
    if probability < 30:
        return "Healthy - Low Burnout Risk"
    elif probability < 50:
        return "Caution - Moderate Risk"
    elif probability < 70:
        return "Warning - High Risk"
    else:
        return "Danger - Severe Burnout Risk"


if __name__ == "__main__":
    # Test the predictor
    print("Testing Burnout Predictor")
    print("=" * 60)
    
    test_cases = [
        (8, 8, 4, "Balanced person"),
        (5, 12, 1, "Overworked person"),
        (4, 14, 0, "Extreme burnout case"),
        (9, 6, 5, "Well-rested person"),
    ]
    
    for sleep, work, relax, description in test_cases:
        result = predict_burnout_risk(sleep, work, relax)
        print(f"\n{description}:")
        print(f"  Sleep: {sleep}h | Work: {work}h | Relax: {relax}h")
        print(f"  {result['status_emoji']} Risk: {result['probability']:.1f}% ({result['risk_level'].upper()})")
        print(f"  Advice: {result['recommendation']}")
