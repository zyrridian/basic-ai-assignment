# 🧠 Burnout & Lifestyle Balance AI

A comprehensive binary classification AI system that predicts burnout risk based on lifestyle habits using a manually implemented neural network with backpropagation.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Project Overview

This project implements a **binary classification neural network** from scratch (no TensorFlow/PyTorch) to predict whether a person is at risk of burnout based on three simple lifestyle inputs:

- **Sleep Hours**: Hours of sleep per night
- **Work Hours**: Hours spent working or studying per day
- **Relax Hours**: Hours spent on pure relaxation (no screens, no work)

The AI outputs a **burnout probability** from 0-100%, classified into four risk levels:

- 🟢 **Healthy** (< 30%): Well-balanced lifestyle
- 🟡 **Caution** (30-50%): Minor warning signs
- 🟠 **Warning** (50-70%): High burnout risk
- 🔴 **Danger** (> 70%): Severe burnout risk

### Why This Project?

- **Educational**: Demonstrates manual implementation of backpropagation and gradient descent
- **Practical**: Addresses real-world mental health awareness
- **Interactive**: Includes web interface for easy testing
- **Complete**: Jupyter notebook for learning + production API for deployment

## 🏗️ Architecture

### Neural Network Design

```
Input Layer (3 neurons)
    ↓
Hidden Layer (4 neurons, Sigmoid activation)
    ↓
Output Layer (1 neuron, Sigmoid activation)
```

**Mathematical Foundation**:

- **Activation Function**: Sigmoid σ(x) = 1 / (1 + e^(-x))
- **Training Algorithm**: Backpropagation with Gradient Descent
- **Learning Rate**: 0.5
- **Training Epochs**: 20,000
- **Loss Function**: Mean Squared Error (MSE)

### Project Structure

```
basic-ai-assignment/
│
├── notebooks/
│   └── burnout_model.ipynb      # Interactive learning notebook
│
├── src/
│   ├── __init__.py
│   ├── model.py                 # BurnoutNetwork class
│   ├── predictor.py             # Prediction utilities
│   └── api.py                   # FastAPI backend
│
├── models/
│   └── burnout_weights.npz      # Trained model weights (generated)
│
├── web/
│   └── index.html               # Web interface
│
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone or navigate to the project directory
cd d:\Projects\basic-ai-assignment

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model (Jupyter Notebook)

```powershell
# Start Jupyter Notebook
jupyter notebook notebooks/burnout_model.ipynb
```

Run all cells in the notebook to:

- Understand the neural network theory
- Train the model with backpropagation
- Visualize training progress
- Test predictions interactively
- Save trained weights to `models/burnout_weights.npz`

### 3. Alternative: Train via Python Script

```powershell
# Train and save model directly
python src/model.py
```

### 4. Start the Web Application

```powershell
# Start FastAPI server
cd src
python api.py
```

The server will start at:

- **Web Interface**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 5. Use the Web Interface

1. Open http://localhost:8000 in your browser
2. Adjust the sliders for sleep, work, and relax hours
3. Click "Check Burnout Risk"
4. See your personalized risk assessment with recommendations!

## 📊 Training Data

The model is trained on 13 carefully designed lifestyle patterns:

**Healthy Patterns** (Target: 0)

- 8h sleep, 8h work, 4h relax - Balanced lifestyle
- 9h sleep, 6h work, 5h relax - Well-rested, light work
- 7h sleep, 7h work, 3h relax - Decent balance

**Burnout Patterns** (Target: 1)

- 4h sleep, 14h work, 0h relax - Severe burnout
- 5h sleep, 13h work, 1h relax - High burnout risk
- 3h sleep, 15h work, 0h relax - Extreme burnout

## 🔬 How It Works

### Forward Pass (Feedforward)

1. **Input Normalization**: Divide hours by 24 to get 0-1 range
2. **Hidden Layer**: `h = sigmoid(X · W1)`
3. **Output Layer**: `y = sigmoid(h · W2)`

### Backward Pass (Backpropagation)

1. **Calculate Error**: `error = target - predicted`
2. **Output Gradient**: `δ_out = error · sigmoid'(output)`
3. **Hidden Gradient**: `δ_hidden = (δ_out · W2^T) · sigmoid'(hidden)`
4. **Update Weights**:
   - `W2 += hidden^T · δ_out · learning_rate`
   - `W1 += input^T · δ_hidden · learning_rate`

## 🌐 API Usage

### Predict Endpoint

**Request**:

```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "sleep_hours": 6,
  "work_hours": 10,
  "relax_hours": 2
}
```

**Response**:

```json
{
  "success": true,
  "probability": 65.4,
  "risk_level": "warning",
  "status_emoji": "🟠",
  "recommendation": "You're at risk of burnout. Prioritize rest and relaxation immediately.",
  "inputs": {
    "sleep_hours": 6,
    "work_hours": 10,
    "relax_hours": 2
  }
}
```

### Health Check

```bash
GET http://localhost:8000/health
```

## 🧪 Testing

### Test the Predictor Module

```powershell
cd src
python predictor.py
```

### Test the Neural Network

```powershell
python model.py
```

### Interactive Testing

Use the Jupyter notebook for interactive testing with visualizations.

## 📚 Educational Value

This project demonstrates:

- ✅ Manual implementation of backpropagation (no high-level ML frameworks)
- ✅ Sigmoid activation function and its derivative
- ✅ Gradient descent optimization
- ✅ Binary classification with neural networks
- ✅ Data normalization techniques
- ✅ RESTful API design with FastAPI
- ✅ Full-stack integration (ML + Backend + Frontend)

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Matrix operations and numerical computing
- **Matplotlib**: Training visualization
- **Jupyter Notebook**: Interactive development and learning
- **FastAPI**: Modern, fast web framework
- **Uvicorn**: ASGI server
- **HTML/CSS/JavaScript**: Web interface

## 🎯 Use Cases

1. **Learning**: Understand neural networks from first principles
2. **Health Awareness**: Personal burnout risk assessment
3. **Portfolio**: Demonstrate AI/ML skills to employers
4. **Research**: Base for expanding to multi-class classification
5. **Wellness Apps**: Integration into health/productivity tools

## 🔄 Extending the Project

Ideas for enhancement:

- Add more features (exercise, diet, social activities)
- Implement data collection from users
- Add time-series analysis for tracking progress
- Create mobile app version
- Deploy to cloud (Heroku, AWS, Azure)
- Add user authentication and history tracking
- Implement A/B testing for different architectures

## 📝 License

This project is open-source and available under the MIT License.

## 👤 Author

Created as a comprehensive AI assignment demonstrating binary classification with manual backpropagation.

## 🤝 Contributing

Feel free to fork, modify, and submit pull requests!

## 📧 Support

For questions or issues, please open an issue in the repository.

---

**Remember**: This AI is for educational and awareness purposes. For serious mental health concerns, please consult a professional.

🧠 **Happy Learning & Stay Balanced!** 🌿
