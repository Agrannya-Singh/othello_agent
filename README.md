# othello_backend

google collab note book for model training : https://colab.research.google.com/drive/1Limm2oFEha1HED5FOWjXWig_DiYBTMSe?usp=sharing
# Othello API 

A machine learning-powered Othello (Reversi) game API built with Flask and TensorFlow. This repository serves as the backend API and ML training engine for the [Othello-Dojo](https://github.com/Agrannya-Singh/Othello-Dojo) frontend application.

## 🎯 Overview

This API provides a self-learning Othello engine that uses neural networks to improve its gameplay over time. Unlike traditional rule-based AI, this implementation leverages machine learning to adapt and evolve its strategy through self-play and game analysis.

## 🚀 Features

- **Machine Learning Engine**: TensorFlow-powered neural network for board evaluation
- **Self-Training**: AI can improve through self-play sessions
- **RESTful API**: Clean HTTP endpoints for game interaction
- **Multiple Difficulty Levels**: Configurable AI difficulty settings
- **Real-time Training**: Update the model with game outcomes
- **Session Management**: Persistent game states across requests

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **ML Framework**: TensorFlow/Keras
- **Game Logic**: NumPy for efficient board operations
- **Session Management**: Flask sessions with secure cookies
- **Additional**: Pygame for game utilities

## 📋 Requirements
lask==2.3.2
numpy==1.24.3
tensorflow==2.15.0
flask-cors==4.0.0
gunicorn==20.1.0
pygame==2.6.1


<img width="816" height="738" alt="image" src="https://github.com/user-attachments/assets/e04d8aa9-92f1-4369-9922-2037de4d6dcc" />
<img width="815" height="518" alt="image" src="https://github.com/user-attachments/assets/4affd77f-6cbe-414b-a588-378bce89e2be" />

This API works in conjunction with Othello-Dojo https://github.com/Agrannya-Singh/Othello-Dojo, a Next.js frontend that provides the user interface for playing against the AI.


<img width="777" height="383" alt="image" src="https://github.com/user-attachments/assets/26a4ccf8-fdfe-417a-974b-fbe1eb3109a1" />


🤖 AI Behavior

The AI employs a hybrid approach:

    Traditional Minimax: For immediate tactical moves
    Neural Network: For long-term position evaluation
    Self-Learning: Adapts strategy based on game outcomes
    Heuristic Fallback: Uses corner control and mobility when ML is unavailable


