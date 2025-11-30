# Bayesian Optimization for ResNet-18 on Fashion-MNIST

This project implements **Bayesian Optimization (BO)** to tune the learning rate of a custom ResNet-18 neural network on the Fashion-MNIST dataset.

It uses a **Gaussian Process** as a surrogate model and **Weighted Expected Improvement (WEI)** as the acquisition function to efficiently search for the optimal hyperparameter.

## Features

*   **Model:** Custom implementation of ResNet-18.
*   **Dataset:** Fashion-MNIST (automatically downloaded).
*   **Optimization:** Bayesian Optimization with a Sobol sequence initial design.
*   **Visualization:** Real-time plotting of the Gaussian Process posterior and Acquisition Function.

## Installation

1.  **Clone the repository** (or navigate to your project folder).
2.  **Install dependencies** using the requirements file:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Standard Execution
To run the optimization with the default configuration (Budget: 10, Initial Steps: 3, Bounds: $10^{-4}$ to $10^{-1}$), simply run:

```bash
python main.py
```

## Sample Output
A full log of a successful optimization run is available in [output.txt](output.txt). 
This file demonstrates the convergence of the model over the 10-iteration budget.