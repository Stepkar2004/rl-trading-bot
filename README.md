# Reinforcement Learning Trading Bot

This project implements a cryptocurrency trading bot using a Deep Q-Network (DQN), a reinforcement learning algorithm. The bot is trained on historical price data and learns to make trading decisions to maximize its profit.

## Features

*   Fetches historical cryptocurrency data from the Bitfinex API.
*   Calculates various technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, OBV) to be used as features for the model.
*   Splits the data into training and testing sets.
*   Trains a DQN model using the `stable-baselines3` library.
*   Evaluates the trained model's performance on the test set.
*   Visualizes the bot's performance with interactive plots.

## Technologies Used

*   Python
*   Jupyter Notebook
*   Pandas
*   NumPy
*   Scikit-learn
*   TensorFlow
*   Plotly
*   Stable-Baselines3
*   Gym

## Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/rl-trading-bot.git
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook get_market_data.ipynb
    ```

## Results

The trading bot's performance is evaluated on the test set. The following plots show the portfolio value, rewards, and actions taken by the bot over time.

*(You can add the generated plots here)*
