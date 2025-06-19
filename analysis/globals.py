import os
from datetime import datetime

# Global variables
parent_path = os.getcwd()
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
graph_dir = f"{parent_path}/graphs/{timestamp}/backtesting"

# Create needed directories
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(f"{parent_path}/graphs/rsi", exist_ok=True)
