import os
from datetime import datetime, timedelta

# Global variables
parent_path = os.getcwd()
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
graph_dir = f"{parent_path}/graphs/{timestamp}/backtesting"

# Create needed directories
os.makedirs(graph_dir, exist_ok=True)

# Helper function to format dates like '6 dec 2024'
def format_start_date(dt):
    return dt.strftime("%-d %b %Y").lower()  # use '%#d' on Windows instead of '%-d'

now = datetime.now()

# Dynamic date_combinations variable
date_combinations = [
    ('5m',  format_start_date(now - timedelta(minutes=5 * 28))),
    ('15m', format_start_date(now - timedelta(minutes=15 * 144))),  # ~7 hours ago
    ('1d',  format_start_date(now - timedelta(days=365)))
    ('2d',  format_start_date(now - timedelta(days=430)))
]
