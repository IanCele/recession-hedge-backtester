import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
import os
import json

# Config
TICKERS = ["LMT", "NOC", "GLD", "NVDA", "SPY"]
WEIGHTS = {"LMT": 0.2, "NOC": 0.2, "GLD": 0.3, "NVDA": 0.3}
CRISIS_PERIODS = {
    "2008 Recession": ("2007-12-01", "2009-06-01"),
    "2020 COVID Crash": ("2020-02-01", "2020-04-01"),
    "2022 Ukraine War": ("2022-02-01", "2022-12-01")
}

def fetch_data():
    """Download historical data and save as CSV for caching"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=10*365)).strftime("%Y-%m-%d")
    
    if not os.path.exists("data/historical_prices.csv"):
        data = yf.download(TICKERS, start=start_date, end=end_date)["Adj Close"]
        data.to_csv("data/historical_prices.csv")
    return pd.read_csv("data/historical_prices.csv", index_col=0, parse_dates=True)

def run_backtest(data):
    """Core backtest logic"""
    returns = data.pct_change()
    portfolio_returns = (returns * pd.Series(WEIGHTS)).sum(axis=1)
    comparison = pd.DataFrame({"Portfolio": portfolio_returns, "SPY": returns["SPY"]})
    
    # Calculate metrics
    cumulative = (1 + comparison).cumprod()
    sharpe = (comparison.mean() / comparison.std()) * (252**0.5)
    max_dd = (cumulative / cumulative.cummax() - 1).min() * 100
    
    return cumulative, sharpe, max_dd

def generate_visualizations(cumulative):
    """Create static and interactive plots"""
    # Matplotlib (static)
    plt.figure(figsize=(12,6))
    cumulative.plot()
    for name, (start, end) in CRISIS_PERIODS.items():
        plt.axvspan(start, end, color="red", alpha=0.1)
    plt.title("Recession Hedge Portfolio vs S&P 500")
    plt.savefig("output/performance_chart.png")
    
    # Plotly (interactive)
    fig = px.line(cumulative, title="Live Portfolio Tracker")
    fig.write_html("output/interactive_chart.html")

def save_metrics(sharpe, max_dd):
    """Save metrics for dashboard"""
    metrics = {
        "sharpe": round(sharpe["Portfolio"], 2),
        "spy_sharpe": round(sharpe["SPY"], 2),
        "drawdown": f"{round(max_dd['Portfolio'], 1)}%",
        "spy_drawdown": f"{round(max_dd['SPY'], 1)}%",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("output/metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    data = fetch_data()
    cumulative, sharpe, max_dd = run_backtest(data)
    generate_visualizations(cumulative)
    save_metrics(sharpe, max_dd)
    
    # Print latest metrics
    print(f"Latest Sharpe Ratios:\n{sharpe}\n")
    print(f"Maximum Drawdowns:\n{max_dd}\n")
    print("Visualizations saved to /output")
