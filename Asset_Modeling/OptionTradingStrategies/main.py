import numpy as np
from statistics import mean
from numpy import std
from tqdm import tqdm

# Import your custom classes
from Asset_Modeling.Asset_class import asset_BS
from Options.Options_class import Option_eu
from Options.Book_class import Book
from Graphics.Graphics import plot_multiple_lists, plot_2d

# Constants
N_MC = 1000
STEP_SIZE = 30 / 365

# Hedging Strategies
def no_hedging(price_path, book, step):
    book.asset.St = price_path[0]
    book.set_t(0.0)

    initial_price = book.option_price_close_formulae()
    prices = [initial_price]

    for price in price_path[1:]:
        book.asset.St = price
        book.update_t(step / len(price_path))
        prices.append(book.option_price_close_formulae())

    pnl = np.array(prices) - initial_price
    return pnl


def single_hedge(price_path, book, step):
    book.asset.St = price_path[0]
    book.set_t(0.0)
    if abs(book.Delta_DF()) > 1.0:
        book.asset.quantity -= int(book.Delta_DF())

    initial_price = np.abs(book.option_price_close_formulae())
    prices = [initial_price]

    for price in price_path[1:]:
        book.asset.St = price
        book.update_t(step / len(price_path))
        prices.append(np.abs(book.option_price_close_formulae()))

    pnl = np.array(prices) - initial_price
    return pnl


def continuous_hedge(price_path, book, step, delta_limit=1.0):
    book.asset.St = price_path[0]
    book.set_t(0.0)

    if abs(book.Delta_DF()) > 1.0:
        book.asset.quantity -= int(book.Delta_DF())

    price_len = len(price_path)
    purchase = np.array([np.abs(book.option_price_close_formulae())] * price_len)
    prices = [purchase[0]]

    for i in range(1, price_len):
        book.asset.St = price_path[i]
        book.update_t(step / price_len)

        if abs(book.Delta_DF()) > delta_limit:
            delta_qty = int(book.Delta_DF())
            purchase[i:] += delta_qty * book.asset.St
            book.asset.quantity -= delta_qty

        prices.append(np.abs(book.option_price_close_formulae()))

    pnl = np.array(prices) - purchase
    return pnl

def compute_results(strategy_names, *pnl_paths):
    for name, pnl_path in zip(strategy_names, pnl_paths):
        final_pnl = [path[-1] for path in pnl_path]
        pnl_array = np.array(final_pnl)

        avg_pnl = np.mean(pnl_array)
        std_pnl = np.std(pnl_array)
        min_pnl = np.min(pnl_array)
        max_pnl = np.max(pnl_array)
        sharpe_ratio = avg_pnl / std_pnl if std_pnl != 0 else np.nan
        pct_profitable = np.mean(pnl_array > 0) * 100

        print(f"{'='*70}")
        print(f"Strategy: {name}")
        print(f"Average PNL        : {avg_pnl:.4f}")
        print(f"PNL Std Dev        : {std_pnl:.4f}")
        print(f"Min PNL            : {min_pnl:.4f}")
        print(f"Max PNL            : {max_pnl:.4f}")
        print(f"Sharpe Ratio       : {sharpe_ratio:.4f}")
        print(f"% Profitable Trades: {pct_profitable:.2f}%")
        print(f"{'='*70}\n")


# Main Execution
if __name__ == '__main__':
    strat1_pnl, strat2_pnl, strat3_pnl, strat4_pnl = [], [], [], []
    stock_price_paths = []
    for _ in tqdm(range(N_MC), desc="Simulating scenarios"):
        # Asset and Option setup
        stock = asset_BS(100, 0, mu=0.1, sigma=0.2)
        call_option = Option_eu(100, 'Call EU', stock, 100, 0.5, 0.02, sigma=0.2)
        book = Book([call_option])

        # Simulate asset price
        stock.simu_asset(STEP_SIZE)

        # Run strategies
        stock_price_paths.append(stock.history)
        strat1_pnl.append(no_hedging(stock.history, book, STEP_SIZE))
        strat2_pnl.append(single_hedge(stock.history, book, STEP_SIZE))
        strat3_pnl.append(continuous_hedge(stock.history, book, STEP_SIZE))
        strat4_pnl.append(continuous_hedge(stock.history, book, STEP_SIZE, delta_limit=3.0))

    plot_multiple_lists(stock_price_paths)
    plot_multiple_lists(strat1_pnl, strat2_pnl, strat3_pnl, strat4_pnl)

    # Compute results
    strategy_labels = ["No Hedging", "Single Hedge", "Continuous Hedge", "Continuous Hedge delta 3"]
    compute_results(strategy_labels, strat1_pnl, strat2_pnl, strat3_pnl, strat4_pnl)