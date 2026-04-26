import numpy as np
from dataclasses import dataclass

FAIR_VALUE = 920
MIN_RESERVE = 670
MAX_RESERVE = 920
STEP = 5

RESERVES = np.arange(MIN_RESERVE, MAX_RESERVE + STEP, STEP)


@dataclass
class PlayerStrategy:
    b1: float
    b2: float


def sample_other_second_bids(n_players: int, distribution: str = "normal"):
    """
    Simulates the other players' second bids.
    You can change this distribution based on what you think people will submit.
    """

    if distribution == "normal":
        # Example: most people bid around 850, with some spread
        bids = np.random.normal(loc=850, scale=25, size=n_players)

    elif distribution == "uniform":
        bids = np.random.uniform(670, 920, size=n_players)

    elif distribution == "mixed":
        # Some conservative, some aggressive, some random
        n1 = int(0.1 * n_players)
        n2 = int(0.4 * n_players)
        n3 = int(0.4*n_players)
        n4 = n_players - n1 - n2 - n3

        bids = np.concatenate([
            np.random.normal(856, 3, n1), # no brain average
            np.random.normal(836 , 6 , n2), # claude response
            np.random.normal(845, 2, n3,), # first iteration
            np.random.uniform(750, 920, n4), #
        ])

    else:
        raise ValueError("Unknown distribution")

    return np.clip(bids, MIN_RESERVE, MAX_RESERVE)


def simulate_one_trial(
    my_strategy: PlayerStrategy,
    n_counterparties: int,
    n_other_players: int,
    distribution: str = "normal",
    random_counterparty_count: bool = True,
):
    b1 = my_strategy.b1
    b2 = my_strategy.b2

    other_b2s = sample_other_second_bids(n_other_players, distribution)
    avg_b2 = np.mean(np.append(other_b2s, b2))

    if random_counterparty_count:
        actual_n = np.random.randint(1, n_counterparties + 1)
    else:
        actual_n = n_counterparties

    reserves = np.random.choice(RESERVES, size=actual_n, replace=True)

    pnl = 0
    trades_b1 = 0
    trades_b2 = 0

    for reserve in reserves:
        # First bid happens first
        if b1 > reserve:
            pnl += FAIR_VALUE - b1
            trades_b1 += 1
            continue

        # Second bid only applies if first bid did not trade
        if b2 > reserve:
            raw_profit = FAIR_VALUE - b2

            if b2 > avg_b2:
                penalty = 1.0
            else:
                penalty = ((FAIR_VALUE - avg_b2) / (FAIR_VALUE - b2)) ** 3

            pnl += raw_profit * penalty
            trades_b2 += 1

    return pnl, trades_b1, trades_b2, avg_b2


def simulate_one_trial_fast(
    my_strategy: PlayerStrategy,
    n_counterparties: int,
    n_other_players: int,
    distribution: str = "normal",
    random_counterparty_count: bool = True,
):
    b1 = my_strategy.b1
    b2 = my_strategy.b2

    other_b2s = sample_other_second_bids(n_other_players, distribution)
    avg_b2 = np.mean(np.append(other_b2s, b2))

    if random_counterparty_count:
        actual_n = np.random.randint(1, n_counterparties + 1)
    else:
        actual_n = n_counterparties

    reserves = np.random.choice(RESERVES, size=actual_n, replace=True)

    # First bid trades
    b1_mask = b1 > reserves
    trades_b1 = np.sum(b1_mask)

    # Second bid only sees counterparties not taken by b1
    remaining = ~b1_mask
    b2_mask = remaining & (b2 > reserves)
    trades_b2 = np.sum(b2_mask)

    pnl_b1 = trades_b1 * (FAIR_VALUE - b1)

    raw_profit = FAIR_VALUE - b2

    if b2 > avg_b2:
        penalty = 1.0
    else:
        penalty = ((FAIR_VALUE - avg_b2) / (FAIR_VALUE - b2)) ** 3

    pnl_b2 = trades_b2 * raw_profit * penalty

    pnl = pnl_b1 + pnl_b2

    return pnl, trades_b1, trades_b2, avg_b2

def simulate_strategy(
    b1,
    b2,
    trials=10_000,
    n_counterparties=100,
    n_other_players=1000,
    distribution="normal",
):
    strategy = PlayerStrategy(b1=b1, b2=b2)

    pnls = []
    b1_trades = []
    b2_trades = []
    avg_b2s = []

    for _ in range(trials):
        pnl, t1, t2, avg_b2 = simulate_one_trial_fast(
            strategy,
            n_counterparties=n_counterparties,
            n_other_players=n_other_players,
            distribution=distribution,
        )

        pnls.append(pnl)
        b1_trades.append(t1)
        b2_trades.append(t2)
        avg_b2s.append(avg_b2)

    return {
        "b1": b1,
        "b2": b2,
        "mean_pnl": np.mean(pnls),
        "std_pnl": np.std(pnls),
        "mean_b1_trades": np.mean(b1_trades),
        "mean_b2_trades": np.mean(b2_trades),
        "mean_avg_b2": np.mean(avg_b2s),
    }


def random_search(
    samples=1000,
    trials=200,
    distribution="mixed",
    min_bid=670,
    max_bid=920,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    results = []

    for _ in range(samples):
        b1 = np.random.randint(min_bid, max_bid + 1)
        b2 = np.random.randint(b1, max_bid + 1)  # enforce b2 >= b1

        result = simulate_strategy(
            b1=b1,
            b2=b2,
            trials=trials,
            distribution=distribution,
        )

        results.append(result)

    return sorted(results, key=lambda x: x["mean_pnl"], reverse=True)

def smart_search(
    random_samples=1500,
    random_trials=200,
    top_k=20,
    local_radius=10,
    local_trials=1000,
    final_trials=10000,
    distribution="mixed",
):
    print("Stage 1: Random search...")

    random_results = random_search(
        samples=random_samples,
        trials=random_trials,
        distribution=distribution,
    )

    print("\nBest random results:")
    for r in random_results[:10]:
        print(
            f"b1={r['b1']}, b2={r['b2']}, "
            f"mean_pnl={r['mean_pnl']:.2f}, "
            f"b1_trades={r['mean_b1_trades']:.2f}, "
            f"b2_trades={r['mean_b2_trades']:.2f}, "
            f"avg_b2={r['mean_avg_b2']:.2f}"
        )

    print("\nStage 2: Local search around best candidates...")

    local_results = local_search_around(
        candidates=random_results[:top_k],
        radius=local_radius,
        trials=local_trials,
        distribution=distribution,
    )

    print("\nBest local results:")
    for r in local_results[:10]:
        print(
            f"b1={r['b1']}, b2={r['b2']}, "
            f"mean_pnl={r['mean_pnl']:.2f}, "
            f"b1_trades={r['mean_b1_trades']:.2f}, "
            f"b2_trades={r['mean_b2_trades']:.2f}, "
            f"avg_b2={r['mean_avg_b2']:.2f}"
        )

    print("\nStage 3: Final high-trial evaluation...")

    finalists = local_results[:20]
    final_results = []

    for candidate in finalists:
        result = simulate_strategy(
            b1=candidate["b1"],
            b2=candidate["b2"],
            trials=final_trials,
            distribution=distribution,
        )
        final_results.append(result)

    final_results = sorted(final_results, key=lambda x: x["mean_pnl"], reverse=True)

    print("\nFinal best results:")
    for r in final_results[:10]:
        print(
            f"b1={r['b1']}, b2={r['b2']}, "
            f"mean_pnl={r['mean_pnl']:.2f}, "
            f"std_pnl={r['std_pnl']:.2f}, "
            f"b1_trades={r['mean_b1_trades']:.2f}, "
            f"b2_trades={r['mean_b2_trades']:.2f}, "
            f"avg_b2={r['mean_avg_b2']:.2f}"
        )

    return final_results
'''
def grid_search(
    b1_range=range(670, 921),
    b2_range=range(670, 921),
    trials=100,
    distribution="normal",
):
    results = []

    for b1 in b1_range:
        for b2 in b2_range:
            # Usually you want b2 >= b1 because b1 happens first.
            # If b1 is too high, it steals trades that b2 could have won more cheaply.
            if b2 < b1:
                continue

            result = simulate_strategy(
                b1=b1,
                b2=b2,
                trials=trials,
                distribution=distribution,
            )
            results.append(result)

    results = sorted(results, key=lambda x: x["mean_pnl"], reverse=True)
    return results


if __name__ == "__main__":
    # Test one strategy
    result = simulate_strategy(
        b1=800,
        b2=860,
        trials=100,
        distribution="mixed",
    )

    print("Single strategy result:")
    for k, v in result.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nSearching best strategy...")

    best_results = grid_search(
        b1_range=range(670, 921,1),
        b2_range=range(670, 921,1),
        trials=100,
        distribution="mixed",
    )

    print("\nTop 10 strategies:")
    for r in best_results[:10]:
        print(
            f"b1={r['b1']}, b2={r['b2']}, "
            f"mean_pnl={r['mean_pnl']:.2f}, "
            f"b1_trades={r['mean_b1_trades']:.2f}, "
            f"b2_trades={r['mean_b2_trades']:.2f}, "
            f"avg_b2={r['mean_avg_b2']:.2f}"
        )
'''
