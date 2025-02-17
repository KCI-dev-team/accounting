import pandas as pd
import json
from datetime import datetime


def find_n_sum_dp(arr, target, n, start):
    """
    Find a combination of exactly n transactions from arr (list of (date, amount))
    that sum to target using dynamic programming.

    Converts amounts to integer cents.

    Returns a list containing one valid combination (list of (date, amount) tuples) if found,
    otherwise an empty list.
    """
    # Convert target to integer cents
    T = int(round(target * 100))
    # dp[k] maps a sum in cents to a combination (list of transactions) that uses exactly k items.
    dp = [dict() for _ in range(n + 1)]
    dp[0][0] = []  # zero transactions yield sum 0

    for i in range(start, len(arr)):
        amt_cents = int(round(arr[i][1] * 100))
        # iterate in reverse to avoid using the same transaction twice
        for count in range(n - 1, -1, -1):
            for s, comb in list(dp[count].items()):
                new_sum = s + amt_cents
                if new_sum > T:
                    continue
                if new_sum not in dp[count + 1]:
                    dp[count + 1][new_sum] = comb + [arr[i]]
    if T in dp[n]:
        return [dp[n][T]]
    return []


def get_n_sum_dp(gl, totals, n):
    """
    For each target in totals, find exactly n GL transactions (from gl) whose amounts sum to the target,
    considering only GL transactions with Date <= target date.

    Returns a dictionary mapping (target_date, target_amount) to a combination (list of (date, amount)).
    """
    result = {}
    total_targets = len(totals)
    print(f"\nLooking for combinations of {n} transactions using DP...")

    for i, (target_date, target_amount) in enumerate(
        totals[["Date", "Amount"]].values, 1
    ):
        if i % 10 == 0:
            print(
                f"Processing target {i}/{total_targets}: {target_amount} on {target_date.date()}"
            )

        valid_transactions = gl[gl["Date"] <= target_date].copy()
        if len(valid_transactions) < n:
            result[(target_date, target_amount)] = []
            continue

        # Create a list of (date, amount) and sort by amount
        transactions = list(zip(valid_transactions["Date"], valid_transactions["Net"]))
        transactions.sort(key=lambda x: x[1])

        combination = find_n_sum_dp(transactions, target_amount, n, 0)
        result[(target_date, target_amount)] = combination[0] if combination else []

    return result


def find_all_n_sums_dp(gl, totals, max_n=10):
    """
    Iteratively try n from 1 to max_n using the DP approach.
    Once a transaction is used in a solution, it cannot be reused.
    Once a total amount is matched, it is removed from further consideration.
    """
    result = {}
    used_transactions = set()  # Stores (date, amount) tuples already used
    remaining_transactions = gl.copy()
    remaining_totals = totals.copy()

    print(f"\nStarting DP search for combinations up to {max_n} transactions")
    print(f"Total targets to match: {len(totals)}")
    print(f"Total GL entries available: {len(gl)}")

    for n in range(1, max_n + 1):
        if len(remaining_totals) == 0:
            print("\nAll totals have been matched!")
            break

        if used_transactions:
            remaining_transactions = remaining_transactions[
                ~remaining_transactions.apply(
                    lambda row: (row["Date"], row["Net"]) in used_transactions, axis=1
                )
            ]

        if len(remaining_transactions) < n:
            print(f"\nNot enough remaining transactions for n={n}. Stopping search.")
            break

        print(f"\nTrying combinations of {n} transactions using DP...")
        print(f"Remaining targets to match: {len(remaining_totals)}")
        print(f"Remaining unused transactions: {len(remaining_transactions)}")

        solutions = get_n_sum_dp(remaining_transactions, remaining_totals, n)
        found_solutions = sum(1 for s in solutions.values() if s)
        if found_solutions > 0:
            result[n] = solutions
            matched_keys = set()
            for key, trans_list in solutions.items():
                if trans_list:
                    matched_keys.add(key)
                    used_transactions.update(tuple(t) for t in trans_list)
            remaining_totals = remaining_totals[
                ~remaining_totals.apply(
                    lambda row: (row["Date"], row["Amount"]) in matched_keys, axis=1
                )
            ]
            print(f"Found {found_solutions} new matches using {n} transactions!")
    return result


# --- Example usage and demonstration ---

if __name__ == "__main__":
    # Read CSV with explicit date parsing and data types
    gl = pd.read_csv(
        "data/gl_deb.csv",
        dtype={"Date": str},  # Read Date as string first
        parse_dates=False,
    )  # Don't auto-parse dates

    # Clean the data before converting dates
    gl["Date"] = pd.to_datetime(gl["Date"], format="%m/%d/%Y", errors="coerce")
    # Remove rows with invalid dates
    gl = gl.dropna(subset=["Date"])

    # If any NaT (Not a Time) values appear, you can check which rows had parsing errors
    if gl["Date"].isna().any():
        print("Warning: Some dates could not be parsed:")
        print(gl[gl["Date"].isna()]["Date"])

    # Convert numeric columns to float, handling any commas in numbers
    gl["Debit"] = gl["Debit"].str.replace(",", "").astype(float)
    gl["Credit"] = gl["Credit"].str.replace(",", "").astype(float)
    gl["Net"] = gl["Net"].str.replace(",", "").astype(float)

    totals = pd.read_csv(
        "data/ttls_deb.csv",
        dtype={"Date": str},  # Read Date as string first
        parse_dates=False,
    )
    totals["Date"] = pd.to_datetime(totals["Date"], format="%m/%d/%Y", errors="coerce")
    totals = totals.dropna(subset=["Date"])
    totals["Amount"] = totals["Amount"].replace(",", "").astype(float)

    solutions_dp = find_all_n_sums_dp(gl, totals, max_n=3)

    # Consolidate found solutions into one dictionary.
    final_solutions_dp = {}
    for n_solutions in solutions_dp.values():
        for key, trans in n_solutions.items():
            if trans:
                final_solutions_dp[key] = trans

    print("\nFound DP solutions:")
    for (date, amount), trans in final_solutions_dp.items():
        print(f"\nTarget: {amount} on {date.date()}")
        print("Matching transactions:")
        for trans_date, trans_amount in trans:
            print(f"  {trans_amount} on {trans_date.date()}")

    # Format and save as JSON
    def format_for_json(solutions_dict):
        formatted = {}
        for (date, amount), trans in solutions_dict.items():
            date_str = date.strftime("%Y-%m-%d")
            key = f"{date_str}_{amount}"
            formatted_transactions = [
                {
                    "date": trans_date.strftime("%Y-%m-%d"),
                    "amount": float(trans_amount),
                }
                for trans_date, trans_amount in trans
            ]
            formatted[key] = {
                "target_date": date_str,
                "target_amount": float(amount),
                "matching_transactions": formatted_transactions,
            }
        return formatted

    json_results = format_for_json(final_solutions_dp)
    with open("data/matching_results_dp.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print("\nResults saved to 'data/matching_results_dp.json'")
