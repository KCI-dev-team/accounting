import pandas as pd
import json
from datetime import datetime
import time
import os
import csv
from itertools import count

def parse_csv(file_path: str, file_type: str) -> pd.DataFrame:
    """
    Parse CSV files for bank reconciliation, standardizing the format.
    
    Args:
        file_path (str): Path to the CSV file
        file_type (str): Either "GL" or "Bank" to indicate file type
    
    Returns:
        pd.DataFrame: Standardized DataFrame with required columns
    """
    if file_type not in ["GL", "Bank"]:
        raise ValueError("file_type must be either 'GL' or 'Bank'")

    # Read CSV and clean column names
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if file_type == "GL":
        df = df.rename(columns={"Npostid": "Control", "Postdate": "Date"})
        # The date is already in the Date column in format "2/3/2025 0:00"
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M", errors="coerce").dt.date

        
        # Calculate Net amount (Debit - Credit)
        df["Net"] = df["Debit"] + df["Credit"]
        # Select final columns
        return df[["Date", "Net", "Control"]]

    else:  # Bank
        # Standardize date format
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce").dt.date
        
        # Select and rename required columns
        if "Amount" not in df.columns:
            raise ValueError("Bank file must have an 'Amount' column")
        # Convert amount to float if str
        if df["Amount"].dtype == object:  # Check if Amount is string/object type
            df["Amount"] = df["Amount"].str.replace(",", "").astype(float)
        else:
            df["Amount"] = df["Amount"].astype(float)  # Ensure float type
        # Select final columns
        return df[["Date", "Amount"]]


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
        transaction = arr[i]  # Now includes (date, amount, control)
        # iterate in reverse to avoid using the same transaction twice
        for count in range(n - 1, -1, -1):
            for s, comb in list(dp[count].items()):
                new_sum = s + amt_cents
                if new_sum > T:
                    continue
                if new_sum not in dp[count + 1]:
                    dp[count + 1][new_sum] = comb + [transaction]
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
                f"Processing target {i}/{total_targets}: {target_amount} on {target_date}"
            )

        # valid_transactions = gl[
        #     (gl["Date"] >= target_date - pd.Timedelta(days=5)) & 
        #     (gl["Date"] <= target_date + pd.Timedelta(days=5))
        # ].copy()
        valid_transactions = gl.copy()
        if len(valid_transactions) < n:
            result[(target_date, target_amount)] = []
            continue

        # Create a list of (date, amount, control) and sort by amount
        transactions = list(
            zip(
                valid_transactions["Date"],
                valid_transactions["Net"],
                valid_transactions["Control"],
            )
        )
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
    used_transactions = set()  # Stores (date, amount, control) tuples already used
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
                    lambda row: (row["Date"], row["Net"], row["Control"])
                    in used_transactions,
                    axis=1,
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


def find_voided_totals(unmatched_totals, max_n=5):
    """
    Find combinations of unmatched totals that sum to 0.
    Args:
        unmatched_totals: List of (date, amount) tuples
        max_n: Maximum number of transactions to combine
    Returns:
        List of lists, where each inner list contains (date, amount) tuples that sum to 0
    """
    voided_groups = []
    remaining_totals = set(unmatched_totals)
    
    def find_zero_sums(target_sum, items, current_combo, start_idx, n):
        if n == 0:
            if abs(target_sum) < 0.01:  # Account for floating point precision
                voided_groups.append(current_combo[:])
            return
        
        for i in range(start_idx, len(items)):
            current_combo.append(items[i])
            find_zero_sums(target_sum + items[i][1], items, current_combo, i + 1, n - 1)
            current_combo.pop()
    
    # Convert to list for indexing
    totals_list = list(unmatched_totals)
    
    # Try combinations of different sizes
    for n in range(2, max_n + 1):
        find_zero_sums(0, totals_list, [], 0, n)
    
    # Remove overlapping groups (prefer smaller groups)
    final_groups = []
    used_totals = set()
    
    # Sort groups by size and then by date
    voided_groups.sort(key=lambda x: (len(x), x[0][0]))
    
    for group in voided_groups:
        if not any(total in used_totals for total in group):
            final_groups.append(group)
            used_totals.update(group)
    
    return final_groups


def process_files_bidirectional(gl_df: pd.DataFrame, totals_df: pd.DataFrame, max_n=10):
    """
    Process GL and totals DataFrames to find matching transactions in both directions.
    Returns unmatched items from both GL and totals.
    
    Args:
        gl_df (pd.DataFrame): GL transactions with Date, Net, and Control columns
        totals_df (pd.DataFrame): Bank totals with Date and Amount columns
        max_n (int): Maximum number of transactions to combine
    """
    # Save original copies for reference
    original_gl = gl_df.copy()
    original_totals = totals_df.copy()

    # PHASE 1: Normal mode - GL transactions matching to totals
    print("\n=== NORMAL MODE: GL entries matching to totals ===")
    print(f"Total targets to match: {len(totals_df)}")
    print(f"Total GL entries available: {len(gl_df)}")
    
    start_time = time.time()
    normal_solutions_dp = find_all_n_sums_dp(gl_df, totals_df, max_n=max_n)
    end_time = time.time()
    print(f"Time taken for normal mode: {end_time - start_time} seconds")

    # Consolidate normal solutions
    normal_final_solutions = {}
    for n_solutions in normal_solutions_dp.values():
        for key, trans in n_solutions.items():
            if trans:
                normal_final_solutions[key] = trans

    # Get remaining transactions and totals after normal mode
    used_gl_controls = set()
    for _, trans_list in normal_final_solutions.items():
        if trans_list:
            for _, _, control in trans_list:
                used_gl_controls.add(control)
    
    remaining_gl = original_gl[~original_gl["Control"].isin(used_gl_controls)]
    
    matched_totals_keys = set(normal_final_solutions.keys())
    remaining_totals = original_totals[~original_totals.apply(
        lambda row: (row["Date"], row["Amount"]) in matched_totals_keys, axis=1
    )]
    
    print(f"\nAfter normal mode:")
    print(f"Matched {len(matched_totals_keys)} totals")
    print(f"Used {len(used_gl_controls)} GL transactions")
    print(f"Remaining totals: {len(remaining_totals)}")
    print(f"Remaining GL transactions: {len(remaining_gl)}")

    # PHASE 2: Reverse mode - remaining totals matching to remaining GL transactions
    print("\n=== REVERSE MODE: Remaining totals matching to remaining GL entries ===")
    
    # Prepare dataframes for reverse mode
    reverse_gl = remaining_totals.copy()
    reverse_gl = reverse_gl.rename(columns={"Amount": "Net"})
    reverse_gl["Control"] = None  # Bank transactions don't have control numbers
    
    reverse_totals = remaining_gl.copy()
    reverse_totals = reverse_totals.rename(columns={"Net": "Amount"})
    
    start_time = time.time()
    reverse_solutions_dp = find_all_n_sums_dp(reverse_gl, reverse_totals, max_n=max_n)
    end_time = time.time()
    print(f"Time taken for reverse mode: {end_time - start_time} seconds")

    # Consolidate reverse solutions
    reverse_final_solutions = {}
    for n_solutions in reverse_solutions_dp.values():
        for key, trans in n_solutions.items():
            if trans:
                # In reverse mode, the key is (gl_date, gl_amount) and we need to preserve the control
                gl_date, gl_amount = key
                gl_control = reverse_totals[
                    (reverse_totals["Date"] == gl_date) & 
                    (reverse_totals["Amount"] == gl_amount)
                ]["Control"].iloc[0]
                
                # Create new transactions with the bank amounts and the GL control
                new_trans = []
                for bank_date, bank_amount, _ in trans:
                    new_trans.append((bank_date, bank_amount, gl_control))
                reverse_final_solutions[key] = new_trans
    
    # Get final unmatched totals and GL items
    matched_in_reverse = set()
    matched_gl_in_reverse = set()  # Store the actual GL transactions matched in reverse
    for key, trans_list in reverse_final_solutions.items():
        if trans_list:
            matched_in_reverse.add(key)
            for t in trans_list:
                matched_gl_in_reverse.add((t[0], t[1], t[2]))  # date, amount, control
    
    # Get totals that weren't matched in either normal or reverse mode
    all_totals_keys = set(zip(original_totals["Date"], original_totals["Amount"]))
    matched_totals_keys = matched_totals_keys.union(
        {(date, amount) for date, amount, _ in 
         [item for sublist in reverse_final_solutions.values() if sublist 
          for item in sublist]}
    )
    unmatched_totals = sorted(
        all_totals_keys - matched_totals_keys, key=lambda x: (x[0], x[1])
    )
    
    # Get GL transactions that weren't matched in either mode
    all_gl_keys = set(zip(original_gl["Date"], original_gl["Net"], original_gl["Control"]))
    matched_gl_keys = set()
    
    # Add GL transactions matched in normal mode
    for trans_list in normal_final_solutions.values():
        if trans_list:
            matched_gl_keys.update(tuple(t) for t in trans_list)
    
    # Add GL transactions matched as targets in reverse mode
    matched_gl_keys.update(matched_gl_in_reverse)
    
    unmatched_gl = sorted(
        all_gl_keys - matched_gl_keys, 
        key=lambda x: (x[0], x[1])
    )
    
    print(f"\nAfter reverse mode:")
    print(f"Matched an additional {len(reverse_final_solutions)} GL transactions")
    print(f"Final unmatched totals: {len(unmatched_totals)}")
    print(f"Final unmatched GL entries: {len(unmatched_gl)}")
    
    # Find voided transactions in unmatched totals
    voided_groups = find_voided_totals(unmatched_totals, max_n=5)
    
    # Remove voided transactions from unmatched_totals
    voided_totals = set()
    for group in voided_groups:
        voided_totals.update(group)
    
    unmatched_totals = sorted(
        set(unmatched_totals) - voided_totals,
        key=lambda x: (x[0], x[1])
    )
    
    print(f"Found {len(voided_groups)} groups of voided transactions")
    print(f"Final unmatched totals after removing voided: {len(unmatched_totals)}")
    
    return normal_final_solutions, reverse_final_solutions, unmatched_totals, unmatched_gl, voided_groups


def print_bidirectional_results(normal_solutions, reverse_solutions, unmatched_totals, unmatched_gl, voided_groups):
    """Print results from bidirectional matching."""
    # Print normal mode results
    print("\n=== NORMAL MODE RESULTS ===")
    print(f"Found {len(normal_solutions)} matches where GL transactions sum to totals")
    for (date, amount), trans in normal_solutions.items():
        print(f"\nTarget: {amount} on {date} (from totals file)")
        print(f"Matching transactions from GL file:")
        for trans_date, trans_amount, control in trans:
            print(f"  {trans_amount} on {trans_date} (Control: {control})")
    
    # Print reverse mode results
    print("\n=== REVERSE MODE RESULTS ===")
    print(f"Found {len(reverse_solutions)} matches where totals sum to GL transactions")
    for (date, amount), trans in reverse_solutions.items():
        print(f"\nTarget: {amount} on {date} (from GL file)")
        print(f"Matching transactions from totals file:")
        for trans_date, trans_amount, control in trans:
            print(f"  {trans_amount} on {trans_date} (Control: {control})")
    
    # Print unmatched totals
    print("\n=== UNMATCHED TOTALS ===")
    print("-" * 50)
    print(f"{'Date':12} {'Amount':>12}")
    print("-" * 50)
    for date, amount in unmatched_totals:
        print(f"{date!s:12} {amount:>12.2f}")
    print("-" * 50)
    print(f"Total unmatched totals: {len(unmatched_totals)}")
    print(f"Total unmatched amount: ${sum(amount for _, amount in unmatched_totals):,.2f}")

    # Print unmatched GL entries
    print("\n=== UNMATCHED GL ENTRIES ===")
    print("-" * 70)
    print(f"{'Date':12} {'Amount':>12} {'Control':>12}")
    print("-" * 70)
    for date, amount, control in unmatched_gl:
        print(f"{date!s:12} {amount:>12.2f} {control:>12}")
    print("-" * 70)
    print(f"Total unmatched GL entries: {len(unmatched_gl)}")
    print(f"Total unmatched amount: ${sum(amount for _, amount, _ in unmatched_gl):,.2f}")

    # Print voided totals groups
    print("\n=== VOIDED TOTALS GROUPS ===")
    print(f"Found {len(voided_groups)} groups of transactions that sum to 0")
    for i, group in enumerate(voided_groups, 1):
        print(f"\nGroup {i}:")
        print("-" * 50)
        print(f"{'Date':12} {'Amount':>12}")
        print("-" * 50)
        for date, amount in group:
            print(f"{date!s:12} {amount:>12.2f}")
        print("-" * 50)
        print(f"Sum: ${sum(amount for _, amount in group):,.2f}")


def save_bidirectional_results(normal_solutions, reverse_solutions, unmatched_totals, unmatched_gl, 
                             voided_groups, max_n):
    """
    Process results from bidirectional matching and return DataFrames.
    
    Args:
        normal_solutions (dict): Solutions from normal mode
        reverse_solutions (dict): Solutions from reverse mode
        unmatched_totals (list): Unmatched totals
        unmatched_gl (list): Unmatched GL transactions
        voided_groups (list): Voided totals groups
        max_n (int): Maximum number of transactions combined
    
    Returns:
        tuple: (reconciliation_df, unmatched_totals_df, unmatched_gl_df, voided_groups_df)
    """
    # Create reconciliation rows
    rows = []
    recon_id_counter = count(1)

    # Process normal mode solutions
    for (bank_date, bank_amount), gl_transactions in normal_solutions.items():
        if not gl_transactions:
            continue
        recon_id = next(recon_id_counter)
        for gl_date, gl_amount, control in gl_transactions:
            rows.append({
                'reconciliation_id': recon_id,
                'bank_amount': float(bank_amount),
                'bank_date': bank_date.strftime("%Y-%m-%d"),
                'gl_amount': float(gl_amount),
                'gl_date': gl_date.strftime("%Y-%m-%d"),
                'control': control,
                'final_value': float(bank_amount)
            })

    # Process reverse mode solutions
    for (gl_date, gl_amount), bank_transactions in reverse_solutions.items():
        if not bank_transactions:
            continue
        recon_id = next(recon_id_counter)
        for bank_date, bank_amount, control in bank_transactions:
            rows.append({
                'reconciliation_id': recon_id,
                'bank_amount': float(bank_amount),
                'bank_date': bank_date.strftime("%Y-%m-%d"),
                'gl_amount': float(gl_amount),
                'gl_date': gl_date.strftime("%Y-%m-%d"),
                'control': control,
                'final_value': float(gl_amount)
            })

    # Create DataFrames
    reconciliation_df = pd.DataFrame(rows)
    
    # Create unmatched totals DataFrame
    unmatched_totals_df = pd.DataFrame(
        [(date.strftime("%Y-%m-%d"), float(amount)) for date, amount in unmatched_totals],
        columns=['date', 'amount']
    )
    
    # Create unmatched GL entries DataFrame
    unmatched_gl_df = pd.DataFrame(
        [(date.strftime("%Y-%m-%d"), float(amount), control) for date, amount, control in unmatched_gl],
        columns=['date', 'amount', 'control']
    )
    
    # Create voided groups DataFrame
    voided_rows = []
    for i, group in enumerate(voided_groups, 1):
        voided_rows.extend([(i, date.strftime("%Y-%m-%d"), float(amount)) for date, amount in group])
    voided_groups_df = pd.DataFrame(voided_rows, columns=['group_id', 'date', 'amount'])
    
    return reconciliation_df, unmatched_totals_df, unmatched_gl_df, voided_groups_df


if __name__ == "__main__":
    # Example usage
    gl_file = os.path.join("data", "fpk_gl.csv")
    totals_file = os.path.join("data", "fpk_bank.csv")
    max_n = 10
    output_prefix = os.path.join("data", "fpk_results")

    # Parse input files
    gl_df = parse_csv(gl_file, "GL")
    totals_df = parse_csv(totals_file, "Bank")
    print(gl_df.head())
    # Process DataFrames in both directions
    normal_solutions, reverse_solutions, unmatched_totals, unmatched_gl, voided_groups = process_files_bidirectional(
        gl_df, totals_df, max_n
    )
    
    # Print results
    print_bidirectional_results(
        normal_solutions, reverse_solutions, unmatched_totals, unmatched_gl, voided_groups
    )
    
    # Save results
    reconciliation_df, unmatched_totals_df, unmatched_gl_df, voided_groups_df = save_bidirectional_results(
        normal_solutions, reverse_solutions, unmatched_totals, unmatched_gl, voided_groups, max_n
    )
    
    # Print reconciliation DataFrame
    print("\n=== RECONCILIATION DATAFRAME ===")
    print(reconciliation_df)

    # Print unmatched totals DataFrame
    print("\n=== UNMATCHED TOTALS DATAFRAME ===")
    print(unmatched_totals_df)

    # Print unmatched GL entries DataFrame
    print("\n=== UNMATCHED GL ENTRIES DATAFRAME ===")
    print(unmatched_gl_df)

    # Print voided groups DataFrame
    print("\n=== VOIDED GROUPS DATAFRAME ===")
    print(voided_groups_df)
