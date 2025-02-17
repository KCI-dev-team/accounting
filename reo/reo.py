import pandas as pd
import os


def extract_monthly_values(file_path, lookup_value):
    # Read the Excel file
    df = pd.read_excel(file_path, header=None)

    # Find the row with months (looking for a row that starts with a month name)
    month_row = None
    for idx, row in df.iterrows():
        # Check if the row has month names (B through M should be months)
        if isinstance(row[1], str) and any(
            month in row[1].lower()
            for month in [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ]
        ):
            month_row = idx
            break

    if month_row is None:
        print("Could not find month headers in the file.")
        return None

    # Truncate everything above the month row and set proper headers
    df = df.iloc[month_row:]
    df.columns = ["Description"] + list(
        df.iloc[0, 1:14]
    )  # First row becomes headers, including total column
    df = df.iloc[1:].reset_index(drop=True)  # Remove the header row from data

    # Find the row with the lookup value in column A (Description)
    matched_rows = df[df["Description"] == lookup_value]

    if matched_rows.empty:
        print(f"No row found with {lookup_value} in Description column.")
        return None
    elif len(matched_rows) > 1:
        print("Multiple rows found. Using the first match.")

    # Get the monthly values (columns B through M) and the total column
    monthly_values = matched_rows.iloc[
        0, 1:14
    ]  # Changed from 1:13 to 1:14 to include total

    return monthly_values


if __name__ == "__main__":
    file_path = os.path.join(
        "reo", "data", "12_Month_Statement_mia139_Accrual (1).xlsx"
    )
    lookup_values = [
        "Mortgage Interest",
        "Preferred Interest",
        "Total Net Operating Income",
    ]

    results = {}
    for lookup_value in lookup_values:
        result = extract_monthly_values(file_path, lookup_value)
        if result is not None:
            results[lookup_value] = result

    print("Extracted monthly values:")
    for key, value in results.items():
        print(f"\n{key}:")
        print(value)

    file_path = os.path.join("reo", "data", "12_Month_Statement_mia139_Accrual.xlsx")
    lookup_values = ["EBITDA", "Interest Expense"]

    results = {}
    for lookup_value in lookup_values:
        result = extract_monthly_values(file_path, lookup_value)
        if result is not None:
            results[lookup_value] = result

    print("Extracted monthly values:")
    for key, value in results.items():
        print(f"\n{key}:")
        print(value)
