import streamlit as st
import pandas as pd
import io
import contextlib
import time

# Import your reconciliation logic here
from bank_rec.n_sum import (
    process_files_bidirectional,
    print_bidirectional_results,
    save_bidirectional_results,
    parse_csv,
)


def load_file(uploaded_file):
    if uploaded_file.name.lower().endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def capture_results(normal_solutions, reverse_solutions, unmatched_totals, unmatched_gl, voided_groups):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_bidirectional_results(
            normal_solutions,
            reverse_solutions,
            unmatched_totals,
            unmatched_gl,
            voided_groups
        )
    return buffer.getvalue()


def main():
    st.title("Bank File Reconciliation")

    gl_file = st.file_uploader("Upload GL file", type=["csv", "xlsx"], key="gl_file")
    totals_file = st.file_uploader("Upload Totals file", type=["csv", "xlsx"], key="totals_file")
    max_n = st.number_input("Max n", min_value=1, value=10, step=1)

    run_button = st.button("Run Reconciliation")

    if gl_file and totals_file and run_button:
        with st.spinner('Processing files...'):
            start_time = time.time()
            
            # Parse the uploaded files using parse_csv
            gl_df = parse_csv(gl_file, "GL")
            totals_df = parse_csv(totals_file, "Bank")
            
            # Process files directly as file-like objects
            normal_solutions, reverse_solutions, unmatched_totals, unmatched_gl, voided_groups = \
                process_files_bidirectional(gl_df, totals_df, max_n)

            # Display reconciliation results
            results = capture_results(
                normal_solutions,
                reverse_solutions,
                unmatched_totals,
                unmatched_gl,
                voided_groups
            )
            end_time = time.time()
            execution_time = end_time - start_time
            st.success(f"Reconciliation completed in {execution_time:.2f} seconds")
            st.code(results, language=None)


if __name__ == "__main__":
    main()
