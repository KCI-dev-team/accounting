import streamlit as st
import pandas as pd
import io
import contextlib
import time
from datetime import datetime

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


def to_excel(df_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


def main():
    st.markdown("<h1 style='text-align: center;'>Bank File Reconciliation</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Upload GL File")
    gl_file = st.file_uploader("Choose your GL file", type=["csv",], key="gl_file", label_visibility="collapsed")
    
    st.subheader("Upload Totals File")
    totals_file = st.file_uploader("Choose your Totals file", type=["csv"], key="totals_file", label_visibility="collapsed")
    
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

            # Get DataFrames from save_bidirectional_results
            reconciliation_df, unmatched_totals_df, unmatched_gl_df, voided_groups_df = save_bidirectional_results(
                normal_solutions, reverse_solutions, unmatched_totals, unmatched_gl, voided_groups, max_n
            )

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

            # Create Excel file for download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_data = to_excel({
                'Reconciliation': reconciliation_df,
                'Unmatched Bank Activity': unmatched_totals_df,
                'Unmatched GL': unmatched_gl_df,
                'Voided Groups': voided_groups_df
            })
            
            # Center the download button using columns
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_data,
                    file_name=f"reconciliation_report_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            # Add a toggle for detailed output
            with st.expander("View Detailed Output", expanded=False):
                st.code(results, language=None)


if __name__ == "__main__":
    main()
