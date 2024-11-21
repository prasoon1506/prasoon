import streamlit as st
import pandas as pd
import io

def preprocess_data(df, sheet_name):
    # Special preprocessing for TOTAL sheet
    if sheet_name == "TOTAL":
        # Find the column containing "JKLC+UCWL" in the first row
        jklc_ucwl_col = None
        first_row = df.iloc[0]
        for col, value in first_row.items():
            if isinstance(value, str) and "JKLC+UCWL" in value:
                jklc_ucwl_col = col
                break
        
        # If JKLC+UCWL column found, remove columns before it (except first two)
        if jklc_ucwl_col is not None:
            col_index = df.columns.get_loc(jklc_ucwl_col)
            df = df.iloc[:, max(0, col_index-1):]
    
    # Remove first 3 rows
    df = df.iloc[3:]
    df = df.reset_index(drop=True)
    
    # Remove rows containing "Zone"
    df = df[~df.iloc[:, 0].str.contains("Zone", case=False, na=False)]
    
    # Remove "Region" rows and adjacent rows
    region_indices = df[df.iloc[:, 0].str.contains("Region", case=False, na=False)].index
    rows_to_remove = []
    for index in region_indices:
        rows_to_remove.extend([index - 2, index - 1, index])    
    df = df.drop(rows_to_remove)
    df = df.reset_index(drop=True)
    
    # Fill first column with last known non-null value
    first_column_values = df.iloc[:, 0].values
    current_fill_value = None
    for i, value in enumerate(first_column_values):
        if pd.notna(value):
            current_fill_value = value
        elif pd.isna(value) and current_fill_value is not None:
            df.iloc[i, 0] = current_fill_value
    
    # Remove "All India" row and its preceding row
    all_india_df = df[df.iloc[:, 0].str.contains("All India", case=False, na=False)]
    if not all_india_df.empty:
        all_india_index = all_india_df.index[0]
        if all_india_index > 0:
            df = df.drop(all_india_index - 1)
            df = df.reset_index(drop=True)

    # Remove last two rows
    df = df[:-2]
    return df

def process_and_merge(df, file_type, sheet_name):
    column_mapping = {
        "JKLC": {
            "Oct-Sep": ["Region Name", "Material Name", 
                        "Oct Trade Quantity", "Sep Trade Quantity",
                        "Oct Trade EBITDA", "Sep Trade EBITDA", "Increase in Trade EBITDA",
                        "Oct Non-Trade Quantity", "Sep Non-Trade Quantity", 
                        "Oct Non-Trade EBITDA", "Sep Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Oct Total Quantity", "Sep Total Quantity", 
                        "Oct Total EBITDA", "Sep Total EBITDA", "Increase in Total EBITDA"],
            "Sep-Aug": ["Region Name", "Material Name", 
                        "Sep Trade Quantity", "Aug Trade Quantity",
                        "Sep Trade EBITDA", "Aug Trade EBITDA", "Increase in Trade EBITDA",
                        "Sep Non-Trade Quantity", "Aug Non-Trade Quantity", 
                        "Sep Non-Trade EBITDA", "Aug Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Sep Total Quantity", "Aug Total Quantity", 
                        "Sep Total EBITDA", "Aug Total EBITDA", "Increase in Total EBITDA"],
            "Aug-Jul": ["Region Name", "Material Name", 
                        "Aug Trade Quantity", "Jul Trade Quantity",
                        "Aug Trade EBITDA", "Jul Trade EBITDA", "Increase in Trade EBITDA",
                        "Aug Non-Trade Quantity", "Jul Non-Trade Quantity", 
                        "Aug Non-Trade EBITDA", "Jul Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Aug Total Quantity", "Jul Total Quantity", 
                        "Aug Total EBITDA", "Jul Total EBITDA", "Increase in Total EBITDA"]
        },
        "UCWL": {
            "Oct-Sep": ["Region Name", "Material Name", 
                        "Oct Trade Quantity", "Sep Trade Quantity",
                        "Oct Trade EBITDA", "Sep Trade EBITDA", "Increase in Trade EBITDA",
                        "Oct Non-Trade Quantity", "Sep Non-Trade Quantity", 
                        "Oct Non-Trade EBITDA", "Sep Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Oct Total Quantity", "Sep Total Quantity", 
                        "Oct Total EBITDA", "Sep Total EBITDA", "Increase in Total EBITDA"],
            "Sep-Aug": ["Region Name", "Material Name", 
                        "Sep Trade Quantity", "Aug Trade Quantity",
                        "Sep Trade EBITDA", "Aug Trade EBITDA", "Increase in Trade EBITDA",
                        "Sep Non-Trade Quantity", "Aug Non-Trade Quantity", 
                        "Sep Non-Trade EBITDA", "Aug Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Sep Total Quantity", "Aug Total Quantity", 
                        "Sep Total EBITDA", "Aug Total EBITDA", "Increase in Total EBITDA"],
            "Aug-Jul": ["Region Name", "Material Name", 
                        "Aug Trade Quantity", "Jul Trade Quantity",
                        "Aug Trade EBITDA", "Jul Trade EBITDA", "Increase in Trade EBITDA",
                        "Aug Non-Trade Quantity", "Jul Non-Trade Quantity", 
                        "Aug Non-Trade EBITDA", "Jul Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Aug Total Quantity", "Jul Total Quantity", 
                        "Aug Total EBITDA", "Jul Total EBITDA", "Increase in Total EBITDA"]
        },
        "TOTAL": {
            "Oct-Sep": ["Region Name", "Material Name", 
                        "Oct Trade Quantity", "Sep Trade Quantity",
                        "Oct Trade EBITDA", "Sep Trade EBITDA", "Increase in Trade EBITDA",
                        "Oct Non-Trade Quantity", "Sep Non-Trade Quantity", 
                        "Oct Non-Trade EBITDA", "Sep Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Oct Total Quantity", "Sep Total Quantity", 
                        "Oct Total EBITDA", "Sep Total EBITDA", "Increase in Total EBITDA"],
            "Sep-Aug": ["Region Name", "Material Name", 
                        "Sep Trade Quantity", "Aug Trade Quantity",
                        "Sep Trade EBITDA", "Aug Trade EBITDA", "Increase in Trade EBITDA",
                        "Sep Non-Trade Quantity", "Aug Non-Trade Quantity", 
                        "Sep Non-Trade EBITDA", "Aug Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Sep Total Quantity", "Aug Total Quantity", 
                        "Sep Total EBITDA", "Aug Total EBITDA", "Increase in Total EBITDA"],
            "Aug-Jul": ["Region Name", "Material Name", 
                        "Aug Trade Quantity", "Jul Trade Quantity",
                        "Aug Trade EBITDA", "Jul Trade EBITDA", "Increase in Trade EBITDA",
                        "Aug Non-Trade Quantity", "Jul Non-Trade Quantity", 
                        "Aug Non-Trade EBITDA", "Jul Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Aug Total Quantity", "Jul Total Quantity", 
                        "Aug Total EBITDA", "Jul Total EBITDA", "Increase in Total EBITDA"]
        }
    }
    
    # Assign column names
    column_names = column_mapping[sheet_name][file_type]
    df.columns = column_names[:len(df.columns)]

    # Create separate DataFrames for Total and Non-Total rows
    total_mask = df.iloc[:, 0].str.contains("Total", case=False, na=False)
    df_total = df[total_mask]
    df_no_total = df[~total_mask]

    # Remove specific increase columns from Total DataFrame if present
    columns_to_remove = ["Increase in Trade EBITDA", "Increase in Non-Trade EBITDA", "Increase in Total EBITDA"]
    df_total = df_total.drop(columns=columns_to_remove, errors='ignore')
    df_no_total = df_no_total.drop(columns=columns_to_remove, errors='ignore')
    
    return df_no_total, df_total

def process_files(file_sequence, uploaded_files):
    # Initialize lists to store DataFrames
    processed_dfs = {
        "JKLC": {"Non-Total": [], "Total": []},
        "UCWL": {"Non-Total": [], "Total": []},
        "TOTAL": {"Non-Total": [], "Total": []}
    }
    
    # Check if we have the right number of files
    if len(uploaded_files) != len(file_sequence):
        st.error(f"Please upload exactly {len(file_sequence)} files.")
        return None
    
    for idx, (file_info, uploaded_file) in enumerate(zip(file_sequence, uploaded_files)):
        try:
            # Read Excel file with multiple sheets
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = ["JKLC", "UCWL", "TOTAL"]
            
            for sheet_name in sheet_names:
                # Read and process each sheet
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df = preprocess_data(df, sheet_name)
                
                # Separate non-total and total DataFrames
                df_no_total, df_total = process_and_merge(df, file_info['type'], sheet_name)
                
                # Store processed DataFrames
                processed_dfs[sheet_name]["Non-Total"].append(df_no_total)
                processed_dfs[sheet_name]["Total"].append(df_total)
        
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            return None
    
    return processed_dfs

def merge_dataframes(processed_dfs):
    merged_dfs = {}
    
    # Merge and create DataFrames for each sheet type
    for sheet_name in ["JKLC", "UCWL", "TOTAL"]:
        # Merge non-total DataFrames
        final_df = processed_dfs[sheet_name]["Non-Total"][0]
        for df in processed_dfs[sheet_name]["Non-Total"][1:]:
            df = df[["Region Name", "Material Name"] + [col for col in df.columns if col not in final_df.columns]]
            final_df = pd.merge(final_df, df, on=["Region Name", "Material Name"], how="left")
        
        # Merge total DataFrames
        final_total_df = processed_dfs[sheet_name]["Total"][0]
        for df in processed_dfs[sheet_name]["Total"][1:]:
            # Identify and remove duplicate columns, keeping the first occurrence
            df = df[["Region Name"] + [col for col in df.columns if col not in final_total_df.columns]]
            final_total_df = pd.merge(final_total_df, df, on="Region Name", how="left")
        
        merged_dfs[f'Non-Total_{sheet_name}'] = final_df
        merged_dfs[f'Total_{sheet_name}'] = final_total_df
    
    return merged_dfs

def main():
    st.title("Data File Merger")
    
    st.sidebar.header("File Processing Instructions")
    st.sidebar.info("""
    1. Upload files in the following order:
    - Oct-Sep file
    - Sep-Aug file
    - Aug-Jul file
    
    2. For each upload, select the correct Excel file
    3. Each uploaded file should contain JKLC, UCWL, and TOTAL sheets
    4. Final merged data will be available for download
    """)
    
    # Predefined file types and descriptions
    file_sequence = [
        {"type": "Oct-Sep", "description": "Oct-Sep file"},
        {"type": "Sep-Aug", "description": "Sep-Aug file"},
        {"type": "Aug-Jul", "description": "Aug-Jul file"}
    ]
    
    # File uploads
    uploaded_files = []
    for file_info in file_sequence:
        uploaded_file = st.file_uploader(f"Upload {file_info['description']}", type=['xlsx'])
        if uploaded_file is not None:
            uploaded_files.append(uploaded_file)
    
    if st.button("Process and Merge Files") and len(uploaded_files) == len(file_sequence):
        with st.spinner('Processing files...'):
            # Process uploaded files
            processed_dfs = process_files(file_sequence, uploaded_files)
            
            if processed_dfs:
                # Merge DataFrames
                merged_dfs = merge_dataframes(processed_dfs)
                
                # Create an in-memory Excel file
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    for sheet_name, df in merged_dfs.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                output.seek(0)
                
                # Download button
                st.download_button(
                    label="Download Merged Excel File",
                    data=output,
                    file_name='final_merged_data.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                
                # Optional: Display preview of merged data
                st.subheader("Merged Data Preview")
                for sheet_name, df in merged_dfs.items():
                    st.write(f"Sheet: {sheet_name}")
                    st.dataframe(df.head())
            else:
                st.error("Failed to process files. Please check your input.")

if __name__ == "__main__":
    main()
