import streamlit as st
import pandas as pd
import io

def preprocess_data(df, sheet_type):
    # First, find the column with JKLC+UCWL in the first row
    jklc_ucwl_col = None
    first_row = df.iloc[0]
    
    for col, value in first_row.items():
        if isinstance(value, str) and "JKLC+UCWL" in str(value):
            jklc_ucwl_col = col
            break
    
    # Remove first 3 rows
    df = df.iloc[3:]
    df = df.reset_index(drop=True)
    
    # For TOTAL sheet, remove columns before JKLC+UCWL
    if sheet_type == 'TOTAL' and jklc_ucwl_col is not None:
        # Find the column index
        col_to_keep = df.columns.get_loc(jklc_ucwl_col)
        df = df.iloc[:, max(0, col_to_keep-1):]
    
    # Remove rows containing "Zone"
    df = df[~df.iloc[:, 0].str.contains("Zone", case=False, na=False)]
    
    # Remove region-related rows
    region_indices = df[df.iloc[:, 0].str.contains("Region", case=False, na=False)].index
    rows_to_remove = []
    for index in region_indices:
        rows_to_remove.extend([index - 2, index - 1, index])    
    df = df.drop(rows_to_remove)
    df = df.reset_index(drop=True)
    
    # Fill NaN values in first column with last known non-NaN value
    first_column_values = df.iloc[:, 0].values
    current_fill_value = None
    for i, value in enumerate(first_column_values):
        if pd.notna(value):
            current_fill_value = value
        elif pd.isna(value) and current_fill_value is not None:
            df.iloc[i, 0] = current_fill_value
    
    # Remove "All India" row and preceding row if present
    all_india_df = df[df.iloc[:, 0].str.contains("All India", case=False, na=False)]
    if not all_india_df.empty:
        all_india_index = all_india_df.index[0]
        if all_india_index > 0:
            df = df.drop(all_india_index - 1)
            df = df.reset_index(drop=True)

    # Remove last two rows
    df = df[:-2]
    return df

def process_and_merge(df, file_type, sheet_type):
    # Define column names based on file type and sheet type
    column_names = [
        "Region Name", "Material Name", 
        f"{file_type.split('-')[0]} Trade Quantity", 
        f"{file_type.split('-')[1]} Trade Quantity",
        f"{file_type.split('-')[0]} Trade EBITDA", 
        f"{file_type.split('-')[1]} Trade EBITDA", 
        "Increase in Trade EBITDA",
        f"{file_type.split('-')[0]} Non-Trade Quantity", 
        f"{file_type.split('-')[1]} Non-Trade Quantity", 
        f"{file_type.split('-')[0]} Non-Trade EBITDA", 
        f"{file_type.split('-')[1]} Non-Trade EBITDA", 
        "Increase in Non-Trade EBITDA", 
        f"{file_type.split('-')[0]} Total Quantity", 
        f"{file_type.split('-')[1]} Total Quantity", 
        f"{file_type.split('-')[0]} Total EBITDA", 
        f"{file_type.split('-')[1]} Total EBITDA", 
        "Increase in Total EBITDA"
    ]
    
    # Assign column names
    df.columns = column_names

    # Remove specific increase columns from Total DataFrame
    columns_to_remove = ["Increase in Trade EBITDA", "Increase in Non-Trade EBITDA", "Increase in Total EBITDA"]
    
    # Total sheet specific handling
    if sheet_type == 'TOTAL':
        # Drop increase columns from total DataFrame
        df = df.drop(columns=columns_to_remove, errors='ignore')
        # Ensure exactly 17 columns
        if len(df.columns) > 17:
            df = df.iloc[:, :17]
    else:
        # Non-Total sheets
        df_total = df[df.iloc[:, 0].str.contains("Total", case=False, na=False)].copy()
        df = df[~df.iloc[:, 0].str.contains("Total", case=False, na=False)]
        
        # Drop increase columns from both DataFrames
        df = df.drop(columns=columns_to_remove, errors='ignore')
        df_total = df_total.drop(columns=columns_to_remove, errors='ignore')
        
        return df, df_total

def process_excel_files(uploaded_files):
    # Initialize dictionary to store processed DataFrames
    processed_dfs = {
        'JKLC': {'Non-Total': [], 'Total': []},
        'UCWL': {'Non-Total': [], 'Total': []},
        'TOTAL': {'Non-Total': [], 'Total': []}
    }
    
    # Predefined file types
    file_types = ["Oct-Sep", "Sep-Aug", "Aug-Jul"]
    
    # Process each uploaded file
    for file in uploaded_files:
        # Read the Excel file
        xls = pd.ExcelFile(file)
        
        # Get the current file type
        file_type = file_types[len(uploaded_files) - len(file_types)]
        
        # Process each sheet type
        for sheet_type in ['JKLC', 'UCWL', 'TOTAL']:
            # Read the sheet
            df = pd.read_excel(xls, sheet_name=sheet_type)
            
            # Preprocess the data
            df = preprocess_data(df, sheet_type)
            
            # Process and merge data
            if sheet_type != 'TOTAL':
                df_no_total, df_total = process_and_merge(df, file_type, sheet_type)
                processed_dfs[sheet_type]['Non-Total'].append(df_no_total)
                processed_dfs[sheet_type]['Total'].append(df_total)
            else:
                # For TOTAL sheet, just process without splitting
                df = process_and_merge(df, file_type, sheet_type)
                processed_dfs[sheet_type]['Non-Total'].append(df)
    
    # Create output DataFrames
    output_dfs = {}
    
    # Merge and process DataFrames
    for sheet_type in ['JKLC', 'UCWL', 'TOTAL']:
        for df_type in ['Non-Total', 'Total']:
            if sheet_type == 'TOTAL' and df_type == 'Total':
                # Skip Total for TOTAL sheet
                continue
            
            # Merge DataFrames
            if len(processed_dfs[sheet_type][df_type]) > 0:
                final_df = processed_dfs[sheet_type][df_type][0]
                for df in processed_dfs[sheet_type][df_type][1:]:
                    if sheet_type == 'TOTAL':
                        # For TOTAL sheet, just concatenate
                        final_df = pd.concat([final_df, df], axis=1)
                    else:
                        # For other sheets, merge on Region and Material Name
                        merge_cols = ["Region Name", "Material Name"]
                        if df_type == 'Total':
                            merge_cols = ["Region Name"]
                        
                        # Identify and merge columns
                        df = df[merge_cols + [col for col in df.columns if col not in final_df.columns]]
                        final_df = pd.merge(final_df, df, on=merge_cols, how="left")
                
                # Store in output DataFrames
                output_key = f"{df_type}_{sheet_type}"
                output_dfs[output_key] = final_df
    
    return output_dfs

def main():
    st.title("Excel Data Processing App")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Excel Files (Oct-Sep, Sep-Aug, Aug-Jul in order)", 
        type=['xlsx'], 
        accept_multiple_files=True
    )
    
    # Process files when uploaded
    if uploaded_files and len(uploaded_files) == 3:
        try:
            # Process the files
            output_dfs = process_excel_files(uploaded_files)
            
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for sheet_name, df in output_dfs.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            output.seek(0)
            
            # Download button
            st.download_button(
                label="Download Processed Excel File",
                data=output,
                file_name='final_merged_data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
            # Display DataFrame information
            st.subheader("Processed Sheets Overview")
            for sheet_name, df in output_dfs.items():
                st.write(f"Sheet: {sheet_name}")
                st.write(f"Columns: {df.columns.tolist()}")
                st.write(f"Shape: {df.shape}")
                st.write("---")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please ensure you've uploaded the correct files in the right order.")

if __name__ == "__main__":
    main()
