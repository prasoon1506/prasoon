import streamlit as st
import pandas as pd
import io

def preprocess_data(df):
    df = df.iloc[3:]
    df = df.reset_index(drop=True)
    df = df[~df.iloc[:, 0].str.contains("Zone", case=False, na=False)]
    
    region_indices = df[df.iloc[:, 0].str.contains("Region", case=False, na=False)].index
    rows_to_remove = []
    for index in region_indices:
        rows_to_remove.extend([index - 2, index - 1, index])    
    df = df.drop(rows_to_remove)
    df = df.reset_index(drop=True)
    
    first_column_values = df.iloc[:, 0].values
    current_fill_value = None
    for i, value in enumerate(first_column_values):
        if pd.notna(value):
            current_fill_value = value
        elif pd.isna(value) and current_fill_value is not None:
            df.iloc[i, 0] = current_fill_value
            
    all_india_df = df[df.iloc[:, 0].str.contains("All India", case=False, na=False)]
    if not all_india_df.empty:
        all_india_index = all_india_df.index[0]
        if all_india_index > 0:
            df = df.drop(all_india_index - 1)
            df = df.reset_index(drop=True)

    df = df[:-2]
    return df

def process_and_merge(df, file_type):
    # Define column names based on file type
    if file_type == "Oct-Sep":
        column_names = ["Region Name", "Material Name", 
                        "Oct Trade Quantity", "Sep Trade Quantity",
                        "Oct Trade EBITDA", "Sep Trade EBITDA", "Increase in Trade EBITDA",
                        "Oct Non-Trade Quantity", "Sep Non-Trade Quantity", 
                        "Oct Non-Trade EBITDA", "Sep Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Oct Total Quantity", "Sep Total Quantity", 
                        "Oct Total EBITDA", "Sep Total EBITDA", "Increase in Total EBITDA"]
    elif file_type == "Sep-Aug":
        column_names = ["Region Name", "Material Name", 
                        "Sep Trade Quantity", "Aug Trade Quantity",
                        "Sep Trade EBITDA", "Aug Trade EBITDA", "Increase in Trade EBITDA",
                        "Sep Non-Trade Quantity", "Aug Non-Trade Quantity", 
                        "Sep Non-Trade EBITDA", "Aug Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Sep Total Quantity", "Aug Total Quantity", 
                        "Sep Total EBITDA", "Aug Total EBITDA", "Increase in Total EBITDA"]
    elif file_type == "Aug-Jul":
        column_names = ["Region Name", "Material Name", 
                        "Aug Trade Quantity", "Jul Trade Quantity",
                        "Aug Trade EBITDA", "Jul Trade EBITDA", "Increase in Trade EBITDA",
                        "Aug Non-Trade Quantity", "Jul Non-Trade Quantity", 
                        "Aug Non-Trade EBITDA", "Jul Non-Trade EBITDA", "Increase in Non-Trade EBITDA", 
                        "Aug Total Quantity", "Jul Total Quantity", 
                        "Aug Total EBITDA", "Jul Total EBITDA", "Increase in Total EBITDA"]
    
    # Assign column names
    df.columns = column_names

    # Create separate DataFrames for Total and Non-Total rows
    total_mask = df.iloc[:, 0].str.contains("Total", case=False, na=False)
    df_total = df[total_mask]
    df_no_total = df[~total_mask]

    # Remove specific increase columns from Total DataFrame if present
    columns_to_remove = ["Increase in Trade EBITDA", "Increase in Non-Trade EBITDA", "Increase in Total EBITDA"]
    df_total = df_total.drop(columns=columns_to_remove, errors='ignore')
    df_no_total = df_no_total.drop(columns=columns_to_remove, errors='ignore')
    
    return df_no_total, df_total

def streamlit_data_merger():
    # Set page configuration
    st.set_page_config(
        page_title="Data Merger App", 
        page_icon=":bar_chart:", 
        layout="wide"
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload-section {
        background-color: #F0F4F8;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980B9;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<h1 class="main-title">📊 Data Merger Application</h1>', unsafe_allow_html=True)

    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    # Initialize session state for files and sheets
    if 'files' not in st.session_state:
        st.session_state.files = [None, None, None]
    if 'selected_sheets' not in st.session_state:
        st.session_state.selected_sheets = [None, None, None]

    # File upload and sheet selection
    file_types = ["Oct-Sep", "Sep-Aug", "Aug-Jul"]
    
    for i in range(3):
        st.subheader(f"Upload {file_types[i]} File")
        uploaded_file = st.file_uploader(
            f"Choose Excel file for {file_types[i]}", 
            type=['xlsx', 'xls'], 
            key=f"file_uploader_{i}"
        )
        
        if uploaded_file is not None:
            st.session_state.files[i] = uploaded_file
            
            # Read all sheets
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            
            # Sheet selection
            selected_sheet = st.selectbox(
                f"Select sheet for {file_types[i]}", 
                sheet_names, 
                key=f"sheet_selector_{i}"
            )
            st.session_state.selected_sheets[i] = selected_sheet

    st.markdown('</div>', unsafe_allow_html=True)

    # Merge button
    if st.button("Merge Data", key="merge_button"):
        # Validate file uploads and sheet selections
        if all(st.session_state.files) and all(st.session_state.selected_sheets):
            processed_dfs = []
            processed_total_dfs = []
            
            try:
                for i in range(3):
                    # Read specific sheet
                    df = pd.read_excel(
                        st.session_state.files[i], 
                        sheet_name=st.session_state.selected_sheets[i]
                    )
                    
                    # Preprocess and process
                    df = preprocess_data(df)
                    df_no_total, df_total = process_and_merge(df, file_types[i])
                    
                    processed_dfs.append(df_no_total)
                    processed_total_dfs.append(df_total)

                # Merge non-total DataFrames
                final_df = processed_dfs[0]
                for df in processed_dfs[1:]:
                    df = df[["Region Name", "Material Name"] + [col for col in df.columns if col not in final_df.columns]]
                    final_df = pd.merge(final_df, df, on=["Region Name", "Material Name"], how="left")
                
                # Merge total DataFrames
                final_total_df = processed_total_dfs[0]
                for df in processed_total_dfs[1:]:
                    df = df[["Region Name"] + [col for col in df.columns if col not in final_total_df.columns]]
                    final_total_df = pd.merge(final_total_df, df, on="Region Name", how="left")
                
                # Create Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    final_df.to_excel(writer, sheet_name='Non-Total', index=False)
                    final_total_df.to_excel(writer, sheet_name='Total', index=False)
                output.seek(0)
                
                # Download button
                st.download_button(
                    label="Download Merged Excel File",
                    data=output,
                    file_name='final_merged_data.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                
                # Display DataFrames
                st.success("Data merged successfully!")
                st.subheader("Non-Total DataFrame Preview")
                st.dataframe(final_df.head())
                st.subheader("Total DataFrame Preview")
                st.dataframe(final_total_df.head())
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload all three files and select sheets!")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_data_merger()
