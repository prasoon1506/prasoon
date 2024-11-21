import streamlit as st
import pandas as pd
import io
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t
def calculate_effect_sizes(data_high, data_low):
    """
    Calculate various effect size metrics for small sample comparison
    """
    # Convert to numpy arrays
    high_ebitda = np.array(data_high)
    low_ebitda = np.array(data_low)
    differences = high_ebitda - low_ebitda
    n = len(data_high)
    
    # 1. Cohen's d
    pooled_std = np.sqrt((np.var(high_ebitda) + np.var(low_ebitda)) / 2)
    cohens_d = (np.mean(high_ebitda) - np.mean(low_ebitda)) / pooled_std
    
    # 2. Hedges' g (bias-corrected for small samples)
    correction_factor = 1 - (3 / (4 * (2 * n - 2) - 1))
    hedges_g = cohens_d * correction_factor
    
    # 3. Probability of Superiority (Common Language Effect Size)
    ps = np.mean([1 if h > l else 0 for h in high_ebitda for l in low_ebitda])
    
    # 4. Non-overlap percentage
    nonoverlap = (stats.norm.cdf(abs(cohens_d)/np.sqrt(2)) * 100)
    
    # 5. Paired samples t-test effect size
    t_stat, p_value = stats.ttest_rel(high_ebitda, low_ebitda)
    dof = n - 1
    effect_size_r = np.sqrt(t_stat**2 / (t_stat**2 + dof))
    
    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'probability_superiority': ps,
        'nonoverlap_percentage': nonoverlap,
        'effect_size_r': effect_size_r,
        'p_value': p_value
    }

def analyze_ebitda_comprehensive(data_high, data_low, n_iterations=10000):
    """
    Comprehensive EBITDA analysis including effect sizes and robust metrics
    """
    effect_sizes = calculate_effect_sizes(data_high, data_low)
    
    # Calculate Cliff's Delta (non-parametric effect size)
    cliffs_delta = 2 * (effect_sizes['probability_superiority'] - 0.5)
    
    # Calculate VDA (Vargha and Delaney's A)
    vda = effect_sizes['probability_superiority']
    
    # Calculate confidence intervals for difference using t-distribution
    diff_mean = np.mean(np.array(data_high) - np.array(data_low))
    diff_std = np.std(np.array(data_high) - np.array(data_low), ddof=1)
    n = len(data_high)
    t_val = t.ppf(0.975, df=n-1)
    ci_margin = t_val * (diff_std / np.sqrt(n))
    
    return {
        **effect_sizes,
        'cliffs_delta': cliffs_delta,
        'vda': vda,
        'mean_difference': diff_mean,
        'ci_lower': diff_mean - ci_margin,
        'ci_upper': diff_mean + ci_margin
    }

def interpret_effect_size(d):
    """Interpret Cohen's d effect size"""
    if abs(d) < 0.2:
        return "negligible"
    elif abs(d) < 0.5:
        return "small"
    elif abs(d) < 0.8:
        return "medium"
    else:
        return "large"

def interpret_vda(vda):
    """Interpret Vargha and Delaney's A"""
    if vda < 0.56:
        return "negligible"
    elif vda < 0.64:
        return "small"
    elif vda < 0.71:
        return "medium"
    else:
        return "large"

def generate_comprehensive_report(results, analysis_type):
    """
    Generate a comprehensive report for Streamlit display
    """
    # Format results
    report = f"### {analysis_type} EBITDA Transfer Analysis\n\n"
    
    # Effect Size Section
    report += "#### Effect Size Metrics\n"
    report += f"- **Cohen's d:** {results['cohens_d']:.3f} ({interpret_effect_size(results['cohens_d'])} effect)\n"
    report += f"- **Hedges' g:** {results['hedges_g']:.3f} (bias-corrected)\n"
    report += f"- **Probability of Superiority:** {results['probability_superiority']:.1%}\n"
    report += f"- **Non-overlap Percentage:** {results['nonoverlap_percentage']:.1f}%\n"
    report += f"- **Effect Size r:** {results['effect_size_r']:.3f}\n"
    report += f"- **VDA (Vargha-Delaney A):** {results['vda']:.3f} ({interpret_vda(results['vda'])} effect)\n\n"
    
    # Statistical Significance Section
    report += "#### Statistical Significance\n"
    report += f"- **P-value:** {results['p_value']:.4f}\n"
    report += f"- **Mean Difference:** {results['mean_difference']:.2f}\n"
    report += f"- **95% Confidence Interval:** ({results['ci_lower']:.2f}, {results['ci_upper']:.2f})\n\n"
    
    # Recommendation Section
    report += "#### Recommendation\n"
    
    # Decision framework
    confidence_level = "High" if results['p_value'] < 0.05 else "Low"
    effect_magnitude = interpret_effect_size(results['cohens_d'])
    practical_significance = results['nonoverlap_percentage'] > 55
    
    # Determine recommendation
    if results['cohens_d'] > 0.5 and results['vda'] > 0.64:
        recommendation = "**STRONG SUPPORT for transferring sales**"
    elif results['cohens_d'] > 0.2 and results['vda'] > 0.56:
        recommendation = "**MODERATE SUPPORT for transferring sales**"
    else:
        recommendation = "**LIMITED SUPPORT for transferring sales**"
    
    report += f"- **Statistical Confidence:** {confidence_level}\n"
    report += f"- **Effect Size:** {effect_magnitude}\n"
    report += f"- **Practical Significance:** {'Yes' if practical_significance else 'No'}\n\n"
    report += f"{recommendation}\n"
    
    return report
def ebitda_transfer_analysis_section(total_df, non_total_df):
    """
    Comprehensive EBITDA Transfer Analysis section
    """
    st.markdown("## 📊 EBITDA Transfer Analysis")
    
    # First level selection: Total or Non-Total DataFrame
    df_selection = st.selectbox("Select Analysis Level", 
                                ["Total Level Analysis", "Material Level Analysis"])
    
    if df_selection == "Total Level Analysis":
        # Validate Total DataFrame
        if total_df is None or total_df.empty:
            st.warning("No Total data available for analysis.")
            return
        
        # Get unique regions for total
        unique_regions = sorted(total_df['Region Name'].unique())
        
        # Region selection
        selected_region = st.selectbox("Select Region", unique_regions)
        
        # Filter for specific region
        region_df = total_df[total_df['Region Name'] == selected_region]
        
        # Prepare Trade and Non-Trade EBITDA columns
        trade_ebitda_cols = [col for col in region_df.columns if 'Trade EBITDA' in col]
        non_trade_ebitda_cols = [col for col in region_df.columns if 'Non-Trade EBITDA' in col]
        
        # Ensure we have enough columns
        if len(trade_ebitda_cols) >= 4 and len(non_trade_ebitda_cols) >= 4:
            # Convert and prepare data
            trade_data = pd.to_numeric(region_df[trade_ebitda_cols], errors='coerce')
            non_trade_data = pd.to_numeric(region_df[non_trade_ebitda_cols], errors='coerce')
            
            # Transpose to get analysis-friendly format
            trade_data_values = trade_data.values.flatten()
            non_trade_data_values = non_trade_data.values.flatten()
            
            # Remove NaNs
            valid_indices = ~(np.isnan(trade_data_values) | np.isnan(non_trade_data_values))
            trade_data_clean = trade_data_values[valid_indices]
            non_trade_data_clean = non_trade_data_values[valid_indices]
            
            # Perform analysis if we have valid data
            if len(trade_data_clean) > 0 and len(non_trade_data_clean) > 0:
                # Comprehensive analysis of transfer potential
                results = analyze_ebitda_comprehensive(trade_data_clean, non_trade_data_clean)
                
                # Generate report
                report = generate_comprehensive_report(results, "Trade to Non-Trade Transfer")
                st.markdown(report)
                
                # Display analyzed data
                st.subheader("Analyzed EBITDA Data")
                analysis_data = pd.DataFrame({
                    'Trade EBITDA': trade_data_clean,
                    'Non-Trade EBITDA': non_trade_data_clean
                })
                st.dataframe(analysis_data)
            else:
                st.warning("Insufficient valid numerical data for analysis.")
        else:
            st.warning("Not enough EBITDA data columns for comprehensive analysis.")
    
    else:  # Material Level Analysis
        # Validate Non-Total DataFrame
        if non_total_df is None or non_total_df.empty:
            st.warning("No Material data available for analysis.")
            return
        
        # Get unique regions
        unique_regions = sorted(non_total_df['Region Name'].unique())
        
        # Region selection
        selected_region = st.selectbox("Select Region", unique_regions)
        
        # Filter for specific region
        region_df = non_total_df[non_total_df['Region Name'] == selected_region]
        
        # Get unique materials for that region
        unique_materials = sorted(region_df['Material Name'].unique())
        
        # Material selection
        selected_material = st.selectbox("Select Material", unique_materials)
        
        # Filter for specific material
        material_df = region_df[region_df['Material Name'] == selected_material]
        
        # Prepare Trade and Non-Trade EBITDA columns
        trade_ebitda_cols = [col for col in material_df.columns if 'Trade EBITDA' in col]
        non_trade_ebitda_cols = [col for col in material_df.columns if 'Non-Trade EBITDA' in col]
        
        # Ensure we have enough columns
        if len(trade_ebitda_cols) >= 4 and len(non_trade_ebitda_cols) >= 4:
            # Convert and prepare data
            trade_data = pd.to_numeric(material_df[trade_ebitda_cols], errors='coerce')
            non_trade_data = pd.to_numeric(material_df[non_trade_ebitda_cols], errors='coerce')
            
            # Transpose to get analysis-friendly format
            trade_data_values = trade_data.values.flatten()
            non_trade_data_values = non_trade_data.values.flatten()
            
            # Remove NaNs
            valid_indices = ~(np.isnan(trade_data_values) | np.isnan(non_trade_data_values))
            trade_data_clean = trade_data_values[valid_indices]
            non_trade_data_clean = non_trade_data_values[valid_indices]
            
            # Perform analysis if we have valid data
            if len(trade_data_clean) > 0 and len(non_trade_data_clean) > 0:
                # Comprehensive analysis of transfer potential
                results = analyze_ebitda_comprehensive(trade_data_clean, non_trade_data_clean)
                
                # Generate report
                report = generate_comprehensive_report(results, f"Trade to Non-Trade Transfer for {selected_material}")
                st.markdown(report)
                
                # Display analyzed data
                st.subheader("Analyzed EBITDA Data")
                analysis_data = pd.DataFrame({
                    'Trade EBITDA': trade_data_clean,
                    'Non-Trade EBITDA': non_trade_data_clean
                })
                st.dataframe(analysis_data)
            else:
                st.warning("Insufficient valid numerical data for analysis.")
        else:
            st.warning("Not enough EBITDA data columns for comprehensive analysis.")
def add_ebitda_transfer_analysis_tab(non_total_df, total_df):
    """
    Add EBITDA Transfer Analysis as a new tab or section
    """
    ebitda_transfer_analysis_section(non_total_df)

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
        page_title="Data Merger & Analysis App", 
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

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Data Merger", "Data Analysis","EBITDA Transfer Analysis"])
    with tab1:
        # Title
        st.markdown('<h1 class="main-title">📊 Data Merger</h1>', unsafe_allow_html=True)

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
                    
                    # Store merged dataframes in session state
                    st.session_state.final_non_total_df = final_df
                    st.session_state.final_total_df = final_total_df
                    
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
    with tab3:
        st.markdown('<h1 class="main-title">💹 EBITDA Transfer Analysis</h1>', unsafe_allow_html=True)
        
        # Check if merged data exists in session state
        if hasattr(st.session_state, 'final_total_df') and hasattr(st.session_state, 'final_non_total_df'):
            # Call EBITDA Transfer Analysis section with both DataFrames
            ebitda_transfer_analysis_section(
                st.session_state.final_total_df, 
                st.session_state.final_non_total_df
            )
        else:
            # Option to upload a file
            uploaded_analysis_file = st.file_uploader(
                "Upload Merged Excel File for EBITDA Transfer Analysis", 
                type=['xlsx', 'xls']
            )
            
            if uploaded_analysis_file:
                try:
                    # Read both sheets
                    total_df = pd.read_excel(uploaded_analysis_file, sheet_name='Total')
                    non_total_df = pd.read_excel(uploaded_analysis_file, sheet_name='Non-Total')
                    
                    # Call EBITDA Transfer Analysis section
                    ebitda_transfer_analysis_section(total_df, non_total_df)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    import traceback
                    st.error(traceback.format_exc())
            else:
                st.info("Please upload a merged Excel file or complete the data merger first.")
    with tab2:
        st.markdown('<h1 class="main-title">📈 Data Analysis</h1>', unsafe_allow_html=True)
        
        # File upload for analysis
        uploaded_analysis_file = st.file_uploader(
            "Upload Merged Excel File for Analysis", 
            type=['xlsx', 'xls']
        )

        # If a file is uploaded or merged dataframes exist in session state
        if uploaded_analysis_file or hasattr(st.session_state, 'final_non_total_df'):
            try:
                # Read the uploaded file or use session state dataframes
                if uploaded_analysis_file:
                    non_total_df = pd.read_excel(uploaded_analysis_file, sheet_name='Non-Total')
                    total_df = pd.read_excel(uploaded_analysis_file, sheet_name='Total')
                else:
                    non_total_df = st.session_state.final_non_total_df
                    total_df = st.session_state.final_total_df

                # Categorize analysis type
                analysis_type = st.selectbox(
                    "Select Analysis Type", 
                    ["Material-wise Analysis", "Total Analysis"]
                )

                if analysis_type == "Material-wise Analysis":
                    # Get unique regions and materials
                    unique_regions = sorted(non_total_df['Region Name'].unique())
                    
                    # Region selection
                    selected_region = st.selectbox("Select Region", unique_regions)
                    
                    # Filter DataFrame by region
                    region_df = non_total_df[non_total_df['Region Name'] == selected_region]
                    
                    # Get unique materials for that region
                    unique_materials = sorted(region_df['Material Name'].unique())
                    
                    # Material selection
                    selected_material = st.selectbox("Select Material", unique_materials)
                    
                    # Filter for specific region and material
                    specific_material_data = region_df[region_df['Material Name'] == selected_material]
                    
                    # Prepare columns for display
                    trade_quantity_cols = [col for col in specific_material_data.columns if 'Trade Quantity' in col and 'Non-Trade Quantity' not in col]
                    trade_ebitda_cols = [col for col in specific_material_data.columns if 'Trade EBITDA' in col and 'Non-Trade EBITDA' not in col]
                    non_trade_quantity_cols = [col for col in specific_material_data.columns if 'Non-Trade Quantity' in col]
                    non_trade_ebitda_cols = [col for col in specific_material_data.columns if 'Non-Trade EBITDA' in col]
                    
                    # Display results
                    st.subheader(f"Analysis for {selected_material} in {selected_region}")
                    
                    # Trade Quantity
                    st.markdown("**Trade Quantity**")
                    trade_quantity_data = specific_material_data[trade_quantity_cols].T
                    trade_quantity_data.columns = ['Value']
                    trade_quantity_data.index.name = 'Month'
                    st.dataframe(trade_quantity_data)
                    
                    # Trade EBITDA
                    st.markdown("**Trade EBITDA**")
                    trade_ebitda_data = specific_material_data[trade_ebitda_cols].T
                    trade_ebitda_data.columns = ['Value']
                    trade_ebitda_data.index.name = 'Month'
                    st.dataframe(trade_ebitda_data)
                    
                    # Non-Trade Quantity
                    st.markdown("**Non-Trade Quantity**")
                    non_trade_quantity_data = specific_material_data[non_trade_quantity_cols].T
                    non_trade_quantity_data.columns = ['Value']
                    non_trade_quantity_data.index.name = 'Month'
                    st.dataframe(non_trade_quantity_data)
                    
                    # Non-Trade EBITDA
                    st.markdown("**Non-Trade EBITDA**")
                    non_trade_ebitda_data = specific_material_data[non_trade_ebitda_cols].T
                    non_trade_ebitda_data.columns = ['Value']
                    non_trade_ebitda_data.index.name = 'Month'
                    st.dataframe(non_trade_ebitda_data)

                else:  
                    # Get unique regions for total
                    unique_total_regions = sorted(total_df['Region Name'].unique())
                    
                    # Region selection for total
                    selected_total_region = st.selectbox("Select Region", unique_total_regions)
                    
                    # Filter DataFrame by region
                    specific_total_region_data = total_df[total_df['Region Name'] == selected_total_region]
                    # Prepare columns for display
                    trade_quantity_cols = [col for col in specific_total_region_data.columns 
                                           if 'Trade Quantity' in col and 'Non-Trade Quantity' not in col]
                    trade_ebitda_cols = [col for col in specific_total_region_data.columns 
                                         if 'Trade EBITDA' in col and 'Non-Trade EBITDA' not in col]
                    non_trade_quantity_cols = [col for col in specific_total_region_data.columns 
                                               if 'Non-Trade Quantity' in col]
                    non_trade_ebitda_cols = [col for col in specific_total_region_data.columns 
                                             if 'Non-Trade EBITDA' in col]
                    
                    # Display results
                    st.subheader(f"Total Analysis for {selected_total_region}")
                    
                    # Trade Quantity
                    st.markdown("**Total Trade Quantity**")
                    trade_quantity_data = specific_total_region_data[trade_quantity_cols].T
                    trade_quantity_data.columns = ['Value']
                    trade_quantity_data.index.name = 'Month'
                    st.dataframe(trade_quantity_data)
                    
                    # Trade EBITDA
                    st.markdown("**Total Trade EBITDA**")
                    trade_ebitda_data = specific_total_region_data[trade_ebitda_cols].T
                    trade_ebitda_data.columns = ['Value']
                    trade_ebitda_data.index.name = 'Month'
                    st.dataframe(trade_ebitda_data)
                    
                    # Non-Trade Quantity
                    st.markdown("**Total Non-Trade Quantity**")
                    non_trade_quantity_data = specific_total_region_data[non_trade_quantity_cols].T
                    non_trade_quantity_data.columns = ['Value']
                    non_trade_quantity_data.index.name = 'Month'
                    st.dataframe(non_trade_quantity_data)
                    
                    # Non-Trade EBITDA
                    st.markdown("**Total Non-Trade EBITDA**")
                    non_trade_ebitda_data = specific_total_region_data[non_trade_ebitda_cols].T
                    non_trade_ebitda_data.columns = ['Value']
                    non_trade_ebitda_data.index.name = 'Month'
                    st.dataframe(non_trade_ebitda_data)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
        else:
            st.info("Please upload a merged Excel file for analysis.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_data_merger()