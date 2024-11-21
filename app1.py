import streamlit as st
import pandas as pd
import io
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

def generate_comprehensive_report(results, region, material):
    """Generate a comprehensive report based on analysis results"""
    report = f"# Comprehensive EBITDA Analysis Report\n\n"
    report += f"**Region:** {region}\n"
    report += f"**Material:** {material}\n\n"
    
    report += "## Effect Size Analysis\n"
    report += f"- **Cohen's d:** {results['cohens_d']:.3f} ({interpret_effect_size(results['cohens_d'])} effect)\n"
    report += f"- **Hedges' g:** {results['hedges_g']:.3f} (bias-corrected for small sample)\n"
    report += f"- **Probability of Superiority:** {results['probability_superiority']:.1%}\n"
    report += f"- **Non-overlap Percentage:** {results['nonoverlap_percentage']:.1f}%\n"
    report += f"- **Effect Size r:** {results['effect_size_r']:.3f}\n"
    report += f"- **VDA (Vargha-Delaney A):** {results['vda']:.3f} ({interpret_vda(results['vda'])} effect)\n\n"
    
    report += "## Statistical Significance\n"
    report += f"- **P-value:** {results['p_value']:.4f}\n"
    report += f"- **Mean Difference:** {results['mean_difference']:.2f}\n"
    report += f"- **95% Confidence Interval:** ({results['ci_lower']:.2f}, {results['ci_upper']:.2f})\n\n"
    
    # Recommendation framework
    confidence_level = "High" if results['p_value'] < 0.05 else "Low"
    effect_magnitude = interpret_effect_size(results['cohens_d'])
    practical_significance = results['nonoverlap_percentage'] > 55
    
    report += "## Recommendation\n"
    report += f"- **Statistical Confidence:** {confidence_level}\n"
    report += f"- **Effect Size:** {effect_magnitude}\n"
    report += f"- **Practical Significance:** {'Yes' if practical_significance else 'No'}\n\n"
    
    # Final recommendation
    if results['cohens_d'] > 0.5 and results['vda'] > 0.64:
        report += "### Final Recommendation\n"
        report += "**STRONG SUPPORT for transferring sales**\n\n"
        report += "The analysis provides robust evidence supporting the transfer of sales. The effect size is substantial, and the statistical confidence is high."
    elif results['cohens_d'] > 0.2 and results['vda'] > 0.56:
        report += "### Final Recommendation\n"
        report += "**MODERATE SUPPORT for transferring sales**\n\n"
        report += "The analysis suggests some potential in transferring sales, but the evidence is not as strong as in a high-support scenario."
    else:
        report += "### Final Recommendation\n"
        report += "**LIMITED SUPPORT for transferring sales**\n\n"
        report += "The current analysis does not provide strong evidence to support transferring sales. Further investigation may be needed."
    
    return report

def run_ebitda_analysis_tab():
    st.markdown('<h1 class="main-title">📊 EBITDA Transfer Analysis</h1>', unsafe_allow_html=True)
    
    # File upload for analysis
    uploaded_analysis_file = st.file_uploader(
        "Upload Merged Excel File for EBITDA Transfer Analysis", 
        type=['xlsx', 'xls']
    )

    # If a file is uploaded or merged dataframes exist in session state
    if uploaded_analysis_file or hasattr(st.session_state, 'final_non_total_df'):
        try:
            # Read the uploaded file or use session state dataframes
            if uploaded_analysis_file:
                non_total_df = pd.read_excel(uploaded_analysis_file, sheet_name='Non-Total')
            else:
                non_total_df = st.session_state.final_non_total_df

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
            
            # Prepare columns for EBITDA analysis
            trade_ebitda_cols = [col for col in specific_material_data.columns if 'Trade EBITDA' in col and 'Non-Trade EBITDA' not in col]
            non_trade_ebitda_cols = [col for col in specific_material_data.columns if 'Non-Trade EBITDA' in col]
            
            # Perform analysis for Trade EBITDA
            if len(trade_ebitda_cols) >= 2:
                # Extract trade EBITDA for the last two months
                trade_ebitda_values = specific_material_data[trade_ebitda_cols[-2:]].values[0]
                trade_analysis = analyze_ebitda_comprehensive(
                    trade_ebitda_values[:1], 
                    trade_ebitda_values[1:2]
                )
                
                # Generate trade EBITDA report
                st.subheader("Trade EBITDA Transfer Analysis")
                trade_report = generate_comprehensive_report(trade_analysis, selected_region, selected_material)
                st.markdown(trade_report)
            
            # Perform analysis for Non-Trade EBITDA
            if len(non_trade_ebitda_cols) >= 2:
                # Extract non-trade EBITDA for the last two months
                non_trade_ebitda_values = specific_material_data[non_trade_ebitda_cols[-2:]].values[0]
                non_trade_analysis = analyze_ebitda_comprehensive(
                    non_trade_ebitda_values[:1], 
                    non_trade_ebitda_values[1:2]
                )
                
                # Generate non-trade EBITDA report
                st.subheader("Non-Trade EBITDA Transfer Analysis")
                non_trade_report = generate_comprehensive_report(non_trade_analysis, selected_region, selected_material)
                st.markdown(non_trade_report)
            
            # Comparison if both trade and non-trade analyses are possible
            if len(trade_ebitda_cols) >= 2 and len(non_trade_ebitda_cols) >= 2:
                st.subheader("Comparative Insights")
                
                # Side-by-side comparison of effect sizes
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Trade EBITDA Cohen's d", f"{trade_analysis['cohens_d']:.3f}")
                    st.metric("Trade EBITDA VDA", f"{trade_analysis['vda']:.3f}")
                
                with col2:
                    st.metric("Non-Trade EBITDA Cohen's d", f"{non_trade_analysis['cohens_d']:.3f}")
                    st.metric("Non-Trade EBITDA VDA", f"{non_trade_analysis['vda']:.3f}")
                
                # Overall recommendation
                if trade_analysis['cohens_d'] > 0.5 and non_trade_analysis['cohens_d'] > 0.5:
                    st.success("**High Potential for Sales Transfer:** Both Trade and Non-Trade EBITDA show strong indicators for sales transfer.")
                elif trade_analysis['cohens_d'] > 0.2 and non_trade_analysis['cohens_d'] > 0.2:
                    st.warning("**Moderate Potential for Sales Transfer:** Trade and Non-Trade EBITDA show moderate indicators.")
                else:
                    st.error("**Limited Potential for Sales Transfer:** Neither Trade nor Non-Trade EBITDA show strong transfer potential.")

        except Exception as e:
            st.error(f"An error occurred during EBITDA transfer analysis: {e}")
    else:
        st.info("Please upload a merged Excel file for EBITDA transfer analysis.")

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
    with tab3:
        run_ebitda_analysis_tab()
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
