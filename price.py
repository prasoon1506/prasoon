import streamlit as st
import pandas as pd
import io
import warnings
import plotly.express as fx
import plotly.graph_objs as go
import plotly.io as pio

def process_excel_file(uploaded_file, requires_editing):
    """
    Process the uploaded Excel file with advanced formatting
    """
    # Suppress warnings
    warnings.simplefilter("ignore")

    # Read the Excel file
    df = pd.read_excel(uploaded_file)

    # If editing is not required, return the original dataframe
    if not requires_editing:
        return df

    # Editing process starts here
    df = df.iloc[1:] 
    df = df.iloc[:, 1:]

    # Set headers
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header

    # Clean up dataframe
    df = df[~df.iloc[:, 1].str.contains('Date', na=False, case=False)]

    # Convert datetime
    datetime_series = pd.to_datetime(df.iloc[:, 1])  
    df.iloc[:, 1] = datetime_series.dt.strftime('%d-%b %Y')  

    # Remove null columns
    df = df.loc[:, df.columns.notnull()] 

    # Remove specific row
    df = df[df.iloc[:, 0] != "JKLC Price Tracker Mar'24 - till 03-12-24"]

    # Fill missing first column values
    mask = df.iloc[:, 0].notna()
    current_value = None
    for i in range(len(df)):     
        if mask.iloc[i]:         
            current_value = df.iloc[i, 0]     
        else:         
            if current_value is not None:             
                df.iloc[i, 0] = current_value 

    # Rename first column
    df = df.rename(columns={df.columns[0]: 'Region(District)'})
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def save_processed_dataframe(df):
    """
    Save processed dataframe to Excel with formatting
    """
    # Create a Pandas Excel writer using XlsxWriter as the engine
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write the dataframe to the Excel file
        df.to_excel(writer, sheet_name='Sheet1', index=False)

        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # Professional color palette
        dark_blue = '#2C3E50'
        white = '#FFFFFF'
        light_gray = '#F2F2F2'

        # Header format
        format_header = workbook.add_format({
            'bold': True, 
            'font_size': 14, 
            'bg_color': dark_blue,  
            'font_color': white,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'border_color': '#000000'
        })

        # General format
        format_general = workbook.add_format({
            'font_size': 12,
            'valign': 'vcenter',
            'align': 'center'
        })

        # Alternating row color format
        format_alternating = workbook.add_format({
            'font_size': 12,
            'bg_color': light_gray,
            'valign': 'vcenter',
            'align': 'center'
        })

        # Apply header format
        worksheet.set_row(0, 30, format_header)

        # Apply alternating row colors to data rows
        for row_num in range(1, len(df) + 1):
            if row_num % 2 == 0:
                worksheet.set_row(row_num, None, format_alternating)
            else:
                worksheet.set_row(row_num, None, format_general)

        # Autofit columns
        for col_num, col_name in enumerate(df.columns):
            # Calculate max width of column content
            max_len = max(
                df[col_name].astype(str).map(len).max(),
                len(str(col_name))
            )
            # Set column width with a little padding
            worksheet.set_column(col_num, col_num, max_len + 2, format_general)

        # Conditional formatting for 'MoM Change' column
        if 'MoM Change' in df.columns:
            mom_change_col_index = df.columns.get_loc('MoM Change')

            # Formats for conditional formatting
            format_negative = workbook.add_format({
                'bg_color': '#FFC7CE',
                'font_size': 12,
                'align': 'center',
                'valign': 'vcenter'
            })
            format_zero = workbook.add_format({
                'bg_color': '#D9D9D9',
                'font_size': 12,
                'align': 'center',
                'valign': 'vcenter'
            })
            format_positive = workbook.add_format({
                'bg_color': '#C6EFCE',
                'font_size': 12,
                'align': 'center',
                'valign': 'vcenter'
            })

            # Apply conditional formatting
            worksheet.conditional_format(1, mom_change_col_index, len(df), mom_change_col_index, {
                'type': 'cell', 
                'criteria': '<', 
                'value': 0, 
                'format': format_negative
            })
            worksheet.conditional_format(1, mom_change_col_index, len(df), mom_change_col_index, {
                'type': 'cell', 
                'criteria': '=', 
                'value': 0, 
                'format': format_zero
            })
            worksheet.conditional_format(1, mom_change_col_index, len(df), mom_change_col_index, {
                'type': 'cell', 
                'criteria': '>', 
                'value': 0, 
                'format': format_positive
            })

        # Close the writer
        writer.close()

    # Prepare the downloaded file
    output.seek(0)
    return output

def main():
    st.set_page_config(page_title="Price Tracker", layout="wide")
    # File Uploader
    uploaded_file = st.file_uploader("Please upload the Price Tracker file", type=['xlsx'])

    if uploaded_file is not None:
        # Ask if the file requires initial editing
        requires_editing = st.radio(
            "Does this file require initial editing?", 
            ["No", "Yes"]
        )

        # Process the file based on editing requirement
        try:
            # Process the file
            df = process_excel_file(uploaded_file, requires_editing == "Yes")
            
            # Create two columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Data Entry")
                # Ask if price has changed for any region
                price_changed = st.radio("Do you want to add new data?", ["No", "Yes"])
                
                if price_changed == "Yes":
                    # Get unique regions
                    if 'Region(District)' not in df.columns:
                        st.warning("No 'Region(District)' column found. Please check your file.")
                    else:
                        unique_regions = df['Region(District)'].unique()
                        
                        if len(unique_regions) == 0:
                            st.warning("No regions found in the dataframe.")
                        else:
                            # Allow multiple region selection
                            selected_regions = st.multiselect("Select Region(s)", unique_regions)
                            
                            # Container for data entry for multiple regions
                            data_entries = []
                            
                            for selected_region in selected_regions:
                                st.markdown(f"### Data Entry for {selected_region}")
                                
                                # Region-specific data entry
                                region_df = df[df['Region(District)'] == selected_region]
                                
                                # Date input
                                date_input = st.text_input(f"Enter Date for {selected_region} (format: DD-Mon YYYY, e.g., 01-Jan 2024)", key=f"date_{selected_region}")
                                
                                # Input for other columns
                                inv_input = st.number_input(f"Enter Inv. value for {selected_region}", value=0.0, format="%.2f", key=f"inv_{selected_region}")
                                rd_input = st.number_input(f"Enter RD value for {selected_region}", value=0.0, format="%.2f", key=f"rd_{selected_region}")
                                sts_input = st.number_input(f"Enter STS value for {selected_region}", value=0.0, format="%.2f", key=f"sts_{selected_region}")
                                reglr_input = st.number_input(f"Enter Reglr value for {selected_region}", value=0.0, format="%.2f", key=f"reglr_{selected_region}")
                                
                                # Calculate Net
                                net_input = inv_input - rd_input - sts_input - reglr_input
                                st.write(f"Calculated Net value for {selected_region}: {net_input}")
                                
                                # Calculate MoM Change
                                if 'Net' in region_df.columns and not region_df['Net'].empty:
                                    last_net_value = region_df['Net'].iloc[-1]
                                    mom_change = net_input - last_net_value
                                    st.write(f"Calculated MoM Change for {selected_region}: {mom_change}")
                                else:
                                    mom_change = 0
                                
                                # Remarks input
                                remarks_input = st.text_area(f"Enter Remarks for {selected_region} (Optional)", key=f"remarks_{selected_region}")
                                
                                # Prepare new row
                                new_row = {
                                    'Region(District)': selected_region,
                                    'Date': date_input,
                                    'Inv.': inv_input,
                                    'RD': rd_input,
                                    'STS': sts_input,
                                    'Reglr': reglr_input,
                                    'Net': net_input,
                                    'MoM Change': mom_change,
                                    'Remarks': remarks_input
                                }
                                
                                data_entries.append(new_row)
                                
                                st.markdown("---")
                            
                            # Button to add new rows
                            if st.button("Add New Rows"):
                                # Convert data entries to DataFrame
                                new_rows_df = pd.DataFrame(data_entries)
                                
                                # Append new rows to existing dataframe
                                df = pd.concat([df, new_rows_df], ignore_index=True)
                                
                                # Save processed dataframe
                                output = save_processed_dataframe(df)
                                
                                # Create download button
                                st.download_button(
                                    label="Download Processed Excel File",
                                    data=output,
                                    file_name='processed_price_tracker.xlsx',
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                )

            with col2:
                st.subheader("Region Analysis")
                
                # Check if dataframe has required columns
                if 'Region(District)' in df.columns and 'Date' in df.columns and 'Net' in df.columns:
                    # Region selection for analysis
                    unique_regions = df['Region(District)'].unique()
                    selected_region_analysis = st.selectbox("Select Region for Analysis", unique_regions)
                    
                    # Filter dataframe for selected region
                    region_analysis_df = df[df['Region(District)'] == selected_region_analysis]
                    
                    # 1. Price Change Count
                    price_change_count = len(region_analysis_df['Date'].unique())
                    st.metric("Total Price Changes", price_change_count)
                    
                    # Convert Date column to datetime
                    region_analysis_df['Date'] = pd.to_datetime(region_analysis_df['Date'], format='%d-%b %Y')
                    
                    # Sort by date
                    region_analysis_df = region_analysis_df.sort_values('Date')
                    
                    # Graph type selection
                    graph_type = st.selectbox("Select Graph Type", 
                        ['Net Value', 'Inv.', 'RD', 'STS', 'Reglr']
                    )
                    
                    # Create line graph based on selected type
                    fig = go.Figure()
                    
                    # Add line for selected graph type
                    fig.add_trace(go.Scatter(
                        x=region_analysis_df['Date'], 
                        y=region_analysis_df[graph_type], 
                        mode='lines+markers+text',
                        text=region_analysis_df[graph_type].round(2),
                        textposition='top center',
                        name=f'{graph_type} Value'
                    ))
                    
                    # Customize the layout
                    fig.update_layout(
                        title=f'{graph_type} Value Trend for {selected_region_analysis}',
                        xaxis_title='Date',
                        yaxis_title=f'{graph_type} Value',
                        height=400
                    )
                    
                    # Display the graph
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download graph options
                    graph_download_format = st.selectbox("Download Graph as", ['PNG', 'PDF'])
                    
                    if st.button("Download Graph"):
                        if graph_download_format == 'PNG':
                            img_bytes = pio.to_image(fig, format='png')
                            st.download_button(
                                label="Download Graph as PNG",
                                data=img_bytes,
                                file_name=f'{selected_region_analysis}_{graph_type}_trend.png',
                                mime='image/png'
                            )
                        else:
                            pdf_bytes = pio.to_image(fig, format='pdf')
                            st.download_button(
                                label="Download Graph as PDF",
                                data=pdf_bytes,
                                file_name=f'{selected_region_analysis}_{graph_type}_trend.pdf',
                                mime='application/pdf'
                            )
                    
                    # Display Remarks
                    st.markdown("### Remarks")
                    remarks_df = region_analysis_df[['Date', 'Remarks']].dropna(subset=['Remarks'])
                    
                    if not remarks_df.empty:
                        # Create a styled container for remarks
                        for _, row in remarks_df.iterrows():
                            st.markdown(f"""
                            <div style="background-color:#f0f2f6; 
                                        border-left: 5px solid #4a4a4a; 
                                        padding: 10px; 
                                        margin-bottom: 10px; 
                                        border-radius: 5px;">
                                <strong>{row['Date'].strftime('%d-%b %Y')}</strong>: 
                                {row['Remarks']}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No remarks found for this region.")
                else:
                    else:
                    st.warning("The uploaded file does not have the required columns for analysis.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
