import streamlit as st
import pandas as pd
import io
import warnings
import plotly.express as fx
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime as dt
def convert_dataframe_to_pdf(df, filename):
    """
    Convert DataFrame to PDF with headers on each page
    """
    # Create a buffer
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Prepare styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    
    # Prepare data for PDF
    data = [df.columns.tolist()]  # Header row
    for _, row in df.iterrows():
        data.append([str(val) for val in row.tolist()])
    
    # Create table
    table = Table(data)
    
    # Style the table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    
    # Prepare PDF content
    content = []
    content.append(table)
    
    # Write PDF
    doc.build(content)
    
    # Get PDF bytes
    buffer.seek(0)
    return buffer
def save_processed_dataframe(df, start_date=None, download_format='xlsx'):

    if 'processed_dataframe' in st.session_state:
        df = st.session_state['processed_dataframe']
    df_to_save = df.copy()
    # Ensure Date column is datetime
    if 'Date' in df_to_save.columns:
        df_to_save['Date'] = pd.to_datetime(df_to_save['Date'], format='%d-%b %Y')
        
        # Filter by start date if specified
        if start_date:
            df_to_save = df_to_save[df_to_save['Date'] >= start_date]
            df_to_save['Date'] = df_to_save['Date'].dt.strftime('%d-%b %Y')
    
    # Create output based on format
    output = io.BytesIO()
    
    if download_format == 'xlsx':
        # Excel saving logic
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_to_save.to_excel(writer, sheet_name='Sheet1', index=False)
            
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
            for row_num in range(1, len(df_to_save) + 1):
                if row_num % 2 == 0:
                    worksheet.set_row(row_num, None, format_alternating)
                else:
                    worksheet.set_row(row_num, None, format_general)

            # Autofit columns
            for col_num, col_name in enumerate(df_to_save.columns):
                # Calculate max width of column content
                max_len = max(
                    df_to_save[col_name].astype(str).map(len).max(),
                    len(str(col_name))
                )
                # Set column width with a little padding
                worksheet.set_column(col_num, col_num, max_len + 2, format_general)

            # Conditional formatting for 'MoM Change' column
            if 'MoM Change' in df_to_save.columns:
                mom_change_col_index = df_to_save.columns.get_loc('MoM Change')

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
                worksheet.conditional_format(1, mom_change_col_index, len(df_to_save), mom_change_col_index, {
                    'type': 'cell', 
                    'criteria': '<', 
                    'value': 0, 
                    'format': format_negative
                })
                worksheet.conditional_format(1, mom_change_col_index, len(df_to_save), mom_change_col_index, {
                    'type': 'cell', 
                    'criteria': '=', 
                    'value': 0, 
                    'format': format_zero
                })
                worksheet.conditional_format(1, mom_change_col_index, len(df_to_save), mom_change_col_index, {
                    'type': 'cell', 
                    'criteria': '>', 
                    'value': 0, 
                    'format': format_positive
                })

            # Close the writer
            writer.close()
    elif download_format == 'pdf':
        output = convert_dataframe_to_pdf(df_to_save, 'processed_price_tracker.pdf')
    
    # Prepare the downloaded file
    output.seek(0)
    return output
def parse_date(date_str):
    """
    Flexible date parsing function to handle multiple date formats
    """
    try:
        # Try multiple common formats
        date_formats = [
            '%d-%b %Y',    # 31-Mar 2024
            '%d-%b-%Y',    # 31-Mar-2024
            '%d-%B %Y',    # 31-March 2024
            '%Y-%m-%d',    # 2024-03-31
            '%m/%d/%Y',    # 03/31/2024
            '%d/%m/%Y',    # 31/03/2024
        ]
        
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        
        # If no format matches, try parse with mixed format
        return pd.to_datetime(date_str, format='mixed', dayfirst=True)
    
    except Exception as e:
        st.warning(f"Could not parse date: {date_str}. Error: {e}")
        return pd.NaT

def process_excel_file(uploaded_file, requires_editing):
    """
    Process the uploaded Excel file with advanced formatting and flexible date parsing
    """
    # Suppress warnings
    warnings.simplefilter("ignore")

    # Read the Excel file
    df = pd.read_excel(uploaded_file)

    # If editing is not required, return the original dataframe
    if not requires_editing:
        # Ensure dates are parsed correctly
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(parse_date)
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

    # Convert datetime with flexible parsing
    df.iloc[:, 1] = df.iloc[:, 1].apply(parse_date)
    #df.iloc[:, 1] = df.iloc[:, 1].dt.strftime('%d-%b %Y')  

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
def main():
    # Set page configuration
    st.set_page_config(page_title="Price Tracker", layout="wide", page_icon="💰")
    
    # Title and description
    st.title("📊 Price Tracker Analysis Tool")
    st.markdown("""
    ### Welcome to the Price Tracker Analysis Tool
    
    **Instructions:**
    1. Upload your Excel price tracking file
    2. Choose whether the file needs initial editing
    3. Add new data, analyze regions, and download processed files
    """)
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Please upload the Price Tracker file", 
        type=['xlsx'], 
        help="Upload an Excel file containing price tracking data"
    )

    if uploaded_file is not None:
        # Ask if the file requires initial editing
        requires_editing = st.radio(
            "Does this file require initial editing?", 
            ["No", "Yes"],
            help="Select 'Yes' if the uploaded file needs preprocessing"
        )

        # Process the file based on editing requirement
        try:
            # Process the file
            df = process_excel_file(uploaded_file, requires_editing == "Yes")
            
            # Validate DataFrame
            required_columns = ['Region(District)', 'Date', 'Inv.', 'RD', 'STS', 'Reglr']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()
            
            # Create two columns for layout
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔄 Data Entry")
                
                # Price change option
                price_changed = st.radio("Do you want to add new data?", ["No", "Yes"])
                
                if price_changed == "Yes":
                    # Get unique regions
                    unique_regions = df['Region(District)'].unique()
                    
                    if len(unique_regions) == 0:
                        st.warning("No regions found in the dataframe.")
                    else:
                        # Allow multiple region selection
                        selected_regions = st.multiselect("Select Region(s)", unique_regions)
                        
                        # Container for data entries
                        data_entries = []
                        
                        for selected_region in selected_regions:
                            st.markdown(f"### Data Entry for {selected_region}")
                            
                            # Region-specific data entry
                            region_df = df[df['Region(District)'] == selected_region]
                            from datetime import datetime
                            date_input = st.text_input(f"Enter Date for {selected_region}", value=datetime.now().strftime("%d-%b %Y"),placeholder="DD-Mon YYYY, e.g., 01-Jan 2024",key=f"date_{selected_region}")
                            
                            # Input for financial values
                            inv_input = st.number_input(
                                f"Enter Inv. value for {selected_region}", 
                                value=0.0, 
                                format="%.2f", 
                                key=f"inv_{selected_region}"
                            )
                            rd_input = st.number_input(
                                f"Enter RD value for {selected_region}", 
                                value=0.0, 
                                format="%.2f", 
                                key=f"rd_{selected_region}"
                            )
                            sts_input = st.number_input(
                                f"Enter STS value for {selected_region}", 
                                value=0.0, 
                                format="%.2f", 
                                key=f"sts_{selected_region}"
                            )
                            reglr_input = st.number_input(
                                f"Enter Reglr value for {selected_region}", 
                                value=0.0, 
                                format="%.2f", 
                                key=f"reglr_{selected_region}"
                            )
                            
                            # Calculate Net
                            net_input = inv_input - rd_input - sts_input - reglr_input
                            st.write(f"Calculated Net value for {selected_region}: {net_input}")
                            
                            # Calculate MoM Change
                            last_net_value = region_df['Net'].iloc[-1] if 'Net' in region_df.columns and not region_df['Net'].empty else 0
                            mom_change = net_input - last_net_value
                            st.write(f"Calculated MoM Change for {selected_region}: {mom_change}")
                            
                            # Remarks input
                            remarks_input = st.text_area(
                                f"Enter Remarks for {selected_region} (Optional)", 
                                key=f"remarks_{selected_region}"
                            )
                            
                            # Prepare new row
                            new_row = {
                                'Region(District)': selected_region,
                                'Date': parse_date(date_input).strftime('%d-%b %Y'),
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
                        if st.button("Add New Rows to Dataframe"):
                            # Ensure data_entries is not empty
                            if not data_entries:
                                st.warning("No new entries to add.")
                                return
                            
                            # Prepare a copy of the original dataframe to modify
                            updated_df = df.copy()
                            
                            # Prepare new rows DataFrame with correct columns
                            new_rows_df = pd.DataFrame(data_entries)
                            for col in df.columns:
                                if col not in new_rows_df.columns:
                                    new_rows_df[col] = None
                            new_rows_df = new_rows_df.reindex(columns=df.columns)
                            
                            # Process each unique region
                            for region in new_rows_df['Region(District)'].unique():
                                # Filter new rows for this specific region
                                region_new_rows = new_rows_df[new_rows_df['Region(District)'] == region]
                                
                                # Find indices of existing rows for this region
                                region_existing_indices = updated_df[updated_df['Region(District)'] == region].index
                                
                                if not region_existing_indices.empty:
                                    # Get the last index for this region
                                    last_region_index = region_existing_indices[-1]
                                    
                                    # Prepare to insert new rows
                                    before_region = updated_df.iloc[:last_region_index+1]
                                    after_region = updated_df.iloc[last_region_index+1:]
                                    
                                    # Combine dataframes
                                    updated_df = pd.concat([
                                        before_region, 
                                        region_new_rows, 
                                        after_region
                                    ]).reset_index(drop=True)
                                else:
                                    # If no existing rows, append to the end
                                    updated_df = pd.concat([updated_df, region_new_rows]).reset_index(drop=True)
                            
                            # Update the dataframe
                            df = updated_df
                            st.session_state['processed_dataframe'] = df
                            st.success(f"{len(data_entries)} new rows added successfully!")
            with col2:
                st.subheader("📈 Region Analysis")
                unique_regions = df['Region(District)'].unique()
                selected_region_analysis = st.selectbox("Select Region for Analysis", unique_regions,key="region")
                region_analysis_df = df[df['Region(District)'] == selected_region_analysis]
                region_analysis_df['Date'] = pd.to_datetime(region_analysis_df['Date'], format='%d-%b %Y')
                current_month = dt.now().month
                current_year = dt.now().year
                last_month = current_month - 1 if current_month > 1 else 12
                last_month_year = current_year if current_month > 1 else current_year - 1
                last_month_data = region_analysis_df[(region_analysis_df['Date'].dt.month == last_month) & (region_analysis_df['Date'].dt.year == last_month_year)]
                current_month_data = region_analysis_df[(region_analysis_df['Date'].dt.month == current_month) & (region_analysis_df['Date'].dt.year == current_year)]
                display_columns = ['Date', 'Inv.', 'RD', 'STS', 'Reglr', 'Net', 'MoM Change']
                st.markdown(f"### Monthly Data for {selected_region_analysis}")
                st.markdown("#### Last Month Data")
                if not last_month_data.empty:
                      last_month_display = last_month_data[display_columns].copy()
                      last_month_display['Date'] = last_month_display['Date'].dt.strftime('%d-%b %Y')
                      last_month_display.set_index('Date', inplace=True)
                      last_month_display['Inv.']= last_month_display['Inv.'].abs().round(0).astype(int)
                      last_month_display['RD'] = last_month_display['RD'].abs().round(0).astype(int)
                      last_month_display['STS'] = last_month_display['STS'].abs().round(0).astype(int)
                      last_month_display['Reglr'] = last_month_display['Reglr'].abs().round(0).astype(int)
                      last_month_display['Net'] = last_month_display['Net'].abs().round(0).astype(int)
                      last_month_display['MoM Change'] = last_month_display['MoM Change'].abs().round(0).astype(int)
                      st.dataframe(last_month_display.style.background_gradient(cmap='Blues'), use_container_width=True)
                      col_last_1, col_last_2 = st.columns(2)
                      with col_last_1:
                       st.metric(f"Total No. of Price Change in (Last Month)", len(last_month_data))
                      with col_last_2:
                       st.metric("Total Change in NOD(Last Month)", last_month_data['MoM Change'].sum())
                else:
                     st.info(f"No data found for last month in {selected_region_analysis}")
                st.markdown("#### Current Month Data")
                if not current_month_data.empty:
                     current_month_display = current_month_data[display_columns].copy()
                     current_month_display['Date'] = current_month_display['Date'].dt.strftime('%d-%b %Y')
                     current_month_display.set_index('Date', inplace=True)
                     current_month_display['Inv.']= current_month_display['Inv.'].abs().round(0).astype(int)
                     current_month_display['RD'] = current_month_display['RD'].abs().round(0).astype(int)
                     current_month_display['STS'] = current_month_display['STS'].abs().round(0).astype(int)
                     current_month_display['Reglr'] = current_month_display['Reglr'].abs().round(0).astype(int)
                     current_month_display['Net'] = current_month_display['Net'].abs().round(0).astype(int)
                     current_month_display['MoM Change'] = current_month_display['MoM Change'].abs().round(0).astype(int)
                     st.dataframe(current_month_display.style.background_gradient(cmap='Blues'), use_container_width=True)
                     col_curr_1, col_curr_2 = st.columns(2)
                     with col_curr_1:
                        st.metric("Total No. of Price Change in (Current Month)", len(current_month_data))
                     with col_curr_2:
                         st.metric("Total Change in NOD(Current Month)", current_month_data['MoM Change'].sum())
                else:
                      st.info(f"No data found for current month in {selected_region_analysis}")
                # Region selection for analysis
                unique_regions = df['Region(District)'].unique()
                selected_region_analysis = st.selectbox("Select Region for Analysis", unique_regions)
                
                # Filter dataframe for selected region
                region_analysis_df = df[df['Region(District)'] == selected_region_analysis]
                
                # Metrics
                col_metrics_1, col_metrics_2 = st.columns(2)
                with col_metrics_1:
                    st.metric("Total Price Changes", len(region_analysis_df))
                st.markdown("### Graph Date Range")
                col_start_month, col_start_year = st.columns(2)
                with col_start_month:
                  start_month = st.selectbox("Select Start Month", ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December'],index=8)
                with col_start_year:
                  start_year = st.number_input("Select Start Year", min_value=2000, max_value=2030, value=2024)
                start_date = pd.to_datetime(f'01-{start_month[:3].lower()} {start_year}', format='%d-%b %Y')
                region_analysis_df = df[df['Region(District)'] == selected_region_analysis]
                region_analysis_df['Date'] = pd.to_datetime(region_analysis_df['Date'], format='%d-%b %Y')
                filtered_df = region_analysis_df[region_analysis_df['Date'] >= start_date].copy()
                if filtered_df.empty:
                    st.warning(f"No data available for {selected_region_analysis} from {start_month} {start_year}")
                else:
                    graph_type = st.selectbox("Select Metric for Analysis", ['Net', 'Inv.', 'RD', 'STS', 'Reglr', 'MoM Change'])
                filtered_df = filtered_df.sort_values('Date')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df['Date'],y=filtered_df[graph_type], mode='lines+markers+text',text=filtered_df[graph_type].round(2),textposition='top center',name=f'{graph_type} Value',line=dict(color='#1E90FF',width=3),marker=dict(size=10,color='#4169E1',symbol='circle',line=dict(color='#FFFFFF',width=2)),hovertemplate=('<b>Date</b>: %{x|%d %B %Y}<br>' +f'<b>{graph_type}</b>: %{{y:.2f}}<br>' +'<extra></extra>')))
                fig.update_layout(
                    title=f'{graph_type} Value Trend for {selected_region_analysis}',
                    xaxis_title='Date',
                    yaxis_title=f'{graph_type} Value',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
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
                remarks_df = remarks_df.sort_values('Date', ascending=False)
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
            
            # Global Download Section
            st.markdown("## 📥 Download Options")
            
            download_options = st.radio("Download File From:", 
                ["Entire Dataframe", "Specific Month"], 
                horizontal=True
            )
            
            start_date = None
            if download_options == "Specific Month":
                col1, col2 = st.columns(2)
                with col1:
                    month_input = st.selectbox("Select Month", 
                        ['January', 'February', 'March', 'April', 'May', 'June', 
                         'July', 'August', 'September', 'October', 'November', 'December']
                    )
                with col2:
                    year_input = st.number_input("Select Year", 
                        min_value=2000, max_value=2030, value=2024
                    )
                start_date = pd.to_datetime(f'01-{month_input[:3].lower()} {year_input}', format='%d-%b %Y')
            
            download_format = st.selectbox("Select Download Format", ['Excel (.xlsx)', 'PDF (.pdf)'])
            format_map = {'Excel (.xlsx)': 'xlsx', 'PDF (.pdf)': 'pdf'}
            selected_format = format_map[download_format]
            
            if st.button("Download Processed File"):
                try:
                    output = save_processed_dataframe(df, start_date, selected_format)
                    st.download_button(
                        label=f"Click to Download {download_format}",
                        data=output,
                        file_name=f'processed_price_tracker.{selected_format}',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if selected_format == 'xlsx' else 'application/pdf'
                    )
                except Exception as e:
                    st.error(f"Error during download: {e}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
