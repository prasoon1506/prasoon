import streamlit as st
import pandas as pd
import io
import warnings
import plotly.express as fx
import plotly.graph_objs as go
import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

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
    """
    Save processed dataframe with options for start date and format
    """
    # Filter dataframe by start date if specified
    if start_date:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b %Y')
        df = df[df['Date'] >= start_date]
        df['Date'] = df['Date'].dt.strftime('%d-%b %Y')
    
    # Create output based on format
    output = io.BytesIO()
    
    if download_format == 'xlsx':
        # Excel saving logic
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
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
    elif download_format == 'pdf':
        output = convert_dataframe_to_pdf(df, 'processed_price_tracker.pdf')
    
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
    df.iloc[:, 1] = df.iloc[:, 1].dt.strftime('%d-%b %Y')  

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
                            
                            # Date input
                            date_input = st.text_input(
                                f"Enter Date for {selected_region}", 
                                placeholder="DD-Mon YYYY, e.g., 01-Jan 2024",
                                key=f"date_{selected_region}"
                            )
                            
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
                            # Append new entries to dataframe
                            for entry in data_entries:
                                df = df.append(entry, ignore_index=True)
                            
                            st.success(f"{len(data_entries)} new rows added successfully!")

            with col2:
                st.subheader("📈 Region Analysis")
                
                # Region selection for analysis
                unique_regions = df['Region(District)'].unique()
                selected_region_analysis = st.selectbox("Select Region for Analysis", unique_regions)
                
                # Filter dataframe for selected region
                region_analysis_df = df[df['Region(District)'] == selected_region_analysis]
                
                # Metrics
                col_metrics_1, col_metrics_2 = st.columns(2)
                with col_metrics_1:
                    st.metric("Total Price Changes", len(region_analysis_df))
                
                # Visualization options
                graph_type = st.selectbox("Select Graph Type", 
                    ['Net', 'Inv.', 'RD', 'STS', 'Reglr']
                )
                
                # Create interactive graph
                fig = go.Figure()
                region_analysis_df['Date'] = pd.to_datetime(region_analysis_df['Date'], format='%d-%b-%Y')
                region_analysis_df = region_analysis_df.sort_values('Date')
                
                fig.add_trace(go.Scatter(
                    x=region_analysis_df['Date'], 
                    y=region_analysis_df[graph_type], 
                    mode='lines+markers+text',
                    text=region_analysis_df[graph_type].round(2),
                    textposition='top center',
                    name=f'{graph_type} Value'
                ))
                
                fig.update_layout(
                    title=f'{graph_type} Value Trend for {selected_region_analysis}',
                    xaxis_title='Date',
                    yaxis_title=f'{graph_type} Value',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

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
