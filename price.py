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
from reportlab.lib.units import inch
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import pandas as pd
from reportlab.lib.units import inch
def convert_dataframe_to_pdf(df, filename):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    columns_to_include = [col for col in df.columns if col not in ['Remarks'] and (col != 'MoM Change' or df.columns.get_loc(col) <= df.columns.get_loc('MoM Change'))]
    data = [columns_to_include]
    for _, row in df.iterrows():
        row_data = [str(row[col]) for col in columns_to_include]
        data.append(row_data)
        if pd.notna(row['Remarks']):
            remarks_data = ['Remarks: ' + str(row['Remarks'])]
            data.append(remarks_data)
    col_widths = []
    for col in columns_to_include:
        max_width = max(len(str(data[i][j])) for i in range(len(data)) for j, c in enumerate(columns_to_include) if c == col)
        col_widths.append(max_width * 0.8 * inch)
    table = Table(data, repeatRows=1, colWidths=col_widths)
    table_style = [('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,0), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 12),('BOTTOMPADDING', (0,0), (-1,0), 12),('GRID', (0,0), (-1,-1), 1, colors.black),('BACKGROUND', (0,-1), (-1,-1), colors.yellow),('ALIGN', (0,-1), (-1,-1), 'LEFT'),('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Oblique'),('SPAN', (0,-1), (-1,-1))]
    if 'MoM Change' in columns_to_include:
        mom_change_index = columns_to_include.index('MoM Change')
        for row_num in range(1, len(data)):
            try:
                if len(data[row_num]) == len(columns_to_include):
                    mom_change_value = float(data[row_num][mom_change_index])
                    if mom_change_value < 0:
                        table_style.append(('BACKGROUND', (mom_change_index, row_num), (mom_change_index, row_num), colors.pink))
                    elif mom_change_value > 0:
                        table_style.append(('BACKGROUND', (mom_change_index, row_num), (mom_change_index, row_num), colors.lightgreen))
                    else:
                        table_style.append(('BACKGROUND', (mom_change_index, row_num), (mom_change_index, row_num), colors.lightgrey))
            except (ValueError, TypeError):
                pass
    table.setStyle(TableStyle(table_style))
    content = [table]
    doc.build(content)
    buffer.seek(0)
    return buffer
def save_processed_dataframe(df, start_date=None, download_format='xlsx'):
    if start_date:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b %Y')
        df = df[df['Date'] >= start_date]
        df['Date'] = df['Date'].dt.strftime('%d-%b %Y')
    output = io.BytesIO()
    if download_format == 'xlsx':
        # Excel saving logic
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            dark_blue = '#2C3E50'
            white = '#FFFFFF'
            light_gray = '#F2F2F2'
            format_header = workbook.add_format({'bold': True, 'font_size': 14, 'bg_color': dark_blue,  'font_color': white,'align': 'center','valign': 'vcenter','border': 1,'border_color': '#000000'})
            format_general = workbook.add_format({'font_size': 12,'valign': 'vcenter','align': 'center'})
            format_alternating = workbook.add_format({'font_size': 12,'bg_color': light_gray,'valign': 'vcenter','align': 'center'})
            worksheet.set_row(0, 30, format_header)
            for row_num in range(1, len(df) + 1):
                if row_num % 2 == 0:
                    worksheet.set_row(row_num, None, format_alternating)
                else:
                    worksheet.set_row(row_num, None, format_general)
            for col_num, col_name in enumerate(df.columns):
                max_len = max(df[col_name].astype(str).map(len).max(),len(str(col_name)))
                worksheet.set_column(col_num, col_num, max_len + 2, format_general)
            if 'MoM Change' in df.columns:
                mom_change_col_index = df.columns.get_loc('MoM Change')
                format_negative = workbook.add_format({'bg_color': '#FFC7CE','font_size': 12,'align': 'center','valign': 'vcenter'})
                format_zero = workbook.add_format({'bg_color': '#D9D9D9','font_size': 12,'align': 'center','valign': 'vcenter'})
                format_positive = workbook.add_format({'bg_color': '#C6EFCE','font_size': 12,'align': 'center','valign': 'vcenter'})
                worksheet.conditional_format(1, mom_change_col_index, len(df), mom_change_col_index, {'type': 'cell', 'criteria': '<','value': 0,'format': format_negative})
                worksheet.conditional_format(1, mom_change_col_index, len(df), mom_change_col_index, {'type': 'cell', 'criteria': '=', 'value': 0,'format': format_zero})
                worksheet.conditional_format(1, mom_change_col_index, len(df), mom_change_col_index, {'type': 'cell','criteria': '>','value': 0,'format': format_positive})
            writer.close()
    elif download_format == 'pdf':
        output = convert_dataframe_to_pdf(df, 'processed_price_tracker.pdf')
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
"""def process_excel_file(uploaded_file, requires_editing):
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
    
    return df"""
def main():
    st.set_page_config(page_title="Price Tracker", layout="wide")
    st.title("📊 Price Tracker Analysis Tool")
    st.markdown("""
    ### Welcome to the Price Tracker Analysis Tool
    
    **Instructions:**
    1. Upload your Excel price tracking file
    2. Choose whether the file needs initial editing
    3. Add new data, analyze regions, and download processed files
    """)
    uploaded_file = st.file_uploader("Please upload the Price Tracker file", type=['xlsx'], help="Upload an Excel file containing price tracking data")
    if uploaded_file is not None:
        requires_editing = st.radio("Does this file require initial editing?", ["No", "Yes"],help="Select 'Yes' if the uploaded file needs preprocessing")
        try:
            df = process_excel_file(uploaded_file, requires_editing == "Yes")
            required_columns = ['Region(District)', 'Date', 'Inv.', 'RD', 'STS', 'Reglr']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔄 Data Entry")
                price_changed = st.radio("Do you want to add new data?", ["No", "Yes"])
                if price_changed == "Yes":
                    unique_regions = df['Region(District)'].unique()
                    if len(unique_regions) == 0:
                        st.warning("No regions found in the dataframe.")
                    else:
                        selected_regions = st.multiselect("Select Region(s)", unique_regions)
                        data_entries = []
                        for selected_region in selected_regions:
                            st.markdown(f"### Data Entry for {selected_region}")
                            region_df = df[df['Region(District)'] == selected_region]
                            from datetime import datetime
                            date_input = st.text_input(f"Enter Date for {selected_region}", value=datetime.now().strftime("%d-%b %Y"),placeholder="DD-Mon YYYY, e.g., 01-Jan 2024",key=f"date_{selected_region}")
                            inv_input = st.number_input(f"Enter Inv. value for {selected_region}", value=0.0, format="%.2f",key=f"inv_{selected_region}")
                            rd_input = st.number_input(f"Enter RD value for {selected_region}", value=0.0, format="%.2f", key=f"rd_{selected_region}")
                            sts_input = st.number_input(f"Enter STS value for {selected_region}", value=0.0, format="%.2f", key=f"sts_{selected_region}")
                            reglr_input = st.number_input(f"Enter Reglr value for {selected_region}",value=0.0,format="%.2f",key=f"reglr_{selected_region}")
                            net_input = inv_input - rd_input - sts_input - reglr_input
                            st.write(f"Calculated Net value for {selected_region}: {net_input}")
                            last_net_value = region_df['Net'].iloc[-1] if 'Net' in region_df.columns and not region_df['Net'].empty else 0
                            mom_change = net_input - last_net_value
                            st.write(f"Calculated MoM Change for {selected_region}: {mom_change}")
                            remarks_input = st.text_area(f"Enter Remarks for {selected_region} (Optional)",key=f"remarks_{selected_region}")
                            new_row = {'Region(District)': selected_region,'Date': parse_date(date_input).strftime('%d-%b %Y'),'Inv.': inv_input,'RD': rd_input,'STS': sts_input,'Reglr': reglr_input,'Net': net_input,'MoM Change': mom_change,'Remarks': remarks_input}
                            data_entries.append(new_row)
                            st.markdown("---")
                        if st.button("Add New Rows to Dataframe"):
                            for entry in data_entries:
                                df = df.append(entry, ignore_index=True)
                            st.success(f"{len(data_entries)} new rows added successfully!")
            with col2:
                st.subheader("📈 Region Analysis")
                unique_regions = df['Region(District)'].unique()
                selected_region_analysis = st.selectbox("Select Region for Analysis", unique_regions)
                region_analysis_df = df[df['Region(District)'] == selected_region_analysis]
                col_metrics_1, col_metrics_2 = st.columns(2)
                with col_metrics_1:
                    st.metric("Total Price Changes", len(region_analysis_df))
                graph_type = st.selectbox("Select Graph Type", ['Net', 'Inv.', 'RD', 'STS', 'Reglr'])
                fig = go.Figure()
                region_analysis_df['Date'] = pd.to_datetime(region_analysis_df['Date'], format='%d-%b %Y')
                region_analysis_df = region_analysis_df.sort_values('Date')
                fig.add_trace(go.Scatter(x=region_analysis_df['Date'], y=region_analysis_df[graph_type], mode='lines+markers+text',text=region_analysis_df[graph_type].round(2),textposition='top center',name=f'{graph_type} Value'))
                
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
