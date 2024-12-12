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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime as dt
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime, timedelta
from reportlab.pdfgen import canvas
from reportlab.lib.colors import green, red, black
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime, timedelta
from reportlab.pdfgen import canvas
from reportlab.lib.colors import green, red, black
import calendar
import io
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import pandas as pd
import io
import pandas as pd
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import pandas as pd
import io
import datetime
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
import pandas as pd
import io
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def get_wsp_data():
    include_wsp = st.checkbox("Include WSP (Wholesale Price) Data")
    
    if include_wsp:
        wsp_file = st.file_uploader("Upload WSP Data File", type=['csv', 'xlsx'])
        
        if wsp_file is not None:
            try:
                # Read the WSP file
                if wsp_file.name.endswith('.csv'):
                    wsp_df = pd.read_csv(wsp_file)
                else:
                    wsp_df = pd.read_excel(wsp_file)
                
                # Validate WSP DataFrame
                required_columns = ['Region(District)', 'Week-1 Nov', 'Week-2 Nov', 'Week-3 Nov', 'Week-4 Nov', 'Week-1 Dec']
                for col in required_columns:
                    if col not in wsp_df.columns:
                        st.error(f"Missing required WSP column: {col}")
                        return None
                
                return wsp_df
            except Exception as e:
                st.error(f"Could not read WSP file: {e}")
                return None
    
    return None
def generate_regional_price_trend_report(df, wsp_df=None):
    try:
        # Validate input DataFrame
        required_columns = ['Date', 'Region(District)', 'Inv.', 'Net']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b %Y')
        
        # Sort DataFrame by Region and Date
        df = df.sort_values(['Region(District)', 'Date'])
        
        # Prepare buffer for PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                                rightMargin=2, leftMargin=2, 
                                topMargin=18, bottomMargin=18)
        
        # Get sample styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = styles['Title']
        region_style = ParagraphStyle(
            'RegionStyle', 
            parent=styles['Heading2'], 
            textColor=colors.blue,
            spaceAfter=12
        )
        month_style = ParagraphStyle(
            'MonthStyle', 
            parent=styles['Heading3'], 
            textColor=colors.green,
            spaceAfter=6
        )
        normal_style = styles['Normal']
        
        # Create larger font style for price progression
        large_price_style = ParagraphStyle(
            'LargePriceStyle',
            parent=styles['Normal'],
            fontSize=14,
            spaceAfter=6
        )
        
        # Style for total change
        total_change_style = ParagraphStyle(
            'TotalChangeStyle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.brown,
            alignment=TA_LEFT,
            spaceAfter=14
        )
        
        # Helper function to get start data point
        def get_start_data_point(df, reference_date):
            """
            Get the start data point for the price progression
            Prioritizes 1st of the month, falls back to last available data of previous month
            """
            # Try to get data for the first day of the month
            first_day_data = df[
                (df['Date'].dt.year == reference_date.year) & 
                (df['Date'].dt.month == reference_date.month) & 
                (df['Date'].dt.day == 1)
            ]
            
            if not first_day_data.empty:
                return first_day_data.iloc[0]
            
            # If no first day data, get the last data point of the previous month
            prev_month = reference_date.replace(day=1) - timedelta(days=1)
            last_data_of_prev_month = df[
                (df['Date'].dt.year == prev_month.year) & 
                (df['Date'].dt.month == prev_month.month)
            ]
            
            if not last_data_of_prev_month.empty:
                return last_data_of_prev_month.iloc[-1]
            
            return None
        
        # Helper function to create metric progression
        def create_comprehensive_metric_progression(region_df, current_date, last_month, metric_column, title):
            story.append(Paragraph(f"{title} Progression from {last_month.strftime('%B %Y')} to {current_date.strftime('%B %Y')}:-", month_style))
            
            # Get the start data point (either 1st of last month or last available data from previous month)
            start_data_point = get_start_data_point(region_df, last_month)
            
            if start_data_point is None:
                story.append(Paragraph("No data available for this period", normal_style))
                story.append(Spacer(1, 12))
                return
            
            # Prepare the data for progression
            progression_df = region_df[
                (region_df['Date'] >= start_data_point['Date']) & 
                (region_df['Date'] <= current_date)
            ].copy().sort_values('Date')
            
            if progression_df.empty:
                story.append(Paragraph("No data available for this period", normal_style))
                story.append(Spacer(1, 12))
                return
            
            # Prepare metric values and dates
            metric_values = progression_df[metric_column].apply(lambda x: f"{x:.0f}").tolist()
            dates = progression_df['Date'].dt.strftime('%d-%b').tolist()
            
            # Prepare metric progression parts
            metric_progression_parts = []
            for i in range(len(metric_values)):
                # Add metric value
                metric_progression_parts.append(metric_values[i])
                
                # Add change annotation on arrow for all except the last value
                if i < len(metric_values) - 1:
                    change = float(metric_values[i+1]) - float(metric_values[i])
                    if change > 0:
                        metric_progression_parts.append(
                            f'<sup><font color="green" size="7">+{change:.0f}</font></sup>→'
                        )
                    elif change < 0:
                        # Red downward change
                        metric_progression_parts.append(
                            f'<sup><font color="red" size="7">{change:.0f}</font></sup>→'
                        )
                    else:
                        # Neutral change
                        metric_progression_parts.append(
                            f'<sup><font size="8">00</font></sup>→'
                        )
            
            # Join the progression parts
            full_progression = " ".join(metric_progression_parts)
            date_progression_text = " ----- ".join(dates)
            
            # Add metric progression with larger font
            story.append(Paragraph(full_progression, large_price_style))
            story.append(Paragraph(date_progression_text, normal_style))
            
            # Calculate total change
            if len(metric_values) > 1:
                total_change = float(metric_values[-1]) - float(metric_values[0])
                if total_change == 0:
                    total_change_text = f"Net Change in {title}: 0 Rs."
                else:
                    total_change_text = f"Net Change in {title}: {total_change:+.0f} Rs."
                
                story.append(Paragraph(total_change_text, total_change_style))
            
            story.append(Spacer(1, 12))
        
        # Helper function to create WSP progression
        def create_wsp_progression(wsp_df, region):
            """
            Create WSP (Wholesale Price) progression section for a specific region
            """
            if wsp_df is None:
                return
            
            # Get WSP data for the specific region
            region_wsp = wsp_df[wsp_df['Region(District)'] == region]
            
            if region_wsp.empty:
                story.append(Paragraph(f"No WSP data available for {region}", normal_style))
                story.append(Spacer(1, 12))
                return
            
            # Extract WSP values for November and December weeks
            wsp_columns = ['Week-1 Nov', 'Week-2 Nov', 'Week-3 Nov', 'Week-4 Nov', 'Week-1 Dec']
            
            # Prepare metric values
            metric_values = region_wsp[wsp_columns].values.flatten().tolist()
            
            # Prepare week labels
            week_labels = ['W-1 Nov', 'W-2 Nov', 'W-3 Nov', 'W-4 Nov', 'W-1 Dec']
            
            # Add WSP Progression title
            story.append(Paragraph(f"WSP Progression from November to December 2024:-", month_style))
            
            # Metric progression parts
            metric_progression_parts = []
            for i in range(len(metric_values)):
                # Add metric value
                metric_progression_parts.append(f"{metric_values[i]:.0f}")
                
                # Add change annotation on arrow for all except the last value
                if i < len(metric_values) - 1:
                    change = float(metric_values[i+1]) - float(metric_values[i])
                    if change > 0:
                        metric_progression_parts.append(
                            f'<sup><font color="green" size="7">+{change:.0f}</font></sup>→'
                        )
                    elif change < 0:
                        # Red downward change
                        metric_progression_parts.append(
                            f'<sup><font color="red" size="7">{change:.0f}</font></sup>→'
                        )
                    else:
                        # Neutral change
                        metric_progression_parts.append(
                            f'<sup><font size="8">00</font></sup>→'
                        )
            
            # Join the progression parts
            full_progression = " ".join(metric_progression_parts)
            week_progression_text = " -- ".join(week_labels)
            
            # Add metric progression with larger font
            story.append(Paragraph(full_progression, large_price_style))
            story.append(Paragraph(week_progression_text, normal_style))
            
            # Calculate total change
            if len(metric_values) > 1:
                total_change = float(metric_values[-1]) - float(metric_values[0])
                if total_change == 0:
                    total_change_text = f"Net Change in WSP: 0 Rs."
                else:
                    total_change_text = f"Net Change in WSP: {total_change:+.0f} Rs."
                
                story.append(Paragraph(total_change_text, total_change_style))
            
            story.append(Spacer(1, 12))
        
        # Initialize story for PDF
        story = []
        story.append(Paragraph("Price Trend Analysis", title_style))
        # Current and last month calculation
        current_date = datetime.now()
        last_month = current_date.replace(day=1) - timedelta(days=1)
        
        # Generate report for each region
        for region in df['Region(District)'].unique():
            region_df = df[df['Region(District)'] == region].copy()
            story.append(Paragraph(f"Price Trend Report: {region}", region_style))
            story.append(Spacer(1, 12))
            
            # Create comprehensive metric progression for different metrics
            create_comprehensive_metric_progression(
                region_df, current_date, last_month, 'Inv.', 'Invoice Price'
            )
            
            create_comprehensive_metric_progression(
                region_df, current_date, last_month, 'Net', 'NOD'
            )
            
            # Create WSP progression if WSP data is available
            create_wsp_progression(wsp_df, region)
            
            # Add page break between regions
            story.append(Paragraph("<pagebreak/>", normal_style))
        
        # Build PDF
        doc.build(story)
        
        # Reset buffer position
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        print(f"Error generating report: {e}")
        raise
def save_regional_price_trend_report(df):
    """
    Save the regional price trend report as a PDF
    
    Args:
    df (pandas.DataFrame): DataFrame containing price tracking data
    
    Returns:
    io.BytesIO: PDF report buffer
    """
    # First, get WSP data (if user wants to include)
    wsp_df = get_wsp_data()
    
    # Generate report with optional WSP data
    return generate_regional_price_trend_report(df, wsp_df)
def convert_dataframe_to_pdf(df, filename):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    data = [df.columns.tolist()]  # Header row
    for _, row in df.iterrows():
        data.append([str(val) for val in row.tolist()])
    table = Table(data)
    table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 12),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black)]))
    content = []
    content.append(table)
    doc.build(content)
    buffer.seek(0)
    return buffer
def save_processed_dataframe(df, start_date=None, download_format='xlsx'):
    if 'processed_dataframe' in st.session_state:
        df = st.session_state['processed_dataframe']
    df_to_save = df.copy()
    if 'Date' in df_to_save.columns:
        df_to_save['Date'] = pd.to_datetime(df_to_save['Date'], format='%d-%b %Y')
        if start_date:
            df_to_save = df_to_save[df_to_save['Date'] >= start_date]
            df_to_save['Date'] = df_to_save['Date'].dt.strftime('%d-%b %Y')
    output = io.BytesIO()
    if download_format == 'xlsx':
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_to_save.to_excel(writer, sheet_name='Sheet1', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            dark_blue = '#2C3E50'
            white = '#FFFFFF'
            light_gray = '#F2F2F2'
            format_header = workbook.add_format({'bold': True, 'font_size': 14,'bg_color': dark_blue,'font_color': white,'align': 'center','valign': 'vcenter','border': 1,'border_color': '#000000'})
            format_general = workbook.add_format({'font_size': 12,'valign': 'vcenter','align': 'center'})
            format_alternating = workbook.add_format({'font_size': 12,'bg_color': light_gray,'valign': 'vcenter','align': 'center'})
            worksheet.set_row(0, 30, format_header)
            for row_num in range(1, len(df_to_save) + 1):
                if row_num % 2 == 0:
                    worksheet.set_row(row_num, None, format_alternating)
                else:
                    worksheet.set_row(row_num, None, format_general)
            for col_num, col_name in enumerate(df_to_save.columns):
                max_len = max(df_to_save[col_name].astype(str).map(len).max(),len(str(col_name)))
                worksheet.set_column(col_num, col_num, max_len + 2, format_general)
            if 'MoM Change' in df_to_save.columns:
                mom_change_col_index = df_to_save.columns.get_loc('MoM Change')
                format_negative = workbook.add_format({'bg_color': '#FFC7CE','font_size': 12,'align': 'center','valign': 'vcenter'})
                format_zero = workbook.add_format({'bg_color': '#D9D9D9','font_size': 12,'align': 'center','valign': 'vcenter'})
                format_positive = workbook.add_format({'bg_color': '#C6EFCE','font_size': 12,'align': 'center','valign': 'vcenter'})
                worksheet.conditional_format(1, mom_change_col_index, len(df_to_save), mom_change_col_index, {'type': 'cell', 'criteria': '<', 'value': 0, 'format': format_negative})
                worksheet.conditional_format(1, mom_change_col_index, len(df_to_save), mom_change_col_index, {'type': 'cell','criteria': '=','value': 0,'format': format_zero})
                worksheet.conditional_format(1, mom_change_col_index, len(df_to_save), mom_change_col_index, {'type': 'cell','criteria': '>','value': 0, 'format': format_positive})
            writer.close()
    elif download_format == 'pdf':
        output = convert_dataframe_to_pdf(df_to_save, 'processed_price_tracker.pdf')
    output.seek(0)
    return output
def parse_date(date_str):
    try:
        date_formats = ['%d-%b %Y','%d-%b-%Y','%d-%B %Y','%Y-%m-%d','%m/%d/%Y','%d/%m/%Y',]
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        return pd.to_datetime(date_str, format='mixed', dayfirst=True)
    except Exception as e:
        st.warning(f"Could not parse date: {date_str}. Error: {e}")
        return pd.NaT
def process_excel_file(uploaded_file, requires_editing):
    warnings.simplefilter("ignore")
    df = pd.read_excel(uploaded_file)
    if not requires_editing:
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(parse_date)
        return df
    df = df.iloc[1:] 
    df = df.iloc[:, 1:]
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df = df[~df.iloc[:, 1].str.contains('Date', na=False, case=False)]
    df.iloc[:, 1] = df.iloc[:, 1].apply(parse_date)
    #df.iloc[:, 1] = df.iloc[:, 1].dt.strftime('%d-%b %Y')  
    df = df.loc[:, df.columns.notnull()] 
    df = df[df.iloc[:, 0] != "JKLC Price Tracker Mar'24 - till 03-12-24"]
    mask = df.iloc[:, 0].notna()
    current_value = None
    for i in range(len(df)):     
        if mask.iloc[i]:         
            current_value = df.iloc[i, 0]     
        else:         
            if current_value is not None:             
                df.iloc[i, 0] = current_value 
    df = df.rename(columns={df.columns[0]: 'Region(District)'})
    df = df.reset_index(drop=True)
    return df
def main():
    st.set_page_config(page_title="Price Tracker", layout="wide", page_icon="💰")
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
                            inv_input = st.number_input(f"Enter Inv. value for {selected_region}",value=0.0,format="%.2f",key=f"inv_{selected_region}")
                            rd_input = st.number_input(f"Enter RD value for {selected_region}",value=0.0, format="%.2f",key=f"rd_{selected_region}")
                            sts_input = st.number_input(f"Enter STS value for {selected_region}",value=0.0, format="%.2f", key=f"sts_{selected_region}")
                            reglr_input = st.number_input(f"Enter Reglr value for {selected_region}",value=0.0,format="%.2f",key=f"reglr_{selected_region}")
                            net_input = inv_input - rd_input - sts_input - reglr_input
                            st.write(f"Calculated Net value for {selected_region}: {net_input}")
                            last_net_value = region_df['Net'].iloc[-1] if 'Net' in region_df.columns and not region_df['Net'].empty else 0
                            mom_change = net_input - last_net_value
                            st.write(f"Calculated MoM Change for {selected_region}: {mom_change}")
                            remarks_input = st.text_area(f"Enter Remarks for {selected_region} (Optional)",key=f"remarks_{selected_region}")
                            new_row = {'Region(District)': selected_region,'Date': parse_date(date_input).strftime('%d-%b %Y'),'Inv.': inv_input,'RD': rd_input,'STS': sts_input,
                                'Reglr': reglr_input,'Net': net_input,'MoM Change': mom_change,'Remarks': remarks_input}
                            data_entries.append(new_row)
                            st.markdown("---")
                        if st.button("Add New Rows to Dataframe"):
                            if not data_entries:
                                st.warning("No new entries to add.")
                                return
                            updated_df = df.copy()
                            new_rows_df = pd.DataFrame(data_entries)
                            for col in df.columns:
                                if col not in new_rows_df.columns:
                                    new_rows_df[col] = None
                            new_rows_df = new_rows_df.reindex(columns=df.columns)
                            for region in new_rows_df['Region(District)'].unique():
                                region_new_rows = new_rows_df[new_rows_df['Region(District)'] == region]
                                region_existing_indices = updated_df[updated_df['Region(District)'] == region].index
                                if not region_existing_indices.empty:
                                    last_region_index = region_existing_indices[-1]
                                    before_region = updated_df.iloc[:last_region_index+1]
                                    after_region = updated_df.iloc[last_region_index+1:]
                                    updated_df = pd.concat([before_region,region_new_rows,after_region]).reset_index(drop=True)
                                else:
                                    updated_df = pd.concat([updated_df, region_new_rows]).reset_index(drop=True)
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
                      last_month_display['MoM Change'] = last_month_display['MoM Change'].round(0).astype(int)
                      st.dataframe(last_month_display.style.background_gradient(cmap='Blues'), use_container_width=True)
                      col_last_1, col_last_2 = st.columns(2)
                      with col_last_1:
                       st.metric(f"Total No. of Price Change in (Last Month)", len(last_month_data))
                      with col_last_2:
                       st.metric("Total Change in NOD(Last Month)(in Rs.)", last_month_data['MoM Change'].sum())
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
                     current_month_display['MoM Change'] = current_month_display['MoM Change'].round(0).astype(int)
                     st.dataframe(current_month_display.style.background_gradient(cmap='Blues'), use_container_width=True)
                     col_curr_1, col_curr_2 = st.columns(2)
                     with col_curr_1:
                        st.metric("Total No. of Price Change in (Current Month)", len(current_month_data))
                     with col_curr_2:
                         st.metric("Total Change in NOD(Current Month)(in Rs.)", current_month_data['MoM Change'].sum())
                else:
                      st.info(f"No data found for current month in {selected_region_analysis}")
                region_analysis_df = df[df['Region(District)'] == selected_region_analysis]
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
                fig.add_trace(go.Scatter(x=filtered_df['Date'],y=filtered_df[graph_type], mode='lines+markers+text',text=filtered_df[graph_type].abs().round(0).astype(int),textposition='top center',name=f'{graph_type} Value',line=dict(color='#1E90FF',width=3),marker=dict(size=10,color='#4169E1',symbol='circle',line=dict(color='#FFFFFF',width=2)),hovertemplate=('<b>Date</b>: %{x|%d %B %Y}<br>' +f'<b>{graph_type}</b>: %{{y:.2f}}<br>' +'<extra></extra>')))
                fig.update_layout(title=f'{graph_type} Value Trend for {selected_region_analysis}',xaxis_title='Date',yaxis_title=f'{graph_type} Value',height=400)
                st.plotly_chart(fig, use_container_width=True)
                graph_download_format = st.selectbox("Download Graph as", ['PNG', 'PDF'])
                if st.button("Download Graph"):
                        if graph_download_format == 'PNG':
                            img_bytes = pio.to_image(fig, format='png')
                            st.download_button(label="Download Graph as PNG",data=img_bytes,file_name=f'{selected_region_analysis}_{graph_type}_trend.png',mime='image/png')
                        else:
                            pdf_bytes = pio.to_image(fig, format='pdf')
                            st.download_button(label="Download Graph as PDF",data=pdf_bytes,file_name=f'{selected_region_analysis}_{graph_type}_trend.pdf',mime='application/pdf')
                st.markdown("### Remarks")
                remarks_df = region_analysis_df[['Date', 'Remarks']].dropna(subset=['Remarks'])
                remarks_df = remarks_df.sort_values('Date', ascending=False)
                if not remarks_df.empty:
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
            st.markdown("## 📥 Download Options")
            download_options = st.radio("Download File From:", ["Entire Dataframe", "Specific Month", "Regional Price Trend Report"], horizontal=True)
            if download_options == "Regional Price Trend Report":
                output = save_regional_price_trend_report(df)
                st.download_button(
        label="Download Regional Price Trend Report (PDF)",
        data=output,
        file_name="regional_price_trend_report.pdf",
        mime="application/pdf"
    )
            start_date = None
            if download_options == "Specific Month":
                col1, col2 = st.columns(2)
                with col1:
                    month_input = st.selectbox("Select Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
                with col2:
                    year_input = st.number_input("Select Year", min_value=2000, max_value=2030, value=2024)
                start_date = pd.to_datetime(f'01-{month_input[:3].lower()} {year_input}', format='%d-%b %Y')
            download_format = st.selectbox("Select Download Format", ['Excel (.xlsx)', 'PDF (.pdf)'])
            format_map = {'Excel (.xlsx)': 'xlsx', 'PDF (.pdf)': 'pdf'}
            selected_format = format_map[download_format]
            if st.button("Download Processed File"):
                try:
                    output = save_processed_dataframe(df, start_date, selected_format)
                    st.download_button(label=f"Click to Download {download_format}",data=output,file_name=f'processed_price_tracker.{selected_format}',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if selected_format == 'xlsx' else 'application/pdf')
                except Exception as e:
                    st.error(f"Error during download: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)
if __name__ == "__main__":
    main()
