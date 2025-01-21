import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
from collections import Counter
DISTRICT_COORDS = {'Ahmadabad': [23.0225, 72.5714], 'Surat': [21.1702, 72.8311],'Jaipur': [26.9124, 75.7873], 'Udaipur': [24.5854, 73.7125],'Gurugram': [28.4595, 77.0266], 'Bathinda': [30.2110, 74.9455],'Delhi East': [28.7041, 77.1025], 'Raipur': [21.2514, 81.6296],'Khorda': [20.1734, 85.6745], 'Sambalpur': [21.4669, 83.9756],'Ghaziabad': [28.6692, 77.4538], 'Haridwar': [29.9457, 78.1642],'Dehradun': [30.3165, 78.0322], 'Balaghat': [21.8314, 80.1857],'Indore': [22.7196, 75.8577], 'Nagpur': [21.1458, 79.0882]}
BRANDS = ['JK LAKSHMI CEMENT','JK LAKSHMI PRO+ CEMENT','ULTRATECH CEMENT','WONDER CEMENT','SHREE CEMENT','AMBUJA CEMENT','JK SUPER CEMENT']
def calculate_price_changes(df, district_name, dealer_or_officer_name, brand_name, is_officer=False):
    try:
        district_code = None
        for code, mapped in {'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat','Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur','Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda','Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur','Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur','Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar','Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat','Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'}.items():
            if mapped == district_name:
                district_code = code
                break
        if district_code is None:
            return None, None
        district_data = df[df['District: Name'] == district_code].copy()
        if is_officer:
            district_data = district_data[(district_data['Owner: Full Name'] == dealer_or_officer_name) &(district_data['Brand: Name'].str.upper() == brand_name.upper()) &((district_data['Account: Account Name'].isna()) |(district_data['Account: Account Name'].str.strip() == '') |(district_data['Account: Account Name'].str.lower().str.contains('officer', na=False)) |(district_data['Account: Account Name'].str.lower().str.contains('manager', na=False)))]
        else:
            district_data = district_data[(district_data['Account: Account Name'] == dealer_or_officer_name) &(district_data['Brand: Name'].str.upper() == brand_name.upper())]
        district_data['Full_Date'] = district_data.apply(convert_to_date, axis=1)
        district_data = district_data.dropna(subset=['Full_Date'])
        district_data = district_data.sort_values('Full_Date')
        jan_data = district_data[district_data['Month'].str.lower() == 'january']
        dec_data = district_data[district_data['Month'].str.lower() == 'december']
        if len(jan_data) >= 2:
            jan_first = jan_data.iloc[0]['Whole Sale Price']
            jan_last = jan_data.iloc[-1]['Whole Sale Price']
            jan_change = jan_last - jan_first
        elif len(jan_data) == 1:
            if len(dec_data) > 0:
                dec_last_price = dec_data.iloc[-1]['Whole Sale Price']
                jan_price = jan_data.iloc[0]['Whole Sale Price']
                jan_change = jan_price - dec_last_price
            else:
                jan_change = None
        else:
            jan_change = None
        dec_end_data = dec_data[dec_data['Date'].isin(['30', '31'])]
        if len(dec_data) > 0 and len(dec_end_data) > 0:
            dec_first = dec_data.iloc[0]['Whole Sale Price']
            dec_last = dec_end_data.iloc[-1]['Whole Sale Price']
            dec_change = dec_last - dec_first
        else:
            dec_change = None
        return jan_change, dec_change
    except Exception as e:
        print(f"Error calculating price changes: {str(e)}")
        return None, None
def display_price_change_summary(jan_change, dec_change):
    if jan_change is None:
        return "No price changes recorded in January"
    summary = []
    if jan_change != 0:
        direction = "increased" if jan_change > 0 else "decreased"
        summary.append(f"WSP has {direction} by {abs(jan_change)} Rs. in Jan")
    if dec_change is not None and dec_change != 0:
        direction = "increase" if dec_change > 0 else "decrease"
        summary.append(f"({abs(dec_change)} Rs. {direction} also happened in late Dec)")
    return " ".join(summary) if summary else "No price changes in January"
def get_district_officers(df, district_name):
    district_code = None
    for code, mapped in {'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat','Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur','Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda','Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur','Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur','Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar','Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat','Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'}.items():
        if mapped == district_name:
            district_code = code
            break
    if district_code is None:
        return []
    district_data = df[df['District: Name'] == district_code].copy()
    officer_data = district_data[(district_data['Account: Account Name'].isna() | district_data['Account: Account Name'].str.strip() == '') &district_data['Owner: Full Name'].notna()]
    officer_counts = officer_data['Owner: Full Name'].value_counts()
    officers = [(name, count) for name, count in officer_counts.items()]
    officers.sort(key=lambda x: (-x[1], x[0]))
    return officers
def create_officer_data_table(df, district_name, officer_name, brand_name):
    try:
        district_code = None
        for code, mapped in {'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat','Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur','Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda','Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur','Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur','Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar','Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat','Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'}.items():
            if mapped == district_name:
                district_code = code
                break
        if district_code is None:
            return pd.DataFrame()
        district_data = df[df['District: Name'] == district_code].copy()
        district_data = district_data[(district_data['Owner: Full Name'] == officer_name) &(district_data['Brand: Name'].str.upper() == brand_name.upper()) &(district_data['Account: Account Name'].isna() | district_data['Account: Account Name'].str.strip() == '')]
        if len(district_data) == 0:
            return pd.DataFrame()
        district_data['Full_Date'] = district_data.apply(convert_to_date, axis=1)
        district_data = district_data.dropna(subset=['Full_Date'])
        district_data = district_data.sort_values('Full_Date', ascending=False)
        columns = {'Full_Date': 'Date','Whole Sale Price': 'WSP'}
        result_data = district_data[columns.keys()].copy()
        result_data = result_data.rename(columns=columns)
        result_data['Date'] = result_data['Date'].dt.strftime('%d-%b-%Y')
        return result_data
    except Exception as e:
        st.error(f"Error processing officer data for {district_name}, {brand_name}: {str(e)}")
        return pd.DataFrame()
def process_district_data(df):
    district_mapping = {'Z0605_Ahmadabad': 'GJ(Ahmadabad)', 'Z0616_Surat': 'GJ(Surat)','Z2020_Jaipur': 'RJ(Jaipur)', 'Z2013_Udaipur': 'RJ(Udaipur)','Z0703_Gurugram': 'HY(Gurgaon)', 'Z1909_Bathinda': 'PB(Bhatinda)','Z3001_East': 'Delhi East', 'Z3302_Raipur': 'CG(Raipur)','Z1810_Khorda': 'ORR(Khorda)', 'Z1804_Sambalpur': 'ORR(Sambalpur)','Z2405_Ghaziabad': 'UP(Gaziabad)', 'Z3506_Haridwar': 'UK(Haridwar)','Z3505_Dehradun': 'UK(Dehradun)', 'Z1230_Balaghat': 'M.P. East(Balaghat)','Z1226_Indore': 'M.P. West(Indore)', 'Z1329_Nagpur': 'M.H. East(Nagpur)'}
    df['District: Name'] = df['District: Name'].fillna('').astype(str).str.strip()
    df['Mapped_District'] = df['District: Name'].map(district_mapping)
    return df.dropna(subset=['Mapped_District'])
def convert_to_date(row):
    try:
        day = int(float(row['Date']))
        month = str(row['Month']).strip()
        year = 2024 if month.lower() == 'december' else 2025
        return pd.to_datetime(f"{year}-{month}-{day}", format="%Y-%B-%d")
    except:
        return None
def get_district_dealers(df, district_name):
    district_code = None
    for code, mapped in {'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat','Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur','Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda',
        'Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur','Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur','Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar',
        'Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat','Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'}.items():
        if mapped == district_name:
            district_code = code
            break
    if district_code is None:
        return []
    district_data = df[df['District: Name'] == district_code].copy()
    dealer_counts = district_data['Account: Account Name'].value_counts()
    dealers = [(name, count) for name, count in dealer_counts.items()]
    dealers.sort(key=lambda x: (-x[1], x[0]))  # Sort by count (descending) then name
    return dealers
def create_brand_data_table(df, district_name, dealer_name, brand_name):
    try:
        district_code = None
        for code, mapped in {'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat','Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur','Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda','Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur','Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur','Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar','Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat','Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'}.items():
            if mapped == district_name:
                district_code = code
                break
        if district_code is None:
            return pd.DataFrame()
        district_data = df[df['District: Name'] == district_code].copy()
        district_data = district_data[(district_data['Account: Account Name'] == dealer_name) &(district_data['Brand: Name'].str.upper() == brand_name.upper())]
        if len(district_data) == 0:
            return pd.DataFrame()
        district_data['Full_Date'] = district_data.apply(convert_to_date, axis=1)
        district_data = district_data.dropna(subset=['Full_Date'])
        district_data = district_data.sort_values('Full_Date', ascending=False)
        columns = {'Full_Date': 'Date','Account: Dealer Category': 'Dealer Category','Whole Sale Price': 'WSP','Retail Price': 'Retail Price','Billing(In Rs)': 'Billing','Owner: Full Name': 'Officer Name'}
        result_data = district_data[columns.keys()].copy()
        result_data = result_data.rename(columns=columns)
        result_data['Date'] = result_data['Date'].dt.strftime('%d-%b-%Y')
        return result_data
    except Exception as e:
        st.error(f"Error processing data for {district_name}, {brand_name}: {str(e)}")
        return pd.DataFrame()
def get_district_officers(df, district_name):
    district_code = None
    for code, mapped in {'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat','Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur','Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda','Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur','Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur','Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar','Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat','Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'}.items():
        if mapped == district_name:
            district_code = code
            break
    if district_code is None:
        return []
    district_data = df[df['District: Name'] == district_code].copy()
    officer_data = district_data[(district_data['Owner: Full Name'].notna()) & ((district_data['Account: Account Name'].isna()) | (district_data['Account: Account Name'].str.strip() == '') | (district_data['Account: Account Name'].str.lower().str.contains('officer', na=False)) |(district_data['Account: Account Name'].str.lower().str.contains('manager', na=False)) )]
    print(f"Found {len(officer_data)} officer entries for district {district_name}")
    officer_counts = officer_data['Owner: Full Name'].value_counts()
    officers = [(name, count) for name, count in officer_counts.items()]
    officers.sort(key=lambda x: (-x[1], x[0]))
    return officers
def create_officer_data_table(df, district_name, officer_name, brand_name):
    try:
        district_code = None
        for code, mapped in {'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat','Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur','Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda','Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur','Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur','Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar','Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat','Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'}.items():
            if mapped == district_name:
                district_code = code
                break
        if district_code is None:
            return pd.DataFrame()
        district_data = df[df['District: Name'] == district_code].copy()
        district_data = district_data[(district_data['Owner: Full Name'] == officer_name) &(district_data['Brand: Name'].str.upper() == brand_name.upper()) &((district_data['Account: Account Name'].isna()) |(district_data['Account: Account Name'].str.strip() == '') |(district_data['Account: Account Name'].str.lower().str.contains('officer', na=False)) |(district_data['Account: Account Name'].str.lower().str.contains('manager', na=False)))]
        print(f"Found {len(district_data)} entries for officer {officer_name} and brand {brand_name}")
        if len(district_data) == 0:
            return pd.DataFrame()
        district_data['Full_Date'] = district_data.apply(convert_to_date, axis=1)
        district_data = district_data.dropna(subset=['Full_Date'])
        district_data = district_data.sort_values('Full_Date', ascending=False)
        columns = {'Full_Date': 'Date','Whole Sale Price': 'WSP'}
        result_data = district_data[columns.keys()].copy()
        result_data = result_data.rename(columns=columns)
        result_data['Date'] = result_data['Date'].dt.strftime('%d-%b-%Y')
        return result_data
    except Exception as e:
        st.error(f"Error processing officer data for {district_name}, {brand_name}: {str(e)}")
        return pd.DataFrame()
def convert_to_date(row):
    try:
        day = int(float(row['Date']))
        month = str(row['Month']).strip()
        year = 2024 if month.lower() == 'december' else 2025
        return pd.to_datetime(f"{year}-{month}-{day}", format="%Y-%B-%d")
    except:
        return None
def get_district_code(district_name):
    district_mapping = {'Ahmadabad': 'Z0605_Ahmadabad', 'Surat': 'Z0616_Surat','Jaipur': 'Z2020_Jaipur', 'Udaipur': 'Z2013_Udaipur','Gurugram': 'Z0703_Gurugram', 'Bathinda': 'Z1909_Bathinda','Delhi East': 'Z3001_East', 'Raipur': 'Z3302_Raipur','Khorda': 'Z1810_Khorda', 'Sambalpur': 'Z1804_Sambalpur','Ghaziabad': 'Z2405_Ghaziabad', 'Haridwar': 'Z3506_Haridwar','Dehradun': 'Z3505_Dehradun', 'Balaghat': 'Z1230_Balaghat','Indore': 'Z1226_Indore', 'Nagpur': 'Z1329_Nagpur'}
    return district_mapping.get(district_name)
def get_most_active_dealer_latest_price(df, district_name, brand_name, days_threshold=3):
    try:
        district_code = get_district_code(district_name)
        if not district_code:
            return None, None, None, None, None
        district_data = df[(df['District: Name'] == district_code) &(df['Brand: Name'].str.upper() == brand_name.upper())].copy()
        if len(district_data) == 0:
            return None, None, None, None, None
        district_data['Full_Date'] = district_data.apply(convert_to_date, axis=1)
        district_data = district_data.dropna(subset=['Full_Date'])
        district_data = district_data.sort_values('Full_Date', ascending=False)
        dealer_data = []
        for dealer in district_data['Account: Account Name'].unique():
            dealer_entries = district_data[district_data['Account: Account Name'] == dealer]
            latest_entry = dealer_entries.iloc[0] if len(dealer_entries) > 0 else None
            if latest_entry is not None:
                dealer_data.append({'dealer': dealer,'entries': len(dealer_entries),'latest_entry': latest_entry})
        dealer_data.sort(key=lambda x: x['entries'], reverse=True)
        for dealer_info in dealer_data:
            latest_entry = dealer_info['latest_entry']
            days_old = (district_data['Full_Date'].max() - latest_entry['Full_Date']).days
            if days_old <= days_threshold:
                return (dealer_info['dealer'],latest_entry['Whole Sale Price'],latest_entry['Full_Date'].strftime('%d-%b-%Y'),dealer_info['entries'],days_old)
        if dealer_data:
            latest_entry = dealer_data[0]['latest_entry']
            days_old = (district_data['Full_Date'].max() - latest_entry['Full_Date']).days
            return (dealer_data[0]['dealer'],latest_entry['Whole Sale Price'],latest_entry['Full_Date'].strftime('%d-%b-%Y'),dealer_data[0]['entries'],days_old)
        return None, None, None, None, None
    except Exception as e:
        print(f"Error getting latest price: {str(e)}")
        return None, None, None, None, None
def create_decision_analysis_page(df, district_name):
    st.header(f"Decision Analysis for {district_name}")
    target_districts = ['Raipur', 'Balaghat', 'Khorda', 'Nagpur', 'Sambalpur']
    is_target = any(d in district_name for d in target_districts)
    jk_brand = 'JK LAKSHMI PRO+ CEMENT' if is_target else 'JK LAKSHMI CEMENT'
    brands = [jk_brand,'ULTRATECH CEMENT','WONDER CEMENT','SHREE CEMENT','AMBUJA CEMENT','JK SUPER CEMENT']
    col1, col2 = st.columns([3, 1])
    with col1:
        data = []
        for brand in brands:
            dealer, price, date, entries, days_old = get_most_active_dealer_latest_price(df, district_name, brand)
            data.append({'Brand': brand,'Latest Price': price if price else 'No data','Date': date if date else '-','Dealer': dealer if dealer else '-','Entries': entries if entries else 0,'Days Old': days_old if days_old is not None else '-'})
        price_df = pd.DataFrame(data)
        def highlight_old_data(val):
            if isinstance(val, (int, float)) and val > 3:
                return 'background-color: #ffcdd2'
            return ''
        st.subheader("Latest WSP Comparison")
        styled_df = price_df.style.applymap(highlight_old_data,subset=['Days Old'])
        st.dataframe(styled_df,use_container_width=True,hide_index=True,column_config={'Brand': 'Brand Name','Latest Price': st.column_config.NumberColumn('WSP (₹)', format="₹%d"),'Date': 'Last Updated','Dealer': 'Most Active Dealer','Entries': '# of Entries','Days Old': 'Days Since Update'})
    with col2:
        st.subheader("Market Statistics")
        valid_prices = [row['Latest Price'] for row in data if isinstance(row['Latest Price'], (int, float))]
        if valid_prices:
            avg_price = sum(valid_prices)/len(valid_prices)
            st.metric("Average Market Price",f"₹{avg_price:.0f}",help="Average WSP across all brands")
            price_range = f"₹{min(valid_prices)} - ₹{max(valid_prices)}"
            st.metric("Price Range",price_range,help="Lowest to highest WSP in market")
            jk_price = next((row['Latest Price'] for row in data if row['Brand'] == jk_brand), None)
            if jk_price and isinstance(jk_price, (int, float)):
                sorted_prices = sorted(valid_prices, reverse=True)
                position = sorted_prices.index(jk_price) + 1
                st.metric(f"{jk_brand} Position",f"{position} of {len(valid_prices)}",help=f"Market position by price (1 = highest price)")
                diff = jk_price - avg_price
                st.metric("Difference from Average",f"₹{abs(diff):.0f}",f"{'Above' if diff > 0 else 'Below'} market average",delta_color="inverse")
        else:
            st.warning("No valid price data available for analysis")
        if any(row['Days Old'] > 3 for row in data if isinstance(row['Days Old'], (int, float))):
            st.warning("⚠️ Some prices are more than 3 days old. Check the 'Days Since Update' column for details.",icon="⚠️")
def process_district_data(df):
    district_mapping = {'Z0605_Ahmadabad': 'GJ(Ahmadabad)', 'Z0616_Surat': 'GJ(Surat)','Z2020_Jaipur': 'RJ(Jaipur)', 'Z2013_Udaipur': 'RJ(Udaipur)','Z0703_Gurugram': 'HY(Gurgaon)', 'Z1909_Bathinda': 'PB(Bhatinda)','Z3001_East': 'Delhi East', 'Z3302_Raipur': 'CG(Raipur)','Z1810_Khorda': 'ORR(Khorda)', 'Z1804_Sambalpur': 'ORR(Sambalpur)','Z2405_Ghaziabad': 'UP(Gaziabad)', 'Z3506_Haridwar': 'UK(Haridwar)','Z3505_Dehradun': 'UK(Dehradun)', 'Z1230_Balaghat': 'M.P. East(Balaghat)','Z1226_Indore': 'M.P. West(Indore)', 'Z1329_Nagpur': 'M.H. East(Nagpur)'}
    df['District: Name'] = df['District: Name'].fillna('').astype(str).str.strip()
    df['Mapped_District'] = df['District: Name'].map(district_mapping)
    return df.dropna(subset=['Mapped_District'])
def main():
    st.set_page_config(layout="wide")
    st.title("Cement Price Analysis Dashboard")
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'selected_district' not in st.session_state:
        st.session_state.selected_district = None
    if 'selected_dealer' not in st.session_state:
        st.session_state.selected_dealer = None
    if 'show_decision_analysis' not in st.session_state:
        st.session_state.show_decision_analysis = False
    file_option = st.radio("Choose input type",["Upload SFDC CSV file", "Upload processed district file"])
    uploaded_file = st.file_uploader("Upload CSV file",type=['csv'],help="Upload either SFDC or processed district file based on your selection above")
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            df = pd.read_csv(uploaded_file, encoding='latin1')
            if file_option == "Upload SFDC CSV file":
                st.session_state.processed_df = process_district_data(df)
            else:
                st.session_state.processed_df = df
            st.success("File processed successfully!")
        map_data = pd.DataFrame([(dist, coord[0], coord[1]) for dist, coord in DISTRICT_COORDS.items()],columns=['District', 'lat', 'lon'])
        st.subheader("District Locations (Click on a point to select)")
        fig = px.scatter_mapbox(map_data,lat='lat',lon='lon',hover_name='District',zoom=4,center={"lat": 23.5937, "lon": 78.9629},mapbox_style="carto-positron",color_discrete_sequence=['red'])
        fig.update_traces(marker=dict(size=12),hovertemplate='<b>%{hovertext}</b><extra></extra>')
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},mapbox=dict(center=dict(lat=23.5937, lon=78.9629),zoom=4))
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar': False})
        if st.session_state.selected_district is None:
            st.session_state.selected_district = list(DISTRICT_COORDS.keys())[0]
        selected_district = st.selectbox("Select a district",options=list(DISTRICT_COORDS.keys()),index=list(DISTRICT_COORDS.keys()).index(st.session_state.selected_district))
        if selected_district != st.session_state.selected_district:
            st.session_state.selected_district = selected_district
            st.session_state.selected_dealer = None
            st.session_state.show_decision_analysis = False
        show_analysis = st.checkbox("Show Decision Making Analysis",value=st.session_state.show_decision_analysis)
        if show_analysis != st.session_state.show_decision_analysis:
            st.session_state.show_decision_analysis = show_analysis
        if st.session_state.show_decision_analysis:
            create_decision_analysis_page(st.session_state.processed_df, selected_district)
        else:
         st.subheader("Dealer Data")
         dealers = get_district_dealers(st.session_state.processed_df, selected_district)
         if dealers:
            dealer_options = [f"{dealer[0]} ({dealer[1]} entries)" for dealer in dealers]
            dealer_names = [dealer[0] for dealer in dealers]
            dealer_index = 0
            if st.session_state.selected_dealer in dealer_names:
                dealer_index = dealer_names.index(st.session_state.selected_dealer)
            selected_dealer_option = st.selectbox("Select a dealer",options=dealer_options,index=dealer_index,key="dealer_select")            
            selected_dealer = dealer_names[dealer_options.index(selected_dealer_option)]
            st.session_state.selected_dealer = selected_dealer
            target_districts = ['Raipur', 'Balaghat', 'Khorda', 'Nagpur', 'Sambalpur']
            is_target = any(d in selected_district for d in target_districts)
            jk_brand = 'JK LAKSHMI PRO+ CEMENT' if is_target else 'JK LAKSHMI CEMENT'
            all_brands = [jk_brand, 'ULTRATECH CEMENT', 'WONDER CEMENT','SHREE CEMENT', 'AMBUJA CEMENT', 'JK SUPER CEMENT']
            dealer_tabs = st.tabs([f"{brand.title()}" for brand in all_brands])
            for tab, brand in zip(dealer_tabs, all_brands):
                with tab:
                    jan_change, dec_change = calculate_price_changes(st.session_state.processed_df,selected_district,selected_dealer,brand,is_officer=False)
                    st.info(display_price_change_summary(jan_change, dec_change))
                    brand_data = create_brand_data_table(st.session_state.processed_df,selected_district,selected_dealer,brand)
                    if len(brand_data) > 0:
                        st.dataframe(brand_data,use_container_width=True,hide_index=True)
                    else:
                        st.info(f"No {brand} data available for this dealer")
         else:
            st.warning(f"No dealers found in {selected_district}")
         st.divider()
         st.subheader("Higher Ranked Officers Data")
         officers = get_district_officers(st.session_state.processed_df, selected_district)
         if officers:
            officer_options = [f"{officer[0]} ({officer[1]} entries)" for officer in officers]
            officer_names = [officer[0] for officer in officers]
            selected_officer_option = st.selectbox("Select an officer",options=officer_options,key="officer_select")
            selected_officer = officer_names[officer_options.index(selected_officer_option)]
            officer_tabs = st.tabs([f"{brand.title()} (Officer Data)" for brand in all_brands])
            for tab, brand in zip(officer_tabs, all_brands):
                with tab:
                    jan_change, dec_change = calculate_price_changes(st.session_state.processed_df,selected_district,selected_officer,brand,is_officer=True)
                    st.info(display_price_change_summary(jan_change, dec_change))
                    officer_data = create_officer_data_table(st.session_state.processed_df,selected_district,selected_officer,brand)
                    if len(officer_data) > 0:
                        st.dataframe(officer_data,use_container_width=True,hide_index=True)
                    else:
                        st.info(f"No {brand} WSP data available from this officer")
         else:
            st.info(f"No officer entries found for {selected_district}")
if __name__ == "__main__":
    main()
