import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
from collections import Counter

DISTRICT_COORDS = {
    'Ahmadabad': [23.0225, 72.5714], 'Surat': [21.1702, 72.8311],
    'Jaipur': [26.9124, 75.7873], 'Udaipur': [24.5854, 73.7125],
    'Gurugram': [28.4595, 77.0266], 'Bathinda': [30.2110, 74.9455],
    'Delhi East': [28.7041, 77.1025], 'Raipur': [21.2514, 81.6296],
    'Khorda': [20.1734, 85.6745], 'Sambalpur': [21.4669, 83.9756],
    'Ghaziabad': [28.6692, 77.4538], 'Haridwar': [29.9457, 78.1642],
    'Dehradun': [30.3165, 78.0322], 'Balaghat': [21.8314, 80.1857],
    'Indore': [22.7196, 75.8577], 'Nagpur': [21.1458, 79.0882]
}

BRANDS = [
    'JK LAKSHMI CEMENT',
    'JK LAKSHMI PRO+ CEMENT',
    'ULTRATECH CEMENT',
    'WONDER CEMENT',
    'SHREE CEMENT',
    'AMBUJA CEMENT',
    'JK SUPER CEMENT'
]
DISTRICT_COORDS = {
    'Ahmadabad': [23.0225, 72.5714], 'Surat': [21.1702, 72.8311],
    'Jaipur': [26.9124, 75.7873], 'Udaipur': [24.5854, 73.7125],
    'Gurugram': [28.4595, 77.0266], 'Bathinda': [30.2110, 74.9455],
    'Delhi East': [28.7041, 77.1025], 'Raipur': [21.2514, 81.6296],
    'Khorda': [20.1734, 85.6745], 'Sambalpur': [21.4669, 83.9756],
    'Ghaziabad': [28.6692, 77.4538], 'Haridwar': [29.9457, 78.1642],
    'Dehradun': [30.3165, 78.0322], 'Balaghat': [21.8314, 80.1857],
    'Indore': [22.7196, 75.8577], 'Nagpur': [21.1458, 79.0882]
}

BRANDS = [
    'JK LAKSHMI CEMENT',
    'JK LAKSHMI PRO+ CEMENT',
    'ULTRATECH CEMENT',
    'WONDER CEMENT',
    'SHREE CEMENT',
    'AMBUJA CEMENT',
    'JK SUPER CEMENT'
]

def get_district_officers(df, district_name):
    district_code = None
    for code, mapped in {
        'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat',
        'Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur',
        'Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda',
        'Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur',
        'Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur',
        'Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar',
        'Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat',
        'Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'
    }.items():
        if mapped == district_name:
            district_code = code
            break

    if district_code is None:
        return []

    district_data = df[df['District: Name'] == district_code].copy()
    
    # Get entries where Account Name is empty/null but has Owner Name
    officer_data = district_data[
        (district_data['Account: Account Name'].isna() | 
         district_data['Account: Account Name'].str.strip() == '') &
        district_data['Owner: Full Name'].notna()
    ]
    
    # Count entries per officer
    officer_counts = officer_data['Owner: Full Name'].value_counts()
    
    # Create list of officers sorted by number of entries
    officers = [(name, count) for name, count in officer_counts.items()]
    officers.sort(key=lambda x: (-x[1], x[0]))  # Sort by count (descending) then name
    
    return officers

def create_officer_data_table(df, district_name, officer_name, brand_name):
    try:
        district_code = None
        for code, mapped in {
            'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat',
            'Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur',
            'Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda',
            'Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur',
            'Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur',
            'Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar',
            'Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat',
            'Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'
        }.items():
            if mapped == district_name:
                district_code = code
                break

        if district_code is None:
            return pd.DataFrame()

        district_data = df[df['District: Name'] == district_code].copy()
        district_data = district_data[
            (district_data['Owner: Full Name'] == officer_name) &
            (district_data['Brand: Name'].str.upper() == brand_name.upper()) &
            (district_data['Account: Account Name'].isna() | 
             district_data['Account: Account Name'].str.strip() == '')
        ]

        if len(district_data) == 0:
            return pd.DataFrame()

        district_data['Full_Date'] = district_data.apply(convert_to_date, axis=1)
        district_data = district_data.dropna(subset=['Full_Date'])
        district_data = district_data.sort_values('Full_Date', ascending=False)

        # Select only Date and WSP columns
        columns = {
            'Full_Date': 'Date',
            'Whole Sale Price': 'WSP'
        }

        result_data = district_data[columns.keys()].copy()
        result_data = result_data.rename(columns=columns)
        result_data['Date'] = result_data['Date'].dt.strftime('%d-%b-%Y')
        
        return result_data

    except Exception as e:
        st.error(f"Error processing officer data for {district_name}, {brand_name}: {str(e)}")
        return pd.DataFrame()
def process_district_data(df):
    district_mapping = {
        'Z0605_Ahmadabad': 'GJ(Ahmadabad)', 'Z0616_Surat': 'GJ(Surat)',
        'Z2020_Jaipur': 'RJ(Jaipur)', 'Z2013_Udaipur': 'RJ(Udaipur)',
        'Z0703_Gurugram': 'HY(Gurgaon)', 'Z1909_Bathinda': 'PB(Bhatinda)',
        'Z3001_East': 'Delhi East', 'Z3302_Raipur': 'CG(Raipur)',
        'Z1810_Khorda': 'ORR(Khorda)', 'Z1804_Sambalpur': 'ORR(Sambalpur)',
        'Z2405_Ghaziabad': 'UP(Gaziabad)', 'Z3506_Haridwar': 'UK(Haridwar)',
        'Z3505_Dehradun': 'UK(Dehradun)', 'Z1230_Balaghat': 'M.P. East(Balaghat)',
        'Z1226_Indore': 'M.P. West(Indore)', 'Z1329_Nagpur': 'M.H. East(Nagpur)'
    }
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
    for code, mapped in {
        'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat',
        'Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur',
        'Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda',
        'Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur',
        'Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur',
        'Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar',
        'Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat',
        'Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'
    }.items():
        if mapped == district_name:
            district_code = code
            break

    if district_code is None:
        return []

    district_data = df[df['District: Name'] == district_code].copy()
    
    # Count entries per dealer across all brands
    dealer_counts = district_data['Account: Account Name'].value_counts()
    
    # Create list of dealers sorted by number of entries
    dealers = [(name, count) for name, count in dealer_counts.items()]
    dealers.sort(key=lambda x: (-x[1], x[0]))  # Sort by count (descending) then name
    
    return dealers

def create_brand_data_table(df, district_name, dealer_name, brand_name):
    try:
        district_code = None
        for code, mapped in {
            'Z0605_Ahmadabad': 'Ahmadabad', 'Z0616_Surat': 'Surat',
            'Z2020_Jaipur': 'Jaipur', 'Z2013_Udaipur': 'Udaipur',
            'Z0703_Gurugram': 'Gurugram', 'Z1909_Bathinda': 'Bathinda',
            'Z3001_East': 'Delhi East', 'Z3302_Raipur': 'Raipur',
            'Z1810_Khorda': 'Khorda', 'Z1804_Sambalpur': 'Sambalpur',
            'Z2405_Ghaziabad': 'Ghaziabad', 'Z3506_Haridwar': 'Haridwar',
            'Z3505_Dehradun': 'Dehradun', 'Z1230_Balaghat': 'Balaghat',
            'Z1226_Indore': 'Indore', 'Z1329_Nagpur': 'Nagpur'
        }.items():
            if mapped == district_name:
                district_code = code
                break

        if district_code is None:
            return pd.DataFrame()

        district_data = df[df['District: Name'] == district_code].copy()
        district_data = district_data[
            (district_data['Account: Account Name'] == dealer_name) &
            (district_data['Brand: Name'].str.upper() == brand_name.upper())
        ]

        if len(district_data) == 0:
            return pd.DataFrame()

        district_data['Full_Date'] = district_data.apply(convert_to_date, axis=1)
        district_data = district_data.dropna(subset=['Full_Date'])
        district_data = district_data.sort_values('Full_Date', ascending=False)

        # Select and rename columns
        columns = {
            'Full_Date': 'Date',
            'Account: Dealer Category': 'Dealer Category',
            'Whole Sale Price': 'WSP',
            'Retail Price': 'Retail Price',
            'Billing(In Rs)': 'Billing',
            'Owner: Full Name': 'Officer Name'
        }

        result_data = district_data[columns.keys()].copy()
        result_data = result_data.rename(columns=columns)
        result_data['Date'] = result_data['Date'].dt.strftime('%d-%b-%Y')
        
        return result_data

    except Exception as e:
        st.error(f"Error processing data for {district_name}, {brand_name}: {str(e)}")
        return pd.DataFrame()
def main():
    st.set_page_config(layout="wide")
    st.title("Cement Price Analysis Dashboard")

    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'selected_district' not in st.session_state:
        st.session_state.selected_district = None
    if 'selected_dealer' not in st.session_state:
        st.session_state.selected_dealer = None

    # File input options
    file_option = st.radio(
        "Choose input type",
        ["Upload SFDC CSV file", "Upload processed district file"]
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload either SFDC or processed district file based on your selection above"
    )

    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            df = pd.read_csv(uploaded_file, encoding='latin1')
            if file_option == "Upload SFDC CSV file":
                st.session_state.processed_df = process_district_data(df)
            else:
                st.session_state.processed_df = df
            st.success("File processed successfully!")

        # Create map data
        map_data = pd.DataFrame(
            [(dist, coord[0], coord[1]) for dist, coord in DISTRICT_COORDS.items()],
            columns=['District', 'lat', 'lon']
        )

        st.subheader("District Locations (Click on a point to select)")
        fig = px.scatter_mapbox(
            map_data,
            lat='lat',
            lon='lon',
            hover_name='District',
            zoom=4,
            center={"lat": 23.5937, "lon": 78.9629},
            mapbox_style="carto-positron",
            color_discrete_sequence=['red']
        )

        fig.update_traces(
            marker=dict(size=12),
            hovertemplate='<b>%{hovertext}</b><extra></extra>'
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            mapbox=dict(
                center=dict(lat=23.5937, lon=78.9629),
                zoom=4
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )

        # Initialize selected_district if None
        if st.session_state.selected_district is None:
            st.session_state.selected_district = list(DISTRICT_COORDS.keys())[0]

        # District selection dropdown
        selected_district = st.selectbox(
            "Select a district",
            options=list(DISTRICT_COORDS.keys()),
            index=list(DISTRICT_COORDS.keys()).index(st.session_state.selected_district)
        )

        # Update selected district
        if selected_district != st.session_state.selected_district:
            st.session_state.selected_district = selected_district
            st.session_state.selected_dealer = None

        # DEALER SECTION
        st.subheader("Dealer Data")
        
        # Get dealers for selected district
        dealers = get_district_dealers(st.session_state.processed_df, selected_district)
        
        if dealers:
            dealer_options = [f"{dealer[0]} ({dealer[1]} entries)" for dealer in dealers]
            dealer_names = [dealer[0] for dealer in dealers]
            
            # Dealer selection dropdown
            dealer_index = 0
            if st.session_state.selected_dealer in dealer_names:
                dealer_index = dealer_names.index(st.session_state.selected_dealer)
                
            selected_dealer_option = st.selectbox(
                "Select a dealer",
                options=dealer_options,
                index=dealer_index,
                key="dealer_select"
            )
            
            # Extract dealer name from selected option
            selected_dealer = dealer_names[dealer_options.index(selected_dealer_option)]
            st.session_state.selected_dealer = selected_dealer

            # Determine which JK brand to show based on district
            target_districts = ['Raipur', 'Balaghat', 'Khorda', 'Nagpur', 'Sambalpur']
            is_target = any(d in selected_district for d in target_districts)
            jk_brand = 'JK LAKSHMI PRO+ CEMENT' if is_target else 'JK LAKSHMI CEMENT'

            # Create tabs for each brand
            all_brands = [jk_brand, 'ULTRATECH CEMENT', 'WONDER CEMENT', 
                         'SHREE CEMENT', 'AMBUJA CEMENT', 'JK SUPER CEMENT']

            dealer_tabs = st.tabs([f"{brand.title()}" for brand in all_brands])

            # Display data for each brand in its respective tab
            for tab, brand in zip(dealer_tabs, all_brands):
                with tab:
                    brand_data = create_brand_data_table(
                        st.session_state.processed_df,
                        selected_district,
                        selected_dealer,
                        brand
                    )

                    if len(brand_data) > 0:
                        st.dataframe(
                            brand_data,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info(f"No {brand} data available for this dealer")

        else:
            st.warning(f"No dealers found in {selected_district}")

        # OFFICER SECTION
        st.divider()
        st.subheader("Higher Ranked Officers Data")
        
        # Get officers for selected district
        officers = get_district_officers(st.session_state.processed_df, selected_district)
        
        if officers:
            officer_options = [f"{officer[0]} ({officer[1]} entries)" for officer in officers]
            officer_names = [officer[0] for officer in officers]
            
            # Officer selection dropdown
            selected_officer_option = st.selectbox(
                "Select an officer",
                options=officer_options,
                key="officer_select"
            )
            
            # Extract officer name from selected option
            selected_officer = officer_names[officer_options.index(selected_officer_option)]

            # Create tabs for each brand (reuse brands from dealer section)
            officer_tabs = st.tabs([f"{brand.title()} (Officer Data)" for brand in all_brands])

            # Display data for each brand in its respective tab
            for tab, brand in zip(officer_tabs, all_brands):
                with tab:
                    officer_data = create_officer_data_table(
                        st.session_state.processed_df,
                        selected_district,
                        selected_officer,
                        brand
                    )

                    if len(officer_data) > 0:
                        st.dataframe(
                            officer_data,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info(f"No {brand} WSP data available from this officer")

        else:
            st.info(f"No officer entries found for {selected_district}")

if __name__ == "__main__":
    main()
