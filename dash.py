import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

# District coordinates (latitude, longitude)
DISTRICT_COORDS = {
    'Ahmadabad': [23.0225, 72.5714],
    'Surat': [21.1702, 72.8311],
    'Jaipur': [26.9124, 75.7873],
    'Udaipur': [24.5854, 73.7125],
    'Gurugram': [28.4595, 77.0266],
    'Bathinda': [30.2110, 74.9455],
    'Delhi East': [28.7041, 77.1025],  # Changed from 'Delhi' to 'Delhi East'
    'Raipur': [21.2514, 81.6296],
    'Khorda': [20.1734, 85.6745],
    'Sambalpur': [21.4669, 83.9756],
    'Ghaziabad': [28.6692, 77.4538],
    'Haridwar': [29.9457, 78.1642],
    'Dehradun': [30.3165, 78.0322],
    'Balaghat': [21.8314, 80.1857],
    'Indore': [22.7196, 75.8577],
    'Nagpur': [21.1458, 79.0882]
}

def process_district_data(df):
    """Process district data with the provided mapping"""
    district_mapping = {
        'Z0605_Ahmadabad': 'GJ(Ahmadabad)',
        'Z0616_Surat': 'GJ(Surat)',
        'Z2020_Jaipur': 'RJ(Jaipur)',
        'Z2013_Udaipur': 'RJ(Udaipur)',
        'Z0703_Gurugram': 'HY(Gurgaon)',
        'Z1909_Bathinda': 'PB(Bhatinda)',
        'Z3001_East': 'Delhi East',  # Changed to match with DISTRICT_COORDS
        'Z3302_Raipur': 'CG(Raipur)',
        'Z1810_Khorda': 'ORR(Khorda)',
        'Z1804_Sambalpur': 'ORR(Sambalpur)',
        'Z2405_Ghaziabad': 'UP(Gaziabad)',
        'Z3506_Haridwar': 'UK(Haridwar)',
        'Z3505_Dehradun': 'UK(Dehradun)',
        'Z1230_Balaghat': 'M.P. East(Balaghat)',
        'Z1226_Indore': 'M.P. West(Indore)',
        'Z1329_Nagpur': 'M.H. East(Nagpur)'
    }
    
    df['District: Name'] = df['District: Name'].fillna('').astype(str).str.strip()
    df['Mapped_District'] = df['District: Name'].map(district_mapping)
    return df.dropna(subset=['Mapped_District'])

def convert_to_date(row):
    """Convert date and month to datetime"""
    try:
        day = int(float(row['Date']))
        month = str(row['Month']).strip()
        year = 2024 if month == 'December' else 2025
        return pd.to_datetime(f"{year}-{month}-{day}", format="%Y-%B-%d")
    except:
        return None

def create_price_table(df, district_name):
    """Create price table for selected district"""
    try:
        # Get the corresponding district code
        district_code = None
        for code, mapped in {
            'Z0605_Ahmadabad': 'Ahmadabad',
            'Z0616_Surat': 'Surat',
            'Z2020_Jaipur': 'Jaipur',
            'Z2013_Udaipur': 'Udaipur',
            'Z0703_Gurugram': 'Gurugram',
            'Z1909_Bathinda': 'Bathinda',
            'Z3001_East': 'Delhi East',
            'Z3302_Raipur': 'Raipur',
            'Z1810_Khorda': 'Khorda',
            'Z1804_Sambalpur': 'Sambalpur',
            'Z2405_Ghaziabad': 'Ghaziabad',
            'Z3506_Haridwar': 'Haridwar',
            'Z3505_Dehradun': 'Dehradun',
            'Z1230_Balaghat': 'Balaghat',
            'Z1226_Indore': 'Indore',
            'Z1329_Nagpur': 'Nagpur'
        }.items():
            if mapped == district_name:
                district_code = code
                break

        if district_code is None:
            return pd.DataFrame()

        # Filter for selected district
        district_data = df[df['District: Name'] == district_code]
        
        if len(district_data) == 0:
            return pd.DataFrame()

        # Convert dates
        district_data['Full_Date'] = district_data.apply(convert_to_date, axis=1)
        district_data = district_data.dropna(subset=['Full_Date'])
        
        # Check if district is in target districts
        target_districts = ['Raipur', 'Balaghat', 'Khorda', 'Nagpur', 'Sambalpur']
        is_target = any(d in district_name for d in target_districts)
        
        # Filter brands
        if is_target:
            brands = ['JK LAKSHMI PRO+ CEMENT', 'SHREE CEMENT', 'ULTRATECH CEMENT']
        else:
            brands = ['JK LAKSHMI CEMENT', 'SHREE CEMENT', 'ULTRATECH CEMENT']
        
        district_data = district_data[
            district_data['Brand: Name'].str.upper().isin(brands)
        ]
        
        if len(district_data) == 0:
            return pd.DataFrame()

        # Pivot table
        price_table = district_data.pivot_table(
            index='Full_Date',
            columns='Brand: Name',
            values='Whole Sale Price',
            aggfunc=lambda x: ', '.join(map(str, set(x)))
        ).reset_index()
        
        if len(price_table) > 0:
            price_table['Full_Date'] = price_table['Full_Date'].dt.strftime('%d-%b-%Y')
        
        return price_table
    except Exception as e:
        st.error(f"Error processing data for {district_name}: {str(e)}")
        return pd.DataFrame()

def find_nearest_district(lat, lon):
    """Find the nearest district to given coordinates"""
    min_dist = float('inf')
    nearest_district = None
    
    for district, coords in DISTRICT_COORDS.items():
        dist = ((coords[0] - lat) ** 2 + (coords[1] - lon) ** 2) ** 0.5
        if dist < 0.5:  # Reduced threshold for more precise selection
            if dist < min_dist:
                min_dist = dist
                nearest_district = district
    
    return nearest_district

def main():
    st.title("Cement Price Analysis Dashboard")
    
    # Initialize session state
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'selected_district' not in st.session_state:
        st.session_state.selected_district = None
    
    # File upload
    uploaded_file = st.file_uploader("Upload SFDC CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Show processing message
        with st.spinner("Converting SFDC file to processed district file..."):
            # Read and process data
            df = pd.read_csv(uploaded_file, encoding='latin1')
            st.session_state.processed_df = process_district_data(df)
            st.success("File processed successfully!")
        
        # Create map data
        map_data = pd.DataFrame(
            [(dist, coord[0], coord[1]) 
             for dist, coord in DISTRICT_COORDS.items()],
            columns=['District', 'lat', 'lon']
        )
        
        # Create map
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

        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            mapbox=dict(
                center=dict(lat=23.5937, lon=78.9629),
                zoom=4
            )
        )
        
        # Add click event handler
        selected_point = st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )
        
        # Handle click events
        if st.session_state.selected_district is None:
            st.session_state.selected_district = list(DISTRICT_COORDS.keys())[0]
            
        # District selection dropdown (synced with map)
        selected_district = st.selectbox(
            "Select a district to view prices",
            options=list(DISTRICT_COORDS.keys()),
            index=list(DISTRICT_COORDS.keys()).index(st.session_state.selected_district)
        )
        
        # Update selected district
        st.session_state.selected_district = selected_district
        
        # Display price table
        if selected_district:
            st.subheader(f"Price Data for {selected_district}")
            price_table = create_price_table(st.session_state.processed_df, selected_district)
            
            if len(price_table) > 0:
                st.dataframe(price_table)
            else:
                st.warning(f"No data available for {selected_district}")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
