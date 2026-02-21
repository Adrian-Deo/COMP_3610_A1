import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
from pathlib import Path

st.set_page_config(page_title="Visualization Charts", page_icon="📊", layout="wide")

st.title("Required Visualizations")

trip_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
trip_data_file = "yellow_tripdata_2024-01.parquet"

zone_data_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
zone_data_file = "taxi_zone_lookup.csv"

@st.cache_data
def download_file(url, file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
            file_size_mb = os.path.getsize(file_path) / 1e6
            print(f"Downloaded: {file_path.name} ({file_size_mb:.1f} MB)")
            return True
        except Exception as e:
            st.error(f"Failed to download {url}: {e}")
            st.stop()
    else:
        file_size_mb = os.path.getsize(file_path) / 1e6
        print(f"File already exists: {file_path.name} ({file_size_mb:.1f} MB)")
        return False

@st.cache_data
def load_data():
    download_file(trip_data_url, trip_data_file)
    download_file(zone_data_url, zone_data_file)

    cols = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'PULocationID', 'DOLocationID',
        'passenger_count', 'trip_distance',
        'fare_amount', 'tip_amount', 'payment_type'
    ]
    df = pd.read_parquet(trip_data_file, columns=cols)
    zone_lookup = pd.read_csv(zone_data_file)[['LocationID', 'Zone']]

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    df['trip_duration_min'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    ).dt.total_seconds() / 60
    df['tip_pct'] = (df['tip_amount'] / df['fare_amount'] * 100).fillna(0)

    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)]
    df = df[(df['trip_duration_min'] > 1) & (df['trip_duration_min'] < 180)]

    df['PULocationID'] = df['PULocationID'].astype('category')
    df['DOLocationID'] = df['DOLocationID'].astype('category')

    payment_map = {1: 'Credit Card', 2: 'Cash', 3: 'No Charge', 4: 'Dispute', 5: 'Unknown'}
    df['payment_name'] = df['payment_type'].map(payment_map)

    return df, zone_lookup

with st.spinner('Loading data for visualizations...'):
    df, zone_lookup = load_data()

# ============== SIDEBAR FILTERS ==============
st.sidebar.header("Filters")

st.sidebar.subheader("Date Range")
min_date = df['pickup_date'].min()
max_date = df['pickup_date'].max()

date_range = st.sidebar.date_input(
    "Pick your dates:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

st.sidebar.subheader("Hour Range")
hour_min, hour_max = st.sidebar.slider(
    "Select hour of day:",
    min_value=0,
    max_value=23,
    value=(0, 23),
    step=1
)

st.sidebar.subheader("Passengers")
passenger_options = ['All'] + sorted(df['passenger_count'].dropna().unique().astype(int).tolist())
selected_passengers = st.sidebar.selectbox("How many riders?", passenger_options)

st.sidebar.subheader("Fare Range")
fare_min, fare_max = st.sidebar.slider(
    "Fare ($):",
    min_value=0.0,
    max_value=float(df['fare_amount'].max()),
    value=(0.0, 300.0),
    step=1.0
)

st.sidebar.subheader("Distance")
dist_min, dist_max = st.sidebar.slider(
    "Trip distance (miles):",
    min_value=0.0,
    max_value=float(df['trip_distance'].max()),
    value=(0.0, 30.0),
    step=0.5
)

st.sidebar.subheader("Payment")
payment_options = df['payment_name'].dropna().unique().tolist()
selected_payments = st.sidebar.multiselect(
    "Payment method(s):",
    options=payment_options,
    default=payment_options
)

# ============== APPLY FILTERS ==============
# No copy — just filter directly
filtered_df = df[
    (df['pickup_date'] >= start_date) &
    (df['pickup_date'] <= end_date) &
    (df['pickup_hour'] >= hour_min) &
    (df['pickup_hour'] <= hour_max) &
    (df['fare_amount'] >= fare_min) &
    (df['fare_amount'] <= fare_max) &
    (df['trip_distance'] >= dist_min) &
    (df['trip_distance'] <= dist_max)
]

if selected_passengers != 'All':
    filtered_df = filtered_df[filtered_df['passenger_count'] == selected_passengers]

if selected_payments:
    filtered_df = filtered_df[filtered_df['payment_name'].isin(selected_payments)]

st.sidebar.divider()
st.sidebar.metric("Filtered Trips", f"{len(filtered_df):,}")
st.sidebar.caption(f"out of {len(df):,} ({len(filtered_df)/len(df)*100:.1f}%)")

# ============== THE ACTUAL CHARTS ==============
if len(filtered_df) == 0:
    st.warning("No trips match those filters.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Pie Chart", "Histogram", "Line Graph", "Heatmap", "Bar Chart"]
)

with tab1:
    st.subheader("Breakdown of payment types")
    st.caption("Credit card payments dominate, while cash has second most amount of trips. Both 'No Charge' and 'Dispute' are very rare, and 'Unknown' is negligible.")

    payment_counts = filtered_df['payment_name'].value_counts().reset_index()
    payment_counts.columns = ['payment_type', 'count']

    fig = px.pie(
        payment_counts,
        values='count',
        names='payment_type',
        title='Payment Method Breakdown',
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)  # Replace with width='stretch' after deprecation

with tab2:
    st.subheader("Distribution of trip distances")
    st.caption("Most trips are short, but there are some long outliers. The median distance is around 1.7 miles.")

    fig = px.histogram(
        filtered_df,
        x='trip_distance',
        nbins=50,
        title='Distance Distribution',
        labels={'trip_distance': 'Distance (miles)', 'count': 'Trips'},
        color_discrete_sequence=['#E74C3C']
    )
    fig.add_vline(
        x=filtered_df['trip_distance'].median(),
        line_dash='dash',
        line_color='blue',
        annotation_text=f"Median: {filtered_df['trip_distance'].median():.2f} mi"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Average fare by hour of day")
    st.caption("The price of the trips tends to be higher during the morning and evening rush hours, likely due to increased demand and traffic congestion.")

    hourly_fare = filtered_df.groupby('pickup_hour')['fare_amount'].mean().reset_index(name='avg_fare')
    fig2 = px.line(
        hourly_fare,
        x='pickup_hour',
        y='avg_fare',
        title='Average Fare by Hour of Day',
        labels={'pickup_hour': 'Hour', 'avg_fare': 'Average Fare ($)'},
        markers=True
    )
    fig2.update_layout(height=400)
    fig2.update_xaxes(tickmode='linear', tick0=0, dtick=1)
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.subheader("Trips by day of week and hour")
    st.caption("The heatmap displays Monday–Friday on weekdays rush hours (8–9am and 5–6pm) are the busiest times for taxi trips, while late nights and weekends are decently not as busy.")

    heatmap_data = filtered_df.groupby(['pickup_weekday', 'pickup_hour']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(columns=range(24), fill_value=0)
    heatmap_data = heatmap_data.reindex(index=range(7), fill_value=0)

    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    heatmap_data.index = [weekday_names[i] for i in heatmap_data.index]

    fig = px.imshow(
        heatmap_data,
        labels=dict(x='Hour of Day', y='Day of Week', color='Trips'),
        x=list(range(24)),
        y=list(heatmap_data.index),
        color_continuous_scale='YlOrRd',
        title='When Are Taxis Busiest?'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Top 10 pickup zones by trip count")
    st.caption("The busiest pickup zones are mostly in Manhattan, especially around Midtown and the Financial District. Some popular areas in Brooklyn and Queens also make the list.")

    zone_counts = filtered_df['PULocationID'].value_counts().reset_index()
    zone_counts.columns = ['LocationID', 'trip_count']
    zone_counts = zone_counts.merge(zone_lookup, on='LocationID', how='left')
    zone_counts['Zone'] = zone_counts['Zone'].fillna(zone_counts['LocationID'].astype(str))
    top_zones = zone_counts.head(10).sort_values('trip_count', ascending=True)

    fig = px.bar(
        top_zones,
        x='trip_count',
        y='Zone',
        orientation='h',
        title='Top 10 Pickup Zones',
        labels={'trip_count': 'Number of Trips', 'Zone': ''},
        color='trip_count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
