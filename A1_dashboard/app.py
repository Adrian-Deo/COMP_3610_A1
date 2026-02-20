import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import requests
import os
from pathlib import Path

st.set_page_config(
    page_title='NYC Taxi Dashboard 2024',
    page_icon='🚕',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)



trip_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
trip_data_file = "yellow_tripdata_2024-01.parquet"

zone_data_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
zone_data_file = "taxi_zone_lookup.csv"

@st.cache_data
def download_file(url, file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Downloading from: {url}")
        response = requests.get(url)
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        file_size_mb = os.path.getsize(file_path) / 1e6
        print(f"Downloaded: {file_path.name} ({file_size_mb:.1f} MB)")
        return True
    else:
        file_size_mb = os.path.getsize(file_path) / 1e6
        print(f"File already exists: {file_path.name} ({file_size_mb:.1f} MB)")
        return False

download_file(trip_data_url, trip_data_file)

df = pl.read_parquet(trip_data_file)

df = df.with_columns([
    pl.col('tpep_pickup_datetime').dt.hour().alias('pickup_hour'),
    (pl.col('tpep_pickup_datetime').dt.weekday() - 1).alias('pickup_weekday'),  
    pl.col('tpep_pickup_datetime').dt.date().alias('pickup_date'),
    ((pl.col('tpep_dropoff_datetime') - pl.col('tpep_pickup_datetime')).dt.total_seconds() / 60).alias('trip_duration_min'),
    ((pl.col('tip_amount') / pl.col('fare_amount')) * 100).fill_null(0).alias('tip_pct')
])

df = df.drop_nulls(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "fare_amount"])
df = df.filter(pl.col("trip_distance").is_not_null() & (pl.col("trip_distance") > 0))
df = df.filter((pl.col('fare_amount') > 0) & (pl.col('fare_amount') < 500))
df = df.filter(pl.col('tpep_dropoff_datetime') > pl.col('tpep_pickup_datetime'))


st.markdown('<p class="main-header">NYC Taxi Trip Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Exploring Yellow Taxi Data from January 2024 For Assignemnt 1</p>', unsafe_allow_html=True)

st.divider()

st.subheader('Key Metrics at a Glance')

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Trips",
        value=f"{len(df):,}",
        help="Number of trips in our sample"
    )

with col2:
    avg_fare = df['fare_amount'].mean()
    st.metric(
        label="Average Fare",
        value=f"${avg_fare:.2f}",
        help="Mean fare - though the median might be more representative"
    )

with col3:
    avg_distance = df['trip_distance'].mean()
    st.metric(
        label="Avg Distance",
        value=f"{avg_distance:.2f} mi",
        help="Most NYC taxi trips are pretty short, actually"
    )

with col4:
    avg_duration = df['trip_duration_min'].mean()
    st.metric(
        label="Avg Duration",
        value=f"{avg_duration:.1f} min",
        help="Includes time stuck in traffic, of course"
    )

with col5:
    total_revenue = df['fare_amount'].sum()
    st.metric(
        label="Total Revenue",
        value=f"{total_revenue:.2f}",
        help="The total fare amount collected in our sample (not including tips)"
    )
