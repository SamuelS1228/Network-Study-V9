import pydeck as pdk
import pandas as pd
import streamlit as st

def plot_network_map(stores_df, centers):
    st.subheader("Store & Warehouse Map")
    store_layer = pdk.Layer(
        'ScatterplotLayer',
        data=stores_df,
        get_position='[Longitude, Latitude]',
        get_color='[0, 128, 255]',
        get_radius=20000,
        opacity=0.6,
    )
    center_df = pd.DataFrame(centers, columns=['Longitude', 'Latitude'])
    warehouse_layer = pdk.Layer(
        'ScatterplotLayer',
        data=center_df,
        get_position='[Longitude, Latitude]',
        get_color='[255, 0, 0]',
        get_radius=40000,
        opacity=0.8,
    )
    view_state = pdk.ViewState(latitude=39, longitude=-98, zoom=3.5)
    r = pdk.Deck(layers=[store_layer, warehouse_layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v10')
    st.pydeck_chart(r)

def plot_summary_charts(stores_df):
    st.subheader("Stores per Warehouse")
    st.bar_chart(stores_df['Warehouse'].value_counts().sort_index())
