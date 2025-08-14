import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime

# Set page config with Formula 1 theme
st.set_page_config(page_title="Formula 1 Dashboard", page_icon="🏁", layout="wide")
st.markdown("<style>body {background-color: #1e1e1e; color: white;} .stApp {background-color: #0a0a0a;} .stMetric {text-align: center;} .footer {position: fixed; bottom: 0; width: 100%; text-align: center;}</style>", unsafe_allow_html=True)

# Load combined data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data/F1_2024_2025_Combined.csv')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Display key statistics
def display_statistics(data):
    num_races = data['Track'].nunique()
    num_drivers = data['Driver'].nunique()
    num_teams = data['Team'].nunique()
    col1, col2, col3 = st.columns(3)
    col1.metric("🏎️ Number of Races", num_races)
    col2.metric("👨‍✈️ Number of Drivers", num_drivers)
    col3.metric("🏢 Number of Teams", num_teams)

# Layout header with F1 logo and dynamic date
def layout_header():
    st.markdown("<h1 style='text-align: center; color: red;'>🏎️ Formula 1 2024-2025 Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: grey;'>Updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 2px solid red;'>", unsafe_allow_html=True)

# Footer for professional touch
def layout_footer():
    st.markdown("<div class='footer'><p style='color: grey;'>Developed by Omar El Founti Khsim - Formula 1 Data Dashboard | <a href='https://www.formula1.com' target='_blank' style='color: red;'>Official F1 Website</a></p></div>", unsafe_allow_html=True)

# Streamlit App Main Function
def plot_3d_scatter(data):
    # Convert non-numeric values and drop NaN
    data['Starting Grid'] = pd.to_numeric(data['Starting Grid'], errors='coerce')
    data['Position'] = pd.to_numeric(data['Position'], errors='coerce')
    data['Points'] = pd.to_numeric(data['Points'], errors='coerce')

    # Drop rows where any of the essential columns contain NaN
    cleaned_data = data.dropna(subset=['Starting Grid', 'Position', 'Points'])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cleaned_data['Starting Grid'], cleaned_data['Position'], cleaned_data['Points'], c=cleaned_data['Points'], cmap='viridis')
    ax.set_xlabel('Starting Grid')
    ax.set_ylabel('Final Position')
    ax.set_zlabel('Points')
    plt.title('3D Scatter: Starting Grid vs Position vs Points')
    st.pyplot(fig)

from pandas.plotting import parallel_coordinates

def plot_parallel(data):
    plt.figure(figsize=(10, 6))
    # Select a subset of columns for comparison
    cols = ['Driver', 'Points', 'Position', 'Starting Grid']
    # Convert non-numeric values to NaN and drop them
    for col in cols[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # Drop rows with NaN values after conversion
    cleaned_data = data[cols].dropna()
    # Plot parallel coordinates
    parallel_coordinates(cleaned_data, class_column='Driver', colormap='viridis')
    plt.title('Parallel Coordinates Plot: Driver Metrics')
    st.pyplot(plt)


def plot_radar(data):
    drivers = data['Driver'].unique()
    selected_driver = st.selectbox("Select a driver for radar chart", drivers)
    # Filter the data for the selected driver
    driver_data = data[data['Driver'] == selected_driver]

    # Ensure numeric values only
    driver_data['Points'] = pd.to_numeric(driver_data['Points'], errors='coerce')
    driver_data['Position'] = pd.to_numeric(driver_data['Position'], errors='coerce')
    driver_data['Starting Grid'] = pd.to_numeric(driver_data['Starting Grid'], errors='coerce')

    # Drop rows with NaN values after conversion
    driver_data = driver_data.dropna(subset=['Points', 'Position', 'Starting Grid'])

    # Calculate mean values for the radar chart
    mean_values = driver_data[['Points', 'Position', 'Starting Grid']].mean()

    labels = ['Points', 'Position', 'Starting Grid']
    stats = [mean_values['Points'], mean_values['Position'], mean_values['Starting Grid']]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats += stats[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='red', alpha=0.4)
    ax.plot(angles, stats, color='red', linewidth=2)
    plt.title(f'Performance Radar: {selected_driver}')
    st.pyplot(fig)


def plot_regression(data):
    # Ensure numeric values only
    data['Starting Grid'] = pd.to_numeric(data['Starting Grid'], errors='coerce')
    data['Position'] = pd.to_numeric(data['Position'], errors='coerce')

    # Drop rows with NaN values after conversion
    cleaned_data = data.dropna(subset=['Starting Grid', 'Position'])

    plt.figure(figsize=(8, 5))
    sns.regplot(x='Starting Grid', y='Position', data=cleaned_data, scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})
    plt.title('Regression: Starting Grid vs Final Position')
    plt.xlabel('Starting Grid')
    plt.ylabel('Final Position')
    st.pyplot(plt)


def main():
    st.sidebar.title("🏁 Formula 1 Dashboard Navigation")
    layout_header()
    data = load_data()
    if data is not None:
        display_statistics(data)
        st.sidebar.subheader("Select Visualization")
        selected_tab = st.sidebar.radio("Choose a tab:", ["🏁 Performance Analysis", "📊 3D Visuals", "🔄 Driver Comparison", "📈 Statistical Insights", "🗂️ Drivers & Teams", "🏅 Driver Wins"])

        if selected_tab == "🏁 Performance Analysis":
            st.subheader("🏁 Race Laps per Driver")
            race_wins = data['Driver'].value_counts()
            fig, ax = plt.subplots()
            race_wins.plot(kind='bar', color='red', ax=ax)
            plt.title('Race Laps per Driver')
            plt.xlabel('Driver')
            plt.ylabel('Wins')
            st.pyplot(fig)

        elif selected_tab == "📊 3D Visuals":
            st.subheader("📊 3D Scatter: Starting Grid vs Position vs Points")
            plot_3d_scatter(data)

        elif selected_tab == "🔄 Driver Comparison":
            st.subheader("🔄 Parallel Coordinates Plot: Driver Metrics")
            plot_parallel(data)
            st.subheader("📊 Radar Chart: Driver Performance")
            plot_radar(data)

        elif selected_tab == "📈 Statistical Insights":
            st.subheader("📈 Regression Analysis: Starting Grid vs Final Position")
            plot_regression(data)

        elif selected_tab == "🗂️ Drivers & Teams":
            st.subheader("🗂️ Drivers and Teams Overview")
            drivers_teams = data[['Driver', 'Team']].drop_duplicates().reset_index(drop=True)
            st.dataframe(drivers_teams)

        elif selected_tab == "🏅 Driver Wins":
            st.subheader("🏅 Driver Wins (2024-2025)")
            # Convert 'Position' to numeric
            data['Position'] = pd.to_numeric(data['Position'], errors='coerce')
            # Filter drivers who won the race (Position == 1)
            driver_wins = data[data['Position'] == 1]['Driver'].value_counts().reset_index()
            driver_wins.columns = ['Driver', 'Wins']
            st.dataframe(driver_wins)

        elif selected_tab == "🏅 Driver Wins":
            st.subheader("🏅 Driver Wins (2024-2025)")
            # Convert 'Position' to numeric
            data['Position'] = pd.to_numeric(data['Position'], errors='coerce')
            # Filter drivers who won the race (Position == 1)
            driver_wins = data[data['Position'] == 1]['Driver'].value_counts().reset_index()
            driver_wins.columns = ['Driver', 'Wins']
            st.dataframe(driver_wins)

        layout_footer()

if __name__ == "__main__":
    main()
