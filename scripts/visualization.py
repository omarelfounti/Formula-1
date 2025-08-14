import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load combined data
def load_data():
    try:
        data = pd.read_csv('data/F1_2024_2025_Combined.csv')
        st.success("Data loaded successfully.")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Display data
def display_data(data):
    st.header("Formula 1 Data (2024-2025)")
    st.dataframe(data.head())

# Plotting Functions

def plot_violin(data):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Team', y='Points', data=data)
    st.pyplot(plt)

def plot_bubble(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Position', y='Points', size='Points', hue='Driver', alpha=0.6)
    plt.title('Driver Performance: Points vs Position')
    st.pyplot(plt)

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

def plot_density(data):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data['Position'], fill=True, color='blue')
    plt.title('Density of Finishing Positions')
    st.pyplot(plt)

# Interactive Plotting

def plot_parallel(data):
    from pandas.plotting import parallel_coordinates
    plt.figure(figsize=(10, 6))
    parallel_coordinates(data[['Driver', 'Points', 'Position', 'Starting Grid']], 'Driver')
    plt.title('Parallel Coordinates Plot for Drivers')
    st.pyplot(plt)

# Statistical Analysis

def plot_regression(data):
    plt.figure(figsize=(8, 5))
    sns.regplot(x='Starting Grid', y='Position', data=data)
    plt.title('Regression: Starting Grid vs Final Position')
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

# Streamlit App Main Function
def main():
    st.title("Advanced Formula 1 Data Visualization (2024-2025)")
    data = load_data()
    if data is not None:
        display_data(data)
        st.header("Advanced Visualizations")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Performance Analysis", "3D Visuals", "Driver Comparison", "Statistical Insights", "Drivers & Teams 2024-2025", "Driver Wins"])
        with tab1:
            st.subheader("Violin Plot: Points Distribution by Team")
            plot_violin(data)
            st.subheader("Bubble Plot: Driver Performance")
            plot_bubble(data)
        with tab2:
            st.subheader("3D Scatter: Starting Grid vs Final Position vs Points")
            plot_3d_scatter(data)
        with tab3:
            st.subheader("Density Plot: Finishing Positions")
            plot_density(data)
            st.subheader("Parallel Coordinates Plot: Driver Metrics")
            plot_parallel(data)
            st.subheader("Radar Chart: Driver Performance")
            plot_radar(data)
        with tab4:
            st.subheader("Regression: Starting Grid vs Final Position")
            plot_regression(data)
        with tab5:
            st.subheader("Drivers and Teams Overview (2024-2025)")
            drivers_teams = data[['Driver', 'Team']].drop_duplicates().reset_index(drop=True)
            st.dataframe(drivers_teams)
        with tab6:
            st.subheader("Driver Wins (2024-2025)")
            driver_wins = data[data['Position'] == 1]['Driver'].value_counts().reset_index()
            driver_wins.columns = ['Driver', 'Wins']
            st.dataframe(driver_wins)

if __name__ == "__main__":
    main()
