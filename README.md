# Crime-Prediction-And-Simulation-Model-For-Urban-Safety

Project Description 

    This project is a machine learning-based crime prediction system that identifies crime hotspots 
    using spatial and temporal data analysis.
    
    It combines deep learning and statistical models to predict crime probability and visualize 
    high-risk areas on an interactive map.
    
    The system uses a ConvLSTM model to capture spatial-temporal patterns, along with a Weather Model and 
    Time Series Model (SARIMAX) to improve prediction accuracy.
    
    It also includes an agent-based simulation model to simulate crime behavior and movement.
    
    An interactive dashboard built with Streamlit allows users to explore crime predictions, hotspot maps, 
    and performance metrics in real time.

Key Features

    Crime hotspot prediction using ConvLSTM 
    Weather-based crime adjustment (Random Forest Model)
    Time series forecasting using SARIMAX
    Interactive 2D & 3D crime maps
    Agent-based crime simulation (Mesa framework)
    Model performance evaluation (Accuracy, Precision, Recall, F1-score)

Technologies Used
    
    Python
    PyTorch
    Scikit-learn
    Streamlit
    Folium & PyDeck
    Pandas, NumPy
    Mesa (Agent-Based Modeling)

Output
    
    Crime risk visualization (Grid Map & Heatmap)
    3D hotspot map
    Top high-risk areas
    Model performance metrics

Use Case

    Helps law enforcement identify crime-prone areas
    Supports data-driven policing decisions
    Useful for research and smart city applications

How to Run

Install requirements:

    pip install -r requirements.txt

Prediction:

    conda activate crime_env
    streamlit run GUI.py

Simulation:

    Python CrimeModelRun.py
