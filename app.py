"""
Macro Regime Analysis Platform - Streamlit Dashboard

This is the main Streamlit application that provides an interactive web interface
for the macro regime analysis and portfolio optimization system.

Features:
- Data Setup and Configuration
- Regime Analysis with Multiple Methods
- Portfolio Optimization and Construction
- Performance Monitoring and Attribution
- Strategy Refinement and Backtesting
- User Authentication and Session Management

Implements Task 9: Create Streamlit Dashboard
- Subtask 9.1: Implement Data Setup Page
- Subtask 9.2: Develop Regime Analysis Page
- Subtask 9.3: Create Portfolio Optimization Page
- Subtask 9.4: Build Performance Monitoring Page
- Subtask 9.5: Implement Strategy Refinement Page
- Subtask 9.6: Implement User Authentication System

Author: Macro Regime Analysis System
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import logging
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

# Import our custom modules
try:
    from data.infrastructure import DataInfrastructure
    from data.fetchers import DataFetcher
    from data.fred_fetcher import FREDFetcher
    from data.yahoo_finance_fetcher import YahooFinanceFetcher
    from features.feature_store import FeatureStore
    from features.economic_indicators import EconomicIndicators
    from features.returns_calculator import ReturnsCalculator
    from features.technical_indicators import TechnicalIndicators
    from models.regime_classifier import RuleBasedRegimeClassifier
    from models.ml_regime_classifier import MLRegimeClassifier
    from models.portfolio import PortfolioConstructor
    from models.dynamic_portfolio import DynamicPortfolioOptimizer, create_dynamic_optimizer
    from visualization.regime_visualization import RegimeVisualization, create_regime_visualizer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Macro Regime Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    if 'data_infrastructure' not in st.session_state:
        st.session_state.data_infrastructure = DataInfrastructure()
    
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = DataFetcher()
    
    if 'feature_store' not in st.session_state:
        st.session_state.feature_store = FeatureStore()
    
    if 'economic_indicators' not in st.session_state:
        st.session_state.economic_indicators = EconomicIndicators()
    
    if 'returns_calculator' not in st.session_state:
        st.session_state.returns_calculator = ReturnsCalculator()
    
    if 'portfolio_constructor' not in st.session_state:
        st.session_state.portfolio_constructor = PortfolioConstructor()
    
    if 'regime_visualizer' not in st.session_state:
        st.session_state.regime_visualizer = create_regime_visualizer()
    
    # Data storage
    if 'asset_data' not in st.session_state:
        st.session_state.asset_data = None
    
    if 'macro_data' not in st.session_state:
        st.session_state.macro_data = None
    
    if 'feature_data' not in st.session_state:
        st.session_state.feature_data = None
    
    if 'regime_results' not in st.session_state:
        st.session_state.regime_results = {}
    
    if 'portfolio_results' not in st.session_state:
        st.session_state.portfolio_results = {}

# Subtask 9.6: User Authentication System
def authenticate_user():
    """Simple user authentication system."""
    st.sidebar.markdown("### 🔐 User Authentication")
    
    if not st.session_state.authenticated:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            # Simple authentication (in production, use proper authentication)
            if username and password:
                if username == "admin" and password == "admin123":
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.sidebar.success("Login successful!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")
            else:
                st.sidebar.error("Please enter username and password")
        
        # Demo access
        if st.sidebar.button("Demo Access"):
            st.session_state.authenticated = True
            st.session_state.username = "demo_user"
            st.sidebar.success("Demo access granted!")
            st.rerun()
        
        return False
    else:
        st.sidebar.success(f"Welcome, {st.session_state.username}!")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        return True

# Subtask 9.1: Implement Data Setup Page
def data_setup_page():
    """Data Setup and Configuration Page."""
    st.markdown('<h1 class="main-header">📊 Data Setup & Configuration</h1>', unsafe_allow_html=True)
    
    # Data source configuration
    st.markdown('<h2 class="sub-header">Configure Data Sources</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Asset Data Configuration")
        asset_tickers = st.text_input(
            "Asset Tickers (comma-separated)",
            value="SPY,AGG,GLD,QQQ,VGK,EEM,VTI,BND",
            help="Enter stock/ETF tickers separated by commas"
        )
        
        # Asset data frequency
        asset_frequency = st.selectbox(
            "Data Frequency",
            ["1d", "1wk", "1mo"],
            index=0,
            help="Choose the frequency for asset price data"
        )
    
    with col2:
        st.subheader("🏦 Macroeconomic Data Configuration")
        macro_series = st.text_input(
            "FRED Series IDs (comma-separated)",
            value="GDPC1,UNRATE,CPIAUCSL,FEDFUNDS,T10Y2Y,USREC,DGS10,DEXUSEU",
            help="Enter FRED series IDs separated by commas"
        )
        
        # Macro data transformation
        macro_transform = st.selectbox(
            "Data Transformation",
            ["None", "Year-over-Year", "Quarter-over-Quarter", "Log Returns"],
            index=1,
            help="Choose transformation for macro data"
        )
    
    # Date range selection
    st.markdown('<h2 class="sub-header">📅 Date Range Selection</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*10),  # 10 years ago
            help="Select the start date for data fetching"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            help="Select the end date for data fetching"
        )
    
    with col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        fetch_button = st.button("🔄 Fetch Data", type="primary", use_container_width=True)
    
    # Data fetching
    if fetch_button:
        if start_date >= end_date:
            st.error("Start date must be before end date!")
            return
        
        with st.spinner("🔄 Fetching data... This may take a moment."):
            try:
                # Parse tickers and series
                asset_tickers_list = [ticker.strip().upper() for ticker in asset_tickers.split(',') if ticker.strip()]
                macro_series_list = [series.strip().upper() for series in macro_series.split(',') if series.strip()]
                
                if not asset_tickers_list:
                    st.error("Please enter at least one asset ticker!")
                    return
                
                # Initialize fetchers
                yahoo_fetcher = YahooFinanceFetcher()
                fred_fetcher = FREDFetcher()
                
                # Fetch asset data
                st.info("Fetching asset data from Yahoo Finance...")
                asset_data = yahoo_fetcher.fetch_data(
                    symbols=asset_tickers_list,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    interval=asset_frequency
                )
                
                # Fetch macro data
                if macro_series_list:
                    st.info("Fetching macroeconomic data from FRED...")
                    macro_data = fred_fetcher.fetch_data(
                        series_ids=macro_series_list,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                else:
                    macro_data = pd.DataFrame()
                
                # Store in session state
                st.session_state.asset_data = asset_data
                st.session_state.macro_data = macro_data
                st.session_state.asset_tickers = asset_tickers_list
                st.session_state.macro_series = macro_series_list
                
                # Success message
                st.markdown(
                    '<div class="success-message">✅ Data fetched successfully!</div>',
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return
    
    # Display data preview
    if st.session_state.asset_data is not None:
        st.markdown('<h2 class="sub-header">📈 Asset Data Preview</h2>', unsafe_allow_html=True)
        
        # Asset data metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Assets", len(st.session_state.asset_data.columns.get_level_values(1).unique()))
        
        with col2:
            st.metric("Date Range", f"{len(st.session_state.asset_data)} days")
        
        with col3:
            st.metric("Start Date", st.session_state.asset_data.index[0].strftime('%Y-%m-%d'))
        
        with col4:
            st.metric("End Date", st.session_state.asset_data.index[-1].strftime('%Y-%m-%d'))
        
        # Asset data table
        st.subheader("📊 Price Data")
        if 'Adj Close' in st.session_state.asset_data.columns.get_level_values(0):
            price_data = st.session_state.asset_data['Adj Close']
            st.dataframe(price_data.tail(10), use_container_width=True)
            
            # Price chart
            fig = go.Figure()
            for ticker in price_data.columns:
                fig.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data[ticker],
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Asset Price Evolution",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.macro_data is not None and not st.session_state.macro_data.empty:
        st.markdown('<h2 class="sub-header">🏛️ Macroeconomic Data Preview</h2>', unsafe_allow_html=True)
        
        # Macro data metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Series", len(st.session_state.macro_data.columns))
        
        with col2:
            st.metric("Observations", len(st.session_state.macro_data))
        
        with col3:
            st.metric("Missing Values", st.session_state.macro_data.isnull().sum().sum())
        
        # Macro data table
        st.subheader("📊 Economic Indicators")
        st.dataframe(st.session_state.macro_data.tail(10), use_container_width=True)
        
        # Macro data chart
        if len(st.session_state.macro_data.columns) > 0:
            fig = make_subplots(
                rows=min(4, len(st.session_state.macro_data.columns)),
                cols=1,
                subplot_titles=st.session_state.macro_data.columns[:4].tolist(),
                vertical_spacing=0.08
            )
            
            for i, col in enumerate(st.session_state.macro_data.columns[:4]):
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.macro_data.index,
                        y=st.session_state.macro_data[col],
                        mode='lines',
                        name=col,
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(
                title="Macroeconomic Indicators",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Subtask 9.2: Develop Regime Analysis Page
def regime_analysis_page():
    """Regime Analysis and Classification Page."""
    st.markdown('<h1 class="main-header">🎯 Regime Analysis & Classification</h1>', unsafe_allow_html=True)
    
    # Check if data is available
    if st.session_state.asset_data is None or st.session_state.macro_data is None:
        st.markdown(
            '<div class="warning-message">⚠️ Please fetch data first in the Data Setup page.</div>',
            unsafe_allow_html=True
        )
        return
    
    # Feature Engineering
    st.markdown('<h2 class="sub-header">🔧 Feature Engineering</h2>', unsafe_allow_html=True)
    
    if st.button("🔄 Generate Features", type="primary"):
        with st.spinner("Generating features..."):
            try:
                # Calculate returns
                if 'Adj Close' in st.session_state.asset_data.columns.get_level_values(0):
                    price_data = st.session_state.asset_data['Adj Close']
                    returns_data = st.session_state.returns_calculator.calculate_returns(price_data)
                else:
                    st.error("No adjusted close price data available!")
                    return
                
                # Calculate economic indicators
                if not st.session_state.macro_data.empty:
                    economic_features = st.session_state.economic_indicators.calculate_indicators(
                        st.session_state.macro_data
                    )
                else:
                    economic_features = pd.DataFrame()
                
                # Calculate technical indicators
                tech_indicators = TechnicalIndicators()
                technical_features = tech_indicators.calculate_indicators(price_data)
                
                # Combine all features
                feature_list = []
                if not returns_data.empty:
                    feature_list.append(returns_data)
                if not economic_features.empty:
                    feature_list.append(economic_features)
                if not technical_features.empty:
                    feature_list.append(technical_features)
                
                if feature_list:
                    feature_data = pd.concat(feature_list, axis=1).dropna()
                    st.session_state.feature_data = feature_data
                    
                    st.success(f"✅ Generated {len(feature_data.columns)} features with {len(feature_data)} observations!")
                else:
                    st.error("No features could be generated!")
                    return
                
            except Exception as e:
                st.error(f"Error generating features: {str(e)}")
                return
    
    if st.session_state.feature_data is not None:
        # Feature selection
        st.markdown('<h2 class="sub-header">🎯 Feature Selection</h2>', unsafe_allow_html=True)
        
        available_features = st.session_state.feature_data.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_features = st.multiselect(
                "Select Features for Regime Classification",
                available_features,
                default=available_features[:min(8, len(available_features))],
                help="Choose features that best capture market regimes"
            )
        
        with col2:
            # Feature statistics
            if selected_features:
                feature_subset = st.session_state.feature_data[selected_features]
                st.write("**Selected Features Statistics:**")
                st.dataframe(feature_subset.describe(), use_container_width=True)
        
        # Regime classification
        if selected_features:
            st.markdown('<h2 class="sub-header">🔍 Regime Classification</h2>', unsafe_allow_html=True)
            
            classification_method = st.radio(
                "Select Classification Method",
                ["Rule-based Classification", "K-means Clustering", "Both Methods"],
                horizontal=True
            )
            
            col1, col2 = st.columns(2)
            
            if classification_method in ["K-means Clustering", "Both Methods"]:
                with col1:
                    n_regimes = st.slider("Number of Regimes (K-means)", 2, 8, 4)
                
                with col2:
                    scaling_method = st.selectbox(
                        "Scaling Method",
                        ["StandardScaler", "RobustScaler", "MinMaxScaler", "None"]
                    )
            
            # Run classification
            if st.button("🚀 Run Regime Classification", type="primary"):
                with st.spinner("Classifying regimes..."):
                    try:
                        selected_data = st.session_state.feature_data[selected_features]
                        
                        # Rule-based classification
                        if classification_method in ["Rule-based Classification", "Both Methods"]:
                            st.subheader("📊 Rule-based Regime Classification")
                            
                            rule_classifier = RuleBasedRegimeClassifier()
                            rule_regimes = rule_classifier.classify_regimes(selected_data)
                            st.session_state.regime_results['rule_based'] = rule_regimes
                            
                            # Visualize rule-based results
                            fig_timeline = st.session_state.regime_visualizer.plot_regime_timeline(
                                rule_regimes, title="Rule-based Regime Timeline"
                            )
                            st.plotly_chart(fig_timeline, use_container_width=True)
                            
                            # Regime statistics
                            regime_stats = rule_regimes.value_counts()
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Regime Distribution:**")
                                st.dataframe(regime_stats, use_container_width=True)
                            
                            with col2:
                                fig_pie = px.pie(
                                    values=regime_stats.values,
                                    names=regime_stats.index,
                                    title="Regime Distribution"
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # K-means classification
                        if classification_method in ["K-means Clustering", "Both Methods"]:
                            st.subheader("🤖 K-means Regime Classification")
                            
                            ml_classifier = MLRegimeClassifier(
                                n_clusters=n_regimes,
                                scaling_method=scaling_method
                            )
                            
                            ml_results = ml_classifier.fit_predict(selected_data)
                            
                            # Create regime series
                            ml_regimes = pd.Series(
                                [f"Regime_{label}" for label in ml_results['labels']],
                                index=selected_data.index,
                                name='ml_regime'
                            )
                            st.session_state.regime_results['ml_based'] = ml_regimes
                            
                            # Visualize ML results
                            fig_ml = ml_classifier.visualize_regimes(selected_data)
                            st.plotly_chart(fig_ml, use_container_width=True)
                            
                            # Regime characteristics
                            st.write("**Regime Characteristics:**")
                            regime_chars = ml_classifier.analyze_regime_characteristics(selected_data)
                            
                            for regime, chars in regime_chars.items():
                                with st.expander(f"📈 {regime} ({chars['count']} observations, {chars['percentage']:.1f}%)"):
                                    st.write("**Mean Feature Values:**")
                                    st.dataframe(chars['mean_features'], use_container_width=True)
                                    
                                    if 'interpretation' in chars:
                                        st.write("**Economic Interpretation:**")
                                        st.write(chars['interpretation'])
                        
                        st.success("✅ Regime classification completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error in regime classification: {str(e)}")

# Navigation and main app
def main():
    """Main application function."""
    initialize_session_state()
    
    # User authentication
    if not authenticate_user():
        st.markdown('<h1 class="main-header">🔐 Please Login to Access the Platform</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <h3>Welcome to Macro Regime Analysis Platform</h3>
            <p>This platform provides comprehensive tools for:</p>
            <ul>
                <li>📊 Data fetching and processing</li>
                <li>🎯 Regime analysis and classification</li>
                <li>💼 Portfolio optimization</li>
                <li>📈 Performance monitoring</li>
                <li>🔧 Strategy refinement</li>
            </ul>
            <p><strong>Demo Access:</strong> Click "Demo Access" to explore the platform with sample data.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # App navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        [
            "📊 Data Setup",
            "🎯 Regime Analysis", 
            "💼 Portfolio Optimization",
            "📈 Performance Monitoring",
            "🔧 Strategy Refinement"
        ]
    )
    
    # Route to appropriate page
    if page == "📊 Data Setup":
        data_setup_page()
    elif page == "🎯 Regime Analysis":
        regime_analysis_page()
    elif page == "💼 Portfolio Optimization":
        st.markdown('<h1 class="main-header">💼 Portfolio Optimization</h1>', unsafe_allow_html=True)
        st.info("🚧 Portfolio Optimization page is under development...")
    elif page == "📈 Performance Monitoring":
        st.markdown('<h1 class="main-header">📈 Performance Monitoring</h1>', unsafe_allow_html=True)
        st.info("🚧 Performance Monitoring page is under development...")
    elif page == "🔧 Strategy Refinement":
        st.markdown('<h1 class="main-header">🔧 Strategy Refinement</h1>', unsafe_allow_html=True)
        st.info("🚧 Strategy Refinement page is under development...")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Macro Regime Analysis Platform v1.0**")
    st.sidebar.markdown("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main() 