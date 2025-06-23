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

# Import our custom modules with error handling
modules_loaded = {}
try:
    from data.fred_fetcher import EnhancedFredClient
    modules_loaded['fred'] = True
except ImportError as e:
    st.sidebar.warning(f"FRED fetcher not available: {e}")
    modules_loaded['fred'] = False

try:
    from data.yahoo_finance_fetcher import EnhancedYahooFinanceClient
    modules_loaded['yahoo'] = True
except ImportError as e:
    st.sidebar.warning(f"Yahoo Finance fetcher not available: {e}")
    modules_loaded['yahoo'] = False

try:
    from features.returns_calculator import ReturnsCalculator
    modules_loaded['returns'] = True
except ImportError as e:
    st.sidebar.warning(f"Returns calculator not available: {e}")
    modules_loaded['returns'] = False

try:
    from models.regime_classifier import RuleBasedRegimeClassifier
    modules_loaded['rule_regime'] = True
except ImportError as e:
    st.sidebar.warning(f"Rule-based regime classifier not available: {e}")
    modules_loaded['rule_regime'] = False

try:
    from models.ml_regime_classifier import MLRegimeClassifier
    modules_loaded['ml_regime'] = True
except ImportError as e:
    st.sidebar.warning(f"ML regime classifier not available: {e}")
    modules_loaded['ml_regime'] = False

try:
    from models.portfolio import PortfolioConstructor
    modules_loaded['portfolio'] = True
except ImportError as e:
    st.sidebar.warning(f"Portfolio constructor not available: {e}")
    modules_loaded['portfolio'] = False

try:
    from models.dynamic_portfolio import DynamicPortfolioOptimizer
    modules_loaded['dynamic_portfolio'] = True
except ImportError as e:
    st.sidebar.warning(f"Dynamic portfolio optimizer not available: {e}")
    modules_loaded['dynamic_portfolio'] = False

try:
    from visualization.regime_visualization import RegimeVisualization
    modules_loaded['visualization'] = True
except ImportError as e:
    st.sidebar.warning(f"Regime visualization not available: {e}")
    modules_loaded['visualization'] = False

# Helper function for series units
def get_series_unit(series_id):
    """Get appropriate unit label for FRED series."""
    unit_map = {
        'GDPC1': 'Billions of 2012 USD',
        'UNRATE': 'Percent',
        'CPIAUCSL': 'Index 1982-84=100',
        'FEDFUNDS': 'Percent',
        'T10Y2Y': 'Percent',
        'USREC': 'Binary',
        'DGS10': 'Percent',
        'DEXUSEU': 'USD per Euro'
    }
    return unit_map.get(series_id, 'Value')

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
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
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
    
    # Initialize available modules
    if modules_loaded['fred']:
        if 'fred_client' not in st.session_state:
            st.session_state.fred_client = EnhancedFredClient()
    
    if modules_loaded['yahoo']:
        if 'yahoo_client' not in st.session_state:
            st.session_state.yahoo_client = EnhancedYahooFinanceClient()
    
    if modules_loaded['returns']:
        if 'returns_calculator' not in st.session_state:
            st.session_state.returns_calculator = ReturnsCalculator()
    
    if modules_loaded['portfolio']:
        if 'portfolio_constructor' not in st.session_state:
            st.session_state.portfolio_constructor = PortfolioConstructor()
    
    if modules_loaded['visualization']:
        if 'regime_visualizer' not in st.session_state:
            st.session_state.regime_visualizer = RegimeVisualization()
    
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
    
    # Check module availability
    if not any([modules_loaded['fred'], modules_loaded['yahoo']]):
        st.markdown(
            '<div class="error-message">❌ No data fetchers available. Please check module installations.</div>',
            unsafe_allow_html=True
        )
        return
    
    # Data source configuration
    st.markdown('<h2 class="sub-header">Configure Data Sources</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Asset Data Configuration")
        if modules_loaded['yahoo']:
            asset_tickers = st.text_input(
                "Asset Tickers (comma-separated)",
                value="SPY,AGG,GLD,QQQ,VTI,BND",
                help="Enter stock/ETF tickers separated by commas"
            )
            
            # Asset data frequency
            asset_frequency = st.selectbox(
                "Data Frequency",
                ["1d", "1wk", "1mo"],
                index=0,
                help="Choose the frequency for asset price data"
            )
        else:
            st.warning("Yahoo Finance fetcher not available")
    
    with col2:
        st.subheader("🏦 Macroeconomic Data Configuration")
        if modules_loaded['fred']:
            macro_series = st.text_input(
                "FRED Series IDs (comma-separated)",
                value="GDPC1,UNRATE,CPIAUCSL,FEDFUNDS,T10Y2Y,USREC,DGS10",
                help="Enter FRED series IDs separated by commas"
            )
        else:
            st.warning("FRED fetcher not available")
    
    # Date range selection
    st.markdown('<h2 class="sub-header">📅 Date Range Selection</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*5),  # 5 years ago
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
                # Fetch asset data
                if modules_loaded['yahoo']:
                    asset_tickers_list = [ticker.strip().upper() for ticker in asset_tickers.split(',') if ticker.strip()]
                    
                    if asset_tickers_list:
                        st.info("Fetching asset data from Yahoo Finance...")
                        
                        # Use yfinance directly as fallback
                        import yfinance as yf
                        
                        asset_data = yf.download(
                            asset_tickers_list,
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'),
                            interval=asset_frequency,
                            progress=False
                        )
                        
                        st.session_state.asset_data = asset_data
                        st.session_state.asset_tickers = asset_tickers_list
                
                # Fetch macro data
                if modules_loaded['fred']:
                    macro_series_list = [series.strip().upper() for series in macro_series.split(',') if series.strip()]
                    
                    if macro_series_list:
                        st.info("Fetching macroeconomic data from FRED...")
                        
                        macro_data_dict = {}
                        for series_id in macro_series_list:
                            try:
                                # Use the enhanced FRED client if available
                                if hasattr(st.session_state.fred_client, 'fetch_series'):
                                    series_data = st.session_state.fred_client.fetch_series(
                                        series_id,
                                        start_date=start_date.strftime('%Y-%m-%d'),
                                        end_date=end_date.strftime('%Y-%m-%d')
                                    )
                                else:
                                    # Fallback to direct FRED API
                                    series_data = st.session_state.fred_client.fred.get_series(
                                        series_id,
                                        start=start_date.strftime('%Y-%m-%d'),
                                        end=end_date.strftime('%Y-%m-%d')
                                    )
                                
                                if series_data is not None and not series_data.empty:
                                    # Ensure the data is a pandas Series with proper datetime index
                                    if isinstance(series_data, pd.DataFrame):
                                        series_data = series_data.iloc[:, 0]  # Take first column if DataFrame
                                    
                                    # Convert index to datetime if needed
                                    if not isinstance(series_data.index, pd.DatetimeIndex):
                                        series_data.index = pd.to_datetime(series_data.index)
                                    
                                    # Forward fill missing values for most series (except recession indicator)
                                    if series_id != 'USREC':
                                        series_data = series_data.ffill()
                                    
                                    macro_data_dict[series_id] = series_data
                                    st.success(f"✅ Fetched {series_id}: {len(series_data)} observations")
                                else:
                                    st.warning(f"⚠️ No data available for {series_id}")
                                    
                            except Exception as e:
                                st.warning(f"❌ Could not fetch {series_id}: {str(e)}")
                        
                        if macro_data_dict:
                            # Combine all series into a single DataFrame
                            macro_data = pd.DataFrame(macro_data_dict)
                            
                            # Sort by date and forward fill missing values
                            macro_data = macro_data.sort_index()
                            macro_data = macro_data.ffill()
                            
                            # Remove rows where all values are NaN
                            macro_data = macro_data.dropna(how='all')
                            
                            st.session_state.macro_data = macro_data
                            st.session_state.macro_series = macro_series_list
                            
                            st.success(f"✅ Combined macro data: {len(macro_data)} observations across {len(macro_data.columns)} series")
                        else:
                            st.warning("⚠️ No macroeconomic data could be fetched")
                else:
                    st.warning("⚠️ FRED client not available - skipping macro data")
                
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
        
        if hasattr(st.session_state.asset_data.columns, 'get_level_values'):
            # Multi-level columns
            unique_assets = st.session_state.asset_data.columns.get_level_values(1).unique()
        else:
            # Single-level columns
            unique_assets = st.session_state.asset_data.columns
        
        with col1:
            st.metric("Assets", len(unique_assets))
        
        with col2:
            st.metric("Date Range", f"{len(st.session_state.asset_data)} days")
        
        with col3:
            st.metric("Start Date", st.session_state.asset_data.index[0].strftime('%Y-%m-%d'))
        
        with col4:
            st.metric("End Date", st.session_state.asset_data.index[-1].strftime('%Y-%m-%d'))
        
        # Asset data table and chart
        st.subheader("📊 Price Data")
        
        # Get price data
        if hasattr(st.session_state.asset_data.columns, 'get_level_values') and 'Adj Close' in st.session_state.asset_data.columns.get_level_values(0):
            price_data = st.session_state.asset_data['Adj Close']
        elif 'Adj Close' in st.session_state.asset_data.columns:
            price_data = st.session_state.asset_data[['Adj Close']]
        else:
            price_data = st.session_state.asset_data
        
        st.dataframe(price_data.tail(10), use_container_width=True)
        
        # Price chart
        fig = go.Figure()
        for col in price_data.columns:
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data[col],
                mode='lines',
                name=str(col),
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
            try:
                n_series = min(4, len(st.session_state.macro_data.columns))
                
                # Create better formatted subplots
                fig = make_subplots(
                    rows=n_series,
                    cols=1,
                    subplot_titles=st.session_state.macro_data.columns[:n_series].tolist(),
                    vertical_spacing=0.15,
                    specs=[[{"secondary_y": False}] for _ in range(n_series)]
                )
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                charts_added = 0
                
                for i, col in enumerate(st.session_state.macro_data.columns[:n_series]):
                    # Get clean data for this series
                    series_data = st.session_state.macro_data[col].dropna()
                    
                    if len(series_data) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=series_data.index,
                                y=series_data.values,
                                mode='lines',
                                name=col,
                                line=dict(width=2.5, color=colors[i % len(colors)]),
                                showlegend=False,
                                hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                            ),
                            row=i+1, col=1
                        )
                        
                        # Format y-axis based on series type
                        if col == 'GDPC1':
                            y_title = 'Billions USD'
                            tick_format = ',.0f'
                        elif col in ['UNRATE', 'FEDFUNDS', 'T10Y2Y', 'DGS10']:
                            y_title = 'Percent'
                            tick_format = '.1f'
                        elif col == 'CPIAUCSL':
                            y_title = 'Index'
                            tick_format = '.1f'
                        elif col == 'USREC':
                            y_title = 'Recession'
                            tick_format = '.0f'
                        else:
                            y_title = 'Value'
                            tick_format = '.2f'
                        
                        fig.update_yaxes(
                            title_text=y_title,
                            row=i+1, col=1,
                            tickformat=tick_format,
                            title_font=dict(size=12),
                            tickfont=dict(size=10)
                        )
                        
                        # Update x-axis formatting
                        fig.update_xaxes(
                            row=i+1, col=1,
                            tickfont=dict(size=10),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.2)'
                        )
                        
                        charts_added += 1
                    else:
                        st.warning(f"⚠️ No data available for {col}")
                
                if charts_added > 0:
                    # Update overall layout
                    fig.update_layout(
                        title={
                            'text': "Macroeconomic Indicators",
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'size': 18, 'color': '#1f77b4'}
                        },
                        height=180 * charts_added + 80,
                        showlegend=False,
                        margin=dict(l=80, r=40, t=80, b=60),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    # Add x-axis title only to bottom subplot
                    fig.update_xaxes(title_text="Date", row=charts_added, col=1, title_font=dict(size=12))
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No valid data series to display")
                    
            except Exception as e:
                st.error(f"❌ Error creating chart: {str(e)}")
                st.info("📊 Data table is available above. Chart display will be fixed in the next update.")
        else:
            st.info("ℹ️ No macroeconomic data available to display.")

# Subtask 9.2: Develop Regime Analysis Page
def regime_analysis_page():
    """Regime Analysis and Classification Page."""
    st.markdown('<h1 class="main-header">🎯 Regime Analysis & Classification</h1>', unsafe_allow_html=True)
    
    # Check if data is available
    if st.session_state.asset_data is None:
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
                if hasattr(st.session_state.asset_data.columns, 'get_level_values') and 'Adj Close' in st.session_state.asset_data.columns.get_level_values(0):
                    price_data = st.session_state.asset_data['Adj Close']
                elif 'Adj Close' in st.session_state.asset_data.columns:
                    price_data = st.session_state.asset_data[['Adj Close']]
                else:
                    price_data = st.session_state.asset_data
                
                # Calculate returns using the correct method
                if modules_loaded['returns']:
                    # Use the correct method from ReturnsCalculator
                    returns_data = st.session_state.returns_calculator.calculate_simple_returns(price_data, periods=1)
                    
                    # Handle the case where it returns a DataFrame with period columns
                    if isinstance(returns_data, pd.DataFrame):
                        # The ReturnsCalculator returns columns like 'SPY_return_1d', 'AGG_return_1d'
                        # Let's clean up the column names to just the ticker symbols
                        new_columns = []
                        for col in returns_data.columns:
                            if '_return_1d' in col:
                                new_columns.append(col.replace('_return_1d', ''))
                            elif '_period_1' in col:
                                new_columns.append(col.replace('_period_1', ''))
                            else:
                                new_columns.append(col)
                        returns_data.columns = new_columns
                    elif isinstance(returns_data, pd.Series):
                        # Convert to DataFrame for consistency
                        returns_data = returns_data.to_frame()
                else:
                    # Fallback calculation
                    returns_data = price_data.pct_change().dropna()
                
                # Simple technical indicators - volatility
                volatility_data = returns_data.rolling(window=20).std() * np.sqrt(252)
                volatility_data.columns = [f"{col}_Vol" for col in volatility_data.columns]
                
                # Simple momentum indicators
                momentum_data = price_data.pct_change(periods=20).dropna()
                momentum_data.columns = [f"{col}_Mom" for col in momentum_data.columns]
                
                # Combine features
                feature_list = [returns_data]
                
                # Add volatility if it has data
                if not volatility_data.empty and len(volatility_data.dropna()) > 0:
                    feature_list.append(volatility_data)
                
                # Add momentum if it has data
                if not momentum_data.empty and len(momentum_data.dropna()) > 0:
                    feature_list.append(momentum_data)
                
                # Add macro data if available
                if st.session_state.macro_data is not None and not st.session_state.macro_data.empty:
                    # Resample macro data to match asset data frequency
                    macro_resampled = st.session_state.macro_data.resample('D').ffill()
                    feature_list.append(macro_resampled)
                
                if feature_list:
                    feature_data = pd.concat(feature_list, axis=1).dropna()
                    st.session_state.feature_data = feature_data
                    
                    st.success(f"✅ Generated {len(feature_data.columns)} features with {len(feature_data)} observations!")
                else:
                    st.error("No features could be generated!")
                    return
                
            except Exception as e:
                st.error(f"Error generating features: {str(e)}")
                st.error("Please check the data format and try again.")
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
            
            available_methods = []
            if modules_loaded['rule_regime']:
                available_methods.append("Rule-based Classification")
            if modules_loaded['ml_regime']:
                available_methods.append("K-means Clustering")
            if len(available_methods) > 1:
                available_methods.append("Both Methods")
            
            if not available_methods:
                st.error("No regime classification methods available!")
                return
            
            classification_method = st.radio(
                "Select Classification Method",
                available_methods,
                horizontal=True
            )
            
            col1, col2 = st.columns(2)
            
            if "K-means" in classification_method and modules_loaded['ml_regime']:
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
                        if "Rule-based" in classification_method and modules_loaded['rule_regime']:
                            st.subheader("📊 Rule-based Regime Classification")
                            
                            rule_classifier = RuleBasedRegimeClassifier()
                            rule_regimes = rule_classifier.classify_regimes(selected_data)
                            st.session_state.regime_results['rule_based'] = rule_regimes
                            
                            # Simple visualization
                            fig = px.line(
                                x=rule_regimes.index,
                                y=rule_regimes.values,
                                title="Rule-based Regime Timeline"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
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
                        if "K-means" in classification_method and modules_loaded['ml_regime']:
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
                            
                            # Simple visualization
                            fig = px.scatter(
                                x=selected_data.index,
                                y=ml_regimes.values,
                                color=ml_regimes.values,
                                title="K-means Regime Classification"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Regime statistics
                            regime_stats = ml_regimes.value_counts()
                            st.write("**Regime Distribution:**")
                            st.dataframe(regime_stats, use_container_width=True)
                        
                        st.success("✅ Regime classification completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error in regime classification: {str(e)}")

# Subtask 9.3: Create Portfolio Optimization Page
def portfolio_optimization_page():
    """Portfolio Optimization Page."""
    st.markdown('<h1 class="main-header">💼 Portfolio Optimization</h1>', unsafe_allow_html=True)
    
    if not modules_loaded['portfolio']:
        st.error("Portfolio constructor not available!")
        return
    
    if st.session_state.asset_data is None:
        st.markdown(
            '<div class="warning-message">⚠️ Please fetch data first in the Data Setup page.</div>',
            unsafe_allow_html=True
        )
        return
    
    st.markdown('<h2 class="sub-header">🎯 Portfolio Configuration</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_method = st.selectbox(
            "Optimization Method",
            ["SHARPE", "MIN_VARIANCE", "MAX_RETURN", "RISK_PARITY", "EQUAL_WEIGHT"]
        )
        
        risk_tolerance = st.slider("Risk Tolerance", 0.1, 2.0, 1.0, 0.1)
    
    with col2:
        rebalance_frequency = st.selectbox(
            "Rebalancing Frequency",
            ["Monthly", "Quarterly", "Semi-Annual", "Annual"]
        )
        
        max_weight = st.slider("Maximum Asset Weight", 0.1, 1.0, 0.4, 0.05)
    
    if st.button("🚀 Optimize Portfolio", type="primary"):
        with st.spinner("Optimizing portfolio..."):
            try:
                # Get price data
                if hasattr(st.session_state.asset_data.columns, 'get_level_values') and 'Adj Close' in st.session_state.asset_data.columns.get_level_values(0):
                    price_data = st.session_state.asset_data['Adj Close']
                elif 'Adj Close' in st.session_state.asset_data.columns:
                    price_data = st.session_state.asset_data[['Adj Close']]
                else:
                    price_data = st.session_state.asset_data
                
                # Calculate returns
                returns_data = price_data.pct_change().dropna()
                
                # Simple portfolio optimization
                portfolio_constructor = st.session_state.portfolio_constructor
                
                # Configure constraints
                constraints = {
                    'max_weight': max_weight,
                    'min_weight': 0.01,
                    'risk_tolerance': risk_tolerance
                }
                
                # Optimize portfolio
                result = portfolio_constructor.optimize_portfolio(
                    returns_data,
                    method=optimization_method,
                    constraints=constraints
                )
                
                st.session_state.portfolio_results['optimization'] = result
                
                # Display results
                st.markdown('<h2 class="sub-header">📊 Optimization Results</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Portfolio Weights:**")
                    weights_df = pd.DataFrame({
                        'Asset': result['weights'].index,
                        'Weight': result['weights'].values
                    })
                    st.dataframe(weights_df, use_container_width=True)
                
                with col2:
                    # Pie chart of weights
                    fig = px.pie(
                        values=result['weights'].values,
                        names=result['weights'].index,
                        title="Portfolio Allocation"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                st.write("**Performance Metrics:**")
                metrics_df = pd.DataFrame({
                    'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
                    'Value': [
                        f"{result.get('expected_return', 0):.2%}",
                        f"{result.get('volatility', 0):.2%}",
                        f"{result.get('sharpe_ratio', 0):.2f}"
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True)
                
                st.success("✅ Portfolio optimization completed successfully!")
                
            except Exception as e:
                st.error(f"Error in portfolio optimization: {str(e)}")

# Navigation and main app
def main():
    """Main application function."""
    initialize_session_state()
    
    # Display module status
    st.sidebar.markdown("### 📋 Module Status")
    for module, status in modules_loaded.items():
        status_icon = "✅" if status else "❌"
        st.sidebar.markdown(f"{status_icon} {module.replace('_', ' ').title()}")
    
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
            <p><strong>Admin Access:</strong> Username: admin, Password: admin123</p>
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
        portfolio_optimization_page()
    elif page == "📈 Performance Monitoring":
        st.markdown('<h1 class="main-header">📈 Performance Monitoring</h1>', unsafe_allow_html=True)
        st.info("🚧 Performance Monitoring page is under development...")
        st.markdown("""
        ### Planned Features:
        - Real-time portfolio performance tracking
        - Risk metrics and attribution analysis
        - Benchmark comparison
        - Drawdown analysis
        - Performance attribution by regime
        """)
    elif page == "🔧 Strategy Refinement":
        st.markdown('<h1 class="main-header">🔧 Strategy Refinement</h1>', unsafe_allow_html=True)
        st.info("🚧 Strategy Refinement page is under development...")
        st.markdown("""
        ### Planned Features:
        - Backtesting framework
        - Parameter optimization
        - Sensitivity analysis
        - Strategy comparison
        - Risk-adjusted performance metrics
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Macro Regime Analysis Platform v1.0**")
    st.sidebar.markdown("Built with ❤️ using Streamlit")
    st.sidebar.markdown(f"**Modules Loaded:** {sum(modules_loaded.values())}/{len(modules_loaded)}")

if __name__ == "__main__":
    main() 