# Import Packages
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Options Volatility Surface Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Black-Scholes Functions
def black_scholes_call(S, K, T, r, sigma, q=0):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma, q=0):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    return put_price

def implied_volatility(option_price, S, K, T, r, q=0, option_type='call'):
    def objective(sigma):
        if option_type == 'call':
            return black_scholes_call(S, K, T, r, sigma, q) - option_price
        else:
            return black_scholes_put(S, K, T, r, sigma, q) - option_price
    
    try:
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
        return iv if iv > 0.001 else np.nan
    except:
        return np.nan

# Greek Calculations
def calculate_greeks(S, K, T, r, sigma, q=0, option_type='call'):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    greeks = {}
    
    if option_type == 'call':
        greeks['delta'] = np.exp(-q*T) * norm.cdf(d1)
        greeks['theta'] = (-S*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) 
                          - r*K*np.exp(-r*T)*norm.cdf(d2) 
                          + q*S*np.exp(-q*T)*norm.cdf(d1))
    else:
        greeks['delta'] = -np.exp(-q*T) * norm.cdf(-d1)
        greeks['theta'] = (-S*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) 
                          + r*K*np.exp(-r*T)*norm.cdf(-d2) 
                          - q*S*np.exp(-q*T)*norm.cdf(-d1))
    
    greeks['gamma'] = norm.pdf(d1) * np.exp(-q*T) / (S * sigma * np.sqrt(T))
    greeks['vega'] = S * norm.pdf(d1) * np.exp(-q*T) * np.sqrt(T)
    greeks['rho'] = K * T * np.exp(-r*T) * (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2))
    
    return greeks

# Data Fetching Functions
@st.cache_data(ttl=300)
def get_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            st.error(f"No data available for {ticker}")
            return None, None, None
        
        spot_price = float(hist['Close'].iloc[-1])
        
        info = stock.info
        dividend_yield = float(info.get('dividendYield', 0)) if info else 0.0
        
        return hist, spot_price, dividend_yield
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None, None

@st.cache_data(ttl=300)
def get_options_data_cached(ticker, option_type='both'):
    try:
        stock = yf.Ticker(ticker)
        expiration_dates = list(stock.options)
        
        if option_type == 'both':
            calls_list = []
            puts_list = []
            
            for date in expiration_dates:
                chain = stock.option_chain(date)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                
                calls['expiration'] = date
                puts['expiration'] = date
                calls['optionType'] = 'call'
                puts['optionType'] = 'put'
                
                calls_list.append(calls)
                puts_list.append(puts)
            
            all_options = pd.concat(calls_list + puts_list, ignore_index=True)
        else:
            options_list = []
            for date in expiration_dates:
                chain = stock.option_chain(date)
                options = chain.calls if option_type == 'call' else chain.puts
                options['expiration'] = date
                options['optionType'] = option_type
                options_list.append(options)
            
            all_options = pd.concat(options_list, ignore_index=True)
        
        return all_options, expiration_dates
    except Exception as e:
        st.error(f"Error fetching options data: {str(e)}")
        return pd.DataFrame(), []

def calculate_time_to_expiry(expiry_date):
    if isinstance(expiry_date, str):
        expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
    else:
        expiry = expiry_date
    
    days_to_expiry = (expiry - datetime.now()).days
    return max(days_to_expiry / 365.0, 0)

# Main Application
def main():
    st.title("Advanced Options Volatility Surface Analyzer")
    st.markdown("### Interactive visualization and analysis of implied volatility surfaces")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input("Ticker Symbol", value="SPY", help="Enter a valid stock ticker")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Validate"):
                with st.spinner("Validating ticker..."):
                    test = yf.Ticker(ticker)
                    if test.history(period="1d").empty:
                        st.error("Invalid ticker!")
                    else:
                        st.success("Valid!")
        
        st.subheader("Market Parameters")
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=5.0, 
            step=0.1,
            help="Current risk-free interest rate"
        ) / 100
        
        st.subheader("Option Configuration")
        option_type = st.selectbox(
            "Option Type", 
            ["Calls", "Puts", "Both"],
            help="Select which option types to analyze"
        )
        
        view_type = st.selectbox(
            "Y-Axis Display", 
            ["Strike Price", "Moneyness", "Delta"],
            help="Choose how to display the y-axis"
        )
        
        with st.expander("Advanced Filters"):
            min_volume = st.number_input("Minimum Volume", min_value=0, value=10)
            min_open_interest = st.number_input("Minimum Open Interest", min_value=0, value=50)
            min_dte = st.slider("Minimum Days to Expiry", 0, 365, 30)
            max_dte = st.slider("Maximum Days to Expiry", 0, 730, 365)
        
        st.subheader("Visualization Options")
        color_scheme = st.selectbox(
            "Color Scheme", 
            ["Viridis", "Plasma", "Inferno", "Turbo", "Rainbow", "Portland"]
        )
        
        show_points = st.checkbox("Show Data Points", value=False)
        smoothing = st.slider("Surface Smoothing", 10, 100, 50)
    
    # Main Analysis
    if st.button("Generate Analysis", type="primary", use_container_width=True):
        with st.spinner("Fetching market data..."):
            hist, spot_price, div_yield = get_stock_data(ticker)
            
            if hist is None:
                return
            
            st.markdown("### Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${spot_price:.2f}")
            with col2:
                daily_change = hist['Close'].pct_change().iloc[-1] * 100
                st.metric("Daily Change", f"{daily_change:.2f}%", delta=f"{daily_change:.2f}%")
            with col3:
                st.metric("Dividend Yield", f"{div_yield*100:.2f}%")
            with col4:
                hist_vol = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Historical Vol", f"{hist_vol:.1f}%")
            
            option_filter = 'call' if option_type == "Calls" else 'put' if option_type == "Puts" else 'both'
            options_data, exp_dates = get_options_data_cached(ticker, option_filter)
            
            if options_data.empty:
                st.error("No options data available")
                return
            
            # Process Options Data
            with st.spinner("Calculating implied volatilities..."):
                options_data['timeToExpiry'] = options_data['expiration'].apply(calculate_time_to_expiry)
                options_data = options_data[
                    (options_data['volume'] >= min_volume) &
                    (options_data['openInterest'] >= min_open_interest) &
                    (options_data['timeToExpiry'] >= min_dte/365) &
                    (options_data['timeToExpiry'] <= max_dte/365) &
                    (options_data['lastPrice'] > 0.01)
                ]
                
                min_strike = spot_price * 0.7
                max_strike = spot_price * 1.3
                options_data = options_data[
                    (options_data['strike'] >= min_strike) &
                    (options_data['strike'] <= max_strike)
                ]
                
                iv_data = []
                for idx, row in options_data.iterrows():
                    iv = implied_volatility(
                        row['lastPrice'],
                        spot_price,
                        row['strike'],
                        row['timeToExpiry'],
                        risk_free_rate,
                        div_yield,
                        row['optionType']
                    )
                    
                    if not np.isnan(iv):
                        greeks = calculate_greeks(
                            spot_price,
                            row['strike'],
                            row['timeToExpiry'],
                            risk_free_rate,
                            iv,
                            div_yield,
                            row['optionType']
                        )
                        
                        iv_data.append({
                            'strike': row['strike'],
                            'timeToExpiry': row['timeToExpiry'],
                            'impliedVol': iv,
                            'optionType': row['optionType'],
                            'moneyness': row['strike'] / spot_price,
                            'delta': greeks['delta'],
                            'gamma': greeks['gamma'],
                            'vega': greeks['vega'],
                            'theta': greeks['theta'],
                            'volume': row['volume'],
                            'openInterest': row['openInterest']
                        })
                
                iv_df = pd.DataFrame(iv_data)
                
                if iv_df.empty:
                    st.error("No valid implied volatility data")
                    return
            
            # Create Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["3D Surface", "2D Analysis", "Greeks", "Data Table"])
            
            with tab1:
                # 3D Surface Plot
                X = iv_df['timeToExpiry'].values
                
                if view_type == "Strike Price":
                    Y = iv_df['strike'].values
                    y_label = "Strike Price ($)"
                elif view_type == "Moneyness":
                    Y = iv_df['moneyness'].values
                    y_label = "Moneyness (K/S)"
                else:
                    Y = iv_df['delta'].values
                    y_label = "Delta"
                
                Z = iv_df['impliedVol'].values * 100
                
                xi = np.linspace(X.min(), X.max(), smoothing)
                yi = np.linspace(Y.min(), Y.max(), smoothing)
                xi, yi = np.meshgrid(xi, yi)
                
                zi = griddata((X, Y), Z, (xi, yi), method='cubic')
                
                fig = go.Figure()
                
                fig.add_trace(go.Surface(
                    x=xi,
                    y=yi,
                    z=zi,
                    colorscale=color_scheme,
                    showscale=True,
                    colorbar=dict(title="IV (%)")
                ))
                
                if show_points:
                    fig.add_trace(go.Scatter3d(
                        x=X,
                        y=Y,
                        z=Z,
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=Z,
                            colorscale=color_scheme,
                            showscale=False
                        ),
                        text=[f"IV: {z:.1f}%" for z in Z],
                        hoverinfo='text'
                    ))
                
                fig.update_layout(
                    title=f"{ticker} Implied Volatility Surface",
                    scene=dict(
                        xaxis_title="Time to Expiry (years)",
                        yaxis_title=y_label,
                        zaxis_title="Implied Volatility (%)",
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility Smile
                st.subheader("Volatility Smile Analysis")
                selected_expiry = st.selectbox(
                    "Select Expiration Date",
                    sorted(iv_df['timeToExpiry'].unique())
                )
                
                smile_data = iv_df[iv_df['timeToExpiry'] == selected_expiry]
                
                fig_smile = go.Figure()
                
                if option_type == "Both":
                    for opt_type in ['call', 'put']:
                        data = smile_data[smile_data['optionType'] == opt_type]
                        fig_smile.add_trace(go.Scatter(
                            x=data['strike'],
                            y=data['impliedVol'] * 100,
                            mode='lines+markers',
                            name=opt_type.capitalize(),
                            line=dict(width=3)
                        ))
                else:
                    fig_smile.add_trace(go.Scatter(
                        x=smile_data['strike'],
                        y=smile_data['impliedVol'] * 100,
                        mode='lines+markers',
                        name=option_type,
                        line=dict(width=3, color='blue')
                    ))
                
                fig_smile.add_vline(
                    x=spot_price,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="ATM"
                )
                
                fig_smile.update_layout(
                    title=f"Volatility Smile - {selected_expiry:.3f} Years to Expiry",
                    xaxis_title="Strike Price ($)",
                    yaxis_title="Implied Volatility (%)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_smile, use_container_width=True)
            
            with tab2:
                # Term Structure
                st.subheader("Term Structure Analysis")
                
                atm_data = iv_df[abs(iv_df['moneyness'] - 1.0) < 0.05]
                term_structure = atm_data.groupby('timeToExpiry')['impliedVol'].mean() * 100
                
                fig_term = go.Figure()
                fig_term.add_trace(go.Scatter(
                    x=term_structure.index * 365,
                    y=term_structure.values,
                    mode='lines+markers',
                    name='ATM IV',
                    line=dict(width=3)
                ))
                
                fig_term.update_layout(
                    title="ATM Implied Volatility Term Structure",
                    xaxis_title="Days to Expiry",
                    yaxis_title="Implied Volatility (%)",
                    height=400
                )
                
                st.plotly_chart(fig_term, use_container_width=True)
                
                # Skew Analysis
                st.subheader("Volatility Skew Analysis")
                
                skew_data = []
                for expiry in sorted(iv_df['timeToExpiry'].unique())[:5]:
                    exp_data = iv_df[iv_df['timeToExpiry'] == expiry]
                    
                    puts = exp_data[(exp_data['optionType'] == 'put') & (abs(exp_data['delta'] + 0.25) < 0.05)]
                    calls = exp_data[(exp_data['optionType'] == 'call') & (abs(exp_data['delta'] - 0.25) < 0.05)]
                    
                    if not puts.empty and not calls.empty:
                        skew = puts['impliedVol'].mean() - calls['impliedVol'].mean()
                        skew_data.append({
                            'expiry': expiry * 365,
                            'skew': skew * 100
                        })
                
                if skew_data:
                    skew_df = pd.DataFrame(skew_data)
                    
                    fig_skew = go.Figure()
                    fig_skew.add_trace(go.Bar(
                        x=skew_df['expiry'],
                        y=skew_df['skew'],
                        name='25-Delta Risk Reversal'
                    ))
                    
                    fig_skew.update_layout(
                        title="Volatility Skew (25-Delta Risk Reversal)",
                        xaxis_title="Days to Expiry",
                        yaxis_title="Skew (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_skew, use_container_width=True)
            
            with tab3:
                # Greeks Analysis
                st.subheader("Greeks Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    greek_expiry = st.selectbox(
                        "Select Expiry for Greeks",
                        sorted(iv_df['timeToExpiry'].unique()),
                        key="greek_exp"
                    )
                with col2:
                    greek_type = st.selectbox(
                        "Greek to Display",
                        ["delta", "gamma", "vega", "theta"],
                        key="greek_type"
                    )
                
                greek_data = iv_df[iv_df['timeToExpiry'] == greek_expiry]
                
                fig_greek = go.Figure()
                
                if option_type == "Both":
                    for opt_type in ['call', 'put']:
                        data = greek_data[greek_data['optionType'] == opt_type]
                        fig_greek.add_trace(go.Scatter(
                            x=data['strike'],
                            y=data[greek_type],
                            mode='lines+markers',
                            name=f"{opt_type.capitalize()} {greek_type.capitalize()}"
                        ))
                else:
                    fig_greek.add_trace(go.Scatter(
                        x=greek_data['strike'],
                        y=greek_data[greek_type],
                        mode='lines+markers',
                        name=f"{greek_type.capitalize()}"
                    ))
                
                fig_greek.add_vline(
                    x=spot_price,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Spot"
                )
                
                fig_greek.update_layout(
                    title=f"{greek_type.capitalize()} vs Strike Price",
                    xaxis_title="Strike Price ($)",
                    yaxis_title=greek_type.capitalize(),
                    height=500
                )
                
                st.plotly_chart(fig_greek, use_container_width=True)
                
                # Greeks Heatmap
                st.subheader("Greeks Heatmap")
                
                pivot_data = iv_df.pivot_table(
                    values=greek_type,
                    index='strike',
                    columns='timeToExpiry',
                    aggfunc='mean'
                )
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns * 365,
                    y=pivot_data.index,
                    colorscale='RdBu',
                    colorbar=dict(title=greek_type.capitalize())
                ))
                
                fig_heatmap.update_layout(
                    title=f"{greek_type.capitalize()} Heatmap",
                    xaxis_title="Days to Expiry",
                    yaxis_title="Strike Price ($)",
                    height=500
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with tab4:
                # Data Table
                st.subheader("Detailed Options Data")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Options", len(iv_df))
                with col2:
                    st.metric("Avg IV", f"{iv_df['impliedVol'].mean()*100:.1f}%")
                with col3:
                    st.metric("IV Range", f"{iv_df['impliedVol'].min()*100:.1f}% - {iv_df['impliedVol'].max()*100:.1f}%")
                
                sort_by = st.selectbox("Sort by", ["impliedVol", "volume", "openInterest", "strike"])
                ascending = st.checkbox("Ascending order", value=True)
                
                display_df = iv_df.sort_values(sort_by, ascending=ascending)
                display_df['impliedVol'] = (display_df['impliedVol'] * 100).round(2)
                display_df['strike'] = display_df['strike'].round(2)
                display_df['timeToExpiry'] = (display_df['timeToExpiry'] * 365).round(0).astype(int)
                
                display_df = display_df.rename(columns={
                    'strike': 'Strike',
                    'timeToExpiry': 'DTE',
                    'impliedVol': 'IV%',
                    'optionType': 'Type',
                    'moneyness': 'Moneyness',
                    'delta': 'Delta',
                    'gamma': 'Gamma',
                    'vega': 'Vega',
                    'theta': 'Theta',
                    'volume': 'Volume',
                    'openInterest': 'OI'
                })
                
                numeric_cols = ['Moneyness', 'Delta', 'Gamma', 'Vega', 'Theta']
                for col in numeric_cols:
                    display_df[col] = display_df[col].round(4)
                
                st.dataframe(
                    display_df[['Strike', 'DTE', 'IV%', 'Type', 'Moneyness', 
                               'Delta', 'Gamma', 'Vega', 'Theta', 'Volume', 'OI']],
                    use_container_width=True,
                    hide_index=True
                )
                
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download Data as CSV",
                    data=csv,
                    file_name=f"{ticker}_options_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()