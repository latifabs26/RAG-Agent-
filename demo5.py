import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
from typing import Dict, List
import yfinance as yf  # Note: This would need to be installed

# Page configuration
st.set_page_config(
    page_title="QuantTrader Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with dark theme and glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main > div {
        padding-top: 1rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .trading-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    .metric-glass {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease;
    }
    
    .metric-glass:hover {
        transform: translateY(-5px);
    }
    
    .profit-positive {
        color: #00ff88 !important;
        font-weight: 600;
    }
    
    .profit-negative {
        color: #ff4757 !important;
        font-weight: 600;
    }
    
    .portfolio-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .alert-card {
        background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(255, 71, 87, 0.3);
    }
    
    .news-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .sidebar .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px;
    }
    
    .watchlist-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem;
        margin: 0.3rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .technical-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

class TradingDataGenerator:
    """Generate realistic trading data for the dashboard"""
    
    @staticmethod
    def generate_portfolio_data():
        """Generate portfolio holdings data"""
        symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'NVDA', 'META', 'NFLX']
        portfolio = []
        
        for symbol in symbols:
            shares = random.randint(10, 500)
            current_price = random.uniform(100, 300)
            purchase_price = current_price * random.uniform(0.8, 1.2)
            
            portfolio.append({
                'symbol': symbol,
                'shares': shares,
                'current_price': current_price,
                'purchase_price': purchase_price,
                'market_value': shares * current_price,
                'cost_basis': shares * purchase_price,
                'unrealized_pnl': shares * (current_price - purchase_price),
                'day_change': random.uniform(-5, 5),
                'sector': random.choice(['Technology', 'Healthcare', 'Finance', 'Energy'])
            })
        
        return pd.DataFrame(portfolio)
    
    @staticmethod
    def generate_price_data(symbol: str, days: int = 30):
        """Generate realistic price data with technical indicators"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate price data with realistic movements
        price = 150  # Starting price
        prices = []
        volumes = []
        
        for i in range(days):
            # Add some volatility
            change = np.random.normal(0, 2)  # 2% average daily volatility
            price = max(price * (1 + change/100), 10)  # Ensure price doesn't go negative
            prices.append(price)
            volumes.append(random.randint(1000000, 10000000))
        
        df = pd.DataFrame({
            'date': dates,
            'open': [p * random.uniform(0.98, 1.02) for p in prices],
            'high': [p * random.uniform(1.01, 1.05) for p in prices],
            'low': [p * random.uniform(0.95, 0.99) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Calculate technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=min(50, days)).mean()
        df['rsi'] = TradingDataGenerator.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = TradingDataGenerator.calculate_macd(df['close'])
        
        return df
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

def create_advanced_candlestick_chart(df: pd.DataFrame, symbol: str):
    """Create professional candlestick chart with technical indicators"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Action', 'Volume', 'RSI', 'MACD'),
        row_width=[0.2, 0.1, 0.1, 0.1]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4757'
        ), row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['sma_20'], name='SMA 20', 
                  line=dict(color='#ffa502', width=2)), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['sma_50'], name='SMA 50', 
                  line=dict(color='#3742fa', width=2)), row=1, col=1
    )
    
    # Volume
    colors = ['#00ff88' if row['close'] >= row['open'] else '#ff4757' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='Volume', 
               marker_color=colors, opacity=0.7), row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['rsi'], name='RSI', 
                  line=dict(color='#ff6348', width=2)), row=3, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['macd'], name='MACD', 
                  line=dict(color='#2ed573', width=2)), row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['macd_signal'], name='Signal', 
                  line=dict(color='#ffa502', width=2)), row=4, col=1
    )
    
    # MACD histogram
    macd_histogram = df['macd'] - df['macd_signal']
    colors = ['green' if val >= 0 else 'red' for val in macd_histogram]
    fig.add_trace(
        go.Bar(x=df['date'], y=macd_histogram, name='Histogram', 
               marker_color=colors, opacity=0.6), row=4, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        yaxis_title='Price ($)',
        template='plotly_dark',
        showlegend=True,
        height=800,
        hovermode='x unified'
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def create_portfolio_treemap(portfolio_df: pd.DataFrame):
    """Create portfolio allocation treemap"""
    
    fig = px.treemap(
        portfolio_df,
        values='market_value',
        names='symbol',
        color='unrealized_pnl',
        color_continuous_scale=['#ff4757', '#ffffff', '#00ff88'],
        color_continuous_midpoint=0,
        title='Portfolio Allocation & Performance'
    )
    
    fig.update_traces(
        texttemplate='<b>%{label}</b><br>$%{value:,.0f}<br>%{color:+.1f}',
        textfont_size=12
    )
    
    fig.update_layout(
        template='plotly_dark',
        font=dict(color='white'),
        height=400
    )
    
    return fig

def create_risk_metrics_radar(portfolio_df: pd.DataFrame):
    """Create risk metrics radar chart"""
    
    # Calculate risk metrics (simplified for demo)
    categories = ['Volatility', 'Sharpe Ratio', 'Beta', 'Alpha', 'Max Drawdown', 'VaR']
    values = [65, 75, 85, 70, 60, 80]  # Demo values
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Portfolio',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    # Benchmark comparison
    benchmark_values = [70, 65, 80, 75, 65, 75]
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values,
        theta=categories,
        fill='toself',
        name='Benchmark',
        line_color='#ffa502',
        fillcolor='rgba(255, 165, 2, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='white')
            ),
            angularaxis=dict(
                tickfont=dict(color='white')
            )
        ),
        template='plotly_dark',
        title='Risk Metrics Comparison',
        showlegend=True,
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="trading-header">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">💹 QuantTrader Pro</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Professional Trading & Portfolio Analytics Platform
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                Live Market Data
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                Real-time Analytics
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                Advanced Algorithms
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = TradingDataGenerator.generate_portfolio_data()
        st.session_state.last_update = datetime.now()
    
    portfolio_df = st.session_state.portfolio_data
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Trading Console")
        
        # Account summary
        total_value = portfolio_df['market_value'].sum()
        total_cost = portfolio_df['cost_basis'].sum()
        total_pnl = portfolio_df['unrealized_pnl'].sum()
        pnl_percent = (total_pnl / total_cost) * 100
        
        st.markdown(f"""
        <div class="metric-glass">
            <h4 style="color: white; margin: 0;">Portfolio Value</h4>
            <h2 style="color: #00ff88; margin: 0;">${total_value:,.0f}</h2>
            <p style="color: {'#00ff88' if total_pnl >= 0 else '#ff4757'}; margin: 0;">
                {'+' if total_pnl >= 0 else ''}{total_pnl:,.0f} ({pnl_percent:+.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Watchlist
        st.markdown("### 👁️ Watchlist")
        watchlist_symbols = ['SPY', 'QQQ', 'IWM', 'GLD', 'BTC-USD']
        
        for symbol in watchlist_symbols:
            price = random.uniform(100, 400)
            change = random.uniform(-3, 3)
            color = '#00ff88' if change >= 0 else '#ff4757'
            
            st.markdown(f"""
            <div class="watchlist-item">
                <div>
                    <strong style="color: white;">{symbol}</strong>
                    <br>
                    <span style="font-size: 0.9rem; color: {color};">
                        {'+' if change >= 0 else ''}{change:.2f}%
                    </span>
                </div>
                <div style="text-align: right;">
                    <strong style="color: white;">${price:.2f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Data", type="primary"):
            st.session_state.portfolio_data = TradingDataGenerator.generate_portfolio_data()
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        if st.button("📊 Generate Report"):
            st.success("Portfolio report generated!")
        
        if st.button("🚨 Set Alert"):
            st.info("Price alert configured!")
        
        # Market status
        st.markdown("### 📈 Market Status")
        st.markdown("""
        <div style="display: flex; justify-content: space-between; color: white;">
            <span>S&P 500</span>
            <span style="color: #00ff88;">+0.85%</span>
        </div>
        <div style="display: flex; justify-content: space-between; color: white;">
            <span>NASDAQ</span>
            <span style="color: #00ff88;">+1.23%</span>
        </div>
        <div style="display: flex; justify-content: space-between; color: white;">
            <span>DOW</span>
            <span style="color: #ff4757;">-0.34%</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Charts", "💼 Portfolio", "🎯 Risk Analysis", "🔔 Alerts & News"])
    
    with tab1:
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = [
            ("Portfolio Value", f"${total_value:,.0f}", f"{pnl_percent:+.2f}%", pnl_percent >= 0),
            ("Day P&L", f"${random.uniform(-5000, 5000):+,.0f}", "+2.34%", True),
            ("Buying Power", f"${random.uniform(10000, 50000):,.0f}", "Available", True),
            ("Win Rate", "68.5%", "+3.2%", True),
            ("Sharpe Ratio", "1.45", "Excellent", True)
        ]
        
        for i, (label, value, change, positive) in enumerate(metrics):
            with [col1, col2, col3, col4, col5][i]:
                color = '#00ff88' if positive else '#ff4757'
                st.markdown(f"""
                <div class="metric-glass">
                    <h5 style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">{label}</h5>
                    <h3 style="color: white; margin: 0.2rem 0;">{value}</h3>
                    <p style="color: {color}; margin: 0; font-size: 0.8rem;">{change}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Charts row
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Portfolio performance chart
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            portfolio_values = []
            base_value = total_value * 0.9
            
            for i in range(30):
                base_value *= (1 + np.random.normal(0.001, 0.02))
                portfolio_values.append(base_value)
            
            performance_df = pd.DataFrame({
                'date': dates,
                'value': portfolio_values
            })
            
            fig_performance = px.line(
                performance_df, x='date', y='value',
                title='Portfolio Performance (30 Days)',
                color_discrete_sequence=['#667eea']
            )
            
            fig_performance.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
        
        with col2:
            # Top performers
            st.markdown("### 🏆 Top Performers")
            top_performers = portfolio_df.nlargest(5, 'unrealized_pnl')
            
            for _, stock in top_performers.iterrows():
                pnl_percent = (stock['unrealized_pnl'] / stock['cost_basis']) * 100
                st.markdown(f"""
                <div class="portfolio-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: white; font-size: 1.1rem;">{stock['symbol']}</strong>
                            <br>
                            <span style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
                                {stock['shares']} shares @ ${stock['current_price']:.2f}
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <strong style="color: #00ff88;">${stock['unrealized_pnl']:+,.0f}</strong>
                            <br>
                            <span style="color: #00ff88; font-size: 0.9rem;">+{pnl_percent:.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Market heatmap
        st.markdown("### 🔥 Market Heatmap")
        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Industrial']
        sector_performance = [random.uniform(-3, 5) for _ in sectors]
        
        fig_heatmap = px.imshow(
            [sector_performance],
            x=sectors,
            y=['Sector Performance'],
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            text_auto=True
        )
        
        fig_heatmap.update_layout(
            template='plotly_dark',
            height=150,
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.markdown("### 📈 Advanced Technical Analysis")
        
        # Symbol selector
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            selected_symbol = st.selectbox("Select Symbol", portfolio_df['symbol'].tolist())
        with col2:
            timeframe = st.selectbox("Timeframe", ['1D', '1W', '1M', '3M', '1Y'])
        
        # Generate price data for selected symbol
        days_map = {'1D': 1, '1W': 7, '1M': 30, '3M': 90, '1Y': 365}
        price_data = TradingDataGenerator.generate_price_data(selected_symbol, days_map[timeframe])
        
        # Technical analysis chart
        chart_fig = create_advanced_candlestick_chart(price_data, selected_symbol)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        # Technical indicators summary
        col1, col2, col3, col4 = st.columns(4)
        
        current_rsi = price_data['rsi'].iloc[-1] if not pd.isna(price_data['rsi'].iloc[-1]) else 50
        
        with col1:
            rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            rsi_color = "#ff4757" if current_rsi > 70 else "#00ff88" if current_rsi < 30 else "#ffa502"
            st.markdown(f"""
            <span class="technical-indicator" style="background: {rsi_color};">
                RSI: {current_rsi:.1f} ({rsi_signal})
            </span>
            """, unsafe_allow_html=True)
        
        with col2:
            macd_signal = "Bullish" if random.random() > 0.5 else "Bearish"
            macd_color = "#00ff88" if macd_signal == "Bullish" else "#ff4757"
            st.markdown(f"""
            <span class="technical-indicator" style="background: {macd_color};">
                MACD: {macd_signal}
            </span>
            """, unsafe_allow_html=True)
        
        with col3:
            sma_signal = "Above SMA" if random.random() > 0.5 else "Below SMA"
            sma_color = "#00ff88" if "Above" in sma_signal else "#ff4757"
            st.markdown(f"""
            <span class="technical-indicator" style="background: {sma_color};">
                Price: {sma_signal}
            </span>
            """, unsafe_allow_html=True)
        
        with col4:
            volume_signal = "High Volume" if random.random() > 0.5 else "Low Volume"
            volume_color = "#667eea"
            st.markdown(f"""
            <span class="technical-indicator" style="background: {volume_color};">
                Volume: {volume_signal}
            </span>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 💼 Portfolio Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Portfolio treemap
            treemap_fig = create_portfolio_treemap(portfolio_df)
            st.plotly_chart(treemap_fig, use_container_width=True)
            
            # Holdings table
            st.markdown("### 📋 Current Holdings")
            
            # Enhanced portfolio table
            display_df = portfolio_df.copy()
            display_df['P&L %'] = ((display_df['current_price'] - display_df['purchase_price']) / display_df['purchase_price'] * 100).round(2)
            display_df['Market Value'] = display_df['market_value'].apply(lambda x: f"${x:,.0f}")
            display_df['Unrealized P&L'] = display_df['unrealized_pnl'].apply(lambda x: f"${x:+,.0f}")
            
            st.dataframe(
                display_df[['symbol', 'shares', 'current_price', 'Market Value', 'Unrealized P&L', 'P&L %', 'sector']],
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Portfolio allocation by sector
            sector_allocation = portfolio_df.groupby('sector')['market_value'].sum()
            
            fig_sector = px.pie(
                values=sector_allocation.values,
                names=sector_allocation.index,
                title='Sector Allocation',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_sector.update_layout(
                template='plotly_dark',
                height=300
            )
            
            st.plotly_chart(fig_sector, use_container_width=True)
            
            # Performance metrics
            st.markdown("### 📊 Performance Metrics")
            
            win_rate = random.uniform(60, 80)
            avg_win = random.uniform(200, 800)
            avg_loss = random.uniform(-150, -50)
            
            metrics_data = {
                'Metric': ['Win Rate', 'Avg Win', 'Avg Loss', 'Profit Factor', 'Max Drawdown'],
                'Value': [f'{win_rate:.1f}%', f'${avg_win:.0f}', f'${avg_loss:.0f}', '1.45', '-8.3%'],
                'Status': ['Good', 'Good', 'Good', 'Excellent', 'Fair']
            }
            
            st.dataframe(metrics_data, hide_index=True)
    
    with tab4:
        st.markdown("### 🎯 Risk Analysis Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk metrics radar
            radar_fig = create_risk_metrics_radar(portfolio_df)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            # Value at Risk chart
            confidence_levels = [90, 95, 99]
            var_values = [random.uniform(2000, 8000) for _ in confidence_levels]
            
            fig_var = px.bar(
                x=[f'{cl}%' for cl in confidence_levels],
                y=var_values,
                title='Value at Risk (VaR)',
                color=var_values,
                color_continuous_scale='Reds'
            )
            
            fig_var.update_layout(
                template='plotly_dark',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_var, use_container_width=True)
        
        # Risk warnings
        st.markdown("### ⚠️ Risk Alerts")
        
        risk_alerts = [
            ("High Concentration Risk", "Technology sector represents 65% of portfolio", "warning"),
            ("Volatility Spike", "TSLA showing 40% higher volatility than usual", "error"),
            ("Correlation Alert", "High correlation detected between AAPL and MSFT", "info")
        ]
        
        for title, message, alert_type in risk_alerts:
            if alert_type == "warning":
                st.warning(f"**{title}**: {message}")
            elif alert_type == "error":
                st.error(f"**{title}**: {message}")
            else:
                st.info(f"**{title}**: {message}")
    
    with tab5:
        st.markdown("### 🔔 Alerts & Market News")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🚨 Active Alerts")
            
            alerts = [
                ("AAPL crossed above $150", "2 minutes ago", "success"),
                ("Portfolio down 2% from daily high", "15 minutes ago", "warning"),
                ("TSLA volume spike detected", "1 hour ago", "info"),
                ("Margin requirement increased", "2 hours ago", "error")
            ]
            
            for alert, time, alert_type in alerts:
                color = {
                    'success': '#00ff88',
                    'warning': '#ffa502',
                    'info': '#667eea',
                    'error': '#ff4757'
                }[alert_type]
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 1rem; margin: 0.5rem 0; border-left: 4px solid {color}; border-radius: 5px;">
                    <strong style="color: white;">{alert}</strong>
                    <br>
                    <small style="color: rgba(255,255,255,0.6);">{time}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📰 Market News")
            
            news_items = [
                ("Fed announces interest rate decision", "MarketWatch", "5 min ago"),
                ("Tech stocks rally on earnings optimism", "Reuters", "1 hour ago"),
                ("Oil prices surge amid supply concerns", "Bloomberg", "2 hours ago"),
                ("Crypto market shows signs of recovery", "CoinDesk", "3 hours ago")
            ]
            
            for headline, source, time in news_items:
                st.markdown(f"""
                <div class="news-card">
                    <strong style="color: white;">{headline}</strong>
                    <br>
                    <small style="color: rgba(255,255,255,0.6);">{source} • {time}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Economic calendar
        st.markdown("### 📅 Economic Calendar")
        
        economic_events = pd.DataFrame({
            'Time': ['09:30', '10:00', '14:00', '16:00'],
            'Event': ['Jobless Claims', 'CPI Data', 'Fed Speech', 'Earnings: GOOGL'],
            'Impact': ['Medium', 'High', 'High', 'Medium'],
            'Forecast': ['220K', '3.2%', 'Hawkish', 'Beat Expected']
        })
        
        st.dataframe(economic_events, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()