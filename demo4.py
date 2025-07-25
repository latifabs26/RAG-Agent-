import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="AI Document Analytics Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric > div {
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .header-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .insight-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sophisticated sample data for the dashboard"""
    
    # Document analytics data
    documents = []
    categories = ['Technical Reports', 'Research Papers', 'Legal Documents', 'Marketing Materials', 'User Manuals']
    sentiment_labels = ['Positive', 'Neutral', 'Negative']
    
    for i in range(100):
        doc = {
            'id': f'DOC_{i:03d}',
            'title': f'Document {i+1}',
            'category': random.choice(categories),
            'word_count': random.randint(500, 5000),
            'readability_score': random.uniform(30, 90),
            'sentiment': random.choice(sentiment_labels),
            'confidence': random.uniform(0.6, 0.99),
            'processing_time': random.uniform(0.5, 5.0),
            'topics': random.sample(['AI', 'Machine Learning', 'Data Science', 'Technology', 'Business', 'Innovation', 'Research'], 3),
            'created_date': datetime.now() - timedelta(days=random.randint(1, 365)),
            'similarity_cluster': random.randint(1, 8)
        }
        documents.append(doc)
    
    return pd.DataFrame(documents)

def create_network_graph(df: pd.DataFrame) -> go.Figure:
    """Create an interactive network graph of document relationships"""
    
    G = nx.Graph()
    
    # Add nodes (documents)
    for idx, row in df.head(20).iterrows():
        G.add_node(row['id'], 
                  category=row['category'],
                  title=row['title'][:30] + '...' if len(row['title']) > 30 else row['title'])
    
    # Add edges based on category similarity
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, min(i+4, len(nodes))):  # Connect to 3 nearest nodes
            if random.random() > 0.3:  # 70% chance of connection
                G.add_edge(nodes[i], nodes[j], weight=random.uniform(0.3, 1.0))
    
    # Get positions
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Create traces for edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create traces for nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [G.nodes[node]['title'] for node in G.nodes()]
    node_colors = [hash(G.nodes[node]['category']) % 7 for node in G.nodes()]
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=1, color='rgba(125, 125, 125, 0.3)'),
                            hoverinfo='none',
                            mode='lines',
                            showlegend=False))
    
    # Add nodes
    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                            mode='markers+text',
                            hoverinfo='text',
                            hovertext=node_text,
                            text=[node[:8] for node in G.nodes()],
                            textposition="middle center",
                            marker=dict(size=20,
                                       color=node_colors,
                                       colorscale='Viridis',
                                       line=dict(width=2, color='white')),
                            showlegend=False))
    
    fig.update_layout(
        title="Document Relationship Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(text="Interactive Document Network - Hover for details",
                          showarrow=False,
                          xref="paper", yref="paper",
                          x=0.005, y=-0.002,
                          xanchor='left', yanchor='bottom',
                          font=dict(color='gray', size=12)) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_advanced_analytics_charts(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """Create sophisticated analytics visualizations"""
    
    # 1. Multi-dimensional bubble chart
    fig1 = px.scatter(df, 
                     x='word_count', 
                     y='readability_score',
                     size='confidence',
                     color='category',
                     hover_data=['title', 'sentiment'],
                     title="Document Complexity Analysis",
                     color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig1.update_traces(marker=dict(line=dict(width=1, color='white')))
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # 2. Advanced time series with multiple metrics
    daily_stats = df.groupby(df['created_date'].dt.date).agg({
        'word_count': ['mean', 'count'],
        'readability_score': 'mean',
        'confidence': 'mean'
    }).reset_index()
    
    daily_stats.columns = ['date', 'avg_words', 'doc_count', 'avg_readability', 'avg_confidence']
    
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Document Volume', 'Average Word Count', 'Readability Trend', 'Processing Confidence'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )
    
    # Add traces
    fig2.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['doc_count'],
                             mode='lines+markers', name='Documents', line=dict(color='#FF6B6B')), row=1, col=1)
    
    fig2.add_trace(go.Bar(x=daily_stats['date'], y=daily_stats['avg_words'],
                         name='Avg Words', marker_color='#4ECDC4'), row=1, col=2)
    
    fig2.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['avg_readability'],
                             mode='lines', name='Readability', line=dict(color='#45B7D1')), row=2, col=1)
    
    fig2.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['avg_confidence'],
                             mode='lines+markers', name='Confidence', line=dict(color='#96CEB4')), row=2, col=2)
    
    fig2.update_layout(height=600, showlegend=False, title_text="Multi-Metric Time Series Analysis")
    
    # 3. Sankey diagram for document flow
    categories = df['category'].unique()
    sentiments = df['sentiment'].unique()
    
    # Create source, target, and value lists for Sankey
    source_ids = []
    target_ids = []
    values = []
    labels = list(categories) + list(sentiments)
    
    for i, cat in enumerate(categories):
        for j, sent in enumerate(sentiments):
            count = len(df[(df['category'] == cat) & (df['sentiment'] == sent)])
            if count > 0:
                source_ids.append(i)
                target_ids.append(len(categories) + j)
                values.append(count)
    
    fig3 = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3", "#54A0FF", "#5F27CD"]
        ),
        link=dict(
            source=source_ids,
            target=target_ids,
            value=values,
            color="rgba(255, 255, 255, 0.4)"
        )
    )])
    
    fig3.update_layout(title_text="Document Category → Sentiment Flow", font_size=10)
    
    return fig1, fig2, fig3

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 style="color: white; margin: 0;">🧠 AI Document Analytics Hub</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
            Advanced document intelligence with real-time insights and interactive visualizations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data
    if 'df' not in st.session_state:
        st.session_state.df = generate_sample_data()
    
    df = st.session_state.df
    
    # Sidebar controls
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Control Panel")
        
        # Filters
        selected_categories = st.multiselect(
            "Document Categories",
            options=df['category'].unique(),
            default=df['category'].unique()[:3]
        )
        
        date_range = st.date_input(
            "Date Range",
            value=(df['created_date'].min().date(), df['created_date'].max().date()),
            min_value=df['created_date'].min().date(),
            max_value=df['created_date'].max().date()
        )
        
        confidence_threshold = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time metrics
        st.markdown("### 📊 Live Metrics")
        filtered_df = df[
            (df['category'].isin(selected_categories)) & 
            (df['confidence'] >= confidence_threshold)
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Documents", len(filtered_df))
            st.metric("⚡ Avg Confidence", f"{filtered_df['confidence'].mean():.2%}")
        with col2:
            st.metric("📝 Total Words", f"{filtered_df['word_count'].sum():,}")
            st.metric("🎯 Avg Readability", f"{filtered_df['readability_score'].mean():.1f}")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🕸️ Network Analysis", "📈 Advanced Analytics", "🤖 AI Insights"])
    
    with tab1:
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: white; margin: 0;">Total Documents</h3>
                <h1 style="color: white; margin: 0;">{}</h1>
                <p style="color: rgba(255,255,255,0.8); margin: 0;">+12% from last month</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            avg_processing = df['processing_time'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: white; margin: 0;">Avg Processing</h3>
                <h1 style="color: white; margin: 0;">{avg_processing:.1f}s</h1>
                <p style="color: rgba(255,255,255,0.8); margin: 0;">-8% improvement</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            positive_sentiment = len(df[df['sentiment'] == 'Positive']) / len(df) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: white; margin: 0;">Positive Sentiment</h3>
                <h1 style="color: white; margin: 0;">{positive_sentiment:.1f}%</h1>
                <p style="color: rgba(255,255,255,0.8); margin: 0;">+5% this week</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            high_confidence = len(df[df['confidence'] > 0.8]) / len(df) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: white; margin: 0;">High Confidence</h3>
                <h1 style="color: white; margin: 0;">{high_confidence:.1f}%</h1>
                <p style="color: rgba(255,255,255,0.8); margin: 0;">Quality score</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = df['category'].value_counts()
            fig_donut = px.pie(values=category_counts.values, names=category_counts.index,
                              title="Document Categories Distribution", hole=0.4,
                              color_discrete_sequence=px.colors.qualitative.Set3)
            fig_donut.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_donut, use_container_width=True)
        
        with col2:
            # Sentiment analysis
            sentiment_counts = df['sentiment'].value_counts()
            fig_sentiment = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
                                  title="Sentiment Analysis Results",
                                  color=sentiment_counts.index,
                                  color_discrete_map={'Positive': '#2ECC71', 'Neutral': '#F39C12', 'Negative': '#E74C3C'})
            fig_sentiment.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with tab2:
        st.markdown("### 🕸️ Interactive Document Network")
        st.markdown("*Explore relationships between documents based on content similarity and categories*")
        
        network_fig = create_network_graph(df)
        st.plotly_chart(network_fig, use_container_width=True, height=600)
        
        # Network statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔗 Network Density", "0.73")
        with col2:
            st.metric("🎯 Avg Clustering", "0.68")
        with col3:
            st.metric("📊 Connected Components", "3")
    
    with tab3:
        st.markdown("### 📈 Advanced Analytics Suite")
        
        bubble_fig, time_series_fig, sankey_fig = create_advanced_analytics_charts(df)
        
        st.plotly_chart(bubble_fig, use_container_width=True)
        st.plotly_chart(time_series_fig, use_container_width=True)
        st.plotly_chart(sankey_fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🤖 AI-Powered Insights")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="insight-card">
                <h4>🔍 Key Findings</h4>
                <ul>
                    <li><strong>Content Quality:</strong> 78% of documents exceed readability threshold</li>
                    <li><strong>Processing Efficiency:</strong> Average processing time reduced by 23% this quarter</li>
                    <li><strong>Sentiment Trends:</strong> Positive sentiment increased 15% in technical documents</li>
                    <li><strong>Category Insights:</strong> Legal documents show highest confidence scores (0.89 avg)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-card">
                <h4>📊 Predictive Analytics</h4>
                <p>Based on current trends, we predict:</p>
                <ul>
                    <li>Document volume will increase by 34% next month</li>
                    <li>Processing efficiency will improve by additional 12%</li>
                    <li>Research papers category will dominate by Q3</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎯 Action Items")
            st.info("🔧 Optimize processing pipeline for legal documents")
            st.success("✅ Sentiment analysis accuracy improved")
            st.warning("⚠️ Monitor readability scores in technical reports")
            st.error("🚨 Review low-confidence documents in marketing category")
            
            if st.button("🚀 Generate AI Report", type="primary"):
                st.balloons()
                st.success("Comprehensive AI report generated and sent to your dashboard!")

if __name__ == "__main__":
    main()