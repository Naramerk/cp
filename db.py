import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy.stats import gaussian_kde
import pycountry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_demo_data():
    """Generate demo data for the dashboard"""
    np.random.seed(42)
    
    # Generate sample data
    countries = ['USA', 'United Kingdom', 'Germany', 'France', 'Canada', 'Australia', 'Japan', 'India', 'Brazil', 'Russia']
    leaders = ['Joe Biden', 'Donald Trump', 'Emmanuel Macron', 'Rishi Sunak', 'Vladimir Putin', 'Xi Jinping', 'Justin Trudeau']
    
    n_posts = 50000
    
    data = []
    for i in range(n_posts):
        # Random sentiment with some bias based on leader/country
        sentiment = np.random.normal(0, 0.3)
        sentiment = np.clip(sentiment, -1, 1)
        
        # Random country and leader
        country = np.random.choice(countries, p=[0.3, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05, 0.05, 0.05, 0.05])
        leader = np.random.choice(leaders, p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1])
        
        # Random date in last 30 days
        days_ago = np.random.randint(0, 30)
        created_date = datetime.now() - timedelta(days=days_ago)
        
        # Random upvotes (log-normal distribution)
        upvotes = int(np.random.lognormal(2, 1.5))
        
        # Generate sample text
        sentiment_words = {
            'positive': ['great', 'excellent', 'amazing', 'wonderful', 'fantastic'],
            'negative': ['terrible', 'awful', 'horrible', 'worst', 'disappointing'],
            'neutral': ['okay', 'average', 'normal', 'standard', 'typical']
        }
        
        if sentiment > 0.3:
            words = sentiment_words['positive']
        elif sentiment < -0.3:
            words = sentiment_words['negative']
        else:
            words = sentiment_words['neutral']
        
        text = f"This is a sample post about {leader} and {country}. It's {np.random.choice(words)}."
        
        data.append({
            'post_id': f'post_{i}',
            'text': text,
            'sentiment_score': sentiment,
            'country': country,
            'leader': leader,
            'upvotes': upvotes,
            'created_utc': created_date,
            'date': created_date.date(),
            'hour_of_day': created_date.hour,
            'day_of_week': created_date.weekday()
        })
    
    df = pd.DataFrame(data)
    return df

def normalize_country_name(name):
    """Normalize country names to avoid duplicates"""
    if not name:
        return name
        
    name = name.strip().upper()
    
    country_mapping = {
        'US': 'USA',
        'USA': 'USA',
        'UNITED STATES': 'USA',
        'UNITED STATES OF AMERICA': 'USA',
        'UK': 'United Kingdom',
        'GB': 'United Kingdom',
        'GREAT BRITAIN': 'United Kingdom',
        'UNITED KINGDOM': 'United Kingdom',
        'RUSSIA': 'Russia',
        'RF': 'Russia',
        'RUS': 'Russia',
        'RUSSIAN FEDERATION': 'Russia',
        'CHINA': 'China',
        'PRC': 'China',
        "PEOPLE'S REPUBLIC OF CHINA": 'China',
        'UAE': 'United Arab Emirates',
        'UNITED ARAB EMIRATES': 'United Arab Emirates',
        'EU': 'European Union',
        'EUROPEAN UNION': 'European Union',
    }
    
    return country_mapping.get(name, name.title())

def normalize_leader_name(name):
    """Normalize leader names to avoid duplicates"""
    name_mapping = {
        'Macron': 'Emmanuel Macron',
        'Sunak': 'Rishi Sunak',
        'Trump': 'Donald Trump',
        'Biden': 'Joe Biden',
        'Putin': 'Vladimir Putin',
        'Xi': 'Xi Jinping'
    }
    return name_mapping.get(name, name)

def load_data():
    """Load demo data"""
    try:
        df = generate_demo_data()
        logger.info(f"Generated {len(df)} demo records")
        
        # Split multiple countries and leaders
        df['countries'] = df['country'].fillna('').apply(lambda x: [x] if x else [])
        df['leaders'] = df['leader'].fillna('').apply(lambda x: [x] if x else [])
        
        # Explode the dataframe
        df_countries = df.explode('countries').copy()
        df_leaders = df.explode('leaders').copy()
        
        # Clean data
        df_countries = df_countries[df_countries['countries'].str.strip() != '']
        df_leaders = df_leaders[df_leaders['leaders'].str.strip() != '']
        
        # Normalize country and leader names
        df_countries['countries'] = df_countries['countries'].str.strip().apply(normalize_country_name)
        df_leaders['leaders'] = df_leaders['leaders'].str.strip().apply(normalize_leader_name)
        
        return df, df_countries, df_leaders
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None

def create_world_map(df_countries):
    """Create world map visualization"""
    if df_countries is None or df_countries.empty:
        return None
    
    try:
        # Aggregate data by normalized country names
        country_stats = df_countries.groupby('countries').agg({
            'sentiment_score': ['mean', 'count'],
            'upvotes': 'mean'
        }).reset_index()
        
        country_stats.columns = ['country', 'avg_sentiment', 'post_count', 'avg_upvotes']
        
        # Add special handling for special cases
        def get_country_code(country_name):
            special_cases = {
                'USA': 'USA',
                'United Kingdom': 'GBR',
                'Russia': 'RUS',
                'European Union': 'EUR',
                'United Arab Emirates': 'ARE',
            }
            try:
                return special_cases.get(country_name) or pycountry.countries.search_fuzzy(country_name)[0].alpha_3
            except Exception:
                return None
        
        country_stats['iso_alpha'] = country_stats['country'].apply(get_country_code)
        country_stats = country_stats[country_stats['iso_alpha'].notna()]
        
        fig = px.choropleth(
            country_stats,
            locations='iso_alpha',
            color='avg_sentiment',
            hover_name='country',
            color_continuous_scale=[[0, '#d73027'], [0.5, '#ffffbf'], [1, '#4575b4']],
            range_color=[-1, 1],
            custom_data=['post_count', 'avg_sentiment']
        )
        
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
                         "Sentiment: %{customdata[1]:.2f}<br>" +
                         "Posts: %{customdata[0]:,}<br>" +
                         "<extra></extra>"
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(
                showframe=True,
                framecolor='#666666',
                showcoastlines=True,
                coastlinecolor='#666666',
                showland=True,
                landcolor='#f0f0f0',
                showocean=True,
                oceancolor='#e6f3ff',
                showcountries=True,
                countrycolor='#666666',
                countrywidth=0.8,
                projection_type='equirectangular'
            ),
            coloraxis_colorbar=dict(
                title="Sentiment",
                ticktext=['Very Negative', 'Neutral', 'Very Positive'],
                tickvals=[-1, 0, 1],
                lenmode='fraction',
                len=0.75
            )
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating world map: {e}")
        return None

def create_leader_stats(df_leaders):
    """Create leader statistics"""
    if df_leaders is None or df_leaders.empty:
        return pd.DataFrame()
    
    try:
        df_leaders['leaders'] = df_leaders['leaders'].apply(normalize_leader_name)
        
        leader_stats = df_leaders.groupby('leaders').agg({
            'sentiment_score': ['mean', 'count'],
            'upvotes': 'mean',
        }).reset_index()
        
        leader_stats.columns = ['leader', 'sentiment', 'mentions', 'avg_upvotes']
        leader_stats = leader_stats.sort_values('mentions', ascending=False)
        
        return leader_stats
    except Exception as e:
        logger.error(f"Error creating leader stats: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(
        page_title="Reddit Political Sentiment Analysis - DEMO",
        page_icon="📊",
        layout="wide"
    )
    
    # Demo warning
    st.warning("🚧 This is a DEMO version with generated sample data. Connect to a real database for live data.")
    
    # Load data
    df, df_countries, df_leaders = load_data()
    
    if df is None:
        st.error("Error generating demo data")
        return
    
    # Header
    st.title("Reddit Political Sentiment Analysis - DEMO")
    st.markdown("Demo analysis of political discussions with sample data")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Posts", f"{len(df):,}")
    with col2:
        st.metric("Countries Mentioned", f"{df_countries['countries'].nunique():,}")
    with col3:
        st.metric("Leaders Mentioned", f"{df_leaders['leaders'].nunique():,}")
    with col4:
        st.metric("Average Sentiment", f"{df['sentiment_score'].mean():.2f}")
    with col5:
        st.metric("Average Upvotes", f"{df['upvotes'].mean():.0f}")
    
    # World Map and Top Countries
    st.subheader("Global Sentiment Analysis")
    
    map_col, stats_col = st.columns([2, 1])
    
    with map_col:
        world_map = create_world_map(df_countries)
        if world_map:
            st.plotly_chart(world_map, use_container_width=True)
    
    with stats_col:
        country_stats = df_countries.groupby('countries').agg({
            'sentiment_score': ['mean', 'count']
        }).reset_index()
        country_stats.columns = ['Country', 'Sentiment', 'Posts']
        country_stats = country_stats[country_stats['Posts'] >= 10].sort_values('Sentiment')
        
        st.dataframe(
            country_stats.style.format({
                'Sentiment': '{:.2f}',
                'Posts': '{:,.0f}'
            }),
            hide_index=True
        )
    
    # Leader Analysis
    st.subheader("Leader Analysis")
    
    leader_stats = create_leader_stats(df_leaders)
    
    if not leader_stats.empty:
        # Top 3 Leaders Cards
        top_leaders = leader_stats.head(3)
        
        cols = st.columns(3)
        for i, (_, leader) in enumerate(top_leaders.iterrows()):
            with cols[i]:
                st.markdown(f"### #{i+1} {leader['leader']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mentions", f"{leader['mentions']:,}")
                with col2:
                    st.metric("Sentiment", f"{leader['sentiment']:.2f}")
                    st.metric("Avg. Upvotes", f"{leader['avg_upvotes']:.0f}")
        
        # Leader Rankings Table
        st.markdown("### Leader Rankings")
        
        overall_ranking = leader_stats.copy()
        overall_ranking['Rank'] = range(1, len(overall_ranking) + 1)
        overall_ranking = overall_ranking[[
            'Rank', 'leader', 'mentions',
            'sentiment', 'avg_upvotes'
        ]]
        overall_ranking.columns = [
            'Rank', 'Leader', 'Mentions',
            'Sentiment', 'Avg. Upvotes'
        ]
        
        st.dataframe(
            overall_ranking.style
            .format({
                'Mentions': '{:,.0f}',
                'Sentiment': '{:.2f}',
                'Avg. Upvotes': '{:.0f}'
            })
            .background_gradient(
                subset=['Sentiment'],
                cmap='RdYlBu',
                vmin=-1,
                vmax=1
            ),
            hide_index=True,
            height=400
        )
    
    # Time Analysis
    st.subheader("Sentiment Over Time")
    
    daily_stats = df.groupby('date').agg({
        'sentiment_score': ['mean', 'count']
    }).reset_index()
    daily_stats.columns = ['Date', 'Sentiment', 'Posts']
    
    fig = px.line(
        daily_stats,
        x='Date',
        y='Sentiment',
        custom_data=['Posts']
    )
    
    fig.update_traces(
        line_color='#4575b4',
        hovertemplate="Date: %{x}<br>" +
                     "Sentiment: %{y:.2f}<br>" +
                     "Posts: %{customdata[0]:,}<br>" +
                     "<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution
    st.subheader("Sentiment Distribution")
    
    fig = px.histogram(
        df,
        x='sentiment_score',
        nbins=30,
        color_discrete_sequence=['#4575b4'],
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Number of Posts",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main() 
