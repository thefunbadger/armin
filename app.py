import streamlit as st
import requests
import pandas as pd
import datetime
from requests_oauthlib import OAuth2Session
import warnings
from pymongo import MongoClient
import plotly.express as px
from dotenv import load_dotenv
import os
import numpy as np
from scipy.signal import savgol_filter
from fpdf import FPDF
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# Password Protection Function
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("Login")
        password = st.text_input("Enter Password", type="password")
        if st.button("Submit"):
            if password == st.secrets["APP_PASSWORD"]:
                st.session_state["authenticated"] = True
            else:
                st.error("Incorrect password")
        return False
    return True

# Password Check
if not check_password():
    st.stop()

# Load sensitive information from Streamlit secrets
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
REDIRECT_URI = st.secrets["REDIRECT_URI"]
MONGO_CONNECTION_STRING = st.secrets["MONGO_CONNECTION_STRING"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Ensure all required environment variables are set
required_env_vars = ['CLIENT_ID', 'CLIENT_SECRET', 'REDIRECT_URI', 'MONGO_CONNECTION_STRING', 'OPENAI_API_KEY']
missing_vars = [var for var in required_env_vars if not st.secrets.get(var)]
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}. Please set them in st.secrets before running the app.")
    st.stop()

# Define OAuth2 scopes
SCOPES = [
    'email',
    'public_profile',
    'pages_show_list',
    'instagram_basic',
    'instagram_manage_insights'
]

# Define Valid Metrics
VALID_METRICS = {
    'IMAGE': [
        'impressions', 'reach', 'saved', 'likes', 'comments',
        'profile_views', 'website_clicks'
    ],
    'VIDEO': [
        'plays', 'clips_replays_count', 'ig_reels_video_view_total_time',
        'ig_reels_avg_watch_time', 'video_views', 'saved', 'reach',
        'likes', 'comments', 'profile_views', 'website_clicks'
    ],
    'REELS': [
        'plays', 'clips_replays_count', 'ig_reels_video_view_total_time',
        'ig_reels_avg_watch_time', 'video_views', 'saved', 'reach',
        'likes', 'comments', 'profile_views', 'website_clicks'
    ]
}

# MongoDB Helper Functions
class MongoDBHelper:
    def __init__(self, connection_string, db_name):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]

    def get_collection(self, collection_name):
        return self.db[collection_name]

    def find_one(self, collection_name, query):
        return self.get_collection(collection_name).find_one(query)

    def update_one(self, collection_name, query, update, upsert=False):
        return self.get_collection(collection_name).update_one(query, update, upsert=upsert)

    def insert_many(self, collection_name, documents):
        return self.get_collection(collection_name).insert_many(documents)

def get_mongo_collection(collection_name):
    return mongo_helper.get_collection(collection_name)

# Initialize the MongoDBHelper with your connection string and database
mongo_helper = MongoDBHelper(MONGO_CONNECTION_STRING, 'thefunbadger')

# Function to exchange short-lived token for a long-lived token
def exchange_for_long_lived_token(short_lived_token):
    url = (
        "https://graph.facebook.com/oauth/access_token"
        f"?grant_type=fb_exchange_token"
        f"&client_id={CLIENT_ID}"
        f"&client_secret={CLIENT_SECRET}"
        f"&fb_exchange_token={short_lived_token}"
    )

    response = requests.get(url).json()

    if 'access_token' in response:
        expires_in = response.get('expires_in', 5184000)
        return response['access_token'], expires_in
    else:
        st.session_state['api_errors'].append(response.get('error', 'Unknown error'))
        return None, None

# OAuth2 Helper Functions
def get_facebook_auth_url():
    oauth = OAuth2Session(client_id=CLIENT_ID, redirect_uri=REDIRECT_URI, scope=SCOPES)
    authorization_url, state = oauth.authorization_url('https://www.facebook.com/dialog/oauth')
    st.session_state['oauth_state'] = state
    return authorization_url

def get_access_token(code):
    oauth = OAuth2Session(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        state=st.session_state.get('oauth_state')
    )

    try:
        token = oauth.fetch_token(
            'https://graph.facebook.com/oauth/access_token',
            client_secret=CLIENT_SECRET,
            code=code
        )
        expires_in = token.get('expires_in', 5184000)
        return {'access_token': token['access_token'], 'expires_in': expires_in}
    except Exception as e:
        st.session_state['api_errors'].append({'error': str(e)})
        return None

# Functions to interact with Facebook Graph API
def get_user_pages(access_token):
    url = f'https://graph.facebook.com/v20.0/me/accounts?access_token={access_token}'
    response = requests.get(url).json()
    return response.get('data', [])

def get_instagram_account_id(page_id, page_access_token):
    url = f'https://graph.facebook.com/v20.0/{page_id}?fields=instagram_business_account&access_token={page_access_token}'
    response = requests.get(url).json()
    instagram_account = response.get('instagram_business_account')
    if instagram_account:
        return instagram_account['id']
    return None

def get_media(access_token, instagram_account_id):
    media = []
    fields = 'id,caption,timestamp,media_type,media_url,permalink,thumbnail_url'
    url = f"https://graph.facebook.com/v20.0/{instagram_account_id}/media?fields={fields}&access_token={access_token}"
    while url:
        response = requests.get(url)
        if response.status_code == 200:
            json_response = response.json()
            media.extend(json_response.get('data', []))
            url = json_response.get('paging', {}).get('next', '')
        else:
            st.session_state['api_errors'].append(response.json())
            break
    return media

def get_media_insights(access_token, media_id, media_type):
    try:
        metrics = VALID_METRICS.get(media_type)
        if not metrics:
            st.session_state['api_errors'].append(f"Unsupported media type: {media_type}. Skipping insights.")
            return []

        metrics_str = ','.join(metrics)
        url = f"https://graph.facebook.com/v20.0/{media_id}/insights?metric={metrics_str}&access_token={access_token}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            st.session_state['api_errors'].append(response.json())
            return []
    except Exception as e:
        st.session_state['api_errors'].append({'error': str(e)})
        return []

def fetch_account_insights(access_token, instagram_account_id):
    try:
        metrics = ['followers_count']
        metrics_str = ','.join(metrics)
        url = f"https://graph.facebook.com/v20.0/{instagram_account_id}/insights?metric={metrics_str}&access_token={access_token}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get('data', [])
            if data:
                return data[0]['values'][0]['value']
        else:
            st.session_state['api_errors'].append(response.json())
        return None
    except Exception as e:
        st.session_state['api_errors'].append({'error': str(e)})
        return None

def fetch_all_data(access_token, instagram_account_id):
    media_items = get_media(access_token, instagram_account_id)

    if not media_items:
        st.warning("No media items retrieved. Ensure your Instagram account has posts and the necessary permissions.")
        return pd.DataFrame()

    followers = fetch_account_insights(access_token, instagram_account_id)

    all_data = []
    for item in media_items:
        media_id = item['id']
        media_type = item['media_type']

        insights = get_media_insights(access_token, media_id, media_type)

        data = {
            'id': media_id,
            'caption': item.get('caption', ''),
            'timestamp': item['timestamp'],
            'media_type': media_type,
            'media_url': item.get('media_url', ''),
            'permalink': item['permalink'],
            'impressions': None,
            'reach': None,
            'saved': None,
            'likes': None,
            'comments': None,
            'plays': None,
            'clips_replays_count': None,
            'ig_reels_video_view_total_time': None,
            'ig_reels_avg_watch_time': None,
            'video_views': None,
            'profile_views': None,
            'website_clicks': None,
            'hashtags': extract_hashtags(item.get('caption', '')),
            'followers': followers
        }

        for insight in insights:
            metric_name = insight.get('name')
            if metric_name in data:
                data[metric_name] = insight['values'][0]['value']

        all_data.append(data)

    if not all_data:
        st.session_state['api_errors'].append("No insights data fetched. Please check API permissions and Instagram Business account setup.")

    df = pd.DataFrame(all_data)
    return df

def extract_hashtags(caption):
    hashtags = [tag.strip('#') for tag in caption.split() if tag.startswith('#')]
    return ','.join(hashtags)

# Data Analysis Functions
def calculate_metrics(df):
    numeric_columns = [
        'impressions',
        'reach',
        'saved',
        'likes',
        'comments',
        'plays',
        'clips_replays_count',
        'ig_reels_video_view_total_time',
        'ig_reels_avg_watch_time',
        'video_views',
        'profile_views',
        'website_clicks',
        'followers'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def calculate_engagement_rate(df):
    if 'likes' in df.columns and 'comments' in df.columns and 'impressions' in df.columns:
        df['engagement_rate'] = ((df['likes'] + df['comments']) / df['impressions']) * 100
        df['engagement_rate'].fillna(0, inplace=True)
    else:
        st.warning("Missing columns to calculate engagement rate.")
    return df

# Recommendations Function
def get_recommendations(df):
    recommendations = []
    avg_reach = df['reach'].mean()
    if pd.notnull(avg_reach) and avg_reach < 1000:
        recommendations.append('Your average reach is below 1,000. Consider optimizing your posting times and content.')
    elif pd.notnull(avg_reach):
        recommendations.append('Your average reach is healthy. Keep up the good work!')

    hashtags_series = df['hashtags'].str.split(',', expand=True).stack()
    top_hashtags = hashtags_series.value_counts()
    if not top_hashtags.empty:
        top_hashtag = top_hashtags.idxmax()
        recommendations.append(f'Try using the hashtag #{top_hashtag} more often to increase visibility.')

    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        top_hour = df['hour'].mode()[0]
        recommendations.append(f'Consider posting more frequently around {top_hour}:00 hours when your audience is most active.')

    avg_engagement = df['engagement_rate'].mean()
    if pd.notnull(avg_engagement) and avg_engagement < 1.5:
        recommendations.append('Your engagement rate is below industry standards. Engage more with your audience through stories and interactive posts.')
    elif pd.notnull(avg_engagement):
        recommendations.append('Your engagement rate is good. Continue creating engaging content!')

    if 'followers' in df.columns and not df['followers'].isnull().all():
        follower_growth = df['followers'].diff().mean()
        if pd.notnull(follower_growth) and follower_growth < 10:
            recommendations.append('Your follower growth rate is slow. Consider running promotions or collaborating with influencers.')

    return recommendations

# Export Functions
def export_to_csv(df):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='instagram_data.csv',
        mime='text/csv',
    )

def export_to_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=10)

    # Title
    pdf.cell(200, 10, txt="Instagram Data Analysis", ln=True, align='C')

    # Add table headers
    headers = df.columns.tolist()
    for header in headers:
        pdf.cell(25, 10, header, border=1)
    pdf.ln()

    # Add table rows
    for index, row in df.iterrows():
        for item in row:
            text = str(item) if pd.notnull(item) else ''
            pdf.cell(25, 10, text, border=1)
        pdf.ln()

    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    st.download_button(
        label="Download data as PDF",
        data=pdf_buffer,
        file_name='instagram_data.pdf',
        mime='application/pdf',
    )

# Function to save access token to DB
def save_access_token_to_db(token, expires_at, user_id):
    mongo_helper.update_one('auth', {'user_id': user_id}, {'$set': {'token': token, 'expires_at': expires_at}}, upsert=True)

# Function to get access token from DB
def get_access_token_from_db(user_id):
    collection = get_mongo_collection('auth')
    try:
        data = collection.find_one({'user_id': user_id})
        if data:
            token, expires_at = data['token'], data['expires_at']
            if datetime.datetime.now() > datetime.datetime.fromisoformat(expires_at):
                st.error("Token has expired. Please log in again.")
                return None, None
            return token, expires_at
        else:
            st.warning("No access token found in the database.")
            return None, None
    except Exception as e:
        st.error(f"Error fetching access token from MongoDB: {e}")
        return None, None

# Function to save data to DB
def save_data_to_db(data_df, user_id):
    collection = get_mongo_collection('data')
    records = data_df.to_dict(orient='records')
    collection.delete_many({'user_id': user_id})
    for record in records:
        record['user_id'] = user_id
        collection.insert_one(record)

# Function to get data from DB
def get_data_from_db(user_id):
    collection = get_mongo_collection('data')
    try:
        cursor = collection.find({'user_id': user_id})
        records = list(cursor)
        if not records:
            return pd.DataFrame()
        for record in records:
            record.pop('user_id', None)
            record.pop('_id', None)
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        st.error(f"Error fetching data from MongoDB: {e}")
        return pd.DataFrame()

# Visualization Functions with Plotly
def plot_reach_over_time(df):
    if 'reach' not in df.columns or df['reach'].isnull().all():
        st.warning("Reach data is missing or incomplete. Unable to plot reach over time.")
        return None

    try:
        df = df.sort_values(by='timestamp').dropna(subset=['reach', 'timestamp'])

        if df.empty:
            st.warning("Reach data is empty after cleaning. Unable to plot.")
            return None

        window_length = min(7, len(df))
        if window_length < 3:
            reach_smoothed = df['reach']
        else:
            reach_smoothed = savgol_filter(df['reach'], window_length=window_length, polyorder=2)

        fig = px.line(df, x='timestamp', y=reach_smoothed, 
                      title='Reach Over Time', 
                      labels={'timestamp': 'Date', 'reach': 'Reach'},
                      template='plotly_dark')

        fig.update_traces(mode="markers+lines", marker=dict(size=6), line=dict(width=2))
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')

        return fig

    except Exception as e:
        st.error(f"Error plotting reach over time: {e}")
        return None

def plot_engagement(df):
    if 'engagement_rate' in df.columns and not df['engagement_rate'].isnull().all():
        fig = px.line(df, x='timestamp', y='engagement_rate', title='Engagement Rate Over Time',
                      labels={'timestamp': 'Date', 'engagement_rate': 'Engagement Rate (%)'}, 
                      template='plotly_dark')
        fig.update_traces(mode="markers+lines", marker=dict(size=6), line=dict(width=2))
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
        return fig
    else:
        st.warning("Engagement rate data is missing or incomplete.")
        return None

def plot_top_posts(df, metric='reach', top_n=5):
    if metric not in df.columns or df[metric].isnull().all():
        st.warning(f"{metric.capitalize()} data is missing or incomplete. Unable to plot top posts by {metric}.")
        return None
    top_posts = df.sort_values(by=metric, ascending=False).head(top_n)
    fig = px.bar(top_posts, x='id', y=metric, title=f'Top {top_n} Posts by {metric.capitalize()}', labels={'id': 'Post ID', metric: metric.capitalize()}, template='plotly_dark')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_top_hashtags(df):
    hashtags_series = df['hashtags'].str.split(',', expand=True).stack()
    hashtags_counts = hashtags_series.value_counts().reset_index()
    hashtags_counts.columns = ['hashtag', 'count']
    if hashtags_counts.empty:
        st.warning("No hashtags found to display.")
        return None
    fig = px.bar(hashtags_counts.head(10), x='hashtag', y='count', title='Top Hashtags', labels={'hashtag': 'Hashtag', 'count': 'Count'}, template='plotly_dark')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_comprehensive_metrics(df):
    metrics = [
        'impressions', 'reach', 'saved', 'likes', 'comments', 'plays',
        'clips_replays_count', 'ig_reels_video_view_total_time',
        'ig_reels_avg_watch_time', 'video_views', 'profile_views', 'website_clicks'
    ]

    figs = []
    for metric in metrics:
        if metric in df.columns and not df[metric].isnull().all():
            df_sorted = df.sort_values(by='timestamp')
            if len(df_sorted) >= 3:
                metric_smoothed = savgol_filter(df_sorted[metric], window_length=min(7, len(df_sorted)), polyorder=2)
            else:
                metric_smoothed = df_sorted[metric]
            fig = px.line(df_sorted, x='timestamp', y=metric_smoothed, title=f'{metric.capitalize()} Over Time', 
                          labels={'timestamp': 'Date', metric: metric.capitalize()}, 
                          template='plotly_dark')
            fig.update_traces(mode="markers+lines", marker=dict(size=6), line=dict(width=2))
            fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
            figs.append(fig)
        else:
            st.warning(f"{metric.capitalize()} data is missing or incomplete. Unable to plot {metric} over time.")
    return figs

def plot_follower_growth(df):
    if 'followers' in df.columns and not df['followers'].isnull().all():
        fig = px.line(df, x='timestamp', y='followers', title='Follower Growth Over Time', labels={'timestamp': 'Date', 'followers': 'Followers'}, template='plotly_dark')
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
        return fig
    else:
        st.warning("Follower data is missing or incomplete. Unable to plot follower growth over time.")
        return None

# Main Application Function
def main():
    st.title('Ultimate Instagram Analysis Dashboard')

    # Button to clear cache
    if st.button("Clear Cache and Rerun"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

    # Initialize session states
    if 'data_fetched' not in st.session_state:
        st.session_state['data_fetched'] = False
        st.session_state['df'] = pd.DataFrame()
    if 'api_errors' not in st.session_state:
        st.session_state['api_errors'] = []
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = 'default_user'

    user_id = st.session_state['user_id']

    # Check for access token in session or MongoDB
    if 'access_token' not in st.session_state:
        token_data = get_access_token_from_db(user_id)

        if token_data and token_data[0]:
            st.session_state['access_token'] = token_data[0]
            st.session_state['expires_at'] = token_data[1]

            if datetime.datetime.now() > datetime.datetime.fromisoformat(st.session_state['expires_at']):
                st.error('Access token has expired. Please log in again.')
                st.session_state.clear()
                st.experimental_rerun()

    # Authenticate if access token exists
    if 'access_token' in st.session_state and st.session_state['access_token']:
        if 'expires_at' in st.session_state:
            if datetime.datetime.now() > datetime.datetime.fromisoformat(st.session_state['expires_at']):
                st.error('Access token has expired. Please log in again.')
                st.session_state.clear()
                st.experimental_rerun()
            else:
                st.success('Successfully Authenticated!')

                # Load data from database or fetch from API
                if not st.session_state['data_fetched']:
                    with st.spinner('Loading data from database...'):
                        df = get_data_from_db(user_id)
                        if not df.empty:
                            df = calculate_metrics(df)
                            df = calculate_engagement_rate(df)
                            st.session_state['df'] = df
                            st.session_state['data_fetched'] = True
                            st.success('Data loaded from database!')
                        else:
                            st.info('No cached data found. Please update data to fetch from API.')

                # Update data if necessary
                if st.button('Update Data') or not st.session_state['data_fetched']:
                    with st.spinner('Fetching and processing data...'):
                        pages = get_user_pages(st.session_state['access_token'])
                        if not pages:
                            st.error("No Facebook Pages found. Ensure your account manages a page connected to Instagram.")
                            return

                        instagram_account_id = None
                        for page in pages:
                            page_id = page['id']
                            page_access_token = page['access_token']
                            instagram_account_id = get_instagram_account_id(page_id, page_access_token)
                            if instagram_account_id:
                                st.session_state['instagram_account_id'] = instagram_account_id
                                st.session_state['page_access_token'] = page_access_token
                                break

                        if 'instagram_account_id' in st.session_state:
                            access_token = st.session_state['page_access_token']
                            user_instagram_id = st.session_state['instagram_account_id']

                            df = fetch_all_data(access_token, user_instagram_id)
                            if not df.empty:
                                df = calculate_metrics(df)
                                df = calculate_engagement_rate(df)
                                st.session_state['df'] = df
                                st.session_state['data_fetched'] = True
                                st.success('Data fetched successfully!')

                                save_data_to_db(df, user_id)
                                save_access_token_to_db(
                                    token=st.session_state['access_token'],
                                    expires_at=st.session_state['expires_at'],
                                    user_id=user_id
                                )
                            else:
                                st.warning("No data available to display.")

    # Check if data is available for visualization
    if not st.session_state['df'].empty:
        df = st.session_state['df']
        
        # Sidebar filters
        with st.sidebar:
            st.header('Filter Options')
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()
            start_date, end_date = st.date_input('Select Date Range', [min_date, max_date], min_value=min_date, max_value=max_date)
            media_types = df['media_type'].unique().tolist()
            selected_media_types = st.multiselect('Select Media Types', media_types, default=media_types)
            all_hashtags = df['hashtags'].dropna().str.split(',', expand=True).stack().str.strip().unique().tolist()
            selected_hashtags = st.multiselect('Select Hashtags', all_hashtags)
            
            st.header('Export Data')
            export_csv = st.button('Export to CSV')
            export_pdf = st.button('Export to PDF')

        # Apply filters to data
        filtered_df = df[
            (df['timestamp'].dt.date >= start_date) & 
            (df['timestamp'].dt.date <= end_date) & 
            (df['media_type'].isin(selected_media_types))
        ]
        if selected_hashtags:
            filtered_df = filtered_df[filtered_df['hashtags'].str.contains('|'.join(selected_hashtags), na=False)]

        st.write(f"Displaying {len(filtered_df)} out of {len(df)} posts based on selected filters.")

        # Tabs for different features
        tabs = st.tabs(["Key Metrics", "Recommendations"])

        # Key Metrics Tab
        with tabs[0]:
            st.header('Key Metrics Visualization')
            col1, col2 = st.columns(2)
            with col1:
                fig_reach = plot_reach_over_time(filtered_df)
                if fig_reach:
                    st.plotly_chart(fig_reach, use_container_width=True)
            with col2:
                fig_engagement = plot_engagement(filtered_df)
                if fig_engagement:
                    st.plotly_chart(fig_engagement, use_container_width=True)
            col3, col4 = st.columns(2)
            with col3:
                fig_top_reach = plot_top_posts(filtered_df, metric='reach')
                if fig_top_reach:
                    st.plotly_chart(fig_top_reach, use_container_width=True)
            with col4:
                fig_top_hashtags = plot_top_hashtags(filtered_df)
                if fig_top_hashtags:
                    st.plotly_chart(fig_top_hashtags, use_container_width=True)

            st.header('Comprehensive Metrics')
            comprehensive_figs = plot_comprehensive_metrics(filtered_df)
            for fig in comprehensive_figs:
                st.plotly_chart(fig, use_container_width=True)

            fig_follower = plot_follower_growth(filtered_df)
            if fig_follower:
                st.plotly_chart(fig_follower, use_container_width=True)
            
            st.header('Export Data')
            if export_csv:
                export_to_csv(filtered_df)
            if export_pdf:
                export_to_pdf(filtered_df)

        # Recommendations Tab
        with tabs[1]:
            st.header('Recommendations')
            recommendations = get_recommendations(filtered_df)
            if recommendations:
                for rec in recommendations:
                    st.write(f"- {rec}")
            else:
                st.write("No recommendations available.")

    # If not authenticated, provide Facebook login option
    if 'access_token' not in st.session_state:
        st.header('Login with Facebook')
        authorization_url = get_facebook_auth_url()
        st.markdown(f'<a href="{authorization_url}">Login with Facebook</a>', unsafe_allow_html=True)

        query_params = st.experimental_get_query_params()
        if 'code' in query_params:
            code = query_params['code'][0]
            token = get_access_token(code)
            if token:
                long_lived_token, expires_in = exchange_for_long_lived_token(token['access_token'])
                if long_lived_token:
                    st.session_state['access_token'] = long_lived_token
                    st.session_state['expires_at'] = (datetime.datetime.now() + datetime.timedelta(seconds=expires_in)).isoformat()

                    save_access_token_to_db(
                        token=long_lived_token,
                        expires_at=st.session_state['expires_at'],
                        user_id=user_id
                    )

                    st.experimental_set_query_params()
                    st.experimental_rerun()
                else:
                    st.session_state['api_errors'].append('Failed to obtain a long-lived access token.')
            else:
                st.session_state['api_errors'].append('Failed to retrieve access token.')

        if st.session_state['api_errors']:
            with st.expander("View API Errors"):
                for error in st.session_state['api_errors']:
                    st.write(error)

if __name__ == '__main__':
    main()
