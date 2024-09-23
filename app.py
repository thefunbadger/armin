import streamlit as st
import requests
import pandas as pd
import datetime
from requests_oauthlib import OAuth2Session
import warnings
from pymongo import MongoClient
import time
import plotly.express as px
from dotenv import load_dotenv
import os
import numpy as np
from scipy.signal import savgol_filter  # For smoothing lines

# Load environment variables from .env file (for local development)
# Note: In production, prefer using st.secrets
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Password Protection Function
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("Login")
        password = st.text_input("Enter Password", type="password")
        if st.button("Submit"):
            if password == st.secrets["APP_PASSWORD"]:  # Use st.secrets for security
                st.session_state["authenticated"] = True
            else:
                st.error("Incorrect password")
        return False
    return True

# Password Check
if not check_password():
    st.stop()

# Load sensitive information from Streamlit secrets
CLIENT_ID = st.secrets["CLIENT_ID"]  # Facebook App Client ID
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]  # Facebook App Client Secret
REDIRECT_URI = st.secrets["REDIRECT_URI"]  # Replace with your redirect URI
MONGO_CONNECTION_STRING = st.secrets["MONGO_CONNECTION_STRING"]  # MongoDB Atlas connection string
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # OpenAI API Key

# Ensure all required environment variables are set
required_env_vars = ['CLIENT_ID', 'CLIENT_SECRET', 'REDIRECT_URI', 'MONGO_CONNECTION_STRING', 'OPENAI_API_KEY']
missing_vars = [var for var in required_env_vars if not st.secrets.get(var)]
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}. Please set them in st.secrets before running the app.")
    st.stop()

# Define OpenAI API URL for ChatGPT
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

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
    'IMAGE': ['impressions', 'reach', 'saved', 'likes', 'comments'],
    'VIDEO': [
        'plays',
        'clips_replays_count',
        'ig_reels_video_view_total_time',
        'ig_reels_avg_watch_time',
        'video_views',
        'saved',
        'reach',
        'likes',
        'comments'
    ],
    'REELS': [
        'plays',
        'clips_replays_count',
        'ig_reels_video_view_total_time',
        'ig_reels_avg_watch_time',
        'video_views',
        'saved',
        'reach',
        'likes',
        'comments'
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
# Continue from Part 1

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
        expires_in = response.get('expires_in', 5184000)  # Default to 60 days
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
        expires_in = token.get('expires_in', 5184000)  # Default to 60 days
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
            st.write(json_response)

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
            json_response = response.json()
            st.write(json_response)

        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            st.session_state['api_errors'].append(response.json())
            return []
    except Exception as e:
        st.session_state['api_errors'].append({'error': str(e)})
        return []

def fetch_all_data(access_token, instagram_account_id):
    media_items = get_media(access_token, instagram_account_id)

    if not media_items:
        st.warning("No media items retrieved. Ensure your Instagram account has posts and the necessary permissions.")
        return pd.DataFrame()  # Return an empty DataFrame if no media is found

    all_data = []
    for item in media_items:
        media_id = item['id']
        media_type = item['media_type']

        # Fetch insights and log the data
        insights = get_media_insights(access_token, media_id, media_type)
        st.write(insights)  # Log insights to see what is missing

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
            'hashtags': extract_hashtags(item.get('caption', '')),
            'followers': None
        }

        for insight in insights:
            metric_name = insight.get('name')
            if metric_name in data:
                data[metric_name] = insight['values'][0]['value']

        all_data.append(data)

    if not all_data:
        st.session_state['api_errors'].append("No insights data fetched. Please check API permissions and Instagram Business account setup.")

    df = pd.DataFrame(all_data)
    st.write(df)  # Log final data before processing
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
        'followers'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def calculate_engagement_rate(df):
    if 'likes' in df.columns and 'comments' in df.columns and 'impressions' in df.columns:
        # Ensure all are numeric values
        df['engagement_rate'] = ((df['likes'] + df['comments']) / df['impressions']) * 100
        df['engagement_rate'].fillna(0, inplace=True)  # Replace NaN with 0 where impressions are missing
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

    # Hashtag Recommendations
    hashtags_series = df['hashtags'].str.split(',', expand=True).stack()
    top_hashtags = hashtags_series.value_counts()
    if not top_hashtags.empty:
        top_hashtag = top_hashtags.idxmax()
        recommendations.append(f'Try using the hashtag #{top_hashtag} more often to increase visibility.')

    # Best Posting Time
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        top_hour = df['hour'].mode()[0]
        recommendations.append(f'Consider posting more frequently around {top_hour}:00 hours when your audience is most active.')

    return recommendations

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
    # Convert DataFrame to dictionary records
    records = data_df.to_dict(orient='records')
    # Replace existing data for the user
    collection.delete_many({'user_id': user_id})
    # Insert new records with user_id
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
        # Remove 'user_id' from records
        for record in records:
            record.pop('user_id', None)
            record.pop('_id', None)  # Remove MongoDB's internal ID
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        st.error(f"Error fetching data from MongoDB: {e}")
        return pd.DataFrame()

# Function to handle AI Insights using ChatGPT API
def get_ai_insights(prompt):
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.7
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            st.session_state['api_errors'].append(response.json())
            return "AI Error: Unable to fetch insights."
    except Exception as e:
        st.session_state['api_errors'].append({'error': str(e)})
        return "AI Error: An exception occurred."

# Competitor Benchmarking Functions
def get_competitor_data(competitor_account_id, access_token):
    # Placeholder function: Implement data fetching from competitor accounts
    # Requires appropriate permissions and access
    # For demonstration, returning mock data
    mock_data = {
        'competitor_name': f'Competitor {competitor_account_id}',
        'reach': 1500,
        'impressions': 3000,
        'engagement_rate': 2.5
    }
    return mock_data

def compare_performance(df, access_token):
    st.subheader("Competitor Benchmarking")

    competitors = st.text_input("Enter Competitor Instagram Account IDs (comma separated):")
    if competitors:
        competitor_ids = [cid.strip() for cid in competitors.split(',')]
        competitors_data = []
        for cid in competitor_ids:
            data = get_competitor_data(cid, access_token)
            if data:
                competitors_data.append(data)
        
        if competitors_data:
            competitors_df = pd.DataFrame(competitors_data)
            user_metrics = {
                'competitor_name': 'Your Account',
                'reach': df['reach'].mean(),
                'impressions': df['impressions'].mean(),
                'engagement_rate': df['engagement_rate'].mean()
            }
            competitors_df = pd.concat([competitors_df, pd.DataFrame([user_metrics])], ignore_index=True)

            for metric in ['reach', 'impressions', 'engagement_rate']:
                if metric in competitors_df.columns:
                    fig = px.bar(
                        competitors_df, x='competitor_name', y=metric,
                        title=f'Competitor Benchmarking: {metric.capitalize()}',
                        labels={'competitor_name': 'Competitor', metric: metric.capitalize()},
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No competitor data available.")
    else:
        st.info("Enter competitor Instagram Account IDs as comma-separated values without the @ symbol.")
# Visualization Functions with Plotly

def plot_reach_over_time(df):
    """
    Plot Reach Over Time for the given DataFrame.
    Handles missing or incomplete data and applies a smoothing filter for better visualization.
    
    :param df: DataFrame containing 'reach' and 'timestamp' columns.
    :return: Plotly figure or None if data is insufficient.
    """
    if 'reach' not in df.columns or df['reach'].isnull().all():
        st.warning("Reach data is missing or incomplete. Unable to plot reach over time.")
        return None

    try:
        # Ensure data is sorted by timestamp and non-null
        df = df.sort_values(by='timestamp').dropna(subset=['reach', 'timestamp'])
        
        if df.empty:
            st.warning("Reach data is empty after cleaning. Unable to plot.")
            return None

        # Apply smoothing filter (adjust window_length and polyorder as needed)
        window_length = min(7, len(df))  # Ensure the window length is smaller than the dataset
        if window_length < 3:
            st.warning("Not enough data points to apply smoothing. Displaying raw data.")
            reach_smoothed = df['reach']
        else:
            reach_smoothed = savgol_filter(df['reach'], window_length=window_length, polyorder=2)

        # Create the Plotly line chart
        fig = px.line(df, x='timestamp', y=reach_smoothed, 
                      title='Reach Over Time', 
                      labels={'timestamp': 'Date', 'y': 'Reach'},
                      template='plotly_dark')

        # Update chart aesthetics
        fig.update_traces(mode="markers+lines", marker=dict(size=6), line=dict(width=2))
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')

        return fig
    
    except Exception as e:
        st.error(f"Error plotting reach over time: {e}")
        return None

def plot_engagement(df):
    """
    Plot Engagement Rate Over Time.
    
    :param df: DataFrame containing 'engagement_rate' and 'timestamp' columns.
    :return: Plotly figure or None if data is insufficient.
    """
    if 'engagement_rate' not in df.columns or df['engagement_rate'].isnull().all():
        st.warning("Engagement rate data is missing or incomplete.")
        return None

    try:
        # Ensure data is sorted by timestamp and non-null
        df = df.sort_values(by='timestamp').dropna(subset=['engagement_rate', 'timestamp'])
        
        if df.empty:
            st.warning("Engagement rate data is empty after cleaning. Unable to plot.")
            return None

        # Apply smoothing filter if enough data points
        window_length = min(7, len(df))
        if window_length < 3:
            st.warning("Not enough data points to apply smoothing. Displaying raw data.")
            engagement_smoothed = df['engagement_rate']
        else:
            engagement_smoothed = savgol_filter(df['engagement_rate'], window_length=window_length, polyorder=2)

        # Create the Plotly line chart
        fig = px.line(df, x='timestamp', y=engagement_smoothed, 
                      title='Engagement Rate Over Time', 
                      labels={'timestamp': 'Date', 'y': 'Engagement Rate (%)'},
                      template='plotly_dark')

        # Update chart aesthetics
        fig.update_traces(mode="markers+lines", marker=dict(size=6), line=dict(width=2))
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')

        return fig
    
    except Exception as e:
        st.error(f"Error plotting engagement rate over time: {e}")
        return None

def plot_top_posts(df, metric='reach', top_n=5):
    """
    Plot Top N Posts based on a specified metric.
    
    :param df: DataFrame containing posts data.
    :param metric: The metric to sort by (e.g., 'reach', 'impressions').
    :param top_n: Number of top posts to display.
    :return: Plotly figure or None if data is insufficient.
    """
    if metric not in df.columns or df[metric].isnull().all():
        st.warning(f"{metric.capitalize()} data is missing or incomplete. Unable to plot top posts by {metric}.")
        return None
    top_posts = df.sort_values(by=metric, ascending=False).head(top_n)
    fig = px.bar(top_posts, x='id', y=metric, title=f'Top {top_n} Posts by {metric.capitalize()}', labels={'id': 'Post ID', metric: metric.capitalize()}, template='plotly_dark')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_top_hashtags(df):
    """
    Plot the top hashtags used in posts.
    
    :param df: DataFrame containing 'hashtags' column.
    :return: Plotly figure or None if data is insufficient.
    """
    hashtags_series = df['hashtags'].str.split(',', expand=True).stack().str.strip()
    hashtags_counts = hashtags_series.value_counts().reset_index()
    hashtags_counts.columns = ['hashtag', 'count']
    if hashtags_counts.empty:
        st.warning("No hashtags found to display.")
        return None
    fig = px.bar(hashtags_counts.head(10), x='hashtag', y='count', title='Top Hashtags', labels={'hashtag': 'Hashtag', 'count': 'Count'}, template='plotly_dark')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_comprehensive_metrics(df):
    """
    Plot comprehensive metrics over time.
    
    :param df: DataFrame containing various metrics.
    :return: List of Plotly figures.
    """
    metrics = ['impressions', 'reach', 'saved', 'likes', 'comments', 'plays', 'clips_replays_count', 
               'ig_reels_video_view_total_time', 'ig_reels_avg_watch_time', 'video_views']
    
    figs = []
    for metric in metrics:
        if metric in df.columns and not df[metric].isnull().all():
            try:
                df_sorted = df.sort_values(by='timestamp')
                window_length = min(7, len(df_sorted))
                if window_length < 3:
                    metric_smoothed = df_sorted[metric]
                else:
                    metric_smoothed = savgol_filter(df_sorted[metric], window_length=window_length, polyorder=2)
                
                fig = px.line(df_sorted, x='timestamp', y=metric_smoothed, 
                              title=f'{metric.capitalize()} Over Time', 
                              labels={'timestamp': 'Date', 'y': metric.capitalize()}, 
                              template='plotly_dark')
                fig.update_traces(mode="markers+lines", marker=dict(size=6), line=dict(width=2))
                fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
                figs.append(fig)
            except Exception as e:
                st.error(f"Error plotting {metric} over time: {e}")
        else:
            st.warning(f"{metric.capitalize()} data is missing or incomplete. Unable to plot {metric} over time.")
    return figs

def plot_follower_growth(df):
    """
    Plot Follower Growth Over Time.
    
    :param df: DataFrame containing 'followers' and 'timestamp' columns.
    :return: Plotly figure or None if data is insufficient.
    """
    if 'followers' in df.columns and not df['followers'].isnull().all():
        try:
            df_sorted = df.sort_values(by='timestamp').dropna(subset=['followers', 'timestamp'])
            if df_sorted.empty:
                st.warning("Follower data is empty after cleaning. Unable to plot follower growth.")
                return None

            window_length = min(7, len(df_sorted))
            if window_length < 3:
                followers_smoothed = df_sorted['followers']
            else:
                followers_smoothed = savgol_filter(df_sorted['followers'], window_length=window_length, polyorder=2)

            fig = px.line(df_sorted, x='timestamp', y=followers_smoothed, title='Follower Growth Over Time', labels={'timestamp': 'Date', 'y': 'Followers'}, template='plotly_dark')
            fig.update_traces(mode="markers+lines", marker=dict(size=6), line=dict(width=2))
            fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
            return fig
        except Exception as e:
            st.error(f"Error plotting follower growth over time: {e}")
            return None
    else:
        st.warning("Follower data is missing or incomplete. Unable to plot follower growth over time.")
        return None

# Main Application Function
# Updated Main Application Function

def main():
    st.title('Ultimate Instagram Analysis Dashboard')

    # Button to clear cache
    if st.button("Clear Cache and Rerun"):
        st.cache_data.clear()  # Clears cache from Streamlit's data functions
        st.cache_resource.clear()  # Clears cache from resource functions (e.g., database connections)
        st.experimental_rerun()  # Reruns the script

    # Initialize session states
    if 'data_fetched' not in st.session_state:
        st.session_state['data_fetched'] = False
        st.session_state['df'] = pd.DataFrame()
    if 'api_errors' not in st.session_state:
        st.session_state['api_errors'] = []
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = 'default_user'  # Replace with actual user identification

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
                            # Calculate metrics and engagement rate
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

                        # Fetch data if Instagram account is connected
                        if 'instagram_account_id' in st.session_state:
                            access_token = st.session_state['page_access_token']
                            user_instagram_id = st.session_state['instagram_account_id']

                            df = fetch_all_data(access_token, user_instagram_id)
                            if not df.empty:
                                # Calculate metrics and engagement rate
                                df = calculate_metrics(df)
                                df = calculate_engagement_rate(df)
                                st.session_state['df'] = df
                                st.session_state['data_fetched'] = True
                                st.success('Data fetched successfully!')

                                # Save data and token to MongoDB
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
            st.header('AI Settings')
            ai_model = st.selectbox('Select AI Model', options=['gpt-4', 'gpt-3.5-turbo'])
            ai_max_length = st.slider('Max Length', min_value=50, max_value=500, value=300)
            ai_temperature = st.slider('Temperature', min_value=0.1, max_value=1.0, value=0.7)
            ai_top_p = st.slider('Top P', min_value=0.1, max_value=1.0, value=0.9)

        # Apply filters to data
        filtered_df = df[
            (df['timestamp'].dt.date >= start_date) & 
            (df['timestamp'].dt.date <= end_date) & 
            (df['media_type'].isin(selected_media_types))
        ]
        if selected_hashtags:
            filtered_df = filtered_df[filtered_df['hashtags'].str.contains('|'.join(selected_hashtags), na=False)]

        st.write(f"Displaying {len(filtered_df)} out of {len(df)} posts based on selected filters.")

        # Tabs for different features (Content Calendar removed as per instructions)
        tabs = st.tabs(["Key Metrics", "AI Insights", "Competitor Benchmarking"])

        # Key Metrics Tab
        with tabs[0]:
            st.header('Key Metrics Visualization')
            col1, col2 = st.columns(2)
            with col1:
                fig_reach = plot_reach_over_time(filtered_df)
                if fig_reach:
                    st.plotly_chart(fig_reach, use_container_width=True)
            with col2:
                fig_engagement = plot_engagement(filtered_df)  # Updated function for engagement
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

            # Comprehensive metrics
            st.header('Comprehensive Metrics')
            comprehensive_figs = plot_comprehensive_metrics(filtered_df)
            for fig in comprehensive_figs:
                st.plotly_chart(fig, use_container_width=True)

            fig_follower = plot_follower_growth(filtered_df)
            if fig_follower:
                st.plotly_chart(fig_follower, use_container_width=True)

        # AI Insights Tab
        with tabs[1]:
            st.header('AI Insights for Selected Post')
            post_ids = filtered_df['id'].tolist()
            selected_post_id = st.selectbox('Select a Post ID to Get AI Insights', options=post_ids)
            selected_post = filtered_df[filtered_df['id'] == selected_post_id].iloc[0]
            if st.button('Get AI Insight'):
                with st.spinner('Generating insights...'):
                    # Create prompt for AI
                    caption = selected_post.get('caption', 'No caption provided.')
                    metrics = {key: selected_post.get(key, 'N/A') for key in ['impressions', 'reach', 'saved', 'likes', 'comments', 
                                                                               'plays', 'clips_replays_count', 
                                                                               'ig_reels_video_view_total_time', 
                                                                               'ig_reels_avg_watch_time', 'video_views', 'followers']}
                    hashtags_formatted = ' '.join([f"#{tag}" for tag in selected_post['hashtags'].split(',') if tag])
                    prompt = f"""
                    Analyze this Instagram post based on its caption and performance metrics. Provide suggestions for improvement.

                    Caption:
                    {caption}

                    {hashtags_formatted}
                    Metrics:
                    - Impressions: {metrics['impressions']}
                    - Reach: {metrics['reach']}
                    - Saved: {metrics['saved']}
                    - Likes: {metrics['likes']}
                    - Comments: {metrics['comments']}
                    - Plays: {metrics['plays']}
                    - Clips/Replays Count: {metrics['clips_replays_count']}
                    - Reels Video View Total Time: {metrics['ig_reels_video_view_total_time']}
                    - Reels Average Watch Time: {metrics['ig_reels_avg_watch_time']}
                    - Video Views: {metrics['video_views']}
                    - Followers: {metrics.get('followers', 'N/A')}

                    Provide detailed analysis and suggestions.
                    """
                    ai_insight_text = get_ai_insights(prompt)
                    st.subheader('AI Generated Analysis:')
                    st.markdown(f"**Post Caption**: {selected_post['caption']}")
                    st.markdown(f"**AI Insight**: {ai_insight_text}")
                    follow_up_question = st.text_area("Ask a follow-up question about this post (optional):")
                    if st.button('Ask AI'):
                        if follow_up_question.strip():
                            with st.spinner('Getting response from AI...'):
                                follow_up_prompt = f"Q: {follow_up_question}\nA:"
                                follow_up_response = get_ai_insights(follow_up_prompt)
                                st.write(f"**AI Answer:** {follow_up_response}")
                        else:
                            st.warning("Please enter a follow-up question.")

        # Competitor Benchmarking Tab
        with tabs[2]:
            compare_performance(filtered_df, st.session_state['access_token'])

        # Display recommendations
        st.header('Recommendations')
        recommendations = get_recommendations(filtered_df)
        for rec in recommendations:
            st.write(f"- {rec}")

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
