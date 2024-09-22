import streamlit as st
import requests
import pandas as pd
from datetime import timezone
now = datetime.datetime.now(timezone.utc)
from requests_oauthlib import OAuth2Session
import warnings
import os
from dotenv import load_dotenv
import plotly.express as px
from pymongo import MongoClient
import time
import random

# Password Protection
# Load environment variables from .env file (only for local development)
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load sensitive information from environment variables or Streamlit secrets
if st.secrets:
    # When running on Streamlit Cloud with secrets
    CLIENT_ID = st.secrets["CLIENT_ID"]
    CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
    REDIRECT_URI = st.secrets["REDIRECT_URI"]
    MONGO_CONNECTION_STRING = st.secrets["MONGO_CONNECTION_STRING"]
    HF_API_TOKEN = st.secrets["HF_API_TOKEN"]  # Hugging Face API Token
    MELIPAYAMAK_USERNAME = os.getenv('MELIPAYAMAK_USERNAME')
    MELIPAYAMAK_PASSWORD = os.getenv('MELIPAYAMAK_PASSWORD')
else:
    # When running locally
    CLIENT_ID = os.getenv('CLIENT_ID')  # Facebook App Client ID
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')  # Facebook App Client Secret
    REDIRECT_URI = os.getenv('REDIRECT_URI')  # Replace with your redirect URI
    MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')  # MongoDB Atlas connection string
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')  # Hugging Face API Token

# Ensure all required environment variables are set
required_env_vars = ['CLIENT_ID', 'CLIENT_SECRET', 'REDIRECT_URI', 'MONGO_CONNECTION_STRING', 'HF_API_TOKEN']
missing_vars = [var for var in required_env_vars if not (st.secrets.get(var) or os.getenv(var))]
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}. Please set them before running the app.")
    st.stop()
# In your token expiry check, use timezone-aware datetime
now = datetime.datetime.now(timezone.utc)
if datetime.datetime.fromisoformat(expires_at) < now:
    st.error("Token has expired. Please log in again.")
    return None, None
# Define Hugging Face API URL for Persian
HF_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"

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
def get_mongo_client():
    return MongoClient(MONGO_CONNECTION_STRING)

def get_mongo_collection(collection_name):
    client = get_mongo_client()
    db = client['thefunbadger']  # Replace with your database name
    collection = db[collection_name]  # Dynamic collection name
    return collection
def save_otp_to_db(phone_number, otp):
    collection = get_mongo_collection('otps')
    collection.update_one(
        {'phone_number': phone_number},
        {'$set': {'otp': otp, 'created_at': time.time()}},
        upsert=True
    )

def validate_otp(phone_number, otp):
    collection = get_mongo_collection('otps')
    otp_entry = collection.find_one({'phone_number': phone_number})
    
    if otp_entry and otp_entry['otp'] == otp:
        time_elapsed = time.time() - otp_entry['created_at']
        if time_elapsed <= 300:  # OTP valid for 5 minutes
            return True
    return False

# Function to send OTP via Melipayamak API
def send_otp(phone_number, otp):
    MELIPAYAMAK_USERNAME = os.getenv('MELIPAYAMAK_USERNAME')
    MELIPAYAMAK_PASSWORD = os.getenv('MELIPAYAMAK_PASSWORD')
    sender_number = "50004001654470"  # Replace with your actual sender number
    
    url = "https://rest.payamak-panel.com/api/SendSMS/SendSMS"
    payload = {
        'username': MELIPAYAMAK_USERNAME,
        'password': MELIPAYAMAK_PASSWORD,
        'to': phone_number,
        'from': sender_number,
        'text': f'Your OTP is {otp}',
        'isflash': False
    }
    
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Failed to send OTP. Status Code: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Error sending OTP: {e}")
        return False

# OTP Generation
def generate_otp():
    return random.randint(100000, 999999)

# OTP-based Authentication Function
def otp_login():
    st.title("Login with OTP")

    phone_number = st.text_input("Enter your phone number")
    if st.button("Send OTP"):
        otp = generate_otp()
        if send_otp(phone_number, otp):
            save_otp_to_db(phone_number, otp)
            st.success(f"OTP has been sent to {phone_number}.")
        else:
            st.error("Failed to send OTP. Please try again.")

    otp_input = st.text_input("Enter the OTP")
    if st.button("Verify OTP"):
        if validate_otp(phone_number, otp_input):
            st.success("OTP verified successfully. You are logged in!")
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid OTP or OTP expired.")

def save_access_token_to_db(token, expires_at, user_id):
    collection = get_mongo_collection('auth')
    collection.update_one(
        {'user_id': user_id},
        {'$set': {'token': token, 'expires_at': expires_at}},
        upsert=True
    )

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
# AI Integration
def query_huggingface(prompt, max_retries=3):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "options": {"use_cache": False},
        "parameters": {
            "max_length": 200,
            "no_repeat_ngram_size": 2,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    for attempt in range(max_retries):
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0].get('generated_text')
        elif response.status_code == 503:
            st.warning("AI model loading. Please wait...")
            time.sleep(2 ** attempt)
        else:
            return f"Error: {response.json().get('error', 'Unknown error')}"
    return "AI model is not available at the moment."

def save_ai_insight_to_db(post_id, ai_text):
    collection = get_mongo_collection('auth')
    collection.update_one(
        {'post_id': post_id},
        {'$set': {'ai_insight': ai_text}},
        upsert=True
    )

def get_ai_insight_from_db(post_id):
    collection = get_mongo_collection('auth')
    try:
        data = collection.find_one({'post_id': post_id})
        if data and 'ai_insight' in data:
            return data['ai_insight']
        return None
    except Exception as e:
        st.error(f"Error fetching AI insight from MongoDB: {e}")
        return None

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
            media.extend(json_response.get('data', []))
            url = json_response.get('paging', {}).get('next', '')
        else:
            st.session_state['api_errors'].append(response.json())
            break
    return media

def get_media_insights(access_token, media_id, media_type):
    try:
        # Fetch the valid metrics for the given media_type
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
            # Log the error internally without displaying it to the user
            st.session_state['api_errors'].append(response.json())
            return []
    except Exception as e:
        # Log the exception internally
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
        
        # Fetch insights for the current media item
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
            'hashtags': extract_hashtags(item.get('caption', '')),
            'followers': None
        }

        # Populate insights data
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
    # Convert relevant columns to numeric types
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
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_recommendations(df):
    recommendations = []
    
    # Example Recommendation: Analyze reach
    avg_reach = df['reach'].mean()
    if pd.notnull(avg_reach) and avg_reach < 1000:
        recommendations.append('Your average reach is below 1,000. Consider optimizing your posting times and content.')
    elif pd.notnull(avg_reach):
        recommendations.append('Your average reach is healthy. Keep up the good work!')

    # Analyze engagement rate
    if 'likes' in df.columns and 'comments' in df.columns and 'followers' in df.columns:
        df['engagement_rate'] = (df['likes'] + df['comments']) / df['followers'].replace(0, 1) * 100
        avg_engagement = df['engagement_rate'].mean()
        if pd.notnull(avg_engagement) and avg_engagement < 1:
            recommendations.append('Your average engagement rate is below 1%. Engage more with your audience through interactive content.')
        elif pd.notnull(avg_engagement):
            recommendations.append('Your engagement rate is healthy. Continue creating engaging content!')

    # Analyze hashtags
    hashtags_series = df['hashtags'].str.split(',', expand=True).stack()
    top_hashtags = hashtags_series.value_counts()
    if not top_hashtags.empty:
        top_hashtag = top_hashtags.idxmax()
        recommendations.append(f'Try using the hashtag #{top_hashtag} more often to increase visibility.')
    
    # Suggest optimal posting times
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        top_hour = df['hour'].mode()[0]
        recommendations.append(f'Consider posting more frequently around {top_hour}:00 hours when your audience is most active.')
    
    return recommendations

# Visualization Functions with Plotly
def plot_reach_over_time(df):
    if 'reach' in df.columns and not df['reach'].isnull().all():
        fig = px.line(df, x='timestamp', y='reach', title='Reach Over Time', labels={'timestamp': 'Date', 'reach': 'Reach'}, template='plotly_dark')
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Reach data is missing or incomplete. Unable to plot reach over time.")

def plot_engagement_over_time(df):
    if 'engagement_rate' in df.columns and not df['engagement_rate'].isnull().all():
        fig = px.line(df, x='timestamp', y='engagement_rate', title='Engagement Rate Over Time', labels={'timestamp': 'Date', 'engagement_rate': 'Engagement Rate (%)'}, template='plotly_dark')
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Engagement rate data is missing or incomplete. Unable to plot engagement rate over time.")

def plot_top_posts(df, metric='reach', top_n=5):
    if metric not in df.columns or df[metric].isnull().all():
        st.warning(f"{metric.capitalize()} data is missing or incomplete. Unable to plot top posts by {metric}.")
        return
    top_posts = df.sort_values(by=metric, ascending=False).head(top_n)
    fig = px.bar(top_posts, x='id', y=metric, title=f'Top {top_n} Posts by {metric.capitalize()}', labels={'id': 'Post ID', metric: metric.capitalize()}, template='plotly_dark')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def plot_top_hashtags(df):
    hashtags_series = df['hashtags'].str.split(',', expand=True).stack()
    hashtags_counts = hashtags_series.value_counts().reset_index()
    hashtags_counts.columns = ['hashtag', 'count']
    if hashtags_counts.empty:
        st.warning("No hashtags found to display.")
        return
    fig = px.bar(hashtags_counts.head(10), x='hashtag', y='count', title='Top Hashtags', labels={'hashtag': 'Hashtag', 'count': 'Count'}, template='plotly_dark')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def plot_comprehensive_metrics(df):
    metrics = ['impressions', 'reach', 'saved', 'likes', 'comments', 'plays', 'clips_replays_count', 
               'ig_reels_video_view_total_time', 'ig_reels_avg_watch_time', 'video_views']
    
    for metric in metrics:
        if metric in df.columns and not df[metric].isnull().all():
            fig = px.line(df, x='timestamp', y=metric, title=f'{metric.capitalize()} Over Time', labels={'timestamp': 'Date', metric: metric.capitalize()}, template='plotly_dark')
            fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"{metric.capitalize()} data is missing or incomplete. Unable to plot {metric} over time.")

def plot_follower_growth(df):
    if 'followers' in df.columns and not df['followers'].isnull().all():
        fig = px.line(df, x='timestamp', y='followers', title='Follower Growth Over Time', labels={'timestamp': 'Date', 'followers': 'Followers'}, template='plotly_dark')
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Follower data is missing or incomplete. Unable to plot follower growth over time.")

# AI Assistance Functions with Retry Logic
def query_huggingface(prompt, max_retries=5, backoff_factor=2):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "options": {"use_cache": False},
        "parameters": {
            "max_length": 300,
            "no_repeat_ngram_size": 2,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    for attempt in range(1, max_retries + 1):
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            st.warning(f"AI Model is loading. Retrying in {backoff_factor} seconds... (Attempt {attempt}/{max_retries})")
            time.sleep(backoff_factor)
            backoff_factor *= 2  # Exponential backoff
        else:
            try:
                error_message = response.json().get('error', 'Unknown error')
            except:
                error_message = 'Unknown error'
            st.session_state['api_errors'].append({'error': error_message})
            return {'error': error_message}
    return {'error': 'Max retries exceeded. AI model is still loading or unavailable.'}

def ai_insight(selected_post, user_id):
    # Check if AI insight already exists in the database
    existing_insight = get_ai_insight_from_db(selected_post['id'])
    if existing_insight:
        return existing_insight
    
    caption = selected_post.get('caption', 'No caption provided.')
    metrics = {key: selected_post.get(key, 'N/A') for key in ['impressions', 'reach', 'saved', 'likes', 'comments', 
                                                               'plays', 'clips_replays_count', 
                                                               'ig_reels_video_view_total_time', 
                                                               'ig_reels_avg_watch_time', 'video_views', 'followers']}
    # Create a prompt for the AI in Persian
    hashtags_formatted = ' '.join([f"#{tag}" for tag in selected_post['hashtags'].split(',') if tag])
    prompt = f"""
    تحلیل کنید پست اینستاگرام را بر اساس کپشن و معیارهای عملکرد آن. پیشنهاداتی برای بهبود با در نظر گرفتن آخرین به‌روزرسانی‌های اینستاگرام ارائه دهید.
    
    کپشن:
    {caption}
    
    {hashtags_formatted}
    معیارها:
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
    
    یک تحلیل دقیق و پیشنهادات ارائه دهید.
    """
    
    # Query the Hugging Face API
    response = query_huggingface(prompt)
    
    # Extract the generated text
    if isinstance(response, dict) and 'error' in response:
        return f"AI Error: {response['error']}"
    elif isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict) and 'generated_text' in response[0]:
        ai_text = response[0]['generated_text']
    elif isinstance(response, str):
        ai_text = response
    else:
        ai_text = "AI couldn't generate a response. Please try again later."
    
    # Save the AI insight to the database
    save_ai_insight_to_db(selected_post['id'], ai_text)
    
    return ai_text

# Competitor Benchmarking (New Feature)
def compare_performance(df, competitors_data):
    """Compare user performance with competitor data."""
    st.subheader("Competitor Benchmarking")
    competitors_df = pd.DataFrame(competitors_data)
    
    for metric in ['reach', 'impressions', 'engagement_rate']:
        fig = px.bar(
            competitors_df, x='competitor_name', y=metric,
            title=f'Competitor Benchmarking: {metric.capitalize()}',
            labels={'competitor_name': 'Competitor', metric: metric.capitalize()}
        )
        st.plotly_chart(fig, use_container_width=True)

# AI-Powered Insights (New Feature)
def ai_powered_audience_insight(df):
    """Provide AI-powered insights about audience behavior."""
    st.subheader("AI-Powered Audience Insights")
    prompt = "Analyze the audience behavior based on recent posts and engagement."
    
    # Example interaction with Hugging Face API (same as original Hugging Face function)
    ai_response = query_huggingface(prompt)
    
    if isinstance(ai_response, dict) and 'error' in ai_response:
        st.error(f"AI Error: {ai_response['error']}")
    else:
        st.write(ai_response[0]['generated_text'])

# Visual Content Calendar (New Feature)
def display_visual_content_calendar(df):
    """Display a visual content calendar with drag-and-drop post scheduling."""
    st.subheader("Visual Content Calendar")
    
    # Example Calendar Visualization (customizable for a more visual calendar)
    df['scheduled_post_time'] = pd.to_datetime(df['timestamp'])
    calendar_fig = px.timeline(
        df, x_start='scheduled_post_time', x_end='scheduled_post_time', y='media_type',
        title='Instagram Post Calendar', labels={'scheduled_post_time': 'Time', 'media_type': 'Post Type'}
    )
    st.plotly_chart(calendar_fig, use_container_width=True)

# Post Recommendations (New Feature)
def get_post_recommendations(df):
    """Provide actionable insights and recommendations for improving Instagram strategy."""
    recommendations = []
    
    avg_likes = df['likes'].mean()
    if avg_likes < 100:
        recommendations.append("Your average likes are below 100. Consider improving visual quality.")
    else:
        recommendations.append("Good job! Your posts are receiving good engagement.")

    return recommendations

# Main Application Function
def main():
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        otp_login()
    else:
        st.title('Ultimate Instagram Analysis Dashboard')
        
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.experimental_rerun()

        # Initialize session state for data caching and error logging
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

        # If a valid access token exists
        if 'access_token' in st.session_state and st.session_state['access_token']:
            if 'expires_at' in st.session_state:
                if datetime.datetime.now() > datetime.datetime.fromisoformat(st.session_state['expires_at']):
                    st.error('Access token has expired. Please log in again.')
                    st.session_state.clear()
                    st.experimental_rerun()
                else:
                    st.success('Successfully Authenticated!')

                    # Load data from DB if not already fetched
                    if not st.session_state['data_fetched']:
                        with st.spinner('Loading data from database...'):
                            df = get_data_from_db(user_id)
                            if not df.empty:
                                df = calculate_metrics(df)
                                st.session_state['df'] = df
                                st.session_state['data_fetched'] = True
                                st.success('Data loaded from database!')
                            else:
                                st.info('No cached data found. Please update data to fetch from API.')

                    # Add button to update data
                    if st.button('Update Data') or not st.session_state['data_fetched']:
                        with st.spinner('Fetching and processing data...'):
                            # Fetch the list of pages
                            pages = get_user_pages(st.session_state['access_token'])
                            if not pages:
                                st.error("No Facebook Pages found. Ensure your account manages a page connected to Instagram.")
                                return

                            # Get Instagram Business Account ID
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

                                # Fetch and cache data
                                df = fetch_all_data(access_token, user_instagram_id)
                                if not df.empty:
                                    df = calculate_metrics(df)
                                    st.session_state['df'] = df
                                    st.session_state['data_fetched'] = True
                                    st.success('Data fetched successfully!')

                                    # Save the data to MongoDB to persist across sessions
                                    save_data_to_db(df, user_id)

                                    # Save the token to MongoDB to persist across sessions
                                    save_access_token_to_db(
                                        token=st.session_state['access_token'],
                                        expires_at=st.session_state['expires_at'],
                                        user_id=user_id
                                    )
                                else:
                                    st.warning("No data available to display.")

        # Display cached data if available
        if not st.session_state['df'].empty:
            df = st.session_state['df']
            
            # **Advanced Filtering Section**
            st.sidebar.header('Filter Options')
            
            # Date Range Filter
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()
            start_date, end_date = st.sidebar.date_input('Select Date Range', [min_date, max_date], min_value=min_date, max_value=max_date)
            
            # Media Type Filter
            media_types = df['media_type'].unique().tolist()
            selected_media_types = st.sidebar.multiselect('Select Media Types', media_types, default=media_types)
            
            # Hashtag Filter
            all_hashtags = df['hashtags'].dropna().str.split(',', expand=True).stack().unique().tolist()
            selected_hashtags = st.sidebar.multiselect('Select Hashtags', all_hashtags)
            
            # Apply Filters
            filtered_df = df[
                (df['timestamp'].dt.date >= start_date) &
                (df['timestamp'].dt.date <= end_date) &
                (df['media_type'].isin(selected_media_types))
            ]
            
            if selected_hashtags:
                # Filter rows where any of the selected hashtags are present
                filtered_df = filtered_df[filtered_df['hashtags'].str.contains('|'.join(selected_hashtags), na=False)]
            
            st.write(f"Displaying {len(filtered_df)} out of {len(df)} posts based on selected filters.")
            
            # **Visualization and Analysis Using filtered_df**
            # General Metrics
            st.header('Key Metrics Visualization')
            plot_reach_over_time(filtered_df)
            plot_engagement_over_time(filtered_df)
            plot_top_posts(filtered_df, metric='reach')
            plot_top_hashtags(filtered_df)
            plot_comprehensive_metrics(filtered_df)
            plot_follower_growth(filtered_df)
            
            # **AI Assistance Section**
            st.header('AI Insights for Selected Post')
            
            # Allow user to select a post
            post_ids = filtered_df['id'].tolist()
            selected_post_id = st.selectbox('Select a Post ID to Get AI Insights', options=post_ids)
            
            selected_post = filtered_df[filtered_df['id'] == selected_post_id].iloc[0]
            
            if st.button('Get AI Insight'):
                with st.spinner('Generating insights...'):
                    insight = ai_insight(selected_post, user_id)
                    st.subheader('AI Analysis:')
                    st.write(insight)
            
            # **Recommendations Section**
            st.header('Recommendations')
            recommendations = get_recommendations(filtered_df)
            for rec in recommendations:
                st.write(f"- {rec}")

        # Authentication if token is not present
        if 'access_token' not in st.session_state:
            st.header('Login with Facebook')
            authorization_url = get_facebook_auth_url()
            st.markdown(f'<a href="{authorization_url}">Login with Facebook</a>', unsafe_allow_html=True)

            # Handle Redirect after Facebook OAuth
            query_params = st.experimental_get_query_params()
            if 'code' in query_params:
                code = query_params['code'][0]
                token = get_access_token(code)
                if token:
                    # Exchange for long-lived token
                    long_lived_token, expires_in = exchange_for_long_lived_token(token['access_token'])
                    if long_lived_token:
                        st.session_state['access_token'] = long_lived_token
                        st.session_state['expires_at'] = (datetime.datetime.now() + datetime.timedelta(seconds=expires_in)).isoformat()

                        # Save the token to MongoDB
                        save_access_token_to_db(
                            token=long_lived_token,
                            expires_at=st.session_state['expires_at'],
                            user_id=user_id
                        )

                        # Remove code from the URL
                        st.experimental_set_query_params()  # Clears query params like 'code'
                        st.experimental_rerun()  # Refresh the app to clean the URL
                    else:
                        st.session_state['api_errors'].append('Failed to obtain a long-lived access token.')
                else:
                    st.session_state['api_errors'].append('Failed to retrieve access token.')

        # Optional: Display API Errors for Debugging (Hidden by Default)
        if st.session_state['api_errors']:
            with st.expander("View API Errors"):
                for error in st.session_state['api_errors']:
                    st.write(error)

if __name__ == '__main__':
    main()
