import streamlit as st
import requests
import pandas as pd
import datetime
from requests_oauthlib import OAuth2Session
import warnings
import os
from dotenv import load_dotenv
import plotly.express as px
from pymongo import MongoClient
import time

# Load environment variables from .env file (for local development)
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
            if password == "my_secure_password":  # Replace with your actual password
                st.session_state["authenticated"] = True
            else:
                st.error("Incorrect password")
        return False
    return True

# Password Check
if not check_password():
    st.stop()

# Load sensitive information from environment variables or Streamlit secrets
if st.secrets:
    CLIENT_ID = st.secrets["CLIENT_ID"]
    CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
    REDIRECT_URI = st.secrets["REDIRECT_URI"]
    MONGO_CONNECTION_STRING = st.secrets["MONGO_CONNECTION_STRING"]
    HF_API_TOKEN = st.secrets["HF_API_TOKEN"]  # Hugging Face API Token
else:
    CLIENT_ID = os.getenv('CLIENT_ID')  # Facebook App Client ID
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')  # Facebook App Client Secret
    REDIRECT_URI = os.getenv('REDIRECT_URI')  # Replace with your redirect URI
    MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')  # MongoDB Atlas connection string
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')  # Hugging Face API Token

# Ensure all required environment variables are set
required_env_vars = ['CLIENT_ID', 'CLIENT_SECRET', 'REDIRECT_URI', 'MONGO_CONNECTION_STRING', 'HF_API_TOKEN']
missing_vars = [var for var in required_env_vars if (var not in st.secrets and os.getenv(var) is None)]
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}. Please set them before running the app.")
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
    'IMAGE': ['impressions', 'reach', 'saved', 'likes', 'comments'],
    'VIDEO': ['plays', 'video_views', 'saved', 'reach', 'likes', 'comments'],
    'REELS': ['plays', 'video_views', 'saved', 'reach', 'likes', 'comments']
}

# MongoDB Helper Functions
def get_mongo_client():
    return MongoClient(MONGO_CONNECTION_STRING)

def get_mongo_collection(collection_name):
    client = get_mongo_client()
    db = client['thefunbadger']  # Replace with your database name
    collection = db[collection_name]  # Dynamic collection name
    return collection

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
    records = data_df.to_dict(orient='records')
    collection.delete_many({'user_id': user_id})
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
        for record in records:
            record.pop('user_id', None)
            record.pop('_id', None)
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        st.error(f"Error fetching data from MongoDB: {e}")
        return pd.DataFrame()

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
    fields = 'id,caption,timestamp,media_type,media_url,permalink'
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

def fetch_all_data(access_token, instagram_account_id):
    media_items = get_media(access_token, instagram_account_id)
    if not media_items:
        st.warning("No media items retrieved.")
        return pd.DataFrame()

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
            'video_views': None,
            'hashtags': extract_hashtags(item.get('caption', '')),
        }
        for insight in insights:
            metric_name = insight.get('name')
            if metric_name in data:
                data[metric_name] = insight['values'][0]['value']
        all_data.append(data)
    df = pd.DataFrame(all_data)
    return df

def extract_hashtags(caption):
    hashtags = [tag.strip('#') for tag in caption.split() if tag.startswith('#')]
    return ','.join(hashtags)

# Data Analysis Functions
def calculate_metrics(df):
    numeric_columns = ['impressions', 'reach', 'saved', 'likes', 'comments', 'plays', 'video_views']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_recommendations(df):
    recommendations = []
    avg_reach = df['reach'].mean()
    if pd.notnull(avg_reach) and avg_reach < 1000:
        recommendations.append('Your average reach is below 1,000. Consider optimizing your posting times and content.')
    elif pd.notnull(avg_reach):
        recommendations.append('Your average reach is healthy. Keep up the good work!')

    if 'likes' in df.columns and 'comments' in df.columns:
        df['engagement_rate'] = (df['likes'] + df['comments']) / df['reach'].replace(0, 1) * 100
        avg_engagement = df['engagement_rate'].mean()
        if pd.notnull(avg_engagement) and avg_engagement < 1:
            recommendations.append('Your engagement rate is below 1%. Engage more with your audience.')
        elif pd.notnull(avg_engagement):
            recommendations.append('Your engagement rate is healthy. Keep creating engaging content!')

    hashtags_series = df['hashtags'].str.split(',', expand=True).stack()
    top_hashtags = hashtags_series.value_counts()
    if not top_hashtags.empty:
        top_hashtag = top_hashtags.idxmax()
        recommendations.append(f'Try using the hashtag #{top_hashtag} more often.')
    
    return recommendations

# Visualization Functions with Plotly
def plot_reach_over_time(df):
    if 'reach' in df.columns and not df['reach'].isnull().all():
        fig = px.line(df, x='timestamp', y='reach', title='Reach Over Time', labels={'timestamp': 'Date', 'reach': 'Reach'}, template='plotly_white')
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Reach data is missing or incomplete.")

def plot_engagement_over_time(df):
    if 'engagement_rate' in df.columns and not df['engagement_rate'].isnull().all():
        fig = px.line(df, x='timestamp', y='engagement_rate', title='Engagement Rate Over Time', labels={'timestamp': 'Date', 'engagement_rate': 'Engagement Rate (%)'}, template='plotly_white')
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Engagement rate data is missing or incomplete.")

def plot_top_posts(df, metric='reach', top_n=5):
    if metric not in df.columns or df[metric].isnull().all():
        st.warning(f"{metric.capitalize()} data is missing or incomplete.")
        return
    top_posts = df.sort_values(by=metric, ascending=False).head(top_n)
    fig = px.bar(top_posts, x='id', y=metric, title=f'Top {top_n} Posts by {metric.capitalize()}', labels={'id': 'Post ID', metric: metric.capitalize()}, template='plotly_white')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def plot_top_hashtags(df):
    hashtags_series = df['hashtags'].str.split(',', expand=True).stack()
    hashtags_counts = hashtags_series.value_counts().reset_index()
    hashtags_counts.columns = ['hashtag', 'count']
    if hashtags_counts.empty:
        st.warning("No hashtags found.")
        return
    fig = px.bar(hashtags_counts.head(10), x='hashtag', y='count', title='Top Hashtags', labels={'hashtag': 'Hashtag', 'count': 'Count'}, template='plotly_white')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def plot_follower_growth(df):
    if 'followers' in df.columns and not df['followers'].isnull().all():
        fig = px.line(df, x='timestamp', y='followers', title='Follower Growth Over Time', labels={'timestamp': 'Date', 'followers': 'Followers'}, template='plotly_white')
        fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Follower data is missing or incomplete.")

# Export Data to CSV
def export_data(df):
    st.download_button(
        label="Export Data as CSV",
        data=df.to_csv().encode('utf-8'),
        file_name='instagram_data.csv',
        mime='text/csv',
    )

# Main Application Function
def main():
    st.title('Ultimate Instagram Analysis Dashboard')

    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

    if 'data_fetched' not in st.session_state:
        st.session_state['data_fetched'] = False
        st.session_state['df'] = pd.DataFrame()
    if 'api_errors' not in st.session_state:
        st.session_state['api_errors'] = []
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = 'default_user'

    user_id = st.session_state['user_id']

    if 'access_token' not in st.session_state:
        token_data = get_access_token_from_db(user_id)
        if token_data and token_data[0]:
            st.session_state['access_token'] = token_data[0]
            st.session_state['expires_at'] = token_data[1]
            if datetime.datetime.now() > datetime.datetime.fromisoformat(st.session_state['expires_at']):
                st.error('Access token has expired. Please log in again.')
                st.session_state.clear()
                st.experimental_rerun()

    if 'access_token' in st.session_state and st.session_state['access_token']:
        if 'expires_at' in st.session_state:
            if datetime.datetime.now() > datetime.datetime.fromisoformat(st.session_state['expires_at']):
                st.error('Access token has expired. Please log in again.')
                st.session_state.clear()
                st.experimental_rerun()
            else:
                st.success('Successfully Authenticated!')

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

                if st.button('Update Data') or not st.session_state['data_fetched']:
                    with st.spinner('Fetching and processing data...'):
                        pages = get_user_pages(st.session_state['access_token'])
                        if not pages:
                            st.error("No Facebook Pages found.")
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
                                st.warning("No data available.")

    if not st.session_state['df'].empty:
        df = st.session_state['df']
        st.sidebar.header('Filter Options')
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        start_date, end_date = st.sidebar.date_input('Select Date Range', [min_date, max_date], min_value=min_date, max_value=max_date)
        media_types = df['media_type'].unique().tolist()
        selected_media_types = st.sidebar.multiselect('Select Media Types', media_types, default=media_types)
        filtered_df = df[
            (df['timestamp'].dt.date >= start_date) &
            (df['timestamp'].dt.date <= end_date) &
            (df['media_type'].isin(selected_media_types))
        ]
        st.write(f"Displaying {len(filtered_df)} out of {len(df)} posts based on selected filters.")
        st.header('Key Metrics Visualization')
        plot_reach_over_time(filtered_df)
        plot_engagement_over_time(filtered_df)
        plot_top_posts(filtered_df, metric='reach')
        plot_top_hashtags(filtered_df)
        plot_follower_growth(filtered_df)
        st.header('Recommendations')
        recommendations = get_recommendations(filtered_df)
        for rec in recommendations:
            st.write(f"- {rec}")
        st.header('Export Data')
        export_data(filtered_df)

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

    if st.session_state['api_errors']:
        with st.expander("View API Errors"):
            for error in st.session_state['api_errors']:
                st.write(error)

if __name__ == '__main__':
    main()
