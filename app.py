import streamlit as st
import requests
import pandas as pd
import datetime
import os
from requests_oauthlib import OAuth2Session
from pymongo import MongoClient
import plotly.express as px
from dotenv import load_dotenv
import warnings

# Load environment variables
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load sensitive information from environment variables
CLIENT_ID = st.secrets["CLIENT_ID"] if st.secrets else os.getenv('CLIENT_ID')
CLIENT_SECRET = st.secrets["CLIENT_SECRET"] if st.secrets else os.getenv('CLIENT_SECRET')
REDIRECT_URI = st.secrets["REDIRECT_URI"] if st.secrets else os.getenv('REDIRECT_URI')
MONGO_CONNECTION_STRING = st.secrets["MONGO_CONNECTION_STRING"] if st.secrets else os.getenv('MONGO_CONNECTION_STRING')
HF_API_TOKEN = st.secrets["HF_API_TOKEN"] if st.secrets else os.getenv('HF_API_TOKEN')

# Ensure all required environment variables are set
required_env_vars = ['CLIENT_ID', 'CLIENT_SECRET', 'REDIRECT_URI', 'MONGO_CONNECTION_STRING', 'HF_API_TOKEN']
missing_vars = [var for var in required_env_vars if not (st.secrets.get(var) or os.getenv(var))]
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}. Please set them before running the app.")
    st.stop()

# MongoDB Helper Functions
def get_mongo_client():
    try:
        return MongoClient(MONGO_CONNECTION_STRING)
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        st.stop()

def get_mongo_collection(collection_name):
    try:
        client = get_mongo_client()
        db = client['thefunbadger']
        collection = db[collection_name]
        return collection
    except Exception as e:
        st.error(f"Error accessing MongoDB collection: {e}")
        st.stop()

def save_access_token_to_db(token, expires_at, user_id):
    try:
        collection = get_mongo_collection('auth')
        collection.update_one(
            {'user_id': user_id},
            {'$set': {'token': token, 'expires_at': expires_at}},
            upsert=True
        )
    except Exception as e:
        st.error(f"Error saving access token to MongoDB: {e}")

def get_access_token_from_db(user_id):
    try:
        collection = get_mongo_collection('auth')
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

# Data Export Functionality (CSV & Excel)
def export_data(df):
    try:
        csv = df.to_csv(index=False)
        excel = df.to_excel(index=False)
    
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="instagram_data.csv",
            mime="text/csv"
        )
    
        st.download_button(
            label="Download Excel",
            data=excel,
            file_name="instagram_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error exporting data: {e}")

# Function to fetch Instagram data with retries, pagination, and error handling
def fetch_instagram_data(access_token, instagram_account_id, retries=3):
    try:
        media_items = get_media(access_token, instagram_account_id)
        all_data = []

        if not media_items:
            st.warning("No media items retrieved. Ensure your Instagram account has posts and the necessary permissions.")
            return pd.DataFrame()

        # Pagination logic to fetch all media items
        while media_items:
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
                    'hashtags': extract_hashtags(item.get('caption', '')),
                    'followers': None
                }

                for insight in insights:
                    metric_name = insight.get('name')
                    if metric_name in data:
                        data[metric_name] = insight['values'][0]['value']

                all_data.append(data)

            # Check if there's more data to fetch
            next_url = media_items.get('paging', {}).get('next')
            if next_url:
                media_items = requests.get(next_url).json().get('data', [])
            else:
                break

        df = pd.DataFrame(all_data)
        return df

    except Exception as e:
        st.error(f"Error fetching Instagram data: {e}")
        return pd.DataFrame()

# Facebook OAuth2 Login Function
def login_with_facebook():
    try:
        oauth = OAuth2Session(client_id=CLIENT_ID, redirect_uri=REDIRECT_URI, scope=SCOPES)
        authorization_url, state = oauth.authorization_url('https://www.facebook.com/dialog/oauth')
        st.session_state['oauth_state'] = state
        st.markdown(f'<a href="{authorization_url}">Login with Facebook</a>', unsafe_allow_html=True)

        # Handle Redirect after Facebook OAuth
        query_params = st.experimental_get_query_params()
        if 'code' in query_params:
            code = query_params['code'][0]
            token = get_access_token(code)
            if token:
                long_lived_token, expires_in = exchange_for_long_lived_token(token['access_token'])
                if long_lived_token:
                    st.session_state['access_token'] = long_lived_token
                    st.session_state['expires_at'] = (datetime.datetime.now() + datetime.timedelta(seconds=expires_in)).isoformat()

                    # Save the token to MongoDB
                    save_access_token_to_db(
                        token=long_lived_token,
                        expires_at=st.session_state['expires_at'],
                        user_id=st.session_state['user_id']
                    )

                    # Remove code from the URL
                    st.experimental_set_query_params()  # Clears query params like 'code'
                    st.experimental_rerun()  # Refresh the app to clean the URL
                else:
                    st.session_state['api_errors'].append('Failed to obtain a long-lived access token.')
            else:
                st.session_state['api_errors'].append('Failed to retrieve access token.')
    except Exception as e:
        st.error(f"Error during Facebook login: {e}")

# Token exchange for long-lived token
def exchange_for_long_lived_token(short_lived_token):
    try:
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
    except Exception as e:
        st.error(f"Error exchanging short-lived token: {e}")
        return None, None

# Main Function
def main():
    try:
        if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
            login_with_facebook()
        else:
            st.title('Ultimate Instagram Analysis Dashboard')

            if st.button("Clear Cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.experimental_rerun()

            if 'data_fetched' not in st.session_state:
                st.session_state['data_fetched'] = False
                st.session_state['df'] = pd.DataFrame()

            user_id = st.session_state['user_id']

            # Token check and retrieval
            if 'access_token' not in st.session_state:
                token_data = get_access_token_from_db(user_id)

                if token_data and token_data[0]:
                    st.session_state['access_token'] = token_data[0]
                    st.session_state['expires_at'] = token_data[1]

                    if datetime.datetime.now() > datetime.datetime.fromisoformat(st.session_state['expires_at']):
                        st.error('Access token has expired. Please log in again.')
                        st.session_state.clear()
                        st.experimental_rerun()

            if 'access_token' in st.session_state:
                access_token = st.session_state['access_token']
                if 'data_fetched' not in st.session_state or not st.session_state['data_fetched']:
                    st.session_state['df'] = fetch_instagram_data(access_token, st.session_state['instagram_account_id'])
                    st.session_state['data_fetched'] = True

                df = st.session_state['df']
                if not df.empty:
                    export_data(df)  # Allow export of the data
                    plot_reach_over_time(df)
                    plot_engagement_over_time(df)
                    plot_top_posts(df)
    except Exception as e:
        st.error(f"Error in main function: {e}")

# Run the app
if __name__ == '__main__':
    main()
