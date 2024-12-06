import streamlit as st
import requests
import pandas as pd
import datetime
from datetime import timezone
from requests_oauthlib import OAuth2Session
from pymongo import MongoClient
from dotenv import load_dotenv
import plotly.express as px
import time
import random
import os

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Instagram Analysis Dashboard", layout="wide")

# Constants
SCOPES = ['email', 'public_profile', 'pages_show_list', 'instagram_basic', 'instagram_manage_insights']
HF_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
VALID_METRICS = {
    'IMAGE': ['impressions', 'reach', 'saved', 'likes', 'comments'],
    'VIDEO': ['plays', 'video_views', 'reach', 'likes', 'comments', 'saved'],
    'REELS': ['plays', 'video_views', 'reach', 'likes', 'comments', 'saved']
}

# Load sensitive data
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')
MONGO_URI = os.getenv('MONGO_CONNECTION_STRING')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
MELIPAYAMAK_USERNAME = os.getenv('MELIPAYAMAK_USERNAME')
MELIPAYAMAK_PASSWORD = os.getenv('MELIPAYAMAK_PASSWORD')

# MongoDB connection setup
@st.cache_resource
def get_mongo_client() -> MongoClient:
    return MongoClient(MONGO_URI)

@st.cache_resource
def get_mongo_collection(collection_name: str):
    client = get_mongo_client()
    db = client['thefunbadger']
    return db[collection_name]

# Token management
def get_access_token(user_id: str):
    collection = get_mongo_collection('auth')
    data = collection.find_one({'user_id': user_id})
    if data and datetime.datetime.now() < datetime.datetime.fromisoformat(data['expires_at']):
        return data['token']
    return None

def save_access_token(user_id: str, token: str, expires_at: str):
    collection = get_mongo_collection('auth')
    collection.update_one({'user_id': user_id}, {'$set': {'token': token, 'expires_at': expires_at}}, upsert=True)

# OTP management
def generate_otp() -> int:
    return random.randint(100000, 999999)

def send_otp(phone_number: str, otp: int) -> bool:
    payload = {
        'username': MELIPAYAMAK_USERNAME,
        'password': MELIPAYAMAK_PASSWORD,
        'to': phone_number,
        'from': "50004001654470",
        'text': f'Your OTP is {otp}',
        'isflash': False
    }
    response = requests.post("https://rest.payamak-panel.com/api/SendSMS/SendSMS", data=payload)
    return response.status_code == 200

def validate_otp(phone_number: str, otp: int) -> bool:
    collection = get_mongo_collection('otps')
    otp_entry = collection.find_one({'phone_number': phone_number})
    if otp_entry and otp_entry['otp'] == otp and time.time() - otp_entry['created_at'] <= 300:
        return True
    return False

# Data fetching
@st.cache_data
def fetch_media_data(access_token: str, instagram_account_id: str) -> pd.DataFrame:
    media = []
    url = f"https://graph.facebook.com/v20.0/{instagram_account_id}/media?fields=id,caption,timestamp,media_type,media_url&access_token={access_token}"
    while url:
        response = requests.get(url).json()
        media.extend(response.get('data', []))
        url = response.get('paging', {}).get('next', None)
    return pd.DataFrame(media)

@st.cache_data
def fetch_insights(access_token: str, media_id: str, media_type: str):
    metrics = VALID_METRICS.get(media_type, [])
    if not metrics:
        return {}
    url = f"https://graph.facebook.com/v20.0/{media_id}/insights?metric={','.join(metrics)}&access_token={access_token}"
    response = requests.get(url).json()
    return {insight['name']: insight['values'][0]['value'] for insight in response.get('data', [])}

# Main application
def main():
    st.title("Instagram Analysis Dashboard")
    
    # User authentication
    user_id = st.session_state.get("user_id", "default_user")
    access_token = get_access_token(user_id)
    if not access_token:
        st.write("Please authenticate to access the dashboard.")
        return

    # Fetch data
    instagram_account_id = "123456789"  # Replace with dynamic retrieval logic
    media_df = fetch_media_data(access_token, instagram_account_id)
    if media_df.empty:
        st.warning("No media found for your account.")
        return
    
    # Display data
    st.dataframe(media_df)

    # Insights generation
    selected_media_id = st.selectbox("Select Media ID", media_df['id'].tolist())
    if st.button("Generate Insights"):
        insights = fetch_insights(access_token, selected_media_id, media_df.loc[media_df['id'] == selected_media_id, 'media_type'].values[0])
        st.json(insights)

if __name__ == "__main__":
    main()
