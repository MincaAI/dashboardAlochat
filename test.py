import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from datetime import datetime
from collections import defaultdict

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Test Window",
    page_icon="🧪",
    layout="wide"
)

st.title("Metrics of WhatsApp Bot")
st.write(f"Date: 29/04/2024")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX"))

    # Query to get all messages
    query_result = index.query(
        vector=[0.0]*1536,
        namespace="messages",
        top_k=1000,
        include_metadata=True
    )

    # Initialize counters
    tidak_count = 0
    total_messages = 0
    users_saying_tidak = set()
    users_only_tidak = set()
    user_message_counts = defaultdict(int)
    user_tidak_counts = defaultdict(int)
    user_messages = defaultdict(list)  # Store all messages per user

    ya_count = 0
    users_saying_ya = set()
    users_only_ya = set()
    user_ya_counts = defaultdict(int)

    # First pass: collect all messages and counts
    for match in query_result.matches:
        if match.metadata:
            text = match.metadata.get("text", "").strip()
            user_name = match.metadata.get("user_name", "")
            if not user_name:  # Skip if no user name
                continue
                
            total_messages += 1
            user_message_counts[user_name] += 1
            user_messages[user_name].append(text)
            
            # Count "tidak" responses
            if "tidak" in text.lower():
                tidak_count += 1
                users_saying_tidak.add(user_name)
                user_tidak_counts[user_name] += 1
            
            # Count "ya" responses
            if "ya" in text.lower() or "Ya" in text:
                ya_count += 1
                users_saying_ya.add(user_name)
                user_ya_counts[user_name] += 1

    # Second pass: verify users who only said "tidak" or "ya"
    for user, messages in user_messages.items():
        # Check for "tidak" only
        if user in users_saying_tidak:
            all_tidak = all("tidak" in msg.lower() for msg in messages)
            if all_tidak:
                users_only_tidak.add(user)
        
        # Check for "ya" only
        if user in users_saying_ya:
            all_ya = all(("ya" in msg.lower() or "Ya" in msg) for msg in messages)
            if all_ya:
                users_only_ya.add(user)

    # Display debug information
    st.write("Debug Information:")
    st.write(f"Total unique users: {len(user_message_counts)}")
    st.write(f"Users with messages: {list(user_message_counts.keys())}")
    st.write("---")

    # Display "tidak" metrics
    st.subheader("User Response Analysis - 'Tidak'")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", total_messages)
    
    with col2:
        st.metric("Messages with 'tidak'", tidak_count)
    
    with col3:
        st.metric("Unique Users saying 'tidak'", len(users_saying_tidak))
    
    with col4:
        st.metric("Users only saying 'tidak'", len(users_only_tidak))

    # Display "tidak" percentage
    if total_messages > 0:
        percentage = (tidak_count / total_messages) * 100
        st.metric("Percentage of messages with 'tidak'", f"{percentage:.2f}%")

    # Add a separator
    st.write("---")

    # Display "ya" metrics
    st.subheader("User Response Analysis - 'Ya'")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Total Messages", total_messages)
    
    with col6:
        st.metric("Messages with 'ya'", ya_count)
    
    with col7:
        st.metric("Unique Users saying 'ya'", len(users_saying_ya))
    
    with col8:
        st.metric("Users only saying 'ya'", len(users_only_ya))

    # Display "ya" percentage
    if total_messages > 0:
        percentage = (ya_count / total_messages) * 100
        st.metric("Percentage of messages with 'ya'", f"{percentage:.2f}%")

except Exception as e:
    st.error(f"Error connecting to database: {str(e)}") 