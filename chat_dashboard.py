import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
import openai
import asyncio
import socket
import sys
from datetime import datetime
from collections import defaultdict

# Load .env
load_dotenv()

def format_timestamp(timestamp_str):
    try:
        # Convert timestamp to datetime object
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        # Format as: Day, Month Date, Year, Time
        return dt.strftime("%A, %B %d, %Y, %I:%M %p")
    except:
        return timestamp_str  # Return original if parsing fails

def calculate_metrics(query_result):
    # Initialize counters
    total_messages = len(query_result.matches)
    user_rooms = defaultdict(set)  # user -> set of rooms
    room_messages = defaultdict(int)  # room -> total number of messages (both user and agent)
    
    # Process all messages
    for match in query_result.matches:
        if match.metadata:
            user_name = match.metadata.get("user_name")
            room_id = match.metadata.get("room_id")
            
            if user_name and room_id:
                user_rooms[user_name].add(room_id)
                room_messages[room_id] += 1
    
    # Calculate metrics
    total_users = len(user_rooms)
    total_rooms = len(room_messages)
    avg_interaction_per_room = sum(room_messages.values()) / total_rooms if total_rooms > 0 else 0
    
    return {
        "total_users": total_users,
        "total_rooms": total_rooms,
        "avg_interaction_per_room": round(avg_interaction_per_room, 2),
        "total_messages": total_messages
    }

try:
    # Environment setup
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX")

    # Init Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)

    # Title
    st.title("Whatsapp AI bot interaction")

    # Get all unique user names
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Fetching user list...")
        # Query to get all unique user names with a smaller top_k
        query_result = index.query(
            vector=[0.0]*1536,
            namespace="messages",
            top_k=100,  # Reduced from 1000 to 100
            include_metadata=True
        )
        
        progress_bar.progress(50)
        status_text.text("Processing user list...")
        
        # Calculate metrics
        metrics = calculate_metrics(query_result)
        
        # Display metrics in sidebar
        with st.sidebar:
            st.header("📊 Metrics")
            st.metric("Total Users", metrics["total_users"])
            st.metric("Total Rooms", metrics["total_rooms"])
            st.metric("Avg Interaction per Room", metrics["avg_interaction_per_room"])
            st.metric("Total Messages", metrics["total_messages"])
        
        # Extract unique user names
        user_names = set()
        for match in query_result.matches:
            if match.metadata and "user_name" in match.metadata:
                user_names.add(match.metadata["user_name"])
        
        user_names = sorted(list(user_names))
        progress_bar.progress(100)
        status_text.text("Ready!")
        
        # Display total number of users
        st.subheader(f"Total Users: {len(user_names)}")
        
        if not user_names:
            st.error("No users found in the database")
        else:
            # Display user selection dropdown
            selected_user = st.selectbox(
                "Select a user to view conversations:",
                options=user_names,
                format_func=lambda x: f"User: {x}"
            )

            if selected_user:
                with st.spinner("Fetching messages..."):
                    try:
                        # Run the query for selected user
                        query_result = index.query(
                            vector=[0.0]*1536,
                            namespace="messages",
                            filter={"user_name": {"$eq": selected_user}},
                            top_k=100,
                            include_metadata=True
                        )

                        # Get unique room_ids
                        room_ids = set()
                        for match in query_result.matches:
                            if match.metadata and "room_id" in match.metadata:
                                room_ids.add(match.metadata["room_id"])

                        if room_ids:
                            st.subheader(f"Found conversations in {len(room_ids)} rooms:")
                            
                            # Display room selection
                            selected_room = st.selectbox(
                                "Select a room to view messages:",
                                options=list(room_ids),
                                format_func=lambda x: f"Room: {x}"
                            )

                            if selected_room:
                                # Filter messages for selected room
                                room_messages = [
                                    m.metadata for m in query_result.matches
                                    if m.metadata and m.metadata.get("room_id") == selected_room
                                ]
                                
                                # Sort messages by timestamp
                                sorted_messages = sorted(room_messages, key=lambda x: x.get("timestamp", ""))
                                
                                # Create a container for the conversation
                                conversation_container = st.container()
                                
                                with conversation_container:
                                    st.write("---")  # Add a separator before the conversation
                                    
                                    # Display messages in chronological order
                                    for msg in sorted_messages:
                                        role = msg.get("sender_type", "user")
                                        content = msg.get("text", "")
                                        timestamp = msg.get("timestamp", "")
                                        formatted_time = format_timestamp(timestamp)
                                        
                                        # Create columns for message layout
                                        col1, col2 = st.columns([1, 4])
                                        
                                        with col1:
                                            st.write(formatted_time)
                                        
                                        with col2:
                                            if role == "user":
                                                st.write("**User:**")
                                                st.chat_message("user").write(content)
                                            else:
                                                st.write("**Agent:**")
                                                st.chat_message("assistant").write(content)
                                        
                                        st.write("---")  # Add a separator between messages
                                    
                        else:
                            st.info("No conversations found for this user.")
                    except Exception as e:
                        st.error(f"Error during query: {str(e)}")
    except Exception as e:
        st.error(f"Error fetching user list: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

except Exception as e:
    st.error(f"Error initializing Pinecone: {str(e)}")
