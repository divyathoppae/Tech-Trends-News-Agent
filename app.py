import streamlit as st
import json
import os
from typing import List, Dict, Any
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from react_agent import ReActAgent, AgentConfig

# Page config
st.set_page_config(
    page_title="Tech Trends News Agent",
    layout="centered"
)

# Custom CSS for minimalist styling
st.markdown("""
    <style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #2c3e50;
    }
    .user-message {
        background-color: #f8f9fa;
        border-left: 3px solid #007bff;
        color: #2c3e50;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left: 3px solid #28a745;
        color: #2c3e50;
    }
    .chat-message strong {
        color: #1a1a1a;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def load_previous_chats() -> List[Dict[str, Any]]:
    """Load previous chats from JSON files."""
    chats_dir = os.path.join(os.path.dirname(__file__), 'data', 'agent_runs')
    if not os.path.exists(chats_dir):
        return []
    
    chats = []
    for filename in sorted(os.listdir(chats_dir), reverse=True):
        if filename.endswith('.json'):
            filepath = os.path.join(chats_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    chats.append(json.load(f))
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
    
    return chats


# Main UI
st.markdown('<div class="main-header">Tech Trends News Agent</div>', unsafe_allow_html=True)

# Sidebar for previous chats
with st.sidebar:
    st.header("Previous Chats")
    previous_chats = load_previous_chats()
    
    if previous_chats:
        chat_options = [f"{chat['query'][:50]}..." if len(chat['query']) > 50 else chat['query'] 
                       for chat in previous_chats]
        selected_chat_idx = st.selectbox(
            "Select a previous chat:",
            range(len(previous_chats)),
            format_func=lambda x: chat_options[x],
            label_visibility="collapsed"
        )
        
        if st.button("Load Chat", use_container_width=True):
            selected_chat = previous_chats[selected_chat_idx]
            st.session_state.chat_history = [
                {'role': 'user', 'content': selected_chat['query']},
                {'role': 'assistant', 'content': selected_chat['result'].get('answer', 'No answer available.')}
            ]
            st.rerun()
    else:
        st.info("No previous chats found.")

# Display chat history
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# Chat input
query = st.text_input(
    "Ask a question about technology trends:",
    placeholder="Type your question here...",
    key="query_input",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 5])
with col1:
    send_button = st.button("Send", type="primary", use_container_width=True)

if send_button and query:
    # Add user message to chat
    st.session_state.chat_history.append({'role': 'user', 'content': query})
    
    with st.spinner("Thinking..."):
        try:
            # Run agent
            config = AgentConfig(max_steps=6, allow_tools=("search", "finish"), verbose=False)
            agent = ReActAgent(config=config)
            result = agent.run(query)
            
            # Get answer
            answer = result.get('answer', 'No answer provided.')
            
            # Add assistant response to chat
            st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.chat_history.append({'role': 'assistant', 'content': f"Sorry, I encountered an error: {str(e)}"})
            st.rerun()

if not st.session_state.chat_history:
    st.info("Start a conversation by typing a question above.")

