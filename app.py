import streamlit as st
import json
import os
import re
from collections import Counter
from typing import List, Dict, Any
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from react_agent import ReActAgent, AgentConfig

# Page config
st.set_page_config(
    page_title="Tech Trends News Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .step-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []


def extract_topics_from_text(text: str, min_length: int = 3) -> List[str]:
    """Extract meaningful words/topics from text."""
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'now', 'then', 'here', 'there', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now',
        'one', 'two', 'three', 'first', 'second', 'third', 'new', 'old',
        'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know',
        'knew', 'think', 'thought', 'say', 'said', 'tell', 'told', 'make',
        'made', 'take', 'took', 'give', 'gave', 'use', 'used', 'find',
        'found', 'work', 'worked', 'call', 'called', 'try', 'tried'
    }
    
    # Tokenize: extract words (alphanumeric sequences, minimum 3 chars)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out stop words and short words
    topics = [w for w in words if w not in stop_words and len(w) >= min_length]
    
    return topics


def extract_topics_from_results(result: Dict[str, Any]) -> Counter:
    """Extract topics from agent results (answer + search results)."""
    all_topics = []
    
    # Extract from final answer
    answer = result.get('answer', '')
    if answer:
        all_topics.extend(extract_topics_from_text(answer))
    
    # Extract from trajectory search results
    trajectory = result.get('trajectory', [])
    for step in trajectory:
        obs = step.get('observation', '')
        if obs and obs != 'done':
            try:
                obs_data = json.loads(obs)
                if 'results' in obs_data:
                    for res in obs_data['results']:
                        snippet = res.get('snippet', '')
                        if snippet:
                            all_topics.extend(extract_topics_from_text(snippet))
            except (json.JSONDecodeError, TypeError):
                pass
    
    # Count topics
    topic_counts = Counter(all_topics)
    
    # Filter: keep topics that appear 2+ times OR are 5+ characters long
    # This helps capture both frequent short terms and meaningful longer terms
    filtered_counts = {k: v for k, v in topic_counts.items() 
                      if v >= 2 or len(k) >= 5}
    
    return Counter(filtered_counts)


def load_previous_runs() -> List[Dict[str, Any]]:
    """Load previous agent runs from JSON files."""
    runs_dir = os.path.join(os.path.dirname(__file__), 'data', 'agent_runs')
    if not os.path.exists(runs_dir):
        return []
    
    runs = []
    for filename in sorted(os.listdir(runs_dir), reverse=True):
        if filename.endswith('.json'):
            filepath = os.path.join(runs_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    runs.append(json.load(f))
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
    
    return runs


# Main UI
st.markdown('<div class="main-header">🤖 Tech Trends News Agent</div>', unsafe_allow_html=True)

# Sidebar for previous runs
with st.sidebar:
    st.header("📚 Previous Runs")
    previous_runs = load_previous_runs()
    
    if previous_runs:
        run_options = [f"{run['query'][:50]}..." if len(run['query']) > 50 else run['query'] 
                      for run in previous_runs]
        selected_run_idx = st.selectbox(
            "Select a previous run:",
            range(len(previous_runs)),
            format_func=lambda x: run_options[x]
        )
        
        if st.button("Load Selected Run"):
            st.session_state.results = previous_runs[selected_run_idx]
            st.rerun()
    else:
        st.info("No previous runs found.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask a Question")
    
    query = st.text_input(
        "Enter your question about technology trends:",
        placeholder="e.g., What are the latest technology trends?",
        key="query_input"
    )
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_query = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if run_query and query:
        with st.spinner("🤔 Agent is thinking..."):
            try:
                # Run agent with non-verbose config
                config = AgentConfig(max_steps=6, allow_tools=("search", "finish"), verbose=False)
                agent = ReActAgent(config=config)
                result = agent.run(query)
                
                # Store results
                st.session_state.results = {
                    'query': query,
                    'result': result
                }
                
                st.success("✅ Query completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error running agent: {str(e)}")
                st.exception(e)

# Display results
if st.session_state.results:
    result_data = st.session_state.results.get('result', {})
    query_text = st.session_state.results.get('query', 'Unknown Query')
    
    st.markdown("---")
    
    # Display query
    st.subheader(f"📝 Query: {query_text}")
    
    # Display answer
    answer = result_data.get('answer', 'No answer provided.')
    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.markdown("### 💡 Answer")
    st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Extract and display topics
    topic_counts = extract_topics_from_results(result_data)
    
    if topic_counts:
        st.markdown("---")
        st.subheader("📊 Most Common Topics")
        
        # Get top N topics
        top_n = st.slider("Number of topics to display:", 5, 30, 15, key="top_n_slider")
        top_topics = dict(topic_counts.most_common(top_n))
        
        if top_topics:
            # Create bar chart
            import pandas as pd
            
            df = pd.DataFrame({
                'Topic': list(top_topics.keys()),
                'Frequency': list(top_topics.values())
            })
            
            st.bar_chart(df.set_index('Topic'))
            
            # Display as table
            with st.expander("📋 View Topic Details"):
                st.dataframe(df.sort_values('Frequency', ascending=False), use_container_width=True)
    
    # Display trajectory (optional, in expander)
    trajectory = result_data.get('trajectory', [])
    if trajectory:
        with st.expander("🔍 View Agent Steps"):
            for i, step in enumerate(trajectory, 1):
                st.markdown(f"### Step {i}")
                st.markdown(f"**Thought:** {step.get('thought', 'N/A')}")
                st.markdown(f"**Action:** `{step.get('action', 'N/A')}`")
                
                obs = step.get('observation', '')
                if obs and obs != 'done':
                    try:
                        obs_data = json.loads(obs)
                        if 'results' in obs_data:
                            st.markdown(f"**Search Results:** {len(obs_data['results'])} found")
                            with st.expander(f"View {len(obs_data['results'])} results"):
                                for j, res in enumerate(obs_data['results'], 1):
                                    st.markdown(f"**Result {j}:**")
                                    st.markdown(f"- ID: {res.get('id', 'N/A')}")
                                    st.markdown(f"- Score: {res.get('score', 0):.4f}")
                                    st.markdown(f"- Snippet: {res.get('snippet', 'N/A')[:200]}...")
                        else:
                            st.markdown(f"**Observation:** {obs[:500]}...")
                    except (json.JSONDecodeError, TypeError):
                        st.markdown(f"**Observation:** {obs[:500]}...")
                else:
                    st.markdown(f"**Observation:** {obs}")
                
                st.markdown("---")

else:
    st.info("👆 Enter a question above and click 'Search' to get started, or load a previous run from the sidebar.")

