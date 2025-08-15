#!/usr/bin/env python3
"""
Emergency Intersection Priority System - Standalone UI
Modern web interface for the emergency vehicle priority system
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import time
from typing import Dict, List, Optional
import os

# Page configuration
st.set_page_config(
    page_title="🚨 Emergency Intersection Priority System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .priority-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .score-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4ecdc4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .stButton > button {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ee5a24, #44a08d);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚨 Emergency Intersection Priority System</h1>
    <p><em>AI that asks before it acts - Transparent, auditable emergency vehicle priority decisions</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API settings
    api_base_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="URL of the Emergency Intersection API"
    )
    
    # Jurisdiction selection
    jurisdiction = st.selectbox(
        "Traffic Jurisdiction",
        options=["us", "uk"],
        format_func=lambda x: "United States (Right-hand)" if x == "us" else "United Kingdom (Left-hand)",
        help="Select the traffic jurisdiction for priority rules"
    )
    
    # LLM settings
    use_llm = st.checkbox(
        "Enable LLM Cross-check",
        value=False,
        help="Enable optional LLM reasoning validation (requires API key)"
    )
    
    if use_llm:
        st.info("💡 LLM cross-check requires environment variables: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL")
    
    # Traffic side indicator
    traffic_side = "Right-hand" if jurisdiction == "us" else "Left-hand"
    st.info(f"🚦 Traffic Side: {traffic_side}")
    
    # Priority rules reminder
    st.markdown("### 🎯 Priority Rules")
    st.markdown("""
    **Base Priority:**
    1. 🚑 Ambulance (100)
    2. 🚒 Fire Engine (95)
    3. 🚓 Police (90)
    4. 🏛️ Presidential (70)
    5. 🚗 Civilian (10)
    
    **Bonuses:**
    - 🚨 Lights/Siren: +15
    - 📍 Proximity: Closer = higher
    - ⏰ Arrival: Sooner = higher
    """)

def check_api_health(api_url: str) -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def analyze_image(api_url: str, image_file, jurisdiction: str, use_llm: bool) -> Optional[Dict]:
    """Send image to API for analysis."""
    try:
        files = {"file": image_file}
        params = {
            "jurisdiction": jurisdiction,
            "use_llm": use_llm
        }
        
        response = requests.post(
            f"{api_url}/analyze",
            files=files,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def create_priority_visualization(data: Dict) -> go.Figure:
    """Create interactive visualization of priority analysis."""
    ordered_ids = data["ordered_ids"]
    scores = data["scores"]
    
    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Priority Scores", "Priority Order Flow"),
        vertical_spacing=0.1,
        specs=[[{"type": "bar"}], [{"type": "scatter"}]]
    )
    
    # Bar chart for scores
    vehicle_names = [vid.replace('_', ' ').title() for vid in ordered_ids]
    score_values = [scores[vid] for vid in ordered_ids]
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
    
    fig.add_trace(
        go.Bar(
            x=vehicle_names,
            y=score_values,
            marker_color=colors[:len(ordered_ids)],
            text=[f"{score:.1f}" for score in score_values],
            textposition='auto',
            name="Priority Scores"
        ),
        row=1, col=1
    )
    
    # Priority order flow
    for i, (vehicle_id, color) in enumerate(zip(ordered_ids, colors)):
        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[0.5],
                mode='markers+text',
                marker=dict(size=50, color=color, symbol='square'),
                text=f"#{i+1}<br>{vehicle_id.replace('_', ' ').title()}",
                textposition="middle center",
                showlegend=False,
                hoverinfo='text'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title="Emergency Vehicle Priority Analysis",
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(title_text="Vehicles", row=1, col=1)
    fig.update_yaxes(title_text="Priority Score", row=1, col=1)
    fig.update_xaxes(showticklabels=False, range=[-0.5, len(ordered_ids)-0.5], row=2, col=1)
    fig.update_yaxes(showticklabels=False, range=[0, 1], row=2, col=1)
    
    return fig

def display_priority_order(data: Dict):
    """Display priority order with styled cards."""
    ordered_ids = data["ordered_ids"]
    
    st.markdown("### 🎯 Priority Order")
    
    # Create columns for priority cards
    cols = st.columns(len(ordered_ids))
    
    for i, (col, vehicle_id) in enumerate(zip(cols, ordered_ids)):
        with col:
            st.markdown(f"""
            <div class="priority-card">
                <h3>#{i+1}</h3>
                <h4>{vehicle_id.replace('_', ' ').title()}</h4>
                <p>Score: {data['scores'][vehicle_id]:.1f}</p>
            </div>
            """, unsafe_allow_html=True)

def display_analysis_details(data: Dict):
    """Display detailed analysis information."""
    st.markdown("### 📊 Analysis Details")
    
    # Scores table
    scores_df = pd.DataFrame([
        {"Vehicle": vid.replace('_', ' ').title(), "Score": score}
        for vid, score in data["scores"].items()
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Priority Scores")
        st.dataframe(scores_df, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Confidence & Decision")
        
        confidence = data.get("confidence", 1.0)
        confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
        
        st.markdown(f"""
        <div class="score-card">
            <h4>Confidence Level</h4>
            <p class="{confidence_class}"><strong>{confidence:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        requires_confirmation = data.get("requires_human_confirmation", False)
        if requires_confirmation:
            st.warning("⚠️ Human confirmation required")
        else:
            st.success("✅ Automatic decision")
    
    # Reasoning and legal notes
    st.markdown("#### 💭 Reasoning")
    st.info(data.get("reasoning", "No reasoning provided"))
    
    st.markdown("#### ⚖️ Legal Notes")
    st.warning(data.get("legality_notes", "No legal notes provided"))

def export_results(data: Dict, format_type: str):
    """Export results in specified format."""
    if format_type == "JSON":
        return json.dumps(data, indent=2)
    elif format_type == "CSV":
        # Convert to CSV format
        rows = []
        for i, vehicle_id in enumerate(data["ordered_ids"]):
            rows.append({
                "Priority_Rank": i + 1,
                "Vehicle_ID": vehicle_id,
                "Vehicle_Type": vehicle_id.split('_')[0].title(),
                "Score": data["scores"][vehicle_id],
                "Confidence": data.get("confidence", 1.0),
                "Requires_Human_Confirmation": data.get("requires_human_confirmation", False)
            })
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

# Main application
def main():
    # Check API health
    if not check_api_health(api_base_url):
        st.error(f"❌ Cannot connect to API at {api_base_url}")
        st.info("💡 Make sure the API server is running: `python emergency_intersection_standalone.py --server`")
        return
    
    st.success(f"✅ Connected to API at {api_base_url}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results", "ℹ️ About"])
    
    with tab1:
        st.markdown("### 📤 Upload Intersection Image")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an intersection image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of an intersection with vehicles"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Analysis button
            if st.button("🚨 Analyze Priority", type="primary"):
                with st.spinner("Analyzing intersection..."):
                    # Analyze image
                    result = analyze_image(api_base_url, uploaded_file, jurisdiction, use_llm)
                    
                    if result:
                        # Store result in session state
                        st.session_state.analysis_result = result
                        st.session_state.uploaded_image = uploaded_file
                        
                        st.success("✅ Analysis complete!")
                        st.balloons()
                        
                        # Auto-switch to results tab
                        st.switch_page("Results")
    
    with tab2:
        if "analysis_result" in st.session_state:
            data = st.session_state.analysis_result
            
            # Display priority order
            display_priority_order(data)
            
            # Create visualization
            st.markdown("### 📈 Priority Visualization")
            fig = create_priority_visualization(data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display analysis details
            display_analysis_details(data)
            
            # Export options
            st.markdown("### 📤 Export Results")
            export_format = st.selectbox("Export Format", ["JSON", "CSV"])
            
            if st.button("📥 Download Results"):
                export_data = export_results(data, export_format)
                file_extension = "json" if export_format == "JSON" else "csv"
                mime_type = "application/json" if export_format == "JSON" else "text/csv"
                
                st.download_button(
                    label=f"📥 Download {export_format}",
                    data=export_data,
                    file_name=f"emergency_priority_analysis.{file_extension}",
                    mime=mime_type
                )
        else:
            st.info("📤 Upload an image and run analysis to see results here")
    
    with tab3:
        st.markdown("""
        ### 🚨 About Emergency Intersection Priority System
        
        This system provides **transparent, auditable priority decisions** for emergency vehicles at intersections using AI-powered analysis.
        
        #### 🎯 Key Features:
        - **Computer Vision**: Detects vehicles and signals
        - **Deterministic Rules**: Transparent priority scoring
        - **LLM Validation**: Optional reasoning cross-check
        - **Jurisdiction Aware**: Supports US/UK traffic rules
        - **Real-time Analysis**: Fast decision making
        
        #### 🔧 How It Works:
        1. **Image Upload** → Upload intersection image
        2. **Vision Analysis** → Detect vehicles and signals
        3. **Priority Scoring** → Calculate scores for each vehicle
        4. **Tie Breaking** → Apply rules for equal scores
        5. **LLM Cross-Check** → Optional reasoning validation
        6. **Final Output** → Ordered priority list with explanations
        
        #### 🎯 Priority Rules:
        - **Base Priority**: Ambulance (100) > Fire Engine (95) > Police (90) > Presidential (70) > Civilian (10)
        - **Lights Bonus**: +15 points for active lights/sirens
        - **Proximity Bonus**: Closer vehicles get small boost
        - **Arrival Bonus**: Earlier arrival gets small boost
        
        #### 🛡️ Safety & Governance:
        - **Ask-before-act**: System asks for missing facts
        - **Deterministic Fallback**: Works even if LLM is unavailable
        - **Human-in-the-loop**: Flags low-confidence decisions
        - **Transparent Scoring**: All decisions are explainable
        
        #### 📊 Output Format:
        ```json
        {
          "ordered_ids": ["ambulance_1", "fire_engine_1", "police_1"],
          "scores": {"ambulance_1": 118.8, "fire_engine_1": 113.1, "police_1": 92.2},
          "reasoning": "Scores blend emergency level, right-of-way signals, proximity and arrival.",
          "legality_notes": "General rule: yield to emergency vehicles with lights/sirens active.",
          "confidence": 1.0,
          "requires_human_confirmation": false
        }
        ```
        
        ---
        
        **🎯 The system is designed to be transparent, auditable, and safe - "AI that asks before it acts!"**
        """)

if __name__ == "__main__":
    main()
