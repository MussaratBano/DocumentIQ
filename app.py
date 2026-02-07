import streamlit as st
import openai
from openai import OpenAI
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import json
import tempfile
from datetime import datetime
import io

# For PDF processing
from pypdf import PdfReader

# For DOCX processing
from docx import Document

# For chart export
import kaleido
import base64

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="DocumentIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= CUSTOM CSS =========================
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .user-message {
        background-color: transparent;
        border-left: none;
    }
    
    .assistant-message {
        background-color: transparent;
        border-left: none;
    }
    
    /* Custom buttons */
    .stButton > button {
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #fafbfc;
    }
    
    /* Title styling */
    h1 {
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    h2 {
        color: #16213e;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ========================= SESSION STATE INITIALIZATION =========================
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if "document_content" not in st.session_state:
    st.session_state.document_content = ""

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "current_charts" not in st.session_state:
    st.session_state.current_charts = []

if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False

if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = {}  # {filename: {content, dataframe, type}}

if "active_file" not in st.session_state:
    st.session_state.active_file = None

if "api_provider" not in st.session_state:
    st.session_state.api_provider = "Gemini"  # Default to free Gemini

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

if "last_processed_message" not in st.session_state:
    st.session_state.last_processed_message = None

# ========================= HELPER FUNCTIONS =========================

@st.cache_data
def extract_text_from_txt(file_content):
    """Extract text from TXT file"""
    try:
        return file_content.decode('utf-8')
    except:
        return file_content.decode('latin-1')

@st.cache_data
def extract_text_from_pdf(file_content):
    """Extract text from PDF file"""
    pdf_reader = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

@st.cache_data
def extract_text_from_docx(file_content):
    """Extract text from DOCX file"""
    doc = Document(io.BytesIO(file_content))
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

@st.cache_data
def extract_dataframe_from_csv(file_content):
    """Extract DataFrame from CSV file"""
    return pd.read_csv(io.BytesIO(file_content))

def process_uploaded_file(uploaded_file):
    """Process uploaded file based on file type"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    file_content = uploaded_file.read()
    
    try:
        if file_extension == 'txt':
            return extract_text_from_txt(file_content), None
        elif file_extension == 'pdf':
            return extract_text_from_pdf(file_content), None
        elif file_extension == 'docx':
            return extract_text_from_docx(file_content), None
        elif file_extension == 'csv':
            df = extract_dataframe_from_csv(file_content)
            return df.to_string(), df
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None, None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None

def get_file_type_icon(filename):
    """Get icon for file type"""
    ext = filename.split('.')[-1].lower()
    icons = {'txt': '📄', 'pdf': '📕', 'docx': '📗', 'csv': '📊'}
    return icons.get(ext, '📎')

def get_chat_response(api_provider, api_key, user_message, system_prompt=""):
    """Get streaming response from OpenAI or Gemini"""
    if not system_prompt:
        # Get active file content
        active_content = ""
        if st.session_state.active_file and st.session_state.active_file in st.session_state.loaded_files:
            active_content = st.session_state.loaded_files[st.session_state.active_file]['content'][:8000]
        
        files_info = f"\nLoaded files: {', '.join(st.session_state.loaded_files.keys())}"
        
        system_prompt = f"""You are a helpful assistant that analyzes and answers questions about the provided documents.
        
Document content:
{active_content}
{files_info}

Please answer the user's question based on the document content provided above. If the question cannot be answered from the document, politely indicate that."""
    
    if api_provider == "OpenAI":
        return get_openai_response(api_key, user_message, system_prompt)
    else:
        return get_gemini_response(api_key, user_message, system_prompt)

def get_openai_response(api_key, user_message, system_prompt):
    """Stream response from OpenAI"""
    client = OpenAI(api_key=api_key)
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add conversation history
    for msg in st.session_state.conversation_history[-10:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": user_message})
    
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=2048
    )

def get_gemini_response(api_key, user_message, system_prompt):
    """Stream response from Google Gemini"""
    genai.configure(api_key=api_key)
    
    # Try different model names for compatibility
    model_names = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    model = None
    
    for model_name in model_names:
        try:
            test_model = genai.GenerativeModel(model_name)
            model = test_model
            break
        except:
            continue
    
    if model is None:
        raise Exception(
            "❌ No compatible Gemini model found. "
            "Please check your API key at https://ai.google.dev/app/apikey"
        )
    
    # Build conversation context
    chat_history = []
    
    for msg in st.session_state.conversation_history[-10:]:
        if msg["role"] == "user":
            chat_history.append({"role": "user", "parts": [msg["content"]]})
        else:
            chat_history.append({"role": "model", "parts": [msg["content"]]})
    
    # Start chat session
    chat = model.start_chat(history=chat_history)
    
    # Send message with system prompt
    full_prompt = f"{system_prompt}\n\nUser message: {user_message}"
    response = chat.send_message(full_prompt, stream=True)
    
    return response

def create_csv_chart(dataframe, chart_type, x_col=None, y_col=None, title="", color_col=None, size_col=None, aggregate=None):
    """Create interactive Plotly chart from CSV data with advanced options"""
    
    try:
        fig = None
        
        # Basic Charts (require X and Y)
        if chart_type == "line":
            fig = px.line(dataframe, x=x_col, y=y_col, title=title or f"Line Chart: {y_col} vs {x_col}")
        
        elif chart_type == "bar":
            fig = px.bar(dataframe, x=x_col, y=y_col, title=title or f"Bar Chart: {y_col} vs {x_col}")
        
        elif chart_type == "scatter":
            fig = px.scatter(dataframe, x=x_col, y=y_col, 
                           color=color_col, size=size_col,
                           title=title or f"Scatter Plot: {y_col} vs {x_col}")
        
        elif chart_type == "area":
            fig = px.area(dataframe, x=x_col, y=y_col, title=title or f"Area Chart: {y_col} vs {x_col}")
        
        elif chart_type == "box":
            fig = px.box(dataframe, x=x_col, y=y_col, title=title or f"Box Plot: {y_col} by {x_col}")
        
        elif chart_type == "violin":
            fig = px.violin(dataframe, x=x_col, y=y_col, title=title or f"Violin Plot: {y_col} by {x_col}")
        
        elif chart_type == "histogram":
            fig = px.histogram(dataframe, x=y_col, nbins=30, title=title or f"Histogram: {y_col}")
        
        elif chart_type == "density":
            fig = px.density_contour(dataframe, x=x_col, y=y_col, title=title or f"Density Plot: {y_col} vs {x_col}")
        
        elif chart_type == "strip":
            fig = px.strip(dataframe, x=x_col, y=y_col, title=title or f"Strip Plot: {y_col} by {x_col}")
        
        # Single Column Charts
        elif chart_type == "pie":
            if aggregate and aggregate != "none":
                agg_data = dataframe.groupby(x_col)[y_col].sum() if y_col else dataframe.groupby(x_col).size()
                fig = px.pie(values=agg_data.values, names=agg_data.index, title=title or f"Pie Chart: {x_col}")
            else:
                fig = px.pie(dataframe, names=x_col, values=y_col, title=title or f"Pie Chart: {x_col}")
        
        elif chart_type == "sunburst":
            if y_col:
                agg_data = dataframe.groupby(x_col)[y_col].sum().reset_index()
                fig = px.sunburst(agg_data, names=x_col, values=y_col, title=title or f"Sunburst Chart: {x_col}")
        
        elif chart_type == "treemap":
            if y_col:
                agg_data = dataframe.groupby(x_col)[y_col].sum().reset_index()
                fig = px.treemap(agg_data, names=x_col, values=y_col, title=title or f"Treemap: {x_col}")
        
        elif chart_type == "funnel":
            agg_data = dataframe.groupby(x_col)[y_col].sum().reset_index() if y_col else dataframe.groupby(x_col).size().reset_index(name=y_col or 'count')
            fig = px.funnel(agg_data, x=y_col or 'count', y=x_col, title=title or f"Funnel Chart: {x_col}")
        
        elif chart_type == "bubble":
            fig = px.scatter(dataframe, x=x_col, y=y_col, size=size_col, color=color_col,
                           title=title or f"Bubble Chart: {y_col} vs {x_col}")
        
        elif chart_type == "heatmap":
            # Pivot table for heatmap
            if len(dataframe.columns) >= 3:
                pivot_data = dataframe.pivot_table(values=y_col, index=x_col, columns=color_col, aggfunc='sum')
                fig = go.Figure(data=go.Heatmap(z=pivot_data.values, x=pivot_data.columns, y=pivot_data.index))
                fig.update_layout(title=title or f"Heatmap: {y_col}")
        
        # Apply common layout settings
        if fig:
            fig.update_layout(
                template="plotly_white",
                height=550,
                hovermode='closest',
                font=dict(size=12),
                showlegend=True,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            return fig
        
        return None
    
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def export_chart_as_png(fig):
    """Export Plotly figure as PNG"""
    try:
        png_bytes = fig.to_image(format="png", width=1000, height=600)
        return png_bytes
    except Exception as e:
        st.error(f"Error exporting chart: {str(e)}")
        return None

def export_conversation_history():
    """Export conversation history as JSON"""
    export_data = {
        "session_id": st.session_state.current_session_id,
        "timestamp": datetime.now().isoformat(),
        "api_provider": st.session_state.api_provider,
        "loaded_files": list(st.session_state.loaded_files.keys()),
        "active_file": st.session_state.active_file,
        "conversation": st.session_state.conversation_history
    }
    return json.dumps(export_data, indent=2)

def display_file_preview(content, max_chars=5000):
    """Display preview of file content"""
    preview = content[:max_chars]
    if len(content) > max_chars:
        preview += "\n\n... [Content truncated for preview]"
    return preview

# ========================= SIDEBAR =========================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Provider Selection
    api_provider = st.radio(
        "🔑 Select AI Provider",
        ["Gemini", "OpenAI"],
        help="Choose your LLM provider"
    )
    
    # Map display name to internal name
    provider_name = "Gemini" if "Gemini" in api_provider else "OpenAI"
    st.session_state.api_provider = provider_name
    
    # Gemini Configuration
    if provider_name == "Gemini":
        st.info(
            "✨ **Gemini**\n\n"
            "- Free to use (no credit card required)\n\n"
            "✅ [Get API Key](https://ai.google.dev/gemini-api/docs/api-key)"
        )
        gemini_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            key="gemini_key_input",
            help="Get from https://ai.google.dev/app/apikey"
        )
        if gemini_key:
            st.session_state.gemini_api_key = gemini_key
            st.success("✅ Gemini API configured!")
        else:
            st.warning("⚠️ Please enter your Gemini API key")
            with st.expander("📝 How to get your API key?"):
                st.markdown("""
                1. Go to [ai.google.dev/app/apikey](https://ai.google.dev/app/apikey)
                2. Click "Create API key"
                3. Select "Create API key in new Google Cloud project"
                4. Copy the API key
                5. Paste it above
                
                **Note:** Free tier includes 60 requests/minute!
                """)
    
    # OpenAI Configuration
    else:
        st.info(
            "💳 **OpenAI GPT-3.5**\n\n"
            "- Requires paid account\n"
            "- Higher quality responses\n\n"
            "[Get API Key](https://platform.openai.com/api-keys)"
        )
        openai_key = st.text_input(
            "Enter OpenAI API Key",
            type="password",
            key="openai_key_input"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key
            st.success("✅ OpenAI API configured!")
        else:
            st.warning("⚠️ Please enter your OpenAI API key")
    
    st.divider()
    
    # File Upload Section
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (multiple selections supported)",
        type=['txt', 'pdf', 'docx', 'csv'],
        accept_multiple_files=True,
        help="Supported formats: TXT, PDF, DOCX, CSV. You can select multiple files!"
    )
    
    if uploaded_files:
        if st.button("📥 Process Files", use_container_width=True):
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                success_count = 0
                for uploaded_file in uploaded_files:
                    content, dataframe = process_uploaded_file(uploaded_file)
                    if content:
                        st.session_state.loaded_files[uploaded_file.name] = {
                            'content': content,
                            'dataframe': dataframe,
                            'type': uploaded_file.name.split('.')[-1].lower()
                        }
                        if st.session_state.active_file is None:
                            st.session_state.active_file = uploaded_file.name
                        success_count += 1
                
                if success_count > 0:
                    st.success(f"✅ {success_count} file(s) loaded successfully!")
                    st.session_state.conversation_history = []
                    st.rerun()
    
    st.divider()
    
    # Loaded Files Management
    if st.session_state.loaded_files:
        st.markdown("### 📚 Loaded Files")
        
        for filename in st.session_state.loaded_files.keys():
            col_name, col_btn = st.columns([3, 1])
            
            file_info = st.session_state.loaded_files[filename]
            icon = get_file_type_icon(filename)
            is_active = filename == st.session_state.active_file
            
            with col_name:
                file_type = file_info['type'].upper()
                status = "✓" if is_active else ""
                st.write(f"{icon} {filename} {status}")
            
            with col_btn:
                if st.button("✕", key=f"del_{filename}", help="Remove file"):
                    del st.session_state.loaded_files[filename]
                    if st.session_state.active_file == filename:
                        st.session_state.active_file = list(st.session_state.loaded_files.keys())[0] if st.session_state.loaded_files else None
                    st.rerun()
        
        if len(st.session_state.loaded_files) > 1:
            st.markdown("**Select Active File:**")
            active_file = st.selectbox(
                "Choose file to chat with",
                options=list(st.session_state.loaded_files.keys()),
                index=list(st.session_state.loaded_files.keys()).index(st.session_state.active_file) if st.session_state.active_file in st.session_state.loaded_files else 0,
                label_visibility="collapsed"
            )
            st.session_state.active_file = active_file
    
    st.divider()
    
    # Session Management
    st.markdown("### 💾 Session Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.rerun()
    
    with col2:
        if st.button("📥 Export Chat", use_container_width=True):
            export_json = export_conversation_history()
            st.download_button(
                label="⬇️ Download JSON",
                data=export_json,
                file_name=f"chat_{st.session_state.current_session_id}.json",
                mime="application/json",
                use_container_width=True
            )
    
    st.divider()
    
    # Document Information
    if st.session_state.active_file and st.session_state.active_file in st.session_state.loaded_files:
        st.markdown("### 📋 Active Document")
        file_info = st.session_state.loaded_files[st.session_state.active_file]
        st.info(f"**File:** {st.session_state.active_file}\n\n**Tokens:** ~{len(file_info['content'].split()) // 4}")

# ========================= MAIN CONTENT =========================

# Title
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h1 style='font-size: 3.5em; margin-bottom: 0.5rem;'>📊 DocumentIQ</h1>
    <p style='font-size: 1.2em; color: #666; margin-bottom: 1rem;'><b>Intelligent Document Analysis & Visualization Platform</b></p>
    <p style='font-size: 0.95em; color: #999;'>
        📄 Supports: <b>TXT</b> • <b>PDF</b> • <b>DOCX</b> • <b>CSV</b>
    </p>
    <p style='font-size: 0.9em; color: #bbb;'>Upload multiple files • Chat with AI • Create stunning visualizations</p>
</div>
""", unsafe_allow_html=True)

# Check if any API is configured
api_configured = st.session_state.gemini_api_key or st.session_state.openai_api_key

if not api_configured:
    st.warning(
        "⚠️ **API Key Required to Start**\n\n"
        "Choose your AI provider in the **Configuration** section:\n\n"
        "**Gemini:**\n"
        "- [Get Gemini API Key](https://ai.google.dev/gemini-api/docs/api-key)\n\n"
        "**OpenAI (Paid):**\n"
        "- Higher quality responses\n"
        "- [Get OpenAI API Key](https://platform.openai.com/api-keys)"
    )
else:
    # Two-column layout
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        st.markdown("## � Chat")
        
        # Display conversation history
        if st.session_state.active_file and st.session_state.active_file in st.session_state.loaded_files:
            icon = get_file_type_icon(st.session_state.active_file)
            st.markdown(f"**{icon} Chatting about:** `{st.session_state.active_file}`")
            if len(st.session_state.loaded_files) > 1:
                st.caption(f"📚 {len(st.session_state.loaded_files)} files loaded")
            st.divider()
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            if len(st.session_state.conversation_history) == 0:
                st.info("👋 No messages yet. Upload a document and ask your first question!")
            
            for i, message in enumerate(st.session_state.conversation_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>👤 You:</b><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <b>🤖 Assistant:</b><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # Input section
        if st.session_state.active_file and st.session_state.active_file in st.session_state.loaded_files:
            st.markdown("### 📝 Ask a Question")
            
            # Use form to prevent duplicate submissions
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Type your question here",
                    label_visibility="collapsed",
                    height=80,
                    placeholder="Ask anything about the document..."
                )
                submit_button = st.form_submit_button("📤 Send", use_container_width=True)
            
            if submit_button and user_input:
                # Add user message to history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Get and stream response
                with st.spinner("💭 Thinking..."):
                    try:
                        # Get API key based on provider
                        if st.session_state.api_provider == "Gemini":
                            if not st.session_state.gemini_api_key:
                                st.error("❌ Please enter your Gemini API key in the sidebar")
                                if st.session_state.conversation_history[-1]["role"] == "user":
                                    st.session_state.conversation_history.pop()
                            else:
                                stream = get_chat_response(
                                    "Gemini",
                                    st.session_state.gemini_api_key,
                                    user_input
                                )
                                
                                # Stream response for Gemini
                                response_placeholder = st.empty()
                                full_response = ""
                                
                                for chunk in stream:
                                    if hasattr(chunk, 'text'):
                                        full_response += chunk.text
                                    else:
                                        full_response += str(chunk)
                                    
                                    response_placeholder.markdown(f"""
                                    <div class="chat-message assistant-message">
                                        <b>🤖 Assistant:</b><br>
                                        {full_response}
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Add assistant response to history
                                st.session_state.conversation_history.append({
                                    "role": "assistant",
                                    "content": full_response
                                })
                                
                                st.rerun()
                        else:
                            if not st.session_state.openai_api_key:
                                st.error("❌ Please enter your OpenAI API key in the sidebar")
                                if st.session_state.conversation_history[-1]["role"] == "user":
                                    st.session_state.conversation_history.pop()
                            else:
                                stream = get_chat_response(
                                    "OpenAI",
                                    st.session_state.openai_api_key,
                                    user_input
                                )
                                
                                # Stream response for OpenAI
                                response_placeholder = st.empty()
                                full_response = ""
                                
                                for chunk in stream:
                                    if chunk.choices[0].delta.content:
                                        full_response += chunk.choices[0].delta.content
                                        response_placeholder.markdown(f"""
                                        <div class="chat-message assistant-message">
                                            <b>🤖 Assistant:</b><br>
                                            {full_response}
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Add assistant response to history
                                st.session_state.conversation_history.append({
                                    "role": "assistant",
                                    "content": full_response
                                })
                                
                                st.rerun()
                    
                    except Exception as e:
                        error_msg = str(e)
                        # Handle specific errors
                        if "quota" in error_msg.lower() or "429" in error_msg:
                            st.error(
                                "❌ **API Quota Exceeded**\n\n"
                                "Your API has hit its usage quota. Please check your account and billing details."
                            )
                        elif "invalid_api_key" in error_msg or "invalid api" in error_msg.lower():
                            st.error("❌ Invalid API key. Please check your API key in the sidebar.")
                        elif "connection" in error_msg.lower():
                            st.error("❌ Connection error. Please check your internet connection and try again.")
                        else:
                            st.error(f"❌ Error: {error_msg}")
                        
                        # Remove the user message if response failed
                        if st.session_state.conversation_history and st.session_state.conversation_history[-1]["role"] == "user":
                            st.session_state.conversation_history.pop()
        else:
            st.info("📤 Please upload documents first to start chatting.\n\nClick on the **Upload Documents** section in the sidebar to get started!")
    
    # Right column - Tools & Features
    with col2:
        st.markdown("## 🛠️ Tools")
        
        # File Preview
        if st.session_state.active_file and st.session_state.active_file in st.session_state.loaded_files:
            with st.expander("👁️ File Preview", expanded=False):
                file_info = st.session_state.loaded_files[st.session_state.active_file]
                preview = display_file_preview(file_info['content'])
                st.code(preview, language="text")
        
        # CSV Chart Creator
        if st.session_state.active_file and st.session_state.active_file.endswith('.csv'):
            file_info = st.session_state.loaded_files[st.session_state.active_file]
            if file_info['dataframe'] is not None:
                st.markdown("### 📊 Advanced Chart Creator")
                
                df = file_info['dataframe']
                columns = df.columns.tolist()
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                
                # Chart Type Selection with Categories
                st.write("**Select Visualization Type:**")
                chart_category = st.selectbox(
                    "Category",
                    ["Comparison", "Distribution", "Composition", "Correlation"],
                    key="chart_category",
                    label_visibility="collapsed"
                )
                
                chart_options = {
                    "Comparison": ["bar", "line", "scatter", "area", "strip"],
                    "Distribution": ["histogram", "box", "violin", "density"],
                    "Composition": ["pie", "sunburst", "treemap", "funnel"],
                    "Correlation": ["scatter", "bubble", "heatmap"]
                }
                
                chart_type = st.selectbox(
                    "Chart Type",
                    chart_options[chart_category],
                    key="chart_type_select"
                )
                
                # Column Selection based on Chart Type
                st.write("**Configure Data:**")
                
                if chart_type in ["pie", "funnel"]:
                    col_x = st.selectbox("Categories", columns, key="x_axis_select", label_visibility="collapsed")
                    col_y = st.selectbox("Values", numeric_columns, key="y_axis_select", label_visibility="collapsed") if numeric_columns else None
                else:
                    col_x = st.selectbox("X-Axis", columns, key="x_axis_select", label_visibility="collapsed")
                    col_y = st.selectbox("Y-Axis", numeric_columns, key="y_axis_select", label_visibility="collapsed") if numeric_columns else None
                
                # Advanced Options
                with st.expander("⚙️ Advanced Options"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        color_col = st.selectbox(
                            "Color By (optional)",
                            [None] + columns,
                            key="color_col"
                        )
                        
                        aggregate = st.selectbox(
                            "Aggregation",
                            ["none", "sum", "mean", "count", "min", "max"],
                            key="aggregate"
                        )
                    
                    with col2:
                        size_col = st.selectbox(
                            "Size By (optional)",
                            [None] + numeric_columns,
                            key="size_col"
                        )
                        
                        show_legend = st.checkbox("Show Legend", value=True, key="show_legend")
                
                chart_title = st.text_input(
                    "Chart Title (optional)",
                    key="chart_title_input",
                    placeholder="Enter custom title..."
                )
                
                if st.button("📈 Generate Chart", use_container_width=True, key="gen_chart"):
                    fig = create_csv_chart(df, chart_type, col_x, col_y, chart_title, color_col, size_col, aggregate)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.current_charts.append(fig)
                        
                        # Export options
                        col_download, col_settings = st.columns(2)
                        with col_download:
                            png_data = export_chart_as_png(fig)
                            if png_data:
                                st.download_button(
                                    label="⬇️ PNG",
                                    data=png_data,
                                    file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                        
                        with col_settings:
                            st.info(f"📊 **Chart Info:** {chart_type.upper()} | Rows: {len(df)}")
        
        # Conversation Stats
        st.markdown("### 📊 Stats")
        if st.session_state.conversation_history:
            msg_count = len(st.session_state.conversation_history)
            user_msgs = sum(1 for m in st.session_state.conversation_history if m["role"] == "user")
            assistant_msgs = msg_count - user_msgs
            
            st.metric("Total Messages", msg_count)
            col_u, col_a = st.columns(2)
            with col_u:
                st.metric("Your Questions", user_msgs)
            with col_a:
                st.metric("Responses", assistant_msgs)
        
# ========================= FOOTER =========================
st.divider()
st.markdown("""
---
<div style='text-align: center; color: #666;'>
    <small>📊 DocumentIQ - Powered by Google Gemini & OpenAI | Built for intelligent document analysis</small>
</div>
""", unsafe_allow_html=True)
