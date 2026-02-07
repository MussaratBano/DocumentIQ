# Application for LLM Integration in chat with File Uploads using streamlit 
> This application demonstrates how to integrate a Large Language Model (LLM) into a chat interface that supports file uploads using Streamlit. Users can upload files and interact with the LLM to extract information, summarize content, or perform other tasks based on the uploaded files.
## Features
 - File upload support for various formats (e.g., text, PDF, Word).    
- Chat interface for user interaction with the LLM.    
- Integration with a Large Language Model for processing and responding to user queries using API calls.    
- Display of LLM responses in a user-friendly chat format.

## 1. Installation
>Create a virtual environment for the project:
```bash
python -m venv .venv
```
> Activate the virtual environment:\
> for windows gitbash:
```bash
source .venv/Scripts/activate
conda deactivate
```
> for linux or mac:
```bash 
source .venv/bin/activate
```
## 2. Install the required packages
```bash
 pip install -r requirements.txt
```
## 3. Set up API Keys
> Obtain an API key from your LLM provider (e.g., OpenAI or Google Gemini).

# DocumentIQ - Intelligent Document Analysis & Visualization Platform

**DocumentIQ** is a powerful, open-source document intelligence application built with Streamlit 
that transforms how you interact with your files. Powered by advanced AI models (Google Gemini & 
OpenAI), DocumentIQ enables seamless conversations with your documents—whether they're text files, 
PDFs, Word documents, or CSV datasets. Upload multiple files simultaneously, ask intelligent 
questions about their content, and receive real-time AI-powered responses with contextual accuracy. 

Beyond document analysis, DocumentIQ features an advanced data visualization engine leveraging 
Plotly's comprehensive charting library, allowing you to create stunning interactive visualizations 
including bar charts, line graphs, scatter plots, heatmaps, sunburst diagrams, funnels, and much 
more. Perfect for professionals, researchers, students, and data enthusiasts, DocumentIQ combines 
document comprehension with powerful analytics in a single, intuitive interface.

**Key Highlights:**
- 🎯 **Multi-Format Support**: Chat with TXT, PDF, DOCX, and CSV files simultaneously
- 🤖 **AI-Powered Intelligence**: Leverage Google Gemini (free!) or OpenAI for document analysis
- 📊 **Advanced Visualizations**: 15+ Plotly chart types with customizable colors, sizes, and aggregations
- 💾 **Conversation Export**: Save entire chat histories as JSON for future reference
- 📈 **Chart Export**: Download visualizations as high-quality PNG images
- ⚡ **Lightning-Fast Processing**: Optimized caching for superior performance and instant responses
- 🔄 **Multi-File Navigation**: Switch between loaded documents with intelligent context management
- 📱 **Responsive UI**: Clean, professional, and intuitive interface designed for accessibility