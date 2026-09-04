# Multi-Node-Conversational-AI-Framework-LangGraph
https://multi-node-conversational-ai-framework.streamlit.app

## Overview

A sophisticated Streamlit-based conversational AI application that combines LangGraph state management with sentiment analysis and Groq's fast LLM inference. This framework demonstrates multi-node graph-based processing for intelligent conversation handling.

## Architecture

### System Architecture Diagram

```
User Input
    ↓
┌─────────────────────────────────────────┐
│        PREPROCESSING NODE               │
│  • Strip whitespace                     │
│  • Normalize input                      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│    SENTIMENT ANALYSIS NODE              │
│  • Keyword-based sentiment detection    │
│  • Categories: positive/negative/neutral│
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│      CHATBOT NODE (LLM)                 │
│  • Groq API Integration                 │
│  • Model: qwen/qwen3.6-27b              │
│  • Fast inference response generation   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│       LOGGER NODE                       │
│  • Store sentiment in session state     │
│  • Track conversation history           │
└─────────────────────────────────────────┘
    ↓
Streamlit UI Display
  • Chat message display
  • Sentiment badge
  • Chat history
```

### Core Components

#### 1. **State Management (TypedDict)**

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    sentiment: str
```

- Maintains conversation messages
- Tracks sentiment for current user input
- Uses LangGraph's `add_messages` reducer for efficient list handling

#### 2. **Graph Nodes**

##### a. Preprocessing Node

- Cleans user input by stripping whitespace
- Normalizes message content
- Prepares data for analysis

##### b. Sentiment Analysis Node

- Pattern-matching based sentiment detection
- **Positive words**: good, great, excellent, happy, love, awesome
- **Negative words**: bad, poor, sad, hate, awful, terrible
- **Default**: neutral (if no keywords match)

##### c. Chatbot Node

- Integrates Groq's fast LLM API
- Uses ChatGroq model with LangChain
- Generates context-aware responses
- Leverages message history for conversation continuity

##### d. Logger Node

- Stores sentiment in session state
- Enables sentiment-based features
- Maintains audit trail

### Technology Stack

| Component               | Technology | Version |
| ----------------------- | ---------- | ------- |
| **UI Framework**        | Streamlit  | ≥1.28.0 |
| **LLM Framework**       | LangChain  | ≥0.1.0  |
| **Graph Orchestration** | LangGraph  | ≥0.1.0  |
| **LLM Provider**        | Groq API   | gsk\_\* |
| **Language**            | Python     | 3.9+    |

### Data Flow

1. **Input Stage**: User sends message via Streamlit chat interface
2. **Preprocessing**: Message is cleaned and normalized
3. **Analysis**: Sentiment is detected using keyword matching
4. **Processing**: LLM generates response using Groq API
5. **Logging**: Sentiment and response are stored in session
6. **Output**: Message and sentiment displayed in chat interface

### Session State

```python
st.session_state:
  chat_history: List[Dict]
    - role: "user" | "assistant"
    - content: str
    - sentiment: str

  last_sentiment: str ("positive" | "negative" | "neutral")
```

## Features

- ✅ **Graph-Based Workflow**: LangGraph StateGraph for structured multi-node processing
- ✅ **Real-time Sentiment Analysis**: Keyword-based sentiment detection
- ✅ **Fast LLM Inference**: Groq API for rapid response generation
- ✅ **Chat History**: Persistent conversation tracking within session
- ✅ **Sentiment Display**: Visual sentiment indicators for each message
- ✅ **Configurable Model**: Switch between Groq models via sidebar
- ✅ **Error Handling**: Graceful handling of missing API keys

## Setup Instructions

### Prerequisites

- Python 3.9+
- Groq API key

### Local Development

1. **Clone the repository**

   ```bash
   git clone https://github.com/ai-genai-agenticai-intelligence/Multi-Node-Conversational-AI-Framework-LangGraph-.git
   cd Multi-Node-Conversational-AI-Framework-LangGraph-
   ```

2. **Create `.streamlit/secrets.toml`**

   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   GROQ_MODEL = "qwen/qwen3.6-27b"
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   streamlit run streamlit_langgraph_app.py
   ```

5. **Access the app**
   - Local: http://localhost:8501
   - Network: http://192.168.x.x:8501

### Streamlit Cloud Deployment

1. **Push code to GitHub** (already done)

2. **Deploy via Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Connect your GitHub repository
   - Select the repository and main branch

3. **Add Secrets in Streamlit Cloud**
   - Click app menu (≡) → Settings
   - Click "Secrets" in left sidebar
   - Add your configuration:
     ```
     GROQ_API_KEY = "your_groq_api_key_here"
     GROQ_MODEL = "qwen/qwen3.6-27b"
     ```
   - Save (app will auto-redeploy)

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required)
- `GROQ_MODEL`: Model name (default: `qwen/qwen3.6-27b`)

### Sidebar Settings

- **Groq model**: Switch between available Groq models

## Dependencies

```
streamlit>=1.28.0          # Web UI framework
langchain-groq>=0.1.0      # Groq integration
langchain-core>=0.1.0      # Core LangChain utilities
langgraph>=0.1.0           # Graph orchestration
typing-extensions>=4.0.0   # Type hints
```

## Project Structure

```
9.LANGGRAPH/
├── streamlit_langgraph_app.py    # Main application
├── .streamlit/
│   └── secrets.toml              # Local secrets (gitignored)
├── requirements.txt              # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
└── [Jupyter notebooks]           # Development notebooks
    ├── langchain.ipynb
    ├── langgraph.ipynb
    ├── langgraph2.ipynb
    └── lgraph_quickstart.ipynb
```

## API Response Model

### ChatGroq Model Specs

- **Provider**: Groq (API-based)
- **Speed**: Ultra-fast inference (<100ms)
- **Architecture**: Optimized for speed and cost
- **Supported Models**:
  - `meta-llama/llama-3.1-8b-instant`
  - `qwen/qwen3.6-27b`
  - `mixtral-8x7b-32768`

## Error Handling

- ✅ Missing API key: Clear error message with setup instructions
- ✅ Network issues: Streamlit handles gracefully
- ✅ Invalid model: Falls back to default model
- ✅ Session state: Preserved across reruns

## Performance Metrics

- **Preprocessing**: <1ms per message
- **Sentiment Analysis**: <5ms per message
- **LLM Inference**: 50-200ms (via Groq)
- **Total Latency**: ~100-300ms per turn

## Future Enhancements

- [ ] Advanced NLP-based sentiment analysis (transformers)
- [ ] Multi-language support
- [ ] Conversation persistence (database)
- [ ] Analytics dashboard
- [ ] User authentication
- [ ] Custom node types
- [ ] Knowledge base integration (RAG)

## License

MIT License

## Contributing

Pull requests welcome! Please ensure code follows PEP 8 style guidelines.

## Support

For issues and questions, please open a GitHub issue or check the documentation.

---

**Built with ❤️ using LangGraph, Streamlit, and Groq**
