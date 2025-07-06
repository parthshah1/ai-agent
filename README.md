# Filecoin Code & Documentation QA CLI

An AI-powered CLI assistant that helps answer questions about Filecoin by searching through source code, documentation, and GitHub issues/PRs.

## Features

- Search and analyze code from Filecoin repositories (lotus, specs, FIPs, etc.)
- Answer questions with references to exact code locations and documentation
- Provide context from GitHub issues and pull requests
- Support for semantic code search and natural language queries
- Powered by OpenAI GPT models for accurate code understanding
- Local vector storage with FAISS for fast similarity search

## Cost Considerations

This tool uses:
- OpenAI API (GPT models) for answering questions
- Local embedding model (sentence-transformers) for code indexing
- GitHub API with standard limits (5000 requests/hour for authenticated users)

Typical usage costs:
- Question answering: ~$0.002-$0.01 per query (depends on context size)
- Code indexing: Free (uses local embedding model)
- GitHub API: Free for public repositories

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/filecoin-AI.git
cd filecoin-AI
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.template .env
```
Then edit `.env` to add your GitHub token and OpenAI API key. The template includes detailed comments about each configuration option.

## Usage

Run the CLI:
```bash
python -m filecoin_qa
```

Example queries:
- "How does Lotus handle block validation?"
- "What does FIP-0032 propose?"
- "Show me the code for sector sealing"
- "Has there been any issue related to proof verification failures?"

## Configuration

The tool can be configured through environment variables in your `.env` file:

Required:
- `GITHUB_TOKEN`: Your GitHub personal access token
- `OPENAI_API_KEY`: Your OpenAI API key

Optional:
- `REPOS`: Comma-separated list of Filecoin repositories to index
- `EMBEDDING_MODEL`: HuggingFace model to use for embeddings (defaults to all-MiniLM-L6-v2)
- `TOP_K`: Number of similar documents to retrieve per query (defaults to 5)
- `DEBUG`: Enable debug logging
- `OPENAI_MODEL`: OpenAI model to use (defaults to gpt-4-turbo-preview)

### Default Configuration
```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Local embedding model
OPENAI_MODEL=gpt-4-turbo-preview  # For best code understanding
TOP_K=5  # Balanced for performance and cost
DEBUG=false
```

### Resource Usage
- Memory: ~500MB-1GB (depends on indexed repositories)
- Disk: ~100MB-500MB for vector store
- CPU: Moderate usage during embedding
- No GPU required

## Project Structure

```
filecoin-AI/
├── filecoin_qa/
│   ├── __init__.py
│   ├── cli.py           # CLI interface
│   ├── agent.py         # QA agent implementation
│   ├── loaders/         # Document loaders
│   │   ├── __init__.py
│   │   ├── github.py    # GitHub repo loader
│   │   └── docs.py      # Documentation loader
│   ├── indexing/        # Vector store and embedding
│   │   ├── __init__.py
│   │   └── store.py     # FAISS vector store management
│   └── utils/           # Helper utilities
│       ├── __init__.py
│       └── formatting.py # Output formatting
├── tests/               # Test suite
├── requirements.txt     # Project dependencies
├── env.template         # Environment variables template
└── README.md           # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details 