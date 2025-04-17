# DocuFind 📚

DocuFind is an intelligent document processing and search system that allows users to upload, analyze, and query documents with advanced natural language processing capabilities. Built with Python and Flask, it supports multiple languages and provides smart document recommendations.

## 🌟 Features

### Document Processing
- 📄 Upload and process multiple document formats
- 🌐 Automatic language detection
- 🔍 Smart text extraction and indexing
- 🖼️ Image extraction and caption analysis

### Search & Query
- 🔎 Natural language querying
- 🌍 Multilingual search support
- 📊 Context-aware results
- 💡 Smart recommendations

### Document Analysis
- 📝 Automatic document summarization
- 🎯 Key points extraction
- 🔤 Multi-language support
- 📱 Mobile-friendly interface

### Advanced Features
- 🔄 Real-time language translation
- 🌐 URL document fetching
- 📑 Related document suggestions
- 📈 Topic analysis and clustering

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Flask
- Required Python packages (install via requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/brainardphilemon/docufind.git
cd docufind
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

The application will be available at `http://localhost:8000`

## 🛠️ API Endpoints

### Document Management
- `POST /upload` - Upload new documents
- `POST /fetch-document` - Fetch document from URL
- `GET /view-document/<doc_id>` - View specific document

### Search & Query
- `POST /query` - Search across documents
- `GET /recommendations` - Get document recommendations
- `GET /summary/<doc_id>` - Get document summary

### Analysis
- `GET /document-images/<doc_id>` - Extract images from document
- `POST /detect-language` - Detect text language
- `GET /related-documents/<doc_id>` - Find related documents
- `GET /top-topics` - Extract top topics from collection

## 🎯 Usage Examples

### Upload and Query Documents
```python
# Upload a document
POST /upload
# Response: {"doc_id": "123", "title": "example.pdf", "preview": "Document content..."}

# Search documents
POST /query
Data: {"query": "What is machine learning?", "language": "en"}
# Response: [{"doc_id": "123", "title": "ML Guide", "content": "..."}]
```

### Get Document Summary
```python
GET /summary/123?language=en&use_ai=true
# Response: {
#   "summary": "Main document points...",
#   "key_points": ["Point 1", "Point 2"],
#   "language": "en"
# }
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Flask and Python
- Uses advanced NLP techniques for document processing
- Supports multiple languages and formats

---
Made with ❤️ by Brainard Philemon