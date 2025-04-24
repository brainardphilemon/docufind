# DocuFind - Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [API Reference](#api-reference)
5. [Setup Guide](#setup-guide)
6. [Development Guidelines](#development-guidelines)
7. [Troubleshooting](#troubleshooting)

## System Overview

DocuFind is an intelligent document processing and search system built with Python and Flask. It provides advanced natural language processing capabilities for document analysis, search, and recommendations.

### Key Features
- Multi-format document processing
- Natural language querying
- Multilingual support
- Smart recommendations
- Real-time language translation
- Document summarization
- Image extraction and analysis

## Architecture

### High-Level Architecture
```
DocuFind
├── Frontend (Web Interface)
├── Backend (Flask Server)
│   ├── Document Processor
│   ├── Search Engine
│   ├── Language Processor
│   └── Recommendation System
└── Storage Layer
```

### Technology Stack
- **Backend Framework**: Flask (Python)
- **Document Processing**: PyPDF2, python-docx
- **NLP Processing**: NLTK, spaCy
- **Search Engine**: Elasticsearch
- **Image Processing**: PIL, OpenCV
- **Database**: SQLite/PostgreSQL
- **Frontend**: HTML, CSS, JavaScript

## Core Components

### 1. Document Processor
- Handles document upload and parsing
- Supports multiple file formats (PDF, DOCX, TXT)
- Extracts text and metadata
- Processes embedded images
- Performs OCR when necessary

### 2. Search Engine
- Implements full-text search
- Supports fuzzy matching
- Handles multilingual queries
- Provides relevance scoring
- Maintains search index

### 3. Language Processor
- Detects document language
- Performs translation
- Extracts key phrases
- Generates summaries
- Analyzes sentiment

### 4. Recommendation System
- Analyzes document similarities
- Tracks user preferences
- Generates personalized suggestions
- Implements collaborative filtering

## API Reference

### Document Management

#### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: Document file
- language: (optional) Document language
- tags: (optional) Array of tags

Response:
{
    "doc_id": "string",
    "title": "string",
    "preview": "string",
    "status": "success|error"
}
```

#### Fetch Document
```http
POST /fetch-document
Content-Type: application/json

Body:
{
    "url": "string",
    "options": {
        "translate": boolean,
        "summarize": boolean
    }
}

Response:
{
    "doc_id": "string",
    "content": "string",
    "metadata": object
}
```

### Search & Query

#### Search Documents
```http
POST /query
Content-Type: application/json

Body:
{
    "query": "string",
    "language": "string",
    "filters": object,
    "page": number,
    "limit": number
}

Response:
{
    "results": array,
    "total": number,
    "page": number
}
```

#### Get Recommendations
```http
GET /recommendations?doc_id=string
Response:
{
    "related_docs": array,
    "topics": array,
    "confidence": number
}
```

## Setup Guide

### Development Environment Setup

1. **Clone Repository**
```bash
git clone https://github.com/brainardphilemon/docufind.git
cd docufind
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Configuration**
Create `.env` file:
```
FLASK_APP=main.py
FLASK_ENV=development
SECRET_KEY=your_secret_key
DB_URL=sqlite:///docufind.db
```

5. **Initialize Database**
```bash
flask db upgrade
```

### Production Deployment

1. **Server Requirements**
- Python 3.7+
- PostgreSQL
- Redis (for caching)
- Nginx (web server)

2. **Configuration**
Update `config.py` with production settings:
```python
class ProductionConfig:
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    REDIS_URL = os.environ.get('REDIS_URL')
```

3. **Deploy with Gunicorn**
```bash
gunicorn -w 4 -b 127.0.0.1:8000 main:app
```

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Maintain test coverage above 80%

### Git Workflow
1. Create feature branch
2. Implement changes
3. Write/update tests
4. Submit pull request
5. Code review
6. Merge to main

### Testing
```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Check coverage
coverage run -m pytest
coverage report
```

## Troubleshooting

### Common Issues

1. **Document Upload Fails**
- Check file size limits
- Verify file format support
- Ensure proper permissions

2. **Search Not Working**
- Verify Elasticsearch connection
- Check index status
- Review query syntax

3. **Language Detection Issues**
- Ensure sufficient text content
- Check language support
- Verify model installation

### Logging

Logs are stored in `logs/` directory:
- `app.log`: Application logs
- `error.log`: Error logs
- `access.log`: Access logs

### Support

For technical support:
1. Check documentation
2. Review issue tracker
3. Contact development team

---

*Last Updated: April 2025*