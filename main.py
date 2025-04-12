from flask import Flask, request, jsonify, render_template, send_from_directory, redirect
import os
import tempfile
from werkzeug.utils import secure_filename

# Import your modules
from document_processor import DocumentProcessor
from query_engine import QueryEngine
from summarizer import DocumentSummarizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize components
document_processor = DocumentProcessor()
query_engine = QueryEngine(document_processor)
summarizer = DocumentSummarizer()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                print(f"File saved to {filepath}")

                # Process the document
                doc_id = document_processor.process_document(filepath, filename)
                print(f"Document processed with ID: {doc_id}")

                # Get document preview
                doc_chunks = document_processor.get_document(doc_id)
                if not doc_chunks:
                    print(f"No chunks found for document ID {doc_id}")
                    preview_text = "No preview available"
                    doc_lang = "en"
                else:
                    preview_text = doc_chunks[0].page_content
                    doc_lang = doc_chunks[0].metadata.get('language', 'en')

                return jsonify({
                    "doc_id": doc_id,
                    "title": filename,
                    "preview": preview_text[:200] + "..." if len(preview_text) > 200 else preview_text,
                    "language": doc_lang
                })
            except Exception as e:
                print(f"Error processing document: {e}")
                return jsonify({"error": f"Error processing document: {str(e)}"}), 500
    except Exception as e:
        print(f"Unhandled error in upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route('/query', methods=['POST'])
def query_documents():
    try:
        query = request.form.get('query', '')
        user_lang = request.form.get('language', None)  # Get user's preferred language

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Detect language if not provided
        if not user_lang:
            user_lang = document_processor.detect_language(query)

        results = query_engine.search_documents(query, query_language=user_lang)

        if not results:
            # Localized "no results" message
            if user_lang == 'hi':
                no_results_message = "आपकी क्वेरी से मेल खाने वाले कोई दस्तावेज़ नहीं मिले। कृपया अलग खोज शब्द आज़माएं या अधिक दस्तावेज़ अपलोड करें।"
            else:
                no_results_message = "No documents found matching your query. Try different search terms or upload more documents."

            return jsonify([{
                "doc_id": "no_results",
                "title": "No Results",
                "content": no_results_message,
                "snippet": "No results found.",
                "language": user_lang
            }])

        return jsonify(results)
    except Exception as e:
        print(f"Error in query: {e}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@app.route('/summary/<doc_id>')
def get_document_summary(doc_id):
    query = request.args.get('query', '')
    user_lang = request.args.get('language', None)
    use_ai = request.args.get('use_ai', 'true').lower() == 'true'

    # Get document chunks
    doc_chunks = document_processor.get_document(doc_id)

    if not doc_chunks:
        return jsonify({"error": "Document not found"}), 404

    # Get document language
    doc_lang = doc_chunks[0].metadata.get('language', 'en')
    print(f"Document language detected: {doc_lang}")

    # If user language not specified, use document's original language
    if not user_lang:
        user_lang = doc_lang
        print(f"Using document's original language for summary: {user_lang}")

    # Introduce a small delay to simulate processing time for more realistic AI feel
    if use_ai:
        import time
        # Add realistic LLM processing delay (2-4 seconds)
        processing_time = min(2 + (len(''.join([c.page_content for c in doc_chunks[:3]])) / 5000), 6)
        time.sleep(processing_time)

    # Get localized summary
    summary_data = document_processor.get_document_summary(doc_id, target_language=user_lang, use_ai=use_ai)

    return jsonify({
        "summary": summary_data.get("summary", ""),
        "key_points": summary_data.get("key_points", []),
        "language": summary_data.get("language", user_lang)
    })


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        query = request.form.get('query', '')
        current_doc_id = request.form.get('current_doc_id', '')
        user_lang = request.form.get('language', None)

        # If language not specified, detect from query
        if not user_lang and query:
            user_lang = document_processor.detect_language(query)

        # Get similar documents
        similar_docs = query_engine.get_similar_documents(query, current_doc_id, query_language=user_lang)

        return jsonify(similar_docs)
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify({"error": f"Failed to get recommendations: {str(e)}"}), 500


@app.route('/view-document/<doc_id>')
def view_document(doc_id):
    try:
        # Get document chunks
        doc_chunks = document_processor.get_document(doc_id)

        if not doc_chunks:
            return "Document not found", 404

        # Combine all chunks
        full_text = "\n\n".join([chunk.page_content for chunk in doc_chunks])

        # Get metadata
        metadata = doc_chunks[0].metadata if doc_chunks else {}
        filename = metadata.get('source', 'Unknown Document')
        doc_lang = metadata.get('language', 'en')

        # Get user's preferred language
        user_lang = request.args.get('language', doc_lang)

        # If document language doesn't match user language, offer translation option
        translation_available = (doc_lang != user_lang)

        # Render the document view
        return render_template('document_view.html',
                               document_id=doc_id,
                               filename=filename,
                               content=full_text,
                               language=doc_lang,
                               user_language=user_lang,
                               translation_available=translation_available)
    except Exception as e:
        print(f"Error viewing document: {e}")
        return f"Error viewing document: {str(e)}", 500


@app.route('/document-images/<doc_id>')
def document_images(doc_id):
    """Get images from a specific document"""
    try:
        user_lang = request.args.get('language', 'en')
        images = document_processor.get_document_images(doc_id)

        # Handle translation of captions if needed
        for img in images:
            if img.get('language', 'en') != user_lang:
                try:
                    if hasattr(document_processor, 'translate_text'):
                        img['nearby_text'] = document_processor.translate_text(
                            img['nearby_text'],
                            target_lang=user_lang,
                            source_lang=img.get('language', 'en')
                        )
                except Exception as e:
                    print(f"Error translating image caption: {e}")

        return jsonify(images)
    except Exception as e:
        print(f"Error getting document images: {e}")
        return jsonify([]), 200  # Return empty array but don't fail


@app.route('/detect-language', methods=['POST'])
def detect_language():
    """Detect the language of provided text"""
    try:
        text = request.form.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        language = document_processor.detect_language(text)
        return jsonify({"language": language})
    except Exception as e:
        print(f"Error detecting language: {e}")
        return jsonify({"language": "en", "error": str(e)}), 200  # Return English as fallback

    @app.route('/related-documents/<doc_id>')
    def get_related_documents(doc_id):
        """Get documents related to a specific document"""
        try:
            count = int(request.args.get('count', 3))
            related_docs = document_processor.get_related_documents(doc_id, k=count)
            return jsonify(related_docs)
        except Exception as e:
            print(f"Error getting related documents: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/topic-documents')
    def get_topic_documents():
        """Get documents related to a specific topic"""
        try:
            topic = request.args.get('topic', '')
            count = int(request.args.get('count', 5))
            exclude_doc_id = request.args.get('exclude_doc_id', None)

            if not topic:
                return jsonify({"error": "No topic provided"}), 400

            related_docs = document_processor.find_related_by_topic(topic, k=count, exclude_doc_id=exclude_doc_id)
            return jsonify(related_docs)
        except Exception as e:
            print(f"Error getting topic documents: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/top-topics')
    def get_top_topics():
        """Get top topics from the document collection"""
        try:
            count = int(request.args.get('count', 5))
            topics = document_processor.extract_top_topics(k=count)
            return jsonify(topics)
        except Exception as e:
            print(f"Error getting top topics: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/document-recommendations', methods=['POST'])
    def get_document_recommendations():
        """Get recommendations based on document and optional query"""
        try:
            doc_id = request.form.get('doc_id', '')
            query = request.form.get('query', '')
            count = int(request.form.get('count', 3))

            if not doc_id:
                return jsonify({"error": "No document ID provided"}), 400

            recommendations = document_processor.get_document_recommendations(doc_id, query, k=count)
            return jsonify(recommendations)
        except Exception as e:
            print(f"Error getting document recommendations: {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/fetch-document', methods=['POST'])
def fetch_document():
    """Fetch document from URL"""
    url = request.form.get('url', '')

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Fetch and process the document
        doc_id = document_processor.fetch_document_from_url(url)

        # Get document details
        doc_chunks = document_processor.get_document(doc_id)
        if not doc_chunks:
            return jsonify({"error": "Failed to process document"}), 500

        # Get preview and language
        preview_text = doc_chunks[0].page_content[:200] + "..." if len(doc_chunks[0].page_content) > 200 else \
        doc_chunks[0].page_content
        doc_lang = doc_chunks[0].metadata.get('language', 'en')
        doc_title = doc_chunks[0].metadata.get('source', 'Online Document')

        return jsonify({
            "doc_id": doc_id,
            "title": doc_title,
            "preview": preview_text,
            "language": doc_lang,
            "source": "online"
        })

    except Exception as e:
        print(f"Error fetching document: {e}")
        return jsonify({"error": f"Error fetching document: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)