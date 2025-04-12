from typing import List
import os
from langchain.schema import Document
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import requests
import json

# Download ALL required NLTK resources upfront
nltk.download('punkt')
nltk.download('stopwords')


class DocumentSummarizer:
    def __init__(self):
        # Configuration for Ollama - try different model names
        llama_model_options = ["llama3", "llama3:8b", "llama2", "llama2:13b", "llama2:7b"]
        self.ollama_url = "http://localhost:11434/api/generate"  # Default Ollama API endpoint

        # Try to find an available Llama model
        self.model_name = None
        for model in llama_model_options:
            try:
                # Test if this model is available
                print(f"Testing Ollama model: {model}")
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": "Hello",
                        "stream": False
                    },
                    timeout=5
                )

                if response.status_code == 200:
                    self.model_name = model
                    self.initialized = True
                    print(f"Using Ollama model: {model}")
                    break
            except Exception as e:
                print(f"Model {model} not available: {e}")

        # If no Llama model is available, try to find any model
        if self.model_name is None:
            try:
                # Get list of available models
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    if models:
                        self.model_name = models[0]["name"]
                        self.initialized = True
                        print(f"Fallback to available Ollama model: {self.model_name}")
                    else:
                        self.initialized = False
                        print("No models found in Ollama")
                else:
                    self.initialized = False
                    print("Failed to get available models from Ollama")
            except Exception as e:
                self.initialized = False
                print(f"Error connecting to Ollama: {e}")
                print("Please ensure Ollama is running.")
                print("Install Llama with: ollama pull llama2")

    def _generate_with_llama(self, prompt, max_tokens=1000, temperature=0.2):
        """Generate text using Llama via Ollama"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"Error from Ollama API: {response.status_code} {response.text}")
                return ""
        except Exception as e:
            print(f"Error generating with Llama: {e}")
            return ""

    def summarize_document(self, documents: List[Document]) -> str:
        """Generate a comprehensive summary"""
        try:
            if not self.initialized:
                return self._fallback_summary(documents)

            # Combine document contents
            combined_text = " ".join([doc.page_content for doc in documents])

            # Clean up the text
            combined_text = self._clean_text(combined_text)

            # If text is too short or primarily noise, return a simple message
            if len(combined_text.split()) < 20:
                return "This document contains minimal extractable text or may be primarily non-text content."

            # Process longer documents in chunks
            if len(combined_text.split()) > 4000:
                return self._summarize_long_document(combined_text)
            else:
                return self._summarize_with_llama(combined_text)

        except Exception as e:
            print(f"Error generating summary: {e}")
            try:
                return self._fallback_summary(documents)
            except Exception as e2:
                print(f"Fallback summary also failed: {e2}")
                return self._emergency_fallback_summary(documents)

    def _emergency_fallback_summary(self, documents: List[Document]) -> str:
        """Super simple fallback when everything else fails"""
        # Combine all text
        all_text = " ".join([doc.page_content for doc in documents])

        # Simple extractive summary - just take the beginning
        words = all_text.split()
        beginning = " ".join(words[:200])

        # Add some middle content
        if len(words) > 500:
            middle_start = len(words) // 2 - 100
            middle = " ".join(words[middle_start:middle_start + 200])

            # Add some end content
            if len(words) > 1000:
                ending = " ".join(words[-200:])
                return beginning + "\n\n[...]\n\n" + middle + "\n\n[...]\n\n" + ending
            else:
                return beginning + "\n\n[...]\n\n" + middle
        else:
            return beginning

    def _summarize_with_llama(self, text):
        """Generate a summary using Llama"""
        # Limit text length to avoid token limits
        if len(text.split()) > 4000:
            text = " ".join(text.split()[:4000])

        # Create a structured prompt for comprehensive summarization
        prompt = f"""
You are a professional document summarizer. Your task is to create a comprehensive, detailed summary of the following document.
The summary should:
1. Be comprehensive (around 500-800 words)
2. Cover all major points and key information
3. Be well-structured with clear sections
4. Maintain the document's original flow and organization
5. Be written in a clear, professional style

Here is the document to summarize:

{text}

Comprehensive Summary:
"""

        # Generate the summary
        summary = self._generate_with_llama(prompt, max_tokens=1000, temperature=0.1)

        # If summary is too short, try again with different instructions
        if len(summary.split()) < 100:
            prompt = f"""
I need a VERY DETAILED summary of this document. Include ALL important information, key points, data, and conclusions.
The summary should be at least 500 words.

DOCUMENT:
{text}

DETAILED SUMMARY:
"""
            summary = self._generate_with_llama(prompt, max_tokens=1000, temperature=0.1)

        # If still no good summary, use fallback
        if len(summary.split()) < 50:
            return self._fallback_summary([Document(page_content=text)])

        # Clean up the summary
        summary = summary.strip()

        return summary

    def _summarize_long_document(self, text):
        """Summarize a long document by chunking"""
        # Split the document into chunks
        chunks = []
        try:
            chunks = self._split_into_chunks(text)
        except Exception as e:
            print(f"Error splitting document: {e}")
            # Simple fallback chunking
            words = text.split()
            chunk_size = 1500
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            if len(chunk.split()) < 50:  # Skip very small chunks
                continue

            # Summarize this chunk
            try:
                chunk_prompt = f"""
Summarize this section of a document in 150-200 words:

{chunk}

Section Summary:
"""
                chunk_summary = self._generate_with_llama(chunk_prompt, max_tokens=300, temperature=0.1)

                # Check if we got a valid summary
                if len(chunk_summary.split()) >= 30:
                    chunk_summaries.append(chunk_summary)
                else:
                    # Simple fallback - extract first few sentences
                    try:
                        sentences = text.split('.')
                        extract = '. '.join(sentences[:5]) + '.'
                        chunk_summaries.append(extract)
                    except:
                        # Ultra fallback - just take first 200 words
                        chunk_summaries.append(" ".join(chunk.split()[:200]))
            except Exception as e:
                print(f"Error summarizing chunk {i}: {e}")
                # Fallback to extracting some text
                chunk_summaries.append(" ".join(chunk.split()[:200]))

            # Limit to processing ~5 chunks to avoid excessive processing time
            if i >= 4:
                break

        # Combine chunk summaries
        combined_summary = "\n\n".join(chunk_summaries)

        # For very long documents, generate a final summary of summaries
        if self.initialized and len(combined_summary.split()) > 1500:
            try:
                final_prompt = f"""
I've summarized sections of a long document. Create a unified, comprehensive summary from these section summaries.
The final summary should be well-organized, cover all important points, and be around 800 words.

SECTION SUMMARIES:
{combined_summary}

COMPREHENSIVE DOCUMENT SUMMARY:
"""
                final_summary = self._generate_with_llama(final_prompt, max_tokens=1000, temperature=0.1)
                if len(final_summary.split()) >= 100:
                    return final_summary
            except Exception as e:
                print(f"Error generating final summary: {e}")

        # Return the combined summary
        return "# Document Summary\n\n" + combined_summary

    def extract_key_points(self, documents: List[Document], query: str = "", num_points: int = 5) -> List[str]:
        """Extract key points from the document that are relevant to the user's query"""
        try:
            # Combine document contents
            combined_text = " ".join([doc.page_content for doc in documents])

            # Clean up the text
            combined_text = self._clean_text(combined_text)

            # Check for minimal content
            if len(combined_text.split()) < 20:
                return ["This document contains minimal extractable text."]

            # If LLM is not available or for short content, use extractive method
            if not self.initialized or len(combined_text.split()) < 100:
                if query:
                    return self._extract_query_relevant_points(combined_text, query, num_points)
                else:
                    return self._extract_important_sentences(combined_text, num_points)

            # Limit text length
            if len(combined_text.split()) > 4000:
                combined_text = " ".join(combined_text.split()[:4000])

            # Create prompt based on whether we have a query or not
            if query and len(query.strip()) > 0:
                prompt = f"""
Extract exactly {num_points} key points from this document that specifically address this query: "{query}"

Each key point should:
- Be a complete, informative sentence
- Focus on information directly relevant to the query
- Capture an important insight or fact from the document
- Be clear and concise

Document:
{combined_text}

{num_points} KEY POINTS ABOUT "{query}":
1.
"""
            else:
                prompt = f"""
Extract the {num_points} most important key points from this document.

Each key point should:
- Be a complete, informative sentence
- Represent a major insight, finding, or conclusion
- Be clear and concise
- Cover a distinct aspect of the document

Document:
{combined_text}

THE {num_points} MOST IMPORTANT KEY POINTS:
1.
"""

            # Generate key points
            response = self._generate_with_llama(prompt, max_tokens=500, temperature=0.1)

            # Process the response to extract individual points
            points = self._extract_numbered_points(response, num_points)

            # If we couldn't extract enough points, use fallback method
            if len(points) < num_points / 2:
                if query:
                    return self._extract_query_relevant_points(combined_text, query, num_points)
                else:
                    return self._extract_important_sentences(combined_text, num_points)

            return points

        except Exception as e:
            print(f"Error extracting key points: {e}")
            try:
                return self._fallback_key_points(documents, num_points)
            except Exception as e2:
                print(f"Fallback key points extraction also failed: {e2}")
                # Ultra fallback - just return first few sentences
                sentences = combined_text.split('.')[:num_points]
                return [s.strip() + '.' for s in sentences if s.strip()]

    def _extract_numbered_points(self, text, expected_points):
        """Extract numbered points from Llama's output"""
        try:
            # Try to find numbered points like "1. Point one"
            numbered_point_pattern = r'(?:\d+\.\s*)([^\d].*?)(?=\d+\.\s*|$)'
            matches = re.findall(numbered_point_pattern, text, re.DOTALL)

            # Clean up the points
            points = []
            for point in matches:
                # Clean up and split in case the regex caught multiple points
                lines = [line.strip() for line in point.split('\n') if line.strip()]
                points.extend(lines)

            # If we found enough points, return them
            if len(points) >= expected_points / 2:
                return points[:expected_points]

            # If the regex approach failed, try splitting by newlines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_points = []

            for line in lines:
                # Remove numbered prefixes like "1." or "Point 1:"
                cleaned_line = re.sub(r'^\d+\.?\s*', '', line)
                cleaned_line = re.sub(r'^Point \d+:?\s*', '', cleaned_line)

                if cleaned_line and len(cleaned_line) > 15:  # Avoid very short fragments
                    clean_points.append(cleaned_line)

            return clean_points[:expected_points]
        except Exception as e:
            print(f"Error extracting numbered points: {e}")
            # Ultra fallback
            sentences = text.split('.')
            return [s.strip() + '.' for s in sentences[:expected_points] if s.strip()]

    def _extract_query_relevant_points(self, text, query, num_points):
        """Extract sentences that are relevant to the query"""
        try:
            # Get sentences from the document
            sentences = []
            try:
                sentences = sent_tokenize(text)
            except Exception as e:
                print(f"Error tokenizing sentences: {e}")
                # Fallback sentence splitting
                sentences = text.split('.')
                sentences = [s.strip() + '.' for s in sentences if s.strip()]

            # Process query terms
            query_terms = set(query.lower().split())
            try:
                stop_words = set(stopwords.words('english'))
                query_terms = [term for term in query_terms if term not in stop_words]
            except Exception as e:
                print(f"Error processing stopwords: {e}")

            # Score sentences based on query relevance
            scored_sentences = []
            for sentence in sentences:
                # Count query terms in sentence
                sentence_lower = sentence.lower()
                score = sum(1 for term in query_terms if term in sentence_lower)

                # Consider sentence length
                words = sentence.split()
                if 5 <= len(words) <= 40:
                    score += 0.5

                scored_sentences.append((sentence, score))

            # Sort by score (descending)
            scored_sentences.sort(key=lambda x: x[1], reverse=True)

            # Get top sentences
            return [s for s, _ in scored_sentences[:num_points] if s.strip()]
        except Exception as e:
            print(f"Error in query-relevant extraction: {e}")
            # Ultra fallback
            sentences = text.split('.')[:num_points]
            return [s.strip() + '.' for s in sentences if s.strip()]

    def _split_into_chunks(self, text):
        """Split text into manageable chunks for processing"""
        # Use sentence tokenization for more natural chunks
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        # Target approximately 1500 words per chunk
        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_length + sentence_words <= 1500:
                current_chunk.append(sentence)
                current_length += sentence_words
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _extract_important_sentences(self, text: str, num_sentences: int = 5) -> List[str]:
        """Extract important sentences from the text"""
        try:
            # Split into sentences
            sentences = []
            try:
                sentences = sent_tokenize(text)
            except Exception as e:
                print(f"Error tokenizing sentences: {e}")
                # Fallback sentence splitting
                sentences = text.split('.')
                sentences = [s.strip() + '.' for s in sentences if s.strip()]

            if len(sentences) <= num_sentences:
                return sentences

            # Get stopwords
            stop_words = []
            try:
                stop_words = set(stopwords.words('english'))
            except Exception as e:
                print(f"Error getting stopwords: {e}")
                # Common English stopwords as fallback
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                              'to', 'of', 'in', 'for', 'with', 'by', 'at', 'on', 'this', 'that'}

            # Calculate word frequencies (ignoring stopwords)
            word_freq = {}
            for sentence in sentences:
                for word in sentence.lower().split():
                    if word not in stop_words and word.isalnum():
                        word_freq[word] = word_freq.get(word, 0) + 1

            # Score sentences based on word frequencies
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = 0
                for word in sentence.lower().split():
                    if word in word_freq:
                        score += word_freq[word]

                # Normalize by sentence length
                words = len(sentence.split())
                if words > 0:
                    score = score / (words ** 0.5)

                # Give a slight boost to early sentences (introduction) and late sentences (conclusion)
                if i < len(sentences) * 0.2 or i > len(sentences) * 0.8:
                    score *= 1.2

                sentence_scores.append((sentence, score))

            # Get the top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s for s, _ in sentence_scores[:num_sentences]]

            return top_sentences
        except Exception as e:
            print(f"Error extracting important sentences: {e}")
            # Ultra fallback
            sentences = text.split('.')[:num_sentences]
            return [s.strip() + '.' for s in sentences if s.strip()]

    def _clean_text(self, text: str) -> str:
        """Clean up text to remove problematic patterns"""
        if not text:
            return ""

        try:
            # Remove repeating patterns like "Date Page Date Page"
            text = re.sub(r'(Date\s+Page\s*)+', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'(Date\s+Pa\s*e\s*)+', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'(dæssmate|dæssptß|classmate|classp)\s*', ' ', text, flags=re.IGNORECASE)

            # Remove non-printable characters and control characters
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]+', ' ', text)

            # Remove isolated special characters
            text = re.sub(r'(?<!\w)[+\-•#*=_]+(?!\w)', ' ', text)

            # Replace multiple spaces, newlines with single ones
            text = re.sub(r'\s+', ' ', text)

            # Trim whitespace
            text = text.strip()

            return text
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return text

    def _fallback_summary(self, documents: List[Document]) -> str:
        """Generate a comprehensive extractive summary when model fails"""
        try:
            # Combine all text
            all_text = " ".join([doc.page_content for doc in documents])

            # Extract important sentences
            important_sentences = self._extract_important_sentences(all_text, num_sentences=15)

            # Combine into a summary
            summary = " ".join(important_sentences)

            # If summary is still short, add more from the beginning
            if len(summary.split()) < 200:
                words = all_text.split()
                beginning = " ".join(words[:300])
                summary = beginning + "\n\n" + summary

            return summary
        except Exception as e:
            print(f"Error in fallback summary: {e}")
            return self._emergency_fallback_summary(documents)

    def _fallback_key_points(self, documents: List[Document], num_points: int = 5) -> List[str]:
        """Extract key points using keyword-based approach when model fails"""
        try:
            # Combine all text
            all_text = " ".join([doc.page_content for doc in documents])

            # Split into sentences
            sentences = []
            try:
                sentences = sent_tokenize(all_text)
            except:
                sentences = re.split(r'(?<=[.!?])\s+', all_text)
                if not sentences:
                    sentences = all_text.split('.')

            # Simple keyword-based extraction
            keywords = ["important", "key", "significant", "main", "critical",
                        "essential", "crucial", "primary", "major", "vital"]

            # Score sentences based on keywords and length
            scored_sentences = []
            for sentence in sentences:
                score = sum(1 for keyword in keywords if keyword.lower() in sentence.lower())
                # Prioritize medium-length sentences (not too short, not too long)
                length_score = min(1.0, len(sentence.split()) / 20)
                if length_score < 0.5:
                    length_score = 0
                scored_sentences.append((sentence, score + length_score))

            # Sort by score
            scored_sentences.sort(key=lambda x: x[1], reverse=True)

            # If no good sentences found, just take the first few
            if not any(score > 0 for _, score in scored_sentences[:num_points]):
                return [s for s, _ in scored_sentences[:num_points]]

            # Return top sentences with score > 0
            return [s for s, score in scored_sentences[:num_points] if score > 0]
        except Exception as e:
            print(f"Error in fallback key points: {e}")
            # Ultra fallback - first few sentences
            sentences = all_text.split('.')[:num_points]
            return [s.strip() + '.' for s in sentences if s.strip()]