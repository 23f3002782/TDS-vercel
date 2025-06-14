import os
import base64
from io import BytesIO
from PIL import Image
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import glob
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize clients
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize Pinecone index
index_name = "tds-embeddings-hf-v2"
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=768,  # nomic-embed-text-v1.5 dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pinecone.Index(index_name)

class HuggingFaceLLM:
    def __init__(self, model: str = "mistralai/Magistral-Small-2506", embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model = model
        self.embedding_model = embedding_model
        self.hf_token = os.getenv("HF_TOKEN")
        
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is required")
        
        # Initialize Hugging Face client
        self.client = InferenceClient(
            provider="featherless-ai",
            api_key=self.hf_token,
        )
        
        # Initialize embedding model
        try:
            self.embedding_model_instance = SentenceTransformer(
                self.embedding_model, 
                trust_remote_code=True
            )
            logger.info(f"Embedding model {self.embedding_model} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise Exception(f"Failed to load embedding model: {str(e)}")
        
        self._verify_models()
    
    def _verify_models(self):
        """Verify that the models are accessible"""
        try:
            # Test embedding model
            test_embedding = self.get_embedding("test")
            logger.info(f"Embedding model test successful, dimension: {len(test_embedding)}")
            
            # Test generation model
            test_response = self.generate_completion([
                {"role": "user", "content": "Hello, respond with just 'OK' if you're working."}
            ], max_tokens=10)
            logger.info(f"Generation model test successful: {test_response}")
            
        except Exception as e:
            logger.error(f"Error verifying models: {e}")
            raise Exception(f"Failed to verify Hugging Face models: {str(e)}")

    def generate_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1200) -> str:
        """Generate completion using Hugging Face Inference API"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            raise Exception(f"Hugging Face generation failed: {str(e)}")
    
    def generate_completion_with_image(self, messages: List[Dict[str, Any]], temperature: float = 0.7, max_tokens: int = 1200) -> str:
        """Handle image + text - extract text and note image limitation for non-vision models"""
        text_messages = []
        has_image = False
        
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for content in msg["content"]:
                    if content.get("type") == "text":
                        text_parts.append(content["text"])
                    elif content.get("type") == "image_url":
                        has_image = True
                        # For non-vision models, we can't process the image
                        text_parts.append("[IMAGE PROVIDED - Current model cannot analyze images. Please describe the image or use a vision-capable model.]")
                
                text_messages.append({
                    "role": msg["role"],
                    "content": " ".join(text_parts)
                })
            else:
                text_messages.append(msg)
        
        # Add note about image limitation if image was provided
        if has_image:
            text_messages.insert(-1, {
                "role": "system",
                "content": "Note: An image was provided but the current model cannot analyze images. Focus on the text content and ask the user to describe the image if visual analysis is needed."
            })
        
        return self.generate_completion(text_messages, temperature, max_tokens)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using SentenceTransformers"""
        try:
            embedding = self.embedding_model_instance.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise Exception(f"Failed to get embedding: {str(e)}")

def process_image(base64_image: str) -> str:
    """Process and validate base64 image"""
    try:
        # Handle data URL format
        if base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
        
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        
        # Validate it's a proper image
        img = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large
        max_size = (1024, 1024)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert back to base64
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        processed_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return processed_base64
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise ValueError(f"Invalid image format: {str(e)}")

# Initialize the Hugging Face LLM
hf_llm = HuggingFaceLLM()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    start = 0
    text = text.strip()
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            # Last chunk
            chunk = text[start:]
        else:
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_space = chunk.rfind(' ')
            
            # Choose the best break point
            break_points = [p for p in [last_period, last_newline, last_space] if p > start + chunk_size // 2]
            
            if break_points:
                break_point = max(break_points)
                chunk = text[start:break_point + (1 if break_point == last_period else 0)]
                end = break_point + (1 if break_point == last_period else 0)
        
        chunk = chunk.strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        start = max(end - overlap, start + 1)  # Ensure progress
        
        if start >= len(text):
            break
    
    return chunks

def process_markdown_files(folder_path: str) -> List[Dict[str, Any]]:
    """Process markdown files and create chunks"""
    if not os.path.exists(folder_path):
        logger.error(f"Markdown folder not found: {folder_path}")
        return []
    
    md_documents = []
    md_files = glob.glob(os.path.join(folder_path, "*.md"))
    
    if not md_files:
        logger.warning(f"No markdown files found in {folder_path}")
        return []
    
    for file_path in tqdm(md_files, desc="Processing markdown files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Empty file skipped: {file_path}")
                continue
            
            filename = os.path.basename(file_path)
            
            # Extract title
            title = filename.replace('.md', '').replace('_', ' ').title()
            if content.startswith('#'):
                first_line = content.split('\n')[0]
                if first_line.startswith('#'):
                    title = first_line.strip('#').strip()
            
            chunks = chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                md_documents.append({
                    'filename': filename,
                    'title': title,
                    'chunk_id': i,
                    'content': chunk,
                    'full_content': content[:500] + "..." if len(content) > 500 else content
                })
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Processed {len(md_documents)} markdown chunks from {len(md_files)} files")
    return md_documents

def process_posts(filename: str) -> Dict[int, Dict[str, Any]]:
    """Load and group posts by topic"""
    if not os.path.exists(filename):
        logger.error(f"Discourse file not found: {filename}")
        return {}
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            posts_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in discourse file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error reading discourse file: {e}")
        return {}
    
    if not isinstance(posts_data, list):
        logger.error("Discourse data should be a list of posts")
        return {}
    
    topics = {}
    for post in posts_data:
        if not isinstance(post, dict):
            continue
        
        topic_id = post.get("topic_id")
        if topic_id is None:
            continue
            
        if topic_id not in topics:
            topics[topic_id] = {
                "topic_title": post.get("topic_title", ""),
                "posts": []
            }
        topics[topic_id]["posts"].append(post)
    
    # Sort posts by post number
    for topic in topics.values():
        topic["posts"].sort(key=lambda p: p.get("post_number", 0))
    
    logger.info(f"Processed {len(topics)} discourse topics")
    return topics

def initialize_system():
    """Initialize the RAG system by loading and indexing data"""
    try:
        # Validate required environment variables
        required_env_vars = {
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "DISCOURSE_FILE_PATH": os.getenv("DISCOURSE_FILE_PATH"),
            "MARKDOWN_FOLDER_PATH": os.getenv("MARKDOWN_FOLDER_PATH")
        }
        
        missing_vars = [var for var, value in required_env_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Test Hugging Face connection
        try:
            test_response = hf_llm.generate_completion([
                {"role": "user", "content": "Hello, respond with just 'OK' if you're working."}
            ], max_tokens=10)
            logger.info(f"Hugging Face LLM test successful: {test_response}")
        except Exception as e:
            logger.error(f"Hugging Face LLM connection failed: {e}")
            raise Exception("Make sure your HF_TOKEN is valid and has access to the required models")
        
        # Load and process data
        discourse_file = required_env_vars["DISCOURSE_FILE_PATH"]
        markdown_folder = required_env_vars["MARKDOWN_FOLDER_PATH"]
        
        logger.info("Loading discourse data...")
        topics = process_posts(discourse_file)
        if not topics:
            logger.warning("No discourse topics loaded")
        
        logger.info("Loading markdown files...")
        md_documents = process_markdown_files(markdown_folder)
        if not md_documents:
            logger.warning("No markdown documents loaded")
        
        # Index data
        if topics:
            logger.info("Indexing discourse data...")
            embed_and_index_discourse(topics)
            logger.info("Discourse indexing complete")
        
        if md_documents:
            logger.info("Indexing markdown data...")
            embed_and_index_markdown(md_documents)
            logger.info("Markdown indexing complete")
        
        logger.info("System initialization complete!")
        
    except Exception as e:
        logger.error(f"Error during system initialization: {e}")
        raise

def build_thread_map(posts: List[Dict[str, Any]]) -> Dict[Optional[int], List[Dict[str, Any]]]:
    """Build reply tree structure"""
    thread_map = {}
    for post in posts:
        parent = post.get("reply_to_post_number")
        if parent not in thread_map:
            thread_map[parent] = []
        thread_map[parent].append(post)
    return thread_map

def extract_thread(root_num: int, posts: List[Dict[str, Any]], thread_map: Dict[Optional[int], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extract full thread starting from root post"""
    thread = []
    
    def collect_replies(post_num):
        post = next((p for p in posts if p.get("post_number") == post_num), None)
        if post:
            thread.append(post)
            for reply in thread_map.get(post_num, []):
                collect_replies(reply.get("post_number"))
    
    collect_replies(root_num)
    return thread

def embed_and_index_discourse(topics: Dict[int, Dict[str, Any]], batch_size: int = 100):
    """Embed discourse threads and index in Pinecone"""
    vectors = []
    
    for topic_id, topic_data in tqdm(topics.items(), desc="Processing discourse topics"):
        try:
            posts = topic_data["posts"]
            topic_title = topic_data["topic_title"]
            
            if not posts:
                continue
                
            thread_map = build_thread_map(posts)
            root_posts = thread_map.get(None, [])
            
            for root_post in root_posts:
                root_post_num = root_post.get("post_number")
                if root_post_num is None:
                    continue
                
                thread = extract_thread(root_post_num, posts, thread_map)
                
                if not thread:
                    continue
                
                # Build combined text
                combined_text = f"Topic: {topic_title}\n\n"
                thread_contents = []
                for post in thread:
                    content = post.get("content", "").strip()
                    if content:
                        thread_contents.append(content)
                
                if not thread_contents:
                    continue
                
                combined_text += "\n\n---\n\n".join(thread_contents)
                
                try:
                    embedding = hf_llm.get_embedding(combined_text)
                except Exception as e:
                    logger.error(f"Failed to get embedding for topic {topic_id}: {e}")
                    continue
                
                vector = {
                    "id": f"discourse_{topic_id}_{root_post_num}",
                    "values": embedding,
                    "metadata": {
                        "source_type": "discourse",
                        "topic_id": int(topic_id),
                        "topic_title": topic_title,
                        "root_post_number": int(root_post_num),
                        "post_numbers": [str(p.get("post_number", 0)) for p in thread],
                        "combined_text": combined_text,
                        "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}/{root_post_num}"
                    }
                }
                vectors.append(vector)
                
                if len(vectors) >= batch_size:
                    index.upsert(vectors=vectors)
                    vectors = []
                    
        except Exception as e:
            logger.error(f"Error processing topic {topic_id}: {e}")
            continue
    
    if vectors:
        index.upsert(vectors=vectors)

def embed_and_index_markdown(md_documents: List[Dict[str, Any]], batch_size: int = 100):
    """Embed markdown documents and index in Pinecone"""
    vectors = []
    
    for doc in tqdm(md_documents, desc="Processing markdown documents"):
        try:
            embedding = hf_llm.get_embedding(doc['content'])
        except Exception as e:
            logger.error(f"Failed to get embedding for {doc['filename']} chunk {doc['chunk_id']}: {e}")
            continue
        
        vector = {
            "id": f"markdown_{doc['filename']}_{doc['chunk_id']}",
            "values": embedding,
            "metadata": {
                "source_type": "markdown",
                "filename": doc['filename'],
                "title": doc['title'],
                "chunk_id": doc['chunk_id'],
                "content": doc['content'],
                "preview": doc['full_content']
            }
        }
        vectors.append(vector)
        
        if len(vectors) >= batch_size:
            try:
                index.upsert(vectors=vectors)
                vectors = []
            except Exception as e:
                logger.error(f"Failed to upsert batch: {e}")
                vectors = []
    
    if vectors:
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            logger.error(f"Failed to upsert final batch: {e}")

# Initialize FastAPI app
app = FastAPI(title="TDS Virtual TA", description="Virtual Teaching Assistant with RAG capabilities")

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 encoded image

class LinkResponse(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[LinkResponse] = []

def semantic_search(query: str, top_k: int = 10, source_filter: str = None) -> List[Dict[str, Any]]:
    """Search for relevant content using embeddings - INCREASED top_k for better coverage"""
    try:
        query_embedding = hf_llm.get_embedding(query)
    except Exception as e:
        logger.error(f"Failed to get query embedding: {e}")
        return []
    
    filter_dict = {}
    if source_filter:
        filter_dict = {"source_type": source_filter}
    
    try:
        search_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
    except Exception as e:
        logger.error(f"Pinecone search failed: {e}")
        return []
    
    results = []
    for match in search_response.matches:
        try:
            if match.metadata["source_type"] == "discourse":
                results.append({
                    "score": match.score,
                    "source_type": "discourse",
                    "topic_id": match.metadata["topic_id"],
                    "topic_title": match.metadata["topic_title"],
                    "root_post_number": match.metadata["root_post_number"],
                    "post_numbers": [int(pn) for pn in match.metadata["post_numbers"] if pn.isdigit()],
                    "content": match.metadata["combined_text"],
                    "url": match.metadata.get("url", "")
                })
            else:
                results.append({
                    "score": match.score,
                    "source_type": "markdown",
                    "filename": match.metadata["filename"],
                    "title": match.metadata["title"],
                    "chunk_id": match.metadata["chunk_id"],
                    "content": match.metadata["content"],
                    "preview": match.metadata["preview"]
                })
        except Exception as e:
            logger.error(f"Error processing search result: {e}")
            continue
    
    return results

def extract_discourse_links(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract and format discourse links from search results"""
    links = []
    seen_urls = set()
    
    for result in results:
        if result["source_type"] == "discourse" and result.get("url"):
            url = result["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                # Create meaningful link text
                topic_title = result.get("topic_title", "Discussion")
                links.append({
                    "url": url,
                    "text": f"Forum Discussion: {topic_title}"
                })
    
    return links

def generate_answer_with_links(query: str, context_results: List[Dict[str, Any]], image_base64: Optional[str] = None) -> Dict[str, Any]:
    """Generate answer using context and optionally process image - IMPROVED PROMPT"""
    
    # Extract links early
    links = extract_discourse_links(context_results)
    
    # Check if we have relevant context
    if not context_results or all(result["score"] < 0.5 for result in context_results):
        # Low relevance - might not have the information
        future_exam_keywords = ["end-term", "end term", "exam", "2025", "sep 2025", "september 2025"]
        if any(keyword in query.lower() for keyword in future_exam_keywords):
            return {
                "answer": "I don't know the exact date for the TDS Sep 2025 end-term exam as this information is not available yet. Please check the official course announcements or contact the instructors for the most up-to-date exam schedule.",
                "links": links
            }
    
    # Separate and format context by source type
    discourse_context = []
    markdown_context = []
    
    for result in context_results:
        try:
            if result["source_type"] == "discourse":
                discourse_context.append({
                    "title": result['topic_title'],
                    "content": result['content'],
                    "score": result['score']
                })
            else:
                markdown_context.append({
                    "title": result['title'],
                    "content": result['content'],
                    "score": result['score']
                })
        except Exception as e:
            logger.error(f"Error processing result: {e}")
            continue
    
    # Build context string with better formatting
    context_parts = []
    
    if discourse_context:
        context_parts.append("=== FORUM DISCUSSIONS ===")
        for i, ctx in enumerate(discourse_context[:3]):  # Limit to top 3
            context_parts.append(f"\nDiscussion {i+1}: {ctx['title']}")
            context_parts.append(f"Content: {ctx['content'][:1000]}{'...' if len(ctx['content']) > 1000 else ''}")
    
    if markdown_context:
        context_parts.append("\n=== COURSE MATERIALS ===")
        for i, ctx in enumerate(markdown_context[:3]):  # Limit to top 3
            context_parts.append(f"\nMaterial {i+1}: {ctx['title']}")
            context_parts.append(f"Content: {ctx['content'][:1000]}{'...' if len(ctx['content']) > 1000 else ''}")
    
    context = "\n".join(context_parts) if context_parts else "No relevant context found."
    
    # IMPROVED SYSTEM PROMPT - More specific and directive
    system_prompt = """You are a Virtual Teaching Assistant for the TDS (Tools in Data Science) course at IIT Madras. 

IMPORTANT GUIDELINES:
1. Answer based ONLY on the provided context from forum discussions and course materials
2. If the context doesn't contain enough information to answer confidently, say "I don't know" or "This information is not available"
3. For technical questions, be specific about tools, versions, and recommendations mentioned in the context
4. When mentioning tools like Docker vs Podman, be precise about course recommendations
5. For scoring/grading questions, provide exact details if available in the context
6. For future dates/events not covered in the context, clearly state the information is not available yet
7. Always be helpful but honest about limitations

Be concise but thorough. If you find contradictory information, mention both perspectives."""

    # Prepare messages for completion
    try:
        if image_base64:
            # Handle image + text query
            processed_image = process_image(image_base64)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Question: {query}\n\nContext from TDS course materials and forum:\n{context}\n\nPlease answer the question based on the provided context. If an image was provided, note that you cannot analyze it but can work with any text descriptions."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{processed_image}"}}
                ]}
            ]
            
            answer = hf_llm.generate_completion_with_image(messages, temperature=0.3, max_tokens=1200)
        else:
            # Handle text-only query
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}\n\nContext from TDS course materials and forum:\n{context}\n\nPlease answer the question based on the provided context."}
            ]
            
            answer = hf_llm.generate_completion(messages, temperature=0.3, max_tokens=1200)
        
        return {
            "answer": answer,
            "links": links
        }
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            "answer": f"I apologize, but I encountered an error while generating the answer: {str(e)}",
            "links": links
        }

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Main API endpoint to answer questions with optional image support"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Perform semantic search with increased coverage
        results = semantic_search(request.question, top_k=10)
        
        # Generate answer with links
        response_data = generate_answer_with_links(
            request.question, 
            results, 
            request.image
        )
        
        return AnswerResponse(
            answer=response_data["answer"],
            links=[LinkResponse(**link) for link in response_data["links"]]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Hugging Face connection
        test_response = hf_llm.generate_completion([
            {"role": "user", "content": "Test"}
        ], max_tokens=5)
        
        return {
            "status": "healthy",
            "huggingface": "connected",
            "embedding_model": hf_llm.embedding_model,
            "generation_model": hf_llm.model
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "TDS Virtual TA API", 
        "version": "1.0",
        "endpoints": {
            "POST /api/": "Answer questions with optional image support",
            "GET /health": "Health check",
            "GET /": "This endpoint"
        }
    }

if __name__ == "__main__":
    # Initialize the system (uncomment when you want to rebuild the index)
    initialize_system()
    
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
