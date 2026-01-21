import streamlit as st
import os, io, math, re, shutil
from io import BytesIO
from typing import List, Dict, Tuple
from collections import defaultdict

# ----------- CORE LIBS -----------
import PyPDF2
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests

# ----------- OCR & IMAGE -----------
from PIL import Image, ImageEnhance
import pytesseract
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF
import imagehash

# ----------- DOCUMENT PROCESSING -----------
from docx import Document  # python-docx
from pptx import Presentation  # python-pptx
import markdown
from bs4 import BeautifulSoup

# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="📚 Enhanced RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enhanced RAG Syllabus Assistant")
st.caption("Upload multiple documents (PDF, DOCX, PPTX, TXT, MD) and ask multiple questions")

# ================= SECRETS =================
HF_TOKEN = st.secrets["HF_TOKEN"]
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct:fastest"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ================= CACHE MODELS =================
@st.cache_resource
def load_models():
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client(
        Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        )
    )
    return embed, client

embedding_model, chroma_client = load_models()

# ================= CONSTANTS =================
OCR_THRESHOLD = 50
IMAGE_DIR = "data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Common header/footer patterns
HEADER_FOOTER_PATTERNS = [
    r'^\d+$',  # Page numbers
    r'^Page\s+\d+',
    r'©.*\d{4}',  # Copyright notices
    r'^Chapter\s+\d+$',
    r'^Section\s+\d+$',
    r'^\d+\s*/\s*\d+$',  # Page x/y
]

# ================= HEADER/FOOTER REMOVAL =================
def is_header_footer(text: str) -> bool:
    """Detect if a line is likely a header or footer"""
    text = text.strip()
    if len(text) < 3 or len(text) > 100:
        return len(text) < 3
    
    for pattern in HEADER_FOOTER_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False

def clean_text(text: str) -> str:
    """Remove headers, footers, and clean text"""
    lines = text.split('\n')
    cleaned = []
    
    for line in lines:
        if not is_header_footer(line):
            cleaned.append(line)
    
    result = '\n'.join(cleaned)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()

# ================= TEXT EXTRACTION BY FILE TYPE =================
def extract_text_from_txt(file_bytes: bytes) -> List[Dict]:
    """Extract text from TXT files"""
    try:
        text = file_bytes.decode('utf-8')
    except:
        text = file_bytes.decode('latin-1')
    
    text = clean_text(text)
    return [{"page": 1, "text": text}] if text else []

def extract_text_from_md(file_bytes: bytes) -> List[Dict]:
    """Extract text from Markdown files"""
    try:
        md_text = file_bytes.decode('utf-8')
    except:
        md_text = file_bytes.decode('latin-1')
    
    html = markdown.markdown(md_text)
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    text = clean_text(text)
    
    return [{"page": 1, "text": text}] if text else []

def extract_text_from_docx(file_bytes: bytes) -> List[Dict]:
    """Extract text from DOCX files"""
    doc = Document(BytesIO(file_bytes))
    pages = []
    
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    
    text = '\n'.join(full_text)
    text = clean_text(text)
    
    if text:
        pages.append({"page": 1, "text": text})
    
    return pages

def extract_text_from_pptx(file_bytes: bytes) -> List[Dict]:
    """Extract text from PPTX files"""
    prs = Presentation(BytesIO(file_bytes))
    pages = []
    
    for i, slide in enumerate(prs.slides):
        text_parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text_parts.append(shape.text)
        
        text = '\n'.join(text_parts)
        text = clean_text(text)
        
        if text:
            pages.append({"page": i + 1, "text": text})
    
    return pages

def extract_text_from_pdf(pdf_bytes: bytes) -> List[Dict]:
    """Extract text from PDF with OCR fallback"""
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
    except:
        images = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        
        if len(text) < OCR_THRESHOLD and i < len(images):
            img = images[i].convert("L")
            ocr_text = pytesseract.image_to_string(img)
            if len(ocr_text.strip()) > len(text):
                text = ocr_text.strip()
        
        text = clean_text(text)
        
        if text:
            pages.append({"page": i + 1, "text": text})
    
    return pages

def extract_text_from_file(file_bytes: bytes, filename: str) -> List[Dict]:
    """Route to appropriate extractor based on file extension"""
    ext = os.path.splitext(filename)[1].lower()
    
    extractors = {
        '.pdf': extract_text_from_pdf,
        '.txt': extract_text_from_txt,
        '.md': extract_text_from_md,
        '.docx': extract_text_from_docx,
        '.doc': extract_text_from_docx,
        '.pptx': extract_text_from_pptx,
        '.ppt': extract_text_from_pptx,
    }
    
    extractor = extractors.get(ext)
    if extractor:
        try:
            return extractor(file_bytes)
        except Exception as e:
            st.error(f"Error extracting {filename}: {str(e)}")
            return []
    else:
        st.warning(f"Unsupported file type: {ext}")
        return []

# ================= IMAGE UTILITIES =================
def image_entropy(img: Image.Image) -> float:
    hist = img.histogram()
    total = sum(hist)
    entropy = 0
    for c in hist:
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)
    return entropy

def extract_images_from_pdf(pdf_bytes: bytes, doc_name: str) -> List[Dict]:
    """Extract images from PDF"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    seen_hashes = []
    image_data = []
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        
        for img in images:
            base = doc.extract_image(img[0])
            img_bytes = base["image"]
            ext = base["ext"]
            
            pil = Image.open(BytesIO(img_bytes)).convert("RGB")
            
            if image_entropy(pil) < 4.0:
                continue
            
            h = imagehash.phash(pil)
            if any(abs(h - sh) <= 5 for sh in seen_hashes):
                continue
            
            seen_hashes.append(h)
            
            safe_name = re.sub(r'[^\w\-_.]', '_', doc_name)
            path = os.path.join(
                IMAGE_DIR,
                f"{safe_name}_page{page_index+1}_{len(seen_hashes)}.{ext}"
            )
            
            with open(path, "wb") as f:
                f.write(img_bytes)
            
            image_data.append({
                "page": page_index + 1,
                "path": path,
                "doc": doc_name
            })
    
    doc.close()
    return image_data

def ocr_image_text(path: str) -> str:
    try:
        img = Image.open(path).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        text = pytesseract.image_to_string(img, config="--psm 6")
        return re.sub(r"[^a-zA-Z0-9\s\.,:-]", "", text).strip()
    except:
        return ""

# ================= CHUNKING =================
def chunk_pages(pages, chunk_size=500, overlap=100):
    """Chunk text with overlap"""
    chunks = []
    for p in pages:
        words = p["text"].split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunk_data = {"page": p["page"], "text": chunk}
                if "doc" in p:
                    chunk_data["doc"] = p["doc"]
                chunks.append(chunk_data)
    return chunks

# ================= MULTI-QUESTION PARSING =================
def parse_multiple_questions(user_input: str) -> List[str]:
    """Parse multiple questions from user input"""
    # Split by common question delimiters
    questions = re.split(r'\n+|\?+(?=\s*[A-Z0-9])', user_input)
    
    # Clean and filter questions
    parsed = []
    for q in questions:
        q = q.strip()
        if q and len(q) > 5:
            # Add question mark if missing
            if not q.endswith('?'):
                q += '?'
            parsed.append(q)
    
    # If no split found, treat as single question
    if not parsed:
        parsed = [user_input.strip()]
    
    return parsed

# ================= LLM QUERY =================
def llm_query(question: str, chunks: List[Dict]) -> str:
    """Query LLM with retrieved chunks"""
    if not chunks:
        return "Answer not found in the documents."
    
    # Build context with document sources
    context_parts = []
    for c in chunks[:15]:
        doc_info = f" (from {c.get('doc', 'document')})" if 'doc' in c else ""
        context_parts.append(f"Page {c['page']}{doc_info}: {c['text']}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Answer strictly using the document content provided below.
If the answer is not present in the documents, say: "Answer not found in the documents."

DOCUMENT CONTENT:
{context}

QUESTION: {question}

ANSWER:"""
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions strictly based on provided document content."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 400
    }
    
    try:
        r = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=30)
        
        if r.status_code != 200:
            return f"API Error: {r.status_code}"
        
        result = r.json()
        answer = result["choices"][0]["message"]["content"].strip()
        
        # Log usage
        usage = result.get("usage", {})
        if usage:
            st.caption(f"🔢 Tokens - Prompt: {usage.get('prompt_tokens', 0)}, Completion: {usage.get('completion_tokens', 0)}")
        
        return answer
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

def process_multiple_questions(questions: List[str], collection) -> Dict[str, str]:
    """Process multiple questions and return answers"""
    results = {}
    
    for q in questions:
        # Get embeddings and retrieve chunks
        q_emb = embedding_model.encode([q]).tolist()
        res = collection.query(q_emb, n_results=5)
        
        chunks = [
            {
                "page": m.get("page", 1),
                "text": d,
                "doc": m.get("doc", "unknown")
            }
            for d, m in zip(res["documents"][0], res["metadatas"][0])
        ]
        
        # Get answer
        answer = llm_query(q, chunks)
        results[q] = answer
    
    return results

# ================= SESSION STATE =================
if "ready" not in st.session_state:
    st.session_state.ready = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ================= UI =================
st.header("📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload multiple documents",
    type=["pdf", "docx", "doc", "pptx", "ppt", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files and st.button("🚀 Process All Documents"):
    with st.spinner("Processing documents..."):
        all_pages = []
        all_images = []
        
        progress_bar = st.progress(0)
        
        for idx, uploaded in enumerate(uploaded_files):
            st.info(f"Processing: {uploaded.name}")
            
            file_bytes = uploaded.read()
            
            # Extract text
            pages = extract_text_from_file(file_bytes, uploaded.name)
            
            # Add document name to each page
            for p in pages:
                p["doc"] = uploaded.name
            
            all_pages.extend(pages)
            
            # Extract images from PDFs
            if uploaded.name.lower().endswith('.pdf'):
                images = extract_images_from_pdf(file_bytes, uploaded.name)
                all_images.extend(images)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # OCR on images
        for img in all_images:
            text = ocr_image_text(img["path"])
            if text:
                all_pages.append({
                    "page": img["page"],
                    "text": f"[Image OCR] {text}",
                    "doc": img["doc"]
                })
        
        # Chunk all pages
        chunks = chunk_pages(all_pages)
        
        if chunks:
            texts = [c["text"] for c in chunks]
            embeddings = embedding_model.encode(texts).tolist()
            
            # Recreate collection
            try:
                chroma_client.delete_collection("syllabus")
            except:
                pass
            
            col = chroma_client.create_collection("syllabus")
            col.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=[{"page": c["page"], "doc": c.get("doc", "unknown")} for c in chunks],
                ids=[f"c{i}" for i in range(len(texts))]
            )
            
            st.session_state.ready = True
            st.session_state.processed_files = [f.name for f in uploaded_files]
            
            st.success(f"✅ Processed {len(uploaded_files)} documents with {len(chunks)} chunks")
            st.info(f"📄 Files: {', '.join(st.session_state.processed_files)}")
        else:
            st.error("No content extracted from documents")

# ================= QUERY =================
if st.session_state.ready:
    st.divider()
    st.header("💬 Ask Questions")
    
    st.info("💡 Tip: You can ask multiple questions by separating them with new lines or question marks")
    
    user_input = st.text_area(
        "Enter your question(s)",
        height=150,
        placeholder="Example:\nWhat are the course objectives?\nWhen is the final exam?\nWhat textbook is required?"
    )
    
    if st.button("🔍 Get Answers") and user_input:
        questions = parse_multiple_questions(user_input)
        
        st.subheader(f"📝 Found {len(questions)} question(s)")
        
        col = chroma_client.get_collection("syllabus")
        
        with st.spinner("Retrieving answers..."):
            results = process_multiple_questions(questions, col)
        
        # Display results
        for i, (question, answer) in enumerate(results.items(), 1):
            with st.expander(f"❓ Question {i}: {question}", expanded=True):
                st.markdown(f"**Answer:**\n\n{answer}")
                st.divider()

# ================= RESET =================
st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Reset Database"):
        try:
            chroma_client.delete_collection("syllabus")
        except:
            pass
        shutil.rmtree(IMAGE_DIR, ignore_errors=True)
        os.makedirs(IMAGE_DIR, exist_ok=True)
        st.session_state.ready = False
        st.session_state.processed_files = []
        st.success("Reset complete")

with col2:
    if st.session_state.ready:
        st.metric("Documents Loaded", len(st.session_state.processed_files))