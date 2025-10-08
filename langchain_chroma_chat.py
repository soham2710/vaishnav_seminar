import os
import argparse
import gc
from typing import List

# Document reading
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# LangChain + Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Transformers LLM
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# Gradio UI
import gradio as gr

# -------------------- CONFIG --------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-large"
INDEX_DIR = "chroma_index"
PDF_PATH = "docs/attention_is_all_you_need.pdf"  # Update this path
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# -------------------- DOCUMENT PROCESSING --------------------
def load_pdf_efficiently(file_path: str) -> List[Document]:
    """Load PDF page by page and split into chunks efficiently."""
    if PdfReader is None:
        raise RuntimeError("pypdf is required. Install with: pip install pypdf")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")
    
    print(f"Loading PDF: {file_path}")
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_documents = []
    
    # Process in batches of pages to manage memory
    batch_size = 10
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        print(f"Processing pages {batch_start + 1}-{batch_end}...")
        
        # Extract text from batch of pages
        batch_text = []
        for i in range(batch_start, batch_end):
            page_text = reader.pages[i].extract_text()
            if page_text and page_text.strip():
                batch_text.append(page_text)
        
        # Combine batch pages and split into chunks
        if batch_text:
            combined_text = "\n".join(batch_text)
            chunks = text_splitter.split_text(combined_text)
            
            # Create documents with metadata
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(file_path),
                        "page_range": f"{batch_start + 1}-{batch_end}",
                        "chunk": len(all_documents) + chunk_idx
                    }
                )
                all_documents.append(doc)
            
            del combined_text, chunks
        
        del batch_text
        gc.collect()
    
    print(f"✓ Created {len(all_documents)} chunks from PDF")
    return all_documents


# -------------------- INDEXING (CHROMA) --------------------
def build_chroma_index(pdf_path: str, embedding_model_name: str = EMBEDDING_MODEL, 
                       index_path: str = INDEX_DIR) -> Chroma:
    """Build Chroma index from PDF."""
    print("\n=== Building Vector Store ===")
    
    # Load and split PDF
    documents = load_pdf_efficiently(pdf_path)
    
    # Initialize embeddings
    print("\nInitializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Remove old index if exists
    if os.path.exists(index_path):
        import shutil
        shutil.rmtree(index_path)
        print("Removed old index.")
    
    # Create vector store in batches
    print("\nCreating vector embeddings (this may take a few minutes)...")
    batch_size = 50
    vectorstore = None
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}...")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=index_path
            )
        else:
            vectorstore.add_documents(batch)
        
        gc.collect()
    
    vectorstore.persist()
    print(f"\n✓ Vector store created successfully with {len(documents)} chunks!")
    print(f"✓ Saved to: {index_path}\n")
    
    return vectorstore


def load_chroma_index(index_path: str = INDEX_DIR, embedding_model_name: str = EMBEDDING_MODEL) -> Chroma:
    """Load existing Chroma index."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found at '{index_path}'. Please run with --reindex first."
        )
    
    print("Loading existing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = Chroma(persist_directory=index_path, embedding_function=embeddings)
    print("✓ Vector store loaded successfully!\n")
    
    return vectorstore


# -------------------- LLM WRAPPER --------------------
def get_hf_llm_pipeline(model_name: str = LLM_MODEL):
    """Initialize Hugging Face LLM pipeline."""
    print(f"Loading language model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # Use CPU (-1), change to 0 for GPU
        max_length=512,
        do_sample=False,
    )
    print("✓ Language model loaded!\n")
    return HuggingFacePipeline(pipeline=pipe)


# -------------------- CHAT LOGIC --------------------
class LocalChatAssistant:
    def __init__(self, reindex=False):
        # Build or load index
        if reindex or not os.path.exists(INDEX_DIR):
            self.vectorstore = build_chroma_index(PDF_PATH)
        else:
            self.vectorstore = load_chroma_index()

        # Initialize LLM
        self.llm = get_hf_llm_pipeline()
        
        # Create retriever and QA chain
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        
        print("=== Chat Assistant Ready! ===\n")

    def query(self, question: str):
        """Process user question and return answer."""
        if not question.strip():
            return "Please enter a question.", []
        
        try:
            result = self.qa_chain({"query": question})
            answer = result["result"]
            sources = result.get("source_documents", [])
            
            # Format source information
            source_info = []
            for i, doc in enumerate(sources, 1):
                source_info.append(
                    f"**Source {i}** (Pages: {doc.metadata.get('page_range', 'N/A')})\n"
                    f"{doc.page_content[:200]}..."
                )
            
            return answer, source_info
            
        except Exception as e:
            return f"Error processing query: {str(e)}", []


# -------------------- GRADIO UI --------------------
def launch_ui(assistant: LocalChatAssistant):
    """Launch Gradio web interface."""
    
    def chat_fn(message, history):
        history = history or []
        answer, sources = assistant.query(message)
        
        # Format response with sources
        response = answer
        if sources:
            response += "\n\n---\n**Sources:**\n" + "\n\n".join(sources)
        
        history.append((message, response))
        return history, history

    with gr.Blocks(title="Attention Paper AI Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📄 "Attention Is All You Need" - AI Assistant
            
            Ask questions about the Transformer architecture paper!
            
            *Powered by RAG (Retrieval Augmented Generation) with local LLM*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500, label="Chat")
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about transformers, attention mechanisms, etc...",
                        label="Your Question",
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear = gr.Button("Clear Chat")
        
        with gr.Row():
            gr.Markdown(
                """
                ### Example Questions:
                - What is the Transformer architecture?
                - Explain the attention mechanism
                - What are the key components of the encoder?
                - How does multi-head attention work?
                - What are the advantages over RNNs?
                """
            )
        
        # Event handlers
        msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot])
        submit_btn.click(chat_fn, [msg, chatbot], [chatbot, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    print("\n🚀 Launching web interface...")
    print("📍 Open your browser at: http://127.0.0.1:7860")
    print("Press Ctrl+C to stop\n")
    
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


# -------------------- MAIN --------------------
def main(reindex=False):
    """Main application entry point."""
    print("\n" + "="*60)
    print("  📚 RAG-based PDF Chat Assistant")
    print("  Paper: Attention Is All You Need (Vaswani et al., 2017)")
    print("="*60 + "\n")
    
    assistant = LocalChatAssistant(reindex=reindex)
    launch_ui(assistant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat with 'Attention Is All You Need' paper using RAG"
    )
    parser.add_argument(
        "--reindex", 
        action="store_true", 
        help="Rebuild vector store from PDF (required on first run)"
    )
    args = parser.parse_args()
    
    main(reindex=args.reindex)