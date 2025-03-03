import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import tempfile
from fpdf import FPDF
from groq import Groq
from functools import lru_cache

# Set the page config to add title and favicon
st.set_page_config(page_title="AI MCQ Generation from Document", page_icon="🤖", layout="wide")

# Cache the embeddings model to avoid reloading
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sidebar configuration
with st.sidebar:
    st.header("📁 Upload Your Document")
    uploaded_file = st.file_uploader("📥 Upload your PDF or Word document", type=["pdf", "docx"])
    
    st.markdown("---")
    
    st.header("🔑 API Configuration")
    groq_api_token = st.text_input("🔑 Enter your Groq API token:", type="password")
    
    # Add caching session state for the API token
    if groq_api_token:
        st.session_state.groq_api_token = groq_api_token
    elif 'groq_api_token' in st.session_state:
        groq_api_token = st.session_state.groq_api_token

# Main UI Section
st.title("🤖 AI MCQ Generation from Document 📚")
st.markdown("Generate **Multiple Choice Questions (MCQs)** from tech-related documents like PDFs or Word files using AI models. Customize the difficulty level and number of questions, and download the generated MCQs in PDF format!")

# MCQ prompt template
MCQ_TEMPLATE = """
You are an MCQ Generation AI, specialized in creating multiple-choice questions (MCQs) from provided PDF documents. Your sole function is to generate relevant, clear, and accurate MCQs based exclusively on the content of the PDF provided by the user.

Role:
- Serve as a specialized AI that extracts information from a PDF and formulates it into MCQs at varying levels of difficulty (Easy, Medium, Hard), based only on the data in the provided document.

Specific Instructions:
- Easy Level: Generate basic recall questions that test foundational knowledge, definitions, or simple concepts directly stated in the text. 
- Medium Level: Create questions that require a moderate understanding of the content, involving reasoning, comprehension, or application of information.
- Hard Level: Formulate challenging questions that require deeper analysis, critical thinking, or the integration of multiple concepts found in the document.

MCQ Structure:
- Each MCQ has four options: one correct answer and three plausible distractors.
- Format the MCQs as follows:
  1. Question text?
     - A) Option 1
     - B) Option 2
     - C) Option 3
     - D) Option 4
            Correct Answer: (e.g., A)   

Context: {context}
Human: Generate {num_questions} MCQs at {difficulty_level} difficulty based on the given document content.
AI:
"""

# Function to classify document content as tech-related
def is_tech_related(text, api_key):
    client = Groq(api_key=api_key)
    
    # Use a shorter prompt and only first 1000 characters to save tokens
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Is this content tech-related? Answer only yes or no: {text[:1000]}"
            }
        ],
        model="llama-3.3-70b-versatile"
    )
    
    result = response.choices[0].message.content.strip().lower()
    return "yes" in result, result

# Function to generate MCQs based on difficulty level and content
def generate_mcqs(doc_chunks, difficulty_level, num_questions, selected_model, api_key):
    prompt = PromptTemplate(
        input_variables=["context", "difficulty_level", "num_questions"], 
        template=MCQ_TEMPLATE
    )

    # Create vector store from document chunks
    embeddings = get_embeddings()
    faiss_index = FAISS.from_documents(documents=doc_chunks, embedding=embeddings)
    retriever = faiss_index.as_retriever()
    
    # Get context from the document
    context_from_doc = retriever.invoke('Provide diverse content from the entire document.')
    
    formatted_prompt = prompt.format(
        context=context_from_doc,
        difficulty_level=difficulty_level,
        num_questions=num_questions
    )

    # Generate MCQs using Groq API
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": formatted_prompt}],
        model=selected_model
    )
    
    return response.choices[0].message.content.strip()

# Function to convert MCQs to PDF
def convert_to_pdf(mcqs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Generated MCQs", ln=True, align='C')
    pdf.ln(5)
    
    # Add content
    pdf.set_font("Arial", size=12)
    for line in mcqs.split("\n"):
        # Handle long lines with proper wrapping
        pdf.multi_cell(190, 10, txt=line, align='L')
    
    return pdf.output(dest='S').encode('latin1')

# Function to load and process document
def process_document(file_path, file_type):
    # Load the document based on its type
    doc_loader = PyPDFLoader(file_path) if file_type == "application/pdf" else UnstructuredWordDocumentLoader(file_path)
    data = doc_loader.load()
    
    # Split into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return text_splitter.split_documents(data)

# Main app functionality
if uploaded_file and groq_api_token:
    with st.spinner("Processing the document..."):
        try:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            # Process the document
            chunks = process_document(temp_file_path, uploaded_file.type)
            
            # Cache the chunks in session state to avoid reprocessing
            st.session_state.document_chunks = chunks
            
            # Combine the text chunks for tech-related check
            combined_text = " ".join([chunk.page_content for chunk in chunks])
            
            # Check if document is tech-related
            with st.status("Checking if document is tech-related..."):
                is_tech, confidence = is_tech_related(combined_text, groq_api_token)
            
            if not is_tech:
                st.warning("⚠️ The document is not tech-related. Please upload a tech-related document.")
            else:
                st.success("✅ Document processed successfully! The document is tech-related.")
                
                # Options for MCQ generation
                col1, col2 = st.columns(2)
                with col1:
                    difficulty_level = st.selectbox("🎯 Select difficulty level:", ["Easy", "Medium", "Hard"])
                    num_questions = st.slider("⚖️ Select the number of questions:", min_value=5, max_value=30, step=5, value=10)
                
                with col2:
                    selected_model = st.selectbox(
                        "🧠 Select Groq Model:", 
                        [
                            "llama-3.3-70b-versatile", 
                            "llama3-70b-8192", 
                            "llama3-8b-8192", 
                            "gemma2-9b-it"
                        ]
                    )
                
                # Generate MCQs button
                if st.button("✨ Generate MCQs", type="primary"):
                    with st.status("Generating MCQs...") as status:
                        mcqs = generate_mcqs(chunks, difficulty_level, num_questions, selected_model, groq_api_token)
                        status.update(label="MCQs generated successfully!", state="complete")
                    
                    # Store in session state
                    st.session_state.generated_mcqs = mcqs
                    
                    # Display the generated MCQs
                    st.subheader("📜 Generated MCQs:")
                    with st.expander("View all MCQs", expanded=True):
                        st.markdown(mcqs)
                    
                    # Downloadable PDF
                    st.subheader("💾 Export MCQs:")
                    pdf_data = convert_to_pdf(mcqs)
                    filename = f"MCQs_{difficulty_level}_{num_questions}.pdf"
                    st.download_button(
                        "📄 Download MCQs as PDF",
                        data=pdf_data,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        finally:
            # Clean up temporary file
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

else:
    # Display instruction placeholder
    st.info("👈 Please upload a document and provide your Groq API token in the sidebar to get started.")
    
    if not uploaded_file:
        st.warning("⚠️ No document uploaded yet.")
    
    if not groq_api_token:
        st.warning("⚠️ Groq API token is required for generating MCQs.")
    
    # Show example output
    with st.expander("📝 Example Output"):
        st.markdown("""
        1. What is the primary purpose of the AI MCQ Generation tool?
           - A) To create presentation slides
           - B) To generate multiple choice questions from documents
           - C) To translate documents into different languages
           - D) To summarize PDF content
                Correct Answer: B
        """)
