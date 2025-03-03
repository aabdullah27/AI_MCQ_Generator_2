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
import time

# --- App Configuration ---
st.set_page_config(
    page_title="AI MCQ Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
if "processed_chunks" not in st.session_state:
    st.session_state.processed_chunks = None
if "is_tech_related" not in st.session_state:
    st.session_state.is_tech_related = None
if "generated_mcqs" not in st.session_state:
    st.session_state.generated_mcqs = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

# --- Constants ---
MODELS = {
    "LLaMA 3.3 70B": "llama-3.3-70b-versatile",
    "LLaMA 3 70B": "llama3-70b-8192",
    "LLaMA 3 8B": "llama3-8b-8192",
    "Gemmma 9b": "gemma2-9b-it"
}

# --- Utility Functions ---

def load_document(file_path, file_type):
    """Load document based on file type"""
    try:
        if file_type == "application/pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredWordDocumentLoader(file_path)
        
        return loader.load()
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def split_document(documents):
    """Split documents into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    """Create a vector store from document chunks"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents=chunks, embedding=embeddings)

def check_tech_content(text, api_key):
    """Check if document is tech-related"""
    if not api_key:
        st.warning("API key is required for classification.")
        return False, "No API key provided"
    
    try:
        # Initialize the Groq client with API key
        client = Groq(api_key=api_key)
        
        # Use a shorter sample for classification
        sample_text = text[:2000]
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a document classifier. Respond with only 'YES' or 'NO'."
                },
                {
                    "role": "user",
                    "content": f"Is the following document content tech-related?\n\n{sample_text}"
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        return "YES" in result, result
    
    except Exception as e:
        st.error(f"Error during tech classification: {str(e)}")
        return False, f"Error: {str(e)}"

def generate_mcqs(chunks, settings, api_key):
    """Generate MCQs using Groq API"""
    if not chunks or not api_key:
        return "No document chunks or API key provided."
    
    try:
        # Create vector store and retriever
        vector_store = create_vector_store(chunks)
        retriever = vector_store.as_retriever()
        
        # Retrieve relevant content
        document_context = retriever.invoke("Extract comprehensive information from the document")
        
        # Format as a string if it's a list
        if isinstance(document_context, list):
            document_context = "\n\n".join([doc.page_content for doc in document_context])
        
        # Create MCQ generation prompt
        prompt_template = """
        You are an expert in creating multiple-choice questions (MCQs) from educational content.
        
        Create {num_questions} {difficulty} level MCQs based on the following content.
        
        For each MCQ:
        1. Write a clear question
        2. Provide four options (A, B, C, D) where only one is correct
        3. Indicate the correct answer
        4. Ensure questions are diverse and cover different concepts from the document
        
        Document content:
        {context}
        
        Format each question as:
        Q1. [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        Correct Answer: [Letter]
        
        Start generating the MCQs now.
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "difficulty", "num_questions"],
            template=prompt_template
        )
        
        # Create formatted prompt
        formatted_prompt = prompt.format(
            context=document_context,
            difficulty=settings["difficulty"],
            num_questions=settings["num_questions"]
        )
        
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Generate MCQs
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            model=settings["model"],
            temperature=0.7,
            max_tokens=4000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"Error generating MCQs: {str(e)}")
        return f"Error generating MCQs: {str(e)}"

def create_pdf(mcqs, settings):
    """Convert MCQs to PDF format"""
    pdf = FPDF()
    
    # Add a cover page
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Multiple Choice Questions (MCQs)", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Difficulty: {settings['difficulty']}", ln=True)
    pdf.cell(0, 10, f"Number of Questions: {settings['num_questions']}", ln=True)
    pdf.cell(0, 10, f"Date Generated: {time.strftime('%Y-%m-%d')}", ln=True)
    
    # Add MCQs content
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    
    # Clean up and process MCQ text for PDF
    lines = mcqs.split("\n")
    for line in lines:
        # Make question numbers bold
        if line.strip().startswith("Q") and "." in line[:4]:
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(0, 10, line)
            pdf.set_font("Arial", "", 12)
        # Make correct answers bold
        elif line.strip().startswith("Correct Answer:"):
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(0, 10, line)
            pdf.set_font("Arial", "", 12)
        # Regular line
        else:
            pdf.multi_cell(0, 10, line)
    
    return pdf.output(dest="S").encode("latin1")

# --- UI Components ---

def render_sidebar():
    """Render the sidebar UI"""
    st.sidebar.title("📚 MCQ Generator")
    st.sidebar.markdown("---")
    
    # File uploader
    st.sidebar.header("1️⃣ Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF or Word document",
        type=["pdf", "docx"],
        help="The document should be tech-related"
    )
    
    # API configuration
    st.sidebar.header("2️⃣ API Configuration")
    api_key = st.sidebar.text_input(
        "Enter your Groq API key",
        type="password",
        help="Required for document classification and MCQ generation"
    )
    
    # MCQ settings
    st.sidebar.header("3️⃣ MCQ Settings")
    difficulty = st.sidebar.select_slider(
        "Select difficulty level",
        options=["Easy", "Medium", "Hard"],
        value="Medium"
    )
    
    num_questions = st.sidebar.slider(
        "Number of questions",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )
    
    model = st.sidebar.selectbox(
        "Select AI model",
        options=list(MODELS.keys()),
        format_func=lambda x: x,
        index=1
    )
    
    # Process document button
    process_button = st.sidebar.button(
        "Process Document",
        type="primary",
        disabled=(not uploaded_file or not api_key),
        use_container_width=True
    )
    
    # Return all sidebar inputs
    return {
        "uploaded_file": uploaded_file,
        "api_key": api_key,
        "settings": {
            "difficulty": difficulty,
            "num_questions": num_questions,
            "model": MODELS[model]
        },
        "process_button": process_button
    }

def render_main_content():
    """Render the main content area"""
    st.title("🤖 AI MCQ Generator")
    st.markdown(
        """
        Generate high-quality **Multiple Choice Questions (MCQs)** from your technical documents.
        Perfect for educators, trainers, and students to create assessments and study materials.
        """
    )
    
    # Display tabs
    tab1, tab2 = st.tabs(["📝 Generate MCQs", "ℹ️ About"])
    
    with tab1:
        # This will contain the main generation UI
        pass
    
    with tab2:
        st.markdown(
            """
            ### About the AI MCQ Generator
            
            This application helps you generate multiple-choice questions from your technical documents using AI.
            
            **Features:**
            - Upload PDF or Word documents
            - Choose difficulty level (Easy, Medium, Hard)
            - Select the number of questions
            - Download results as a PDF
            
            **How it works:**
            1. Upload your technical document
            2. Enter your Groq API key
            3. Configure MCQ settings
            4. Process the document
            5. Generate and download MCQs
            
            **Requirements:**
            - A valid Groq API key
            - A technical document (PDF or DOCX)
            """
        )
    
    return tab1

def process_document_workflow(inputs, tab1):
    """Handle the document processing workflow"""
    if inputs["process_button"]:
        with tab1:
            with st.status("Processing document...", expanded=True) as status:
                # Step 1: Save uploaded file to temp location
                status.update(label="Saving uploaded file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(inputs["uploaded_file"].name)[1]) as temp_file:
                    temp_file.write(inputs["uploaded_file"].read())
                    temp_file_path = temp_file.name
                
                try:
                    # Step 2: Load the document
                    status.update(label="Loading document...")
                    documents = load_document(temp_file_path, inputs["uploaded_file"].type)
                    
                    if not documents:
                        status.update(label="Failed to load document", state="error")
                        return
                    
                    # Step 3: Split into chunks
                    status.update(label="Splitting document into chunks...")
                    chunks = split_document(documents)
                    
                    # Store chunks in session state
                    st.session_state.processed_chunks = chunks
                    
                    # Step 4: Check if tech-related
                    status.update(label="Checking if document is tech-related...")
                    combined_text = " ".join([chunk.page_content for chunk in chunks])
                    
                    is_tech, tech_response = check_tech_content(combined_text, inputs["api_key"])
                    st.session_state.is_tech_related = is_tech
                    
                    if is_tech:
                        status.update(label="Document processed successfully!", state="complete")
                        st.session_state.document_processed = True
                    else:
                        status.update(label="Document is not tech-related", state="error")
                        st.session_state.document_processed = False
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            
            # Display appropriate message based on tech classification
            if st.session_state.is_tech_related:
                st.success("✅ Document is tech-related and ready for MCQ generation")
                
                # Show generate button
                if st.button("Generate MCQs", type="primary", use_container_width=True):
                    with st.spinner("Generating MCQs..."):
                        mcqs = generate_mcqs(
                            st.session_state.processed_chunks,
                            inputs["settings"],
                            inputs["api_key"]
                        )
                        
                        if mcqs and not mcqs.startswith("Error"):
                            st.session_state.generated_mcqs = mcqs
                            st.success("MCQs generated successfully!")
                            
                            # Display the MCQs
                            with st.expander("Preview Generated MCQs", expanded=True):
                                st.markdown(mcqs)
                            
                            # Create download button for PDF
                            pdf_data = create_pdf(mcqs, inputs["settings"])
                            
                            # File name with metadata
                            file_name = f"MCQs_{inputs['settings']['difficulty']}_{inputs['settings']['num_questions']}.pdf"
                            
                            st.download_button(
                                label="Download MCQs as PDF",
                                data=pdf_data,
                                file_name=file_name,
                                mime="application/pdf",
                                use_container_width=True
                            )
                        else:
                            st.error(f"Failed to generate MCQs: {mcqs}")
            else:
                if st.session_state.document_processed is False:
                    st.error("❌ The document is not tech-related. Please upload a technical document.")

def render_initial_state(tab1):
    """Render initial state when no document is processed"""
    with tab1:
        st.info("👈 Upload a document and enter your Groq API key in the sidebar to get started.")
        
        # Show example of what will be generated
        with st.expander("See example MCQs"):
            st.markdown("""
            ### Example MCQs
            
            Q1. What is the primary purpose of a recursive algorithm?
            A) To execute code in parallel
            B) To solve problems by breaking them into smaller instances of the same problem
            C) To organize code into modular functions
            D) To improve code readability
            Correct Answer: B
            
            Q2. Which of the following is a characteristic of object-oriented programming?
            A) Sequential execution
            B) Absence of variables
            C) Encapsulation
            D) Single-threaded execution
            Correct Answer: C
            """)

# --- Main Application Flow ---

def main():
    # Render sidebar and get inputs
    inputs = render_sidebar()
    
    # Render main content area and get tab reference
    tab1 = render_main_content()
    
    # Check if document has been processed
    if st.session_state.document_processed:
        # If previously processed, show generate button
        with tab1:
            st.success("✅ Document is ready for MCQ generation")
            
            if st.button("Generate MCQs", type="primary", use_container_width=True):
                with st.spinner("Generating MCQs..."):
                    mcqs = generate_mcqs(
                        st.session_state.processed_chunks,
                        inputs["settings"],
                        inputs["api_key"]
                    )
                    
                    if mcqs and not mcqs.startswith("Error"):
                        st.session_state.generated_mcqs = mcqs
                        
                        # Display the MCQs
                        with st.expander("Preview Generated MCQs", expanded=True):
                            st.markdown(mcqs)
                        
                        # Create download button for PDF
                        pdf_data = create_pdf(mcqs, inputs["settings"])
                        file_name = f"MCQs_{inputs['settings']['difficulty']}_{inputs['settings']['num_questions']}.pdf"
                        
                        st.download_button(
                            label="Download MCQs as PDF",
                            data=pdf_data,
                            file_name=file_name,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Failed to generate MCQs: {mcqs}")
    
    # Process document if button is clicked
    elif inputs["process_button"]:
        process_document_workflow(inputs, tab1)
    
    # Otherwise show initial state
    else:
        render_initial_state(tab1)
    
    # Display generated MCQs if they exist
    if st.session_state.generated_mcqs and not inputs["process_button"]:
        with tab1:
            st.subheader("Previously Generated MCQs")
            with st.expander("View MCQs", expanded=False):
                st.markdown(st.session_state.generated_mcqs)
                
                # Recreate PDF download button
                pdf_data = create_pdf(st.session_state.generated_mcqs, inputs["settings"])
                file_name = f"MCQs_{inputs['settings']['difficulty']}_{inputs['settings']['num_questions']}.pdf"
                
                st.download_button(
                    label="Download MCQs as PDF",
                    data=pdf_data,
                    file_name=file_name,
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
