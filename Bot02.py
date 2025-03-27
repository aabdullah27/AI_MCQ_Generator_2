import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import tempfile
from fpdf import FPDF
from groq import Groq

# Set the page config
st.set_page_config(
    page_title="AI MCQ Generator",
    page_icon="🤖",
)

# Initialize session state variables
def initialize_session_state():
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'document_chunks' not in st.session_state:
        st.session_state.document_chunks = None
    if 'mcqs_generated' not in st.session_state:
        st.session_state.mcqs_generated = None
    if 'processing_error' not in st.session_state:
        st.session_state.processing_error = None

initialize_session_state()

# Display title and description
st.title("🤖 AI MCQ Generator from Technical Documents")
st.markdown("""
Generate high-quality multiple choice questions from your technical documents using AI.
Upload a PDF or Word document, customize your settings, and download ready-to-use MCQs!
""")

# Create sidebar
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a technical document", type=["pdf", "docx"])
    
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter your Groq API key", type="password")
    
    st.header("⚙️ MCQ Settings")
    difficulty = st.select_slider(
        "Difficulty level",
        options=["Easy", "Medium", "Hard"],
        value="Medium"
    )
    
    num_questions = st.slider(
        "Number of questions",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )
    
    MODEL_OPTIONS = {
        "LLaMA 3.3 70B": "llama-3.3-70b-versatile",
        "LLaMA 3 70B": "llama3-70b-8192",
        "LLaMA 3 8B": "llama3-8b-8192",
        "Gemma 9B": "gemma2-9b-it"
    }
    
    model_name = st.selectbox(
        "AI model",
        options=list(MODEL_OPTIONS.keys())
    )
    model_id = MODEL_OPTIONS[model_name]

def process_document(file_path, file_type):
    """Process the uploaded document and split it into chunks"""
    try:
        # Load document based on file type
        if file_type == "application/pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredWordDocumentLoader(file_path)
        
        # Load and split the document
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    except Exception as e:
        st.session_state.processing_error = f"Error processing document: {str(e)}"
        return None

def check_tech_content(text, api_key):
    """Check if the document is tech-related using Groq API"""
    try:
        if not api_key:
            st.session_state.processing_error = "API key is required"
            return False, "Missing API key"
            
        # Initialize Groq client with just the API key
        client = Groq(api_key=api_key)
        sample = text[:1500]  # Use a sample for efficiency
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a document classifier. Answer with ONLY 'YES' or 'NO'."
                },
                {
                    "role": "user",
                    "content": f"Is the following document tech-related? Answer with ONLY 'YES' or 'NO'.\n\n{sample}"
                }
            ],
            temperature=0.1,
            max_tokens=7
        )
        
        result = response.choices[0].message.content.strip().upper()
        return "YES" in result, result
    except Exception as e:
        st.session_state.processing_error = f"Error checking document content: {str(e)}"
        return False, str(e)

def generate_mcqs(chunks, difficulty, num_questions, model_id, api_key):
    """Generate MCQs from document chunks using Groq API"""
    try:
        if not api_key:
            st.session_state.processing_error = "API key is required"
            return None
            
        # Create vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
        
        # Retrieve relevant content from document
        retriever = vector_store.as_retriever()
        context_docs = retriever.get_relevant_documents("Major Points discussed in the document")
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create prompt template
        prompt_template = """
        You are an expert in creating multiple-choice questions (MCQs) for technical content.
        
        Create {num_questions} {difficulty} level MCQs based on the following technical content.
        
        For each MCQ:
        1. Write a clear question
        2. Provide four options (A, B, C, D) where only one is correct
        3. Mark the correct answer
        4. Ensure questions cover different concepts from the document
        5. Make the questions challenging but fair
        
        Document content:
        {context}
        
        Format each question as:
        Question 1: [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        Correct Answer: [Letter]
        
        [blank line]
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "difficulty", "num_questions"]
        )
        
        formatted_prompt = prompt.format(
            context=context,
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        # Initialize Groq client with just the API key
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.5,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.session_state.processing_error = f"Error generating MCQs: {str(e)}"
        return None

def create_pdf(mcqs, metadata):
    """Create a PDF document with the generated MCQs"""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add title page
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Multiple Choice Questions", ln=True, align="C")
        pdf.ln(10)
        
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Difficulty Level: {metadata['difficulty']}", ln=True)
        pdf.cell(0, 10, f"Number of Questions: {metadata['num_questions']}", ln=True)
        pdf.cell(0, 10, f"Generated using: {metadata['model_name']}", ln=True)
        pdf.ln(20)
        
        # Add MCQs
        pdf.add_page()
        pdf.set_font("Arial", "", 12)
        
        # Process text for better PDF formatting
        lines = mcqs.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                pdf.ln(5)
                continue
                
            # Make question text bold
            if line.startswith("Question"):
                pdf.set_font("Arial", "B", 12)
                pdf.multi_cell(0, 8, line)
                pdf.set_font("Arial", "", 12)
            # Make correct answer bold
            elif line.startswith("Correct Answer"):
                pdf.set_font("Arial", "B", 12)
                pdf.multi_cell(0, 8, line)
                pdf.set_font("Arial", "", 12)
                pdf.ln(5)  # Extra space after each question
            # Options
            elif line.startswith(("A)", "B)", "C)", "D)")):
                pdf.multi_cell(0, 8, line)
            # Normal text
            else:
                pdf.multi_cell(0, 8, line)
        
        return pdf.output(dest="S").encode("latin1")
    except Exception as e:
        st.session_state.processing_error = f"Error creating PDF: {str(e)}"
        return None

def display_sample_output():
    """Display sample output in expander"""
    with st.expander("See example output", expanded=False):
        st.markdown("""
        ### Example MCQs
        
        Question 1: What is the primary purpose of a database index?
        A) To store data in alphabetical order
        B) To speed up data retrieval operations
        C) To compress the database size
        D) To encrypt sensitive information
        Correct Answer: B
        
        Question 2: Which programming paradigm emphasizes the use of functions and avoids changing state?
        A) Procedural programming
        B) Object-oriented programming
        C) Functional programming
        D) Event-driven programming
        Correct Answer: C
        """)

def main():
    # Main app flow
    if uploaded_file and api_key:
        # Show process button only if file and API key are provided
        if st.button("Process Document", use_container_width=True):
            with st.status("Processing document...", expanded=True) as status:
                # Step 1: Save file to temp location
                status.update(label="Saving uploaded file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                
                try:
                    # Step 2: Process document
                    status.update(label="Processing document...")
                    chunks = process_document(temp_file_path, uploaded_file.type)
                    
                    if not chunks:
                        status.update(label="Failed to process document", state="error")
                        if st.session_state.processing_error:
                            st.error(st.session_state.processing_error)
                    else:
                        # Step 3: Check if tech-related
                        status.update(label="Verifying document content...")
                        combined_text = " ".join([chunk.page_content for chunk in chunks])
                        is_tech, tech_result = check_tech_content(combined_text, api_key)
                        
                        if is_tech:
                            status.update(label="Document processed successfully!", state="complete")
                            st.session_state.document_processed = True
                            st.session_state.document_chunks = chunks
                            st.session_state.processing_error = None
                        else:
                            status.update(label="Document is not tech-related", state="error")
                            st.error("The uploaded document doesn't appear to be tech-related. Please upload a technical document.")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

        # Display MCQ generation section if document is processed
        if st.session_state.document_processed:
            st.success("✅ Document processed successfully! Ready to generate MCQs.")
            
            if st.button("Generate MCQs", type="primary", use_container_width=True):
                with st.spinner("Generating MCQs..."):
                    mcqs = generate_mcqs(
                        st.session_state.document_chunks,
                        difficulty,
                        num_questions,
                        model_id,
                        api_key
                    )
                    
                    if mcqs:
                        st.session_state.mcqs_generated = mcqs
                        
                        # Display the generated MCQs
                        st.subheader("Generated MCQs")
                        with st.expander("View MCQs", expanded=True):
                            st.markdown(mcqs)
                        
                        # Create PDF for download
                        metadata = {
                            "difficulty": difficulty,
                            "num_questions": num_questions,
                            "model_name": model_name
                        }
                        
                        pdf_data = create_pdf(mcqs, metadata)
                        
                        if pdf_data:
                            st.download_button(
                                label="Download MCQs as PDF",
                                data=pdf_data,
                                file_name=f"MCQs_{difficulty}_{num_questions}q.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
            
            # If MCQs were previously generated, show them
            elif st.session_state.mcqs_generated:
                st.subheader("Previously Generated MCQs")
                with st.expander("View MCQs", expanded=True):
                    st.markdown(st.session_state.mcqs_generated)
                
                # Create PDF for download (reusing previous MCQs)
                metadata = {
                    "difficulty": difficulty,
                    "num_questions": num_questions,
                    "model_name": model_name
                }
                
                pdf_data = create_pdf(st.session_state.mcqs_generated, metadata)
                
                if pdf_data:
                    st.download_button(
                        label="Download MCQs as PDF",
                        data=pdf_data,
                        file_name=f"MCQs_{difficulty}_{num_questions}q.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    else:
        # Display instruction if either file or API key is missing
        if not uploaded_file:
            st.info("👈 Please upload a technical document (PDF or DOCX) in the sidebar.")
        
        if not api_key:
            st.info("👈 Please enter your Groq API key in the sidebar.")
        
        display_sample_output()
    
    # Show any processing errors
    if st.session_state.processing_error:
        st.error(st.session_state.processing_error)

if __name__ == "__main__":
    main()
