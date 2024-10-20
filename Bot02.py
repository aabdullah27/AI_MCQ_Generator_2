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

# Set the page config to add title and favicon
st.set_page_config(page_title="AI MCQ Generation from Document", page_icon="🤖", layout="wide")

# Sidebar - File Upload Section
st.sidebar.header("📁 Upload Your Document")
uploaded_file = st.sidebar.file_uploader("📥 Upload your PDF or Word document", type=["pdf", "docx"])
st.sidebar.markdown("---")

# Sidebar - Groq API Key Input
st.sidebar.header("🔑 API Configuration")
groq_api_token = st.sidebar.text_input("🔑 Enter your Groq API token:", type="password")
st.sidebar.markdown("---")

# Main UI Section
st.title("🤖 AI MCQ Generation from Document 📚")
st.markdown("Generate **Multiple Choice Questions (MCQs)** from tech-related documents like PDFs or Word files using AI models. Customize the difficulty level and number of questions, and download the generated MCQs in PDF format!")

# Function to classify document content as tech-related using Groq API
def is_tech_related(text):
    client = Groq(api_key=groq_api_token)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Classify if the following content is tech-related:\n\n{text[:1500]}"
            }
        ],
        model="mixtral-8x7b-32768"
    )
    
    result = response.choices[0].message.content.strip().lower()
    return "yes" in result, result

# Function to generate MCQs based on difficulty level and content using Groq
def generate_mcqs(doc_content, difficulty_level, num_questions=10, selected_model="llama3-groq-8b-8192-tool-use-preview"):
    template = """
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
    
    prompt = PromptTemplate(input_variables=["context", "difficulty_level", "num_questions"], template=template)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(documents=doc_content, embedding=embeddings)

    retriever = faiss_index.as_retriever()
    query = 'Provide diverse MCQs from the entire document.'
    context_from_doc = retriever.invoke(query)

    input_dict = {
        "context": context_from_doc,
        "difficulty_level": difficulty_level,
        "num_questions": num_questions
    }
    
    formatted_prompt = prompt.format(**input_dict)

    client = Groq(api_key=groq_api_token)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        model=selected_model
    )
    
    mcq_result = response.choices[0].message.content.strip()
    return mcq_result

# Function to convert MCQs to PDF
def convert_to_pdf(mcqs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in mcqs.split("\n"):
        pdf.multi_cell(190, 10, txt=line, align='L')
    return pdf.output(dest='S').encode('latin1')

# Main app functionality
if uploaded_file and groq_api_token:
    with st.spinner("Processing the document..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Load the document based on its type
            if uploaded_file.type == "application/pdf":
                doc_loader = PyPDFLoader(temp_file_path)
            else:
                doc_loader = UnstructuredWordDocumentLoader(temp_file_path)

            data = doc_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)

            # Combine the text chunks into one document
            combined_text = " ".join([chunk.page_content for chunk in chunks])

            # Check if the document is tech-related
            is_tech, confidence = is_tech_related(combined_text)

            if not is_tech:
                st.warning(f"The document is not tech-related. Please upload a tech-related document.")
            else:
                difficulty_level = st.selectbox("🎯 Select difficulty level:", ["Easy", "Medium", "Hard"])
                num_questions = st.slider("⚖️ Select the number of questions:", min_value=5, max_value=30, step=1)
                selected_model = st.selectbox("🧠 Select Groq Model:", ["mixtral-8x7b-32768", "llama3-groq-70b-8192-tool-use-preview", "llama3-groq-8b-8192-tool-use-preview", "llama-3.1-70b-versatile"])

                if st.button("✨ Generate MCQs"):
                    mcqs = generate_mcqs(chunks, difficulty_level, num_questions, selected_model)
                    
                    # Preview the first generated MCQ
                    st.subheader("📜 Preview of Generated MCQ:")
                    st.markdown(mcqs.split("\n")[0])  # Display the first MCQ in markdown format

                    # Downloadable PDF
                    st.subheader("💾 Export MCQs:")
                    pdf_data = convert_to_pdf(mcqs)
                    st.download_button("📄 Download MCQs as PDF", data=pdf_data, file_name="mcqs.pdf", mime="application/pdf")

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

else:
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF or Word document to proceed.")
    if not groq_api_token:
        st.warning("⚠️ Please enter your Groq API token to proceed.")
