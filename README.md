# 📄 AI-Driven MCQ Generation System

This project leverages **Streamlit**, **LangChain**, **FAISS**, **Hugging Face**, and **Groq** to generate **Multiple Choice Questions (MCQs)** from tech-related PDF and Word documents. It uses state-of-the-art models for natural language understanding and provides an interactive interface for document processing, MCQ generation, and export options.

## 🚀 Live Demo

Try out the live demo here:  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-mcqgenerator02.streamlit.app/)

## ✨ Features

- **Upload Documents**: Supports PDF and Word document uploads.
- **Tech-Related Classification**: Automatically classifies the document as tech-related using Groq's API.
- **Customizable Question Count**: Select the number of questions (5 to 30).
- **Difficulty Levels**: Choose between Easy, Medium, or Hard questions.
- **Interactive UI**: Preview generated MCQs in markdown format.
- **Export Options**: Download the MCQs as a PDF file.
- **Hugging Face Integration**: Utilizes advanced models from Hugging Face for content processing.
- **Groq Integration**: Classifies and generates MCQs using models from Groq.

## ⚙️ Project Overview
This pro ject implements a system to generate MCQs from tech-related documents. The app uses Groq to classify whether a document is tech-related, and then LangChain and Hugging Face models are used to generate multiple-choice questions based on the document's content.


## 🛠️ How to Run Locally

Follow these steps to run the app on your local machine:

### Clone the Repository

```bash
git clone <your-repository-url>
cd document-question-answering
pip install -r requirements.txt
streamlit run bot.py



