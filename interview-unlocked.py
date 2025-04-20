# Install required packages
# %pip install -qU langchain langgraph langgraph-swarm langchain-google-genai google-generativeai langchain_community faiss-cpu tavily-python google-cloud-speech sounddevice scipy pdfminer.six python-dotenv langchain-openai numpy pandas pytesseract openpyxl langchain-ollama

# Install required packages (uncomment if needed)
#%pip install -qU langchain langgraph langgraph-swarm langchain-google-genai langchain_community faiss-cpu tavily-python google-cloud-speech sounddevice scipy pdfminer.six python-dotenv langchain-openai

import os
import json
import re
import uuid
import numpy as np
import pandas as pd
import sounddevice as sd
import scipy.io.wavfile as wav
from typing import List, Dict, Any, Optional, TypedDict
import pytesseract

# Replace Ollama with Google Generative AI (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_ollama import ChatOllama # Removed unused import
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Use pydantic.v1 for compatibility as suggested by the warning
from pydantic.v1 import BaseModel, Field 
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
# Removed unused langgraph_swarm imports

from google.cloud import speech
from pdfminer.high_level import extract_text
from dotenv import load_dotenv

# Load environment variables (for API keys like Tavily, Google Cloud)
load_dotenv()

# --- Configuration ---
FAISS_RESUME_PATH = "./faiss/resume_embeddings"
FAISS_JD_PATH = "./faiss/jd_embeddings"
FAISS_RUBRIC_PATH = "./faiss/rubric_embeddings"
FAISS_KNOWLEDGE_PATH = "./faiss/knowledge_embeddings"
os.makedirs(os.path.dirname(FAISS_RESUME_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FAISS_JD_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FAISS_RUBRIC_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FAISS_KNOWLEDGE_PATH), exist_ok=True)
# Retrieve the API key loaded by load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM (using Gemini)
# Pass the API key explicitly
# import google.generativeai as genai
# genai.configure(api_key=google_api_key)
# models = genai.list_models()

# for m in models:
#     print(m.name, m.supported_generation_methods)

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Must be valid in your project
    temperature=0,
    google_api_key=google_api_key  # from .env
)

# gemini_model = ChatOllama(model="llama3.1:latest")

# llm = ChatOllama(model="llama3", temperature=0.1)

# Initialize Embeddings Model (using Gemini)
# Pass the API key explicitly
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)
# embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize Checkpointer
checkpointer = InMemorySaver()


# --- Tool Implementations ---

@tool
def extract_text_with_ocr(file_path: str) -> str:
    """
    Extracts text from a file. Uses pdfminer.six for text-based PDF files,
    pytesseract OCR for image-based PDFs (if Tesseract is installed),
    otherwise reads as plain text.
    """
    try:
        if (file_path.lower().endswith('.pdf')):
            print(f"Extracting text from PDF: {file_path}")
            try:
                # Use pdfminer.six for direct text extraction
                text = extract_text(file_path)
                print("PDF text extraction with pdfminer.six finished.")
                return text.strip()
            except Exception as e_pdfminer:
                print(f"pdfminer.six failed: {e_pdfminer}. Falling back to OCR if possible.")
                # Fallback to OCR if pdfminer fails (optional, requires Tesseract)
                try:
                    pytesseract.get_tesseract_version() # Check if Tesseract is available
                    # If you still want OCR as a fallback, you'd need pdf2image back.
                    # For now, we just report the pdfminer error if OCR isn't the primary path.
                    # If you re-introduce pdf2image for fallback:
                    # from pdf2image import convert_from_path
                    # images = convert_from_path(file_path)
                    # full_text = ""
                    # for i, image in enumerate(images):
                    #     print(f"Processing page {i+1}/{len(images)} via OCR fallback...")
                    #     ocr_text = pytesseract.image_to_string(image)
                    #     full_text += ocr_text + "\n"
                    # print("PDF OCR fallback finished.")
                    # return full_text.strip()
                    return f"Error extracting text with pdfminer.six: {e_pdfminer}. OCR fallback not fully implemented without pdf2image."

                except Exception as e_ocr_check:
                     return f"Error extracting text with pdfminer.six: {e_pdfminer}. Tesseract for OCR fallback not found: {e_ocr_check}"
        else:
            # Handle non-PDF files as plain text
            print(f"Reading text file: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"An unexpected error occurred while processing {file_path}: {e}"

@tool
def generate_resume_embeddings_and_save(text: str) -> str:
    """Generates embeddings for the Job Description text and saves/updates the FAISS Resume index."""
    index_path = FAISS_RESUME_PATH # Use the specific path
    try:
        texts = [text] # FAISS expects a list
        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_texts(texts)
        else:
            vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore.save_local(index_path)
        return f"Resume Embeddings generated and saved to {index_path}"
    except Exception as e:
        return f"Error generating/saving Resume embeddings: {e}"

@tool
def generate_jd_embeddings_and_save(text: str) -> str:
    """Generates embeddings for the Job Description text and saves/updates the FAISS JD index."""
    index_path = FAISS_JD_PATH # Use the specific path
    try:
        texts = [text] # FAISS expects a list
        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_texts(texts)
        else:
            vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore.save_local(index_path)
        return f"JD Embeddings generated and saved to {index_path}"
    except Exception as e:
        return f"Error generating/saving JD embeddings: {e}"

@tool
def generate_knowledge_embeddings_and_save(text: str) -> str:
    """Generates embeddings for the Job Description text and saves/updates the FAISS Knowledge index."""
    index_path = FAISS_KNOWLEDGE_PATH # Use the specific path
    try:
        texts = [text] # FAISS expects a list
        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_texts(texts)
        else:
            vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore.save_local(index_path)
        return f"Knowledge Embeddings generated and saved to {index_path}"
    except Exception as e:
        return f"Error generating/saving Knowledge embeddings: {e}"


@tool
def generate_rubric_embeddings_and_save(text: str) -> str:
    """Generates embeddings for the Job Description text and saves/updates the FAISS Rubric index."""
    index_path = FAISS_RUBRIC_PATH # Use the specific path
    try:
        texts = [text] # FAISS expects a list
        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_texts(texts)
        else:
            vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore.save_local(index_path)
        return f"Rubric Embeddings generated and saved to {index_path}"
    except Exception as e:
        return f"Error generating/saving Rubric embeddings: {e}"

@tool
def retrieve_resume_embeddings_from_vector_db(query: str, k: int = 3) -> List[str]:
    """Retrieves relevant documents from the resume FAISS index."""
    try:
        index_path = FAISS_RESUME_PATH
        if not os.path.exists(index_path):
            return ["Resume vector index not found."]
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        results = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"Error retrieving from resume vector DB: {e}"]
    
@tool
def retrieve_jd_embeddings_from_vector_db(query: str, k: int = 3) -> List[str]:
    """Retrieves relevant documents from the JD FAISS index."""
    try:
        index_path = FAISS_JD_PATH
        if not os.path.exists(index_path):
            return ["JD vector index not found."]
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        results = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"Error retrieving from JD vector DB: {e}"]

@tool
def retrieve_knowledge_embeddings_from_vector_db(query: str, k: int = 3) -> List[str]:
    """Retrieves relevant documents from the Knowledge FAISS index."""
    try:
        index_path = FAISS_KNOWLEDGE_PATH
        if not os.path.exists(index_path):
            return ["Knowledge vector index not found."]
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        results = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"Error retrieving from Knowledge vector DB: {e}"]

@tool
def retrieve_rubric_embeddings_from_vector_db(query: str, k: int = 3) -> List[str]:
    """Retrieves relevant documents from the Rubric FAISS index."""
    try:
        index_path = FAISS_RUBRIC_PATH
        if not os.path.exists(index_path):
            return ["Rubric vector index not found."]
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        results = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"Error retrieving from Rubric vector DB: {e}"]


load_dotenv()

# Retrieve the API key from environment variables
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Check if the API key was loaded
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please ensure it is set in your .env file.")

# Tavily Search Tool (already integrated in LangChain)
# Pass the API key during initialization
tavily_tool = TavilySearchResults(
    tavily_api_key=tavily_api_key,
    max_results=10
)


# @tool
# def web_retrieval_tavily_search() -> str:
#     """Used for searching the web for relevant discussion threads about the company"""
#     # Load environment variables from .env file
#     load_dotenv()

#     # Retrieve the API key from environment variables
#     tavily_api_key = os.getenv("TAVILY_API_KEY")

#     # Check if the API key was loaded
#     if not tavily_api_key:
#         raise ValueError("TAVILY_API_KEY not found in environment variables. Please ensure it is set in your .env file.")

#     # Tavily Search Tool (already integrated in LangChain)
#     # Pass the API key during initialization
#     tavily_tool = TavilySearchResults(
#         tavily_api_key=tavily_api_key,
#         max_results=20
#     )

#     print(tavily_tool)
#     # results = tavily_tool.invoke("Amazon system design interview ")

@tool
def company_leetcode_problem_retriever(company: str, role_keywords: Optional[List[str]] = None) -> List[str]:
    """
    Retrieves suggested LeetCode questions for a specific company by reading
    from the './Leetcode-company-problem-set.xlsx' file. Each company's
    questions are expected to be in a sheet named after the company (case-insensitive).
    Questions are assumed to be listed in the first column (A) starting from the first row (A1).
    The role_keywords parameter is currently unused but available for future filtering.
    """
    excel_path = './Leetcode-company-problem-set.xlsx'
    default_questions = ["Reverse Linked List", "Valid Parentheses", "Coin Change"] # Default if company not found

    print(f"Fetching LeetCode questions for {company} from {excel_path}...")

    try:
        # Check if file exists first
        if not os.path.exists(excel_path):
            print(f"Error: Excel file not found at {excel_path}. Returning default questions.")
            return default_questions

        # Read all sheet names first to handle case-insensitivity
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        target_sheet = None
        for name in sheet_names:
            if name.lower() == company.lower():
                target_sheet = name
                break

        if target_sheet:
            # Read the specific sheet, assuming no header and questions start at A1 (index 0)
            df = pd.read_excel(excel_path, sheet_name=target_sheet, header=None)

            if not df.empty and df.shape[1] > 0: # Check if dataframe is not empty and has at least one column
             # Questions are in the first column (index 0)
                questions = df.iloc[:, 0].dropna().astype(str).tolist()
                if questions:
                    print(f"Found {len(questions)} questions for {company} in sheet '{target_sheet}'.")
                    return questions
                else:
                    print(f"Sheet '{target_sheet}' for {company} found, but the first column is empty or contains only NaN values.")
                    return default_questions
            else:
                print(f"Sheet '{target_sheet}' for {company} found but is empty or has no columns.")
                return default_questions
        else:
            print(f"No specific sheet found for '{company}'. Returning default questions.")
            return default_questions

    except FileNotFoundError: # Should be caught by os.path.exists, but kept for robustness
        print(f"Error: Excel file not found at {excel_path}. Returning default questions.")
        return default_questions
    except Exception as e:
        print(f"An error occurred while reading the Excel file for {company}: {e}. Returning default questions.")
        return default_questions


@tool
def record_and_transcribe_audio(duration: int = 15, fs: int = 16000) -> str:
    """Records audio from the microphone for a specified duration and transcribes it using Google Cloud Speech-to-Text."""
    print(f"Recording audio for {duration} seconds... Speak now!")
    audio_file = f"/tmp/interview_answer_{uuid.uuid4()}.wav"
    try:
        # Record audio
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        # Convert to int16 and save
        recording_int16 = np.int16(recording * 32767)
        wav.write(audio_file, fs, recording_int16)
        print("Audio recorded.")

        # Transcribe audio
        print("Transcribing audio...")
        client = speech.SpeechClient() # Assumes GOOGLE_APPLICATION_CREDENTIALS is set
        with open(audio_file, "rb") as f:
            content = f.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=fs,
            language_code="en-US",
            enable_automatic_punctuation=True
        )
        response = client.recognize(config=config, audio=audio)
        os.remove(audio_file) # Clean up temporary file

        if not response.results:
            print("Transcription failed: No speech detected.")
            return "[No speech detected]"

        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        print(f"Transcription complete: {transcript}")
        return transcript.strip()
    except Exception as e:
        if os.path.exists(audio_file):
            os.remove(audio_file)
        error_msg = f"Error during audio recording or transcription: {e}"
        print(error_msg)
        return error_msg


# TODO Implement the below tools

@tool
def generate_ideal_answer(question: str, company_tag: Optional[str] = None) -> str:
    """Generates an ideal answer to the question (simulated by LLM call)."""
    # This would typically involve another LLM call with specific instructions
    # For simplicity here, we'll just return a placeholder or let the main agent handle it.
    return f"[Placeholder: Ideal answer generation for '{question}' considering company '{company_tag}']"

@tool
def rewrite_candidate_answer(question: str, candidate_answer: str) -> str:
    """Rewrites the candidate's answer for improvement (simulated by LLM call)."""
    return f"[Placeholder: Rewritten version of answer for '{question}']"

@tool
def critique_and_advise(question: str, candidate_answer: str, ideal_answer: str, company_tag: Optional[str] = None) -> str:
    """Provides critique and advice based on the answers (simulated by LLM call)."""
    return f"[Placeholder: Critique for answer to '{question}' considering company '{company_tag}']"

# Agent Prompts
preprocessing_prompt = """
Role: PreprocessingAgent. Structure resume and job description data.
Responsibilities:
1. Use `extract_text_with_ocr` on resume (./Mandar_Burande_Resume.pdf) and JD (./jd.txt).
2. Clean extracted text (remove headers/footers/whitespace). Store as `clean_resume`, `clean_jd`.
3. Extract 'COMPANY NAME' from `clean_jd`.
4. Use `generate_resume_embeddings_and_save` on `clean_resume` (save to ./faiss/resume_embeddings).
5. Use `generate_jd_embeddings_and_save` on `clean_jd` (save to ./faiss/jd_embeddings).
HANDOFF: knowledge_agent
"""

knowledge_prompt = """
Role: Knowledge Agent. Extract real-world interview expectations for the company.
Responsibilities:
1. Use `Tavily Tool` to search for interview process insights (Reddit, Glassdoor, Blind). Focus queries on coding/behavioral/system design expectations.
2. Use `retrieve_jd_embeddings_from_vector_db` for JD context.
3. Infer evaluation rubric themes (Ownership, Tradeoffs, Reasoning, Ambiguity, Communication) and communication tips from search results.
4. Use `generate_knowledge_embeddings_and_save` on search results
Output: JSON with `company`, `inferred_rubric` (theme, evidence, reference), `communication_tips`.
HANDOFF: planner_agent
"""

planner_prompt = """
Role: Planner Agent. Curate LeetCode problems, present insights.
Responsibilities:
1. Use resume, jd, and knowledge embeddings for context from vector DB
2. Use `company_leetcode_problem_retriever` for company-specific LeetCode problems.
3. Format and include `inferred_rubric` and `communication_tips` from Knowledge Agent.
4. Use `generate_rubric_embeddings_and_save` on rubric/tips text
Output: `suggested_leetcode`, `company_insights_display`
Strictly HANDOFF to question_agent for generating a relevant, open-ended, behavioral interview question after this
"""

question_prompt = """
Role: Question Agent. Generate relevant, open-ended, behavioral interview questions for skill development
Responsibilities:
1. Use `retrieve_resume_embeddings_from_vector_db`, `retrieve_jd_embeddings_from_vector_db`, `retrieve_knowledge_embeddings_from_vector_db` for context (resume, JD, knowledge/rubric).
2. Synthesize context.
3. Generate one question tailored to the company, JD, resume, and inferred rubric.
Output: JSON with `question`.
"""

evaluation_prompt = """
Role: EvaluationFeedbackAgent. Provide interview feedback.
Inputs: `question`, `candidate_answer`, `company_tag`, `rubric_index_path`.
Responsibilities:
1. Use `retrieve_rubric_embeddings_from_vector_db` for evaluation criteria using `question` and `company_tag`.
2. Use `generate_ideal_answer` for the `question`.
3. Use `rewrite_candidate_answer` for the `candidate_answer`.
4. Use `critique_and_advise` using all inputs. Critique should cover strengths, weaknesses (STAR method, complexity), and improvements (use bullet points, bold key terms).
Output: JSON with `ideal_answer`, `improved_answer`, `detailed_feedback`.
"""


# --- Agent Creation ---


# Creating Agent Nodes
preprocess_tools = [extract_text_with_ocr, generate_resume_embeddings_and_save, generate_jd_embeddings_and_save, create_handoff_tool(agent_name='knowledge_agent', description='Hand over to Knowledge Agent for web search, rubric inference and generating knowledge embeddings')]
preprocess_agent_node = create_react_agent(
    gemini_model,
    preprocess_tools,
    prompt=preprocessing_prompt,
    name='preprocess_agent'
)

knowledge_tools = [tavily_tool, retrieve_jd_embeddings_from_vector_db, generate_knowledge_embeddings_and_save, create_handoff_tool(agent_name='planner_agent', description='Handover to Planner Agent for generating a study plan, curating leetcode problems and inferring company-specific evaluation rubric')]
knowledge_agent_node = create_react_agent(
    gemini_model,
    knowledge_tools,
    prompt=knowledge_prompt,
    name='knowledge_agent'
)

planner_tools = [company_leetcode_problem_retriever, retrieve_resume_embeddings_from_vector_db, retrieve_jd_embeddings_from_vector_db, retrieve_knowledge_embeddings_from_vector_db, 
                 generate_rubric_embeddings_and_save, create_handoff_tool(agent_name='question_agent', description='Hand over to Question Agent for generating relevant, open-ended, behavioral interview questions to improve skill development')]
planner_agent_node = create_react_agent(
    gemini_model,
    planner_tools,
    prompt=planner_prompt,
    name='planner_agent'
)

question_tools = [retrieve_resume_embeddings_from_vector_db, retrieve_jd_embeddings_from_vector_db, retrieve_knowledge_embeddings_from_vector_db]
question_agent_node = create_react_agent(
    gemini_model,
    question_tools,
    prompt=question_prompt,
    name='question_agent'
)

evaluation_tools = [retrieve_resume_embeddings_from_vector_db, retrieve_jd_embeddings_from_vector_db, retrieve_rubric_embeddings_from_vector_db, generate_ideal_answer, rewrite_candidate_answer, critique_and_advise]
evaluation_agent_node = create_react_agent(
    gemini_model,
    evaluation_tools,
    prompt=evaluation_prompt,
    name='evaluation_agent'
)

workflow = create_swarm(
    [preprocess_agent_node, knowledge_agent_node, planner_agent_node, question_agent_node],
    default_active_agent='preprocess_agent'
)

graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": 1}}

turn_1 = graph.invoke(
    {"messages": [
        {
            "role": "user", 
            "content": "The file path to my resume and jd is ./Mandar_Burande_Resume.pdf and ./jd.txt. Give me a list of popular leetcode problems for this company. Based on my experience, skills and projects from my resume, suggest me relevant, open-ended, behavioral interview questions that will help me improve my skills for this company."
        }
    ]},
    config
)

print(turn_1)
print(turn_1['messages'][-1])

