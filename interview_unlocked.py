import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv
from pdfminer.high_level import extract_text
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import uuid
from google.cloud import speech
import pandas as pd
import time

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables and set up Google Cloud credentials
load_dotenv(find_dotenv())

# Set Google Cloud credentials path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(SCRIPT_DIR, "key.json")

# Directory configuration
FAISS_BASE_DIR = os.path.join(SCRIPT_DIR, "faiss")
FAISS_RESUME_PATH = os.path.join(FAISS_BASE_DIR, "resume_embeddings")
FAISS_JD_PATH = os.path.join(FAISS_BASE_DIR, "jd_embeddings")

# Create base directories
os.makedirs(FAISS_BASE_DIR, exist_ok=True)

# Create directories if they don't exist
for path in [FAISS_RESUME_PATH, FAISS_JD_PATH]:
    os.makedirs(path, exist_ok=True)

# Initialize LLM and embeddings
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

def extract_text_from_file(file_path: str) -> str:
    """Extracts text from a file."""
    try:
        if file_path.lower().endswith('.pdf'):
            print(f"Extracting text from PDF: {file_path}")
            text = extract_text(file_path)
            print("PDF text extraction finished.")
            return text.strip()
        else:
            print(f"Reading text file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def generate_embeddings_and_save(text: str, path: str) -> bool:
    """Generates and saves embeddings for the given text."""
    try:
        texts = [text]
        # Create a new vector store each time
        vectorstore = FAISS.from_texts(texts, embeddings)
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        vectorstore.save_local(path)
        return True
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return False

def retrieve_similar_content(query: str, path: str, k: int = 3) -> List[str]:
    """Retrieves similar content from the vector store."""
    try:
        if not os.path.exists(path):
            return []
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        results = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    except Exception as e:
        print(f"Error retrieving content: {e}")
        return []

def generate_interview_questions(resume_text: str, jd_text: str) -> List[Dict[str, str]]:
    """Generates relevant interview questions based on the resume and job description."""
    prompt = f"""
    Based on the following resume and job description, generate 3 relevant interview questions.
    Make the questions specific to the candidate's experience and the job requirements.
    
    Resume:
    {resume_text}
    
    Job Description:
    {jd_text}
    
    Generate these types of questions:
    1. Technical coding question: Focus on algorithms, data structures, or specific technologies from their resume
    2. System design question: Related to their experience and the job's requirements
    3. Behavioral question: Based on their past projects and experience
    
    Format each question with:
    - Clear context/scenario
    - Specific requirements or constraints
    - What you're looking for in the answer
    """
    
    messages = [
        SystemMessage(content="You are an expert technical interviewer who creates clear, specific questions."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    questions = []
    
    # Parse the response to extract questions
    content = response.content
    current_type = None
    current_question = []
    
    for line in content.split('\n'):
        line = line.strip()
        if any(line.startswith(prefix) for prefix in ('1.', '2.', '3.')):
            if current_type and current_question:
                questions.append({
                    "type": current_type,
                    "question": '\n'.join(current_question).strip()
                })
                current_question = []
            
            if line.startswith('1.'):
                current_type = "technical"
            elif line.startswith('2.'):
                current_type = "system_design"
            else:
                current_type = "behavioral"
            
            current_question.append(line.split('.', 1)[1].strip())
        elif line and current_type:
            current_question.append(line)
    
    # Add the last question
    if current_type and current_question:
        questions.append({
            "type": current_type,
            "question": '\n'.join(current_question).strip()
        })
    
    return questions

def record_and_transcribe_audio(duration: int = 30) -> str:
    """Records audio and transcribes it using Google Cloud Speech-to-Text."""
    print(f"\n🎙️ Get ready to answer in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print(f"Recording your answer for up to {duration} seconds...")
    print("Speak clearly and take your time.")
    print("Press Ctrl+C when you're done speaking (or wait for the time to end)")
    
    fs = 16000  # Sample rate
    audio_file = f"/tmp/interview_answer_{uuid.uuid4()}.wav"
    
    try:
        # Start recording
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        
        try:
            # Show progress bar until Ctrl+C or duration ends
            for i in range(duration):
                time.sleep(1)
                remaining = duration - i - 1
                progress = "=" * (i + 1) + "-" * remaining
                print(f"\rRecording: [{progress}] {remaining}s remaining", end="")
        except KeyboardInterrupt:
            # Stop recording early if Ctrl+C is pressed
            sd.stop()
            print("\n\n✅ Recording stopped early!")
        finally:
            sd.wait()  # Wait for recording to finish
        
        print("\n✅ Recording complete!")
        
        # Rest of the function remains the same...
        recording_int16 = np.int16(recording * 32767)
        wav.write(audio_file, fs, recording_int16)
        
        print("🔍 Transcribing audio...")
        client = speech.SpeechClient()
        
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
        os.remove(audio_file)
        
        if not response.results:
            return "[No speech detected]"
            
        transcript = " ".join(result.alternatives[0].transcript for result in response.results)
        print("✅ Transcription complete!")
        return transcript.strip()
        
    except Exception as e:
        if os.path.exists(audio_file):
            os.remove(audio_file)
        error_msg = f"Error during audio recording or transcription: {e}"
        print(error_msg)
        return error_msg

def evaluate_answer(question: str, answer: str, resume_text: str, jd_text: str) -> Dict[str, str]:
    """Evaluates the candidate's answer and provides feedback."""
    evaluation_prompt = f"""
    Evaluate this interview answer considering the candidate's background and the job requirements.
    
    Question: {question}
    
    Candidate's Answer: {answer}
    
    Resume Context:
    {resume_text}
    
    Job Requirements:
    {jd_text}
    
    Please provide:
    1. Strengths of the answer
    2. Areas for improvement
    3. Sample ideal answer
    4. Specific suggestions for better alignment with job requirements
    """
    
    messages = [
        SystemMessage(content="You are an expert interview evaluator."),
        HumanMessage(content=evaluation_prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "evaluation": response.content,
        "original_answer": answer,
        "question": question
    }

def process_interview():
    """Handles the complete interview process."""
    resume_path = os.path.join(SCRIPT_DIR, 'Mandar_Burande_Resume.pdf')
    jd_path = os.path.join(SCRIPT_DIR, 'airbnb-jd.txt')
    
    # Extract text
    print("\n📄 Processing resume and job description...")
    resume_text = extract_text_from_file(resume_path)
    jd_text = extract_text_from_file(jd_path)
    
    if not resume_text or not jd_text:
        return "Failed to extract text from files"
    
    # Generate embeddings
    if not generate_embeddings_and_save(resume_text, FAISS_RESUME_PATH):
        return "Failed to generate resume embeddings"
    if not generate_embeddings_and_save(jd_text, FAISS_JD_PATH):
        return "Failed to generate job description embeddings"
    
    # Generate questions
    print("\n🤔 Generating tailored interview questions...")
    questions = generate_interview_questions(resume_text, jd_text)
    
    evaluations = []
    for q in questions:
        print(f"\n{'='*80}")
        print(f"\n📝 {q['type'].upper()} QUESTION:")
        print(f"\n{q['question']}")
        print(f"\n{'='*80}")
        
        input("\nPress Enter when you're ready to answer this question...")
        
        # Record and transcribe answer
        answer = record_and_transcribe_audio(30)  # 30 seconds for each answer
        print("\n🗣️ Your transcribed answer:")
        print(answer)
        
        # Evaluate answer
        print("\n📊 Evaluating your answer...")
        evaluation = evaluate_answer(q['question'], answer, resume_text, jd_text)
        evaluations.append(evaluation)
        print("\n💡 Evaluation:")
        print(evaluation['evaluation'])
        
        if q != questions[-1]:  # If not the last question
            input("\nPress Enter when you're ready for the next question...")
        
        print("\n" + "="*80 + "\n")
    
    return evaluations

if __name__ == "__main__":
    try:
        print("\n🎯 Welcome to the Interview Practice System!")
        print("We'll ask you three questions - technical, system design, and behavioral.")
        print("You'll have 30 seconds to answer each question.")
        print("After each answer, you'll receive detailed feedback.")
        print("\nMake sure your microphone is working and you're in a quiet environment.")
        input("\nPress Enter when you're ready to begin...")
        
        evaluations = process_interview()
        
        print("\n🎉 Interview practice session complete!")
        print("Review your performance above and practice the areas for improvement.")
        
    except Exception as e:
        print(f"Error during execution: {e}")