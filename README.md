# Interview Unlocked

An AI-powered interview preparation system using LangGraph and multi-agent orchestration.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

3. Place your Google Cloud credentials in `key.json` (needed for speech-to-text)

## Usage

1. Have your resume in PDF format and job description in a text file
2. Run the script:
```bash
python interview_unlocked.py
```

The system will:
- Process your resume and job description
- Generate a personalized study plan
- Suggest relevant LeetCode problems
- Generate and ask interview questions
- Record and evaluate your answers

## Files
- `interview_unlocked.py`: Main script
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (create this)
- `key.json`: Google Cloud credentials (required for speech-to-text)
- `Leetcode-company-problem-set.xlsx`: Company-specific LeetCode problems
