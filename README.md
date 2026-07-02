# AI Customer Support Triage System

An AI-powered customer support triage application built using **Python, Flask, and Google Gemini** that automatically classifies customer support tickets, assigns priorities, summarizes customer issues, recommends actions, and determines when human intervention is required.

This project was developed as part of the **Gateway Group AI Systems Engineering Challenge**, with a focus on building reliable, production-oriented AI software.

---

# Features

- CSV dataset upload
- AI-powered ticket classification
- Priority assignment (P0–P3)
- Automatic issue summarization
- Suggested action generation
- Human Review Queue
- AI Evaluation Dashboard
- Category-wise performance metrics
- Confidence score tracking
- Batch processing for API optimization
- Prompt injection protection
- JSON output validation
- Retry & graceful failure handling
- Export results as JSON
- Export evaluation report

---

# System Architecture

```text
                 Customer Messages (CSV)
                          │
                          ▼
                 Input Validation Layer
                          │
                          ▼
                 Batch Processing Engine
                          │
                          ▼
              Google Gemini 2.5 Flash
                          │
                          ▼
                 JSON Response Validator
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
 Human Review Queue             Evaluation Dashboard
          │                               │
          └───────────────┬───────────────┘
                          ▼
                  Export Results (JSON)
```

---

# AI Workflow

1. Upload a CSV file containing customer support messages.
2. Validate the dataset.
3. Process messages in batches.
4. Send batches to Gemini for analysis.
5. Validate the generated JSON response.
6. Compute confidence scores.
7. Route uncertain cases to the Human Review Queue.
8. Generate evaluation metrics.
9. Export structured results.

---

# Tech Stack

### Backend
- Python
- Flask

### AI
- Google Gemini 2.5 Flash
- Prompt Engineering

### Frontend
- HTML
- Bootstrap 5
- JavaScript

### Data Processing
- Pandas

### Utilities
- JSON
- Logging
- Environment Variables (.env)

### Version Control
- Git
- GitHub

---

# AI Capabilities

For every customer message, the system predicts:

- Category
- Priority
- Summary
- Suggested Action
- Confidence Score
- Human Review Decision

---

# Reliability Features

The application includes multiple safeguards to improve reliability:

- Prompt injection detection
- Structured JSON validation
- Confidence thresholding
- Human-in-the-loop workflow
- Graceful API failure handling
- Retry strategy
- Empty input validation
- Batch processing
- Detailed application logging

---

# Evaluation Dashboard

The dashboard provides:

- Dataset size
- Overall accuracy (when ground truth is available)
- Category-wise accuracy
- Average confidence
- Human review percentage
- API usage
- Retry count
- Processing time
- Latency
- Cost estimation

---

# Project Structure

```
AI_CUSTOMER_SUPPORT/
│
├── app.py
├── triage.py
├── validator.py
├── prompt.py
├── requirements.txt
├── .env.example
│
├── templates/
├── static/
├── uploads/
├── data/
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/dixitaBalapuriya19/AI_CUSTOMER_SUPPORT.git
```

Navigate to the project:

```bash
cd AI_CUSTOMER_SUPPORT
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=YOUR_API_KEY
GEMINI_MODEL=gemini-2.5-flash
```

Run the application:

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

# Input Format

| message |
|---------|
| I was charged twice. |
| The application crashes on startup. |
| I cannot log into my account. |

---

# Sample Output

```json
{
  "category": "Billing",
  "priority": "P1",
  "summary": "Customer reports duplicate payment.",
  "suggested_action": "Verify payment history and process a refund if applicable.",
  "needs_human": false,
  "confidence": 0.95
}
```

---

# Design Decisions

- Selected **Google Gemini 2.5 Flash** for fast inference and structured JSON output.
- Implemented **batch processing** to reduce API calls and improve scalability.
- Added a **Human Review Queue** for low-confidence and high-risk cases.
- Built an **Evaluation Dashboard** to measure AI performance.
- Added **validation and guardrails** for reliable structured outputs.
- Designed the system to **fail safely** using fallback responses instead of terminating execution.

---

# Future Improvements

- Database integration
- Authentication and role-based access
- Asynchronous background processing
- Redis or message queue integration
- Response caching
- Audit logs
- Advanced analytics dashboard
- Fine-tuned classification model
- Multi-provider LLM support

---

# Screenshots

Include screenshots of:

- Home page
- Evaluation Dashboard
- Human Review Queue
- Results Table

---

# Author

**Dixita Balapuriya**

B.Tech Computer Engineering

GitHub: https://github.com/dixitaBalapuriya19

---

# License

This project was developed for educational purposes as part of the **Gateway Group AI Systems Engineering Challenge**.
