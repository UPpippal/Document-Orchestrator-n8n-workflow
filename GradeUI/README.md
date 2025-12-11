# AI-Powered Document Orchestrator

Streamlit app that ingests PDFs, extracts key information with Gemini, evaluates user-defined conditions, and triggers n8n email automation.

## Architecture
- Frontend: Streamlit (`app.py`)
- AI extraction: Google Gemini (via `google-generativeai`)
- PDF parsing: `pypdf`
- Automation: n8n webhook (HTTP POST). Example flow: `Webhook → IF/Filter → Email`

```
PDF upload → text extraction → Gemini JSON → condition check → n8n webhook
```

## Quickstart
1) Install Python 3.10+  
2) `python -m venv .venv && .venv/Scripts/activate` (Windows)  
3) `pip install -r requirements.txt`  
4) Set secrets (either environment variables or `.streamlit/secrets.toml`):
   - `GEMINI_API_KEY` (required)
   - `N8N_WEBHOOK_URL` (required to trigger automation)
   - `DEFAULT_EMAIL_TO` (optional; used by n8n payload)
5) Run `streamlit run app.py`

## n8n Setup (minimal)
1) Create **Webhook** node (POST). Copy its production URL into `N8N_WEBHOOK_URL`.  
2) Add a **Switch/IF** node to branch on payload fields (e.g., `$.condition.met == true`).  
3) Add **Email** node (or Gmail/Outlook) and use payload data, e.g.:
   - Subject: `New document matched condition: {{$json.document.document_type}}`
   - Body: include `{{$json.document.ai_summary}}` and fields from `document.ai_output`.
4) Deploy the workflow; keep the webhook in production mode.

## Streamlit Workflow
1) Upload a PDF.  
2) Click **Run Gemini Extraction** to get structured JSON.  
3) Configure a condition (field path, operator, expected value).  
4) If the condition matches, click **Trigger n8n Automation** to POST the payload to your webhook.

## Payload sent to n8n
```json
{
  "document": {
    "file_name": "invoice.pdf",
    "ai_output": { ... },
    "ai_summary": "Brief summary",
    "raw_text": "first 500 chars..."
  },
  "condition": { "field": "totals.total", "operator": ">", "expected": 1000, "met": true },
  "user": { "email": "from DEFAULT_EMAIL_TO or user input" },
  "metadata": { "timestamp": "ISO-8601" }
}
```

## Notes
- The Gemini prompt is designed to be deterministic and return JSON only; if a code block is returned, the app strips markers before parsing.
- Conditions use dotted paths into the JSON (e.g., `totals.total` or `parties.vendor.name`).
- Keep PDFs under the Streamlit size limit (default 200 MB).

## Deployment
- Local: `streamlit run app.py`
- Streamlit Community Cloud: add `GEMINI_API_KEY` and `N8N_WEBHOOK_URL` as secrets.
- Container: build from `python:3.11-slim`, install requirements, expose port 8501.

