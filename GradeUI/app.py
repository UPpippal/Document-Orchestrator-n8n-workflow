import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import google.generativeai as genai
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv()
st.set_page_config(page_title="AI Document Orchestrator", page_icon="📄", layout="wide")


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch a secret preferring env, then Streamlit secrets (if available)."""
    env_val = os.getenv(key)
    if env_val:
        return env_val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
N8N_WEBHOOK_URL = get_secret("N8N_WEBHOOK_URL")
DEFAULT_EMAIL_TO = get_secret("DEFAULT_EMAIL_TO", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def read_pdf(file) -> str:
    """Extract text from an uploaded PDF file."""
    reader = PdfReader(file)
    text_chunks = []
    for page in reader.pages:
        text_chunks.append(page.extract_text() or "")
    return "\n".join(text_chunks).strip()


def clean_json_text(text: str) -> str:
    """Strip code fences if Gemini wraps JSON in ```json ... ```."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text.strip()


def call_gemini(document_text: str) -> Dict[str, Any]:
    """Send the document text to Gemini to get structured JSON."""
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY.")

    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        "You are an information extraction agent for resumes/CVs. "
        "Return ONLY minified JSON without prose or code fences. "
        "Schema:\n"
        "{"
        '"document_type": "resume",'
        '"candidate": {'
        '"name": string,'
        '"email": string,'
        '"phone": string,'
        '"location": string'
        "},"
        '"work_experience": {'
        '"years": number,'
        '"recent_title": string,'
        '"recent_company": string,'
        '"entries": ['
        '{'
        '"title": string,'
        '"company": string,'
        '"start_date": string (format: "YYYY-MM" or "YYYY"),'
        '"end_date": string (format: "YYYY-MM" or "YYYY" or "Present" or "Current"),'
        '"duration_months": number (calculated from start_date to end_date),'
        '"duration_years": number (calculated from start_date to end_date)'
        '}'
        ']'
        "},"
        '"skills": [string],'
        '"education": {'
        '"degree": string,'
        '"institution": string,'
        '"graduation_year": string'
        "},"
        '"certifications": [string],'
        '"summary": string (<=80 words)'
        "}\n"
        "For each work experience entry, calculate duration_months and duration_years based on start_date and end_date. "
        "If end_date is 'Present' or 'Current', use current date. "
        "If a field is unknown, use null. Respond with valid JSON only."
    )

    response = model.generate_content([prompt, document_text])
    text = response.text or "{}"
    cleaned = clean_json_text(text)
    return json.loads(cleaned)


def extract_field(data: Dict[str, Any], path: str) -> Any:
    """Get nested value from dict using dotted path."""
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def evaluate_condition(ai_output: Dict[str, Any], field: str, operator: str, expected: str) -> bool:
    """Evaluate a simple condition against the AI output."""
    value = extract_field(ai_output, field)
    if value is None:
        return False

    # Normalize types
    def to_number(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    if operator == "equals":
        return str(value).lower() == str(expected).lower()
    if operator == "contains":
        return str(expected).lower() in str(value).lower()
    if operator == ">":
        v_num, e_num = to_number(value), to_number(expected)
        return v_num is not None and e_num is not None and v_num > e_num
    if operator == "<":
        v_num, e_num = to_number(value), to_number(expected)
        return v_num is not None and e_num is not None and v_num < e_num
    return False


def trigger_n8n(payload: Dict[str, Any]) -> requests.Response:
    """POST payload to n8n webhook."""
    if not N8N_WEBHOOK_URL:
        raise RuntimeError("Missing N8N_WEBHOOK_URL.")
    return requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)


def parse_experience_years(ai_output: Dict[str, Any]) -> Optional[float]:
    """Calculate total years of experience by summing durations from experience entries.
    
    First tries to sum duration_months and duration_years from work_experience.entries.
    Falls back to top-level work_experience.years if entries are not available.
    """

    def iter_entries():
        paths = [
            "work_experience.entries",
            "work_experience.roles",
            "experience.entries",
            "experience.roles",
        ]
        for path in paths:
            val = extract_field(ai_output, path)
            if isinstance(val, list):
                for item in val:
                    yield item

    total_months = 0.0
    found = False
    
    # Sum durations from experience entries
    for entry in iter_entries():
        if not isinstance(entry, dict):
            continue
        
        # Try duration_months first (most precise)
        months = entry.get("duration_months") or entry.get("months")
        years = entry.get("duration_years") or entry.get("years")
        
        try:
            if months is not None:
                total_months += float(months)
                found = True
            elif years is not None:
                # Only use years if months not available (to avoid double counting)
                total_months += float(years) * 12
                found = True
        except (TypeError, ValueError):
            continue

    # Return calculated total if we found entries with durations
    if found and total_months > 0:
        return round(total_months / 12, 2)

    # Fallback to top-level years if entries don't have durations
    possible_paths = ["work_experience.years", "experience.years", "experience.total_years"]
    for path in possible_paths:
        val = extract_field(ai_output, path)
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def parse_skills(ai_output: Dict[str, Any]) -> set[str]:
    """Normalize skills into a lowercase set."""
    raw = extract_field(ai_output, "skills") or extract_field(ai_output, "candidate.skills")
    skills: set[str] = set()
    if isinstance(raw, list):
        skills = {str(s).strip().lower() for s in raw if str(s).strip()}
    elif isinstance(raw, str):
        skills = {s.strip().lower() for s in raw.split(",") if s.strip()}
    return skills


def main():
    st.title("📄 AI Document Orchestrator")
    st.markdown(
        "Upload a PDF, let Gemini extract structured data, apply a condition, "
        "and trigger n8n automation when it matches."
    )

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not configured. Add it to env or Streamlit secrets.")
    if not N8N_WEBHOOK_URL:
        st.warning("N8N_WEBHOOK_URL not configured. Automation will be disabled.")

    with st.sidebar:
        st.header("Automation Settings")
        user_email = st.text_input("Notification email", value=DEFAULT_EMAIL_TO)
        st.caption("This email is forwarded to n8n payload (optional).")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        raw_text = read_pdf(uploaded)
        st.success(f"Loaded {uploaded.name} ({len(raw_text)} chars).")
        with st.expander("Preview extracted text"):
            st.text_area("Raw text", raw_text[:8000], height=300)

        if st.button("Run Gemini Extraction", type="primary"):
            with st.spinner("Calling Gemini..."):
                try:
                    ai_output = call_gemini(raw_text)
                    
                    # Calculate RISK LEVEL immediately after extraction
                    # Experience is calculated by summing durations from work_experience.entries
                    candidate_experience = parse_experience_years(ai_output)
                    risk_level = "High" if (candidate_experience is not None and candidate_experience > 2) else "Low"
                    ai_output["risk_level"] = risk_level
                    
                    st.session_state["ai_output"] = ai_output
                    st.session_state["raw_text"] = raw_text
                    st.session_state["file_name"] = uploaded.name
                    st.success("Gemini extraction complete.")
                except Exception as exc:
                    st.error(f"Gemini call failed: {exc}")

    ai_output = st.session_state.get("ai_output")
    if ai_output:
        # Ensure risk_level is calculated and added before displaying JSON
        if "risk_level" not in ai_output:
            candidate_experience = parse_experience_years(ai_output)
            risk_level = "High" if (candidate_experience is not None and candidate_experience > 2) else "Low"
            ai_output["risk_level"] = risk_level
            st.session_state["ai_output"] = ai_output
        
        st.subheader("AI Output (JSON)")
        st.json(ai_output)

        st.subheader("Candidate Selection")
        col1, col2 = st.columns(2)
        with col1:
            min_experience = st.number_input(
                "Minimum required experience (years)",
                min_value=0.0,
                value=2.0,
                step=0.5,
            )
        with col2:
            required_skills_input = st.text_input(
                "Required skills (comma-separated)",
                value="python, sql",
            )

        required_skills = {s.strip().lower() for s in required_skills_input.split(",") if s.strip()}
        candidate_experience = parse_experience_years(ai_output)
        candidate_skills = parse_skills(ai_output)

        missing_skills = sorted(list(required_skills - candidate_skills)) if required_skills else []
        skills_match = len(missing_skills) == 0 if required_skills else True
        exp_match = candidate_experience is not None and candidate_experience > min_experience
        
        # Calculate RISK LEVEL: High if total experience > 2 years, otherwise Low
        # Experience is calculated by summing durations from work_experience.entries
        risk_level = "High" if (candidate_experience is not None and candidate_experience > 2) else "Low"
        
        # Update risk_level in the AI output JSON for display and persistence
        ai_output["risk_level"] = risk_level
        st.session_state["ai_output"] = ai_output  # Update session state to persist changes

        candidate_summary = st.text_area(
            "Candidate summary",
            value=ai_output.get("summary") or "",
            height=120,
        )

        recommendation_default = (
            "Recommended to proceed to interview."
            if exp_match and skills_match
            else "Not recommended. Missing prerequisites."
        )
        recommendation = st.text_area(
            "Recommendation",
            value=recommendation_default,
            height=80,
        )

        st.markdown("**Fit Check**")
        st.write(f"- Experience: {candidate_experience or 'Unknown'} years (needs ≥ {min_experience})")
        st.write(f"- Skills present: {', '.join(sorted(candidate_skills)) or 'None'}")
        st.write(f"- Risk level: {risk_level} (High if > 2 years experience, Low otherwise)")
        if missing_skills:
            st.warning(f"Missing skills: {', '.join(missing_skills)}")
        else:
            st.success("All required skills found." if required_skills else "No required skills specified.")

        condition_met = exp_match  # trigger purely on experience threshold
        st.info(f"Candidate meets experience criteria (> {min_experience} yrs): {condition_met}")

        auto_trigger = st.checkbox(
            "Auto-trigger when candidate meets criteria",
            value=True,
            help="Automatically fire the workflow and email when the candidate passes.",
        )

        trigger_ready = condition_met and N8N_WEBHOOK_URL
        if trigger_ready:
            payload = {
                "document": {
                    "file_name": st.session_state.get("file_name"),
                    "ai_output": ai_output,
                    "ai_summary": ai_output.get("summary"),
                    "raw_text": (st.session_state.get("raw_text") or "")[:500],
                },
                "candidate": {
                    "experience_years": candidate_experience,
                    "min_required_experience": min_experience,
                    "skills_found": sorted(list(candidate_skills)),
                    "required_skills": sorted(list(required_skills)),
                    "missing_skills": missing_skills,
                    "meets_criteria": condition_met,
                    "summary": candidate_summary,
                    "recommendation": recommendation,
                    "risk_level": risk_level,
                },
                "user": {"email": user_email},
                "metadata": {"timestamp": datetime.utcnow().isoformat() + "Z"},
            }

            # Prevent duplicate triggers for the same payload/condition combo.
            trigger_basis = (
                f"{st.session_state.get('file_name')}|"
                f"{json.dumps(ai_output, sort_keys=True)}|"
                f"{min_experience}|{','.join(sorted(required_skills))}"
            )
            trigger_hash = hashlib.md5(trigger_basis.encode()).hexdigest()

            if auto_trigger and st.session_state.get("last_trigger_hash") != trigger_hash:
                try:
                    resp = trigger_n8n(payload)
                    st.success(f"n8n triggered automatically. Status {resp.status_code}")
                    st.session_state["last_trigger_hash"] = trigger_hash
                except Exception as exc:
                    st.error(f"Failed to auto-trigger n8n: {exc}")

            if st.button("Trigger n8n Automation", type="primary"):
                if st.session_state.get("last_trigger_hash") == trigger_hash:
                    st.info("Already triggered for this condition. Change inputs to trigger again.")
                else:
                    try:
                        resp = trigger_n8n(payload)
                        st.success(f"n8n triggered. Status {resp.status_code}")
                        st.session_state["last_trigger_hash"] = trigger_hash
                    except Exception as exc:
                        st.error(f"Failed to trigger n8n: {exc}")


if __name__ == "__main__":
    main()

