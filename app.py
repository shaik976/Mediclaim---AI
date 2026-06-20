import os
import re
import sys
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
from collections import defaultdict

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Policies database (hardcoded)
policies = {
    "SH-10234": {
        "insurer": "Star Health",
        "type": "Family Floater",
        "max_amount": 500000,
        "covered_conditions": ["Diabetes", "Cardiac", "Surgery", "Kidney"]
    },
    "PM-98765": {
        "insurer": "PMJAY",
        "type": "Government Scheme",
        "max_amount": 500000,
        "covered_conditions": ["All conditions"]
    },
    "HD-45612": {
        "insurer": "HDFC Ergo",
        "type": "Individual",
        "max_amount": 300000,
        "covered_conditions": ["Diabetes", "Ortho", "General Surgery"]
    },
    "NI-33210": {
        "insurer": "New India Assurance",
        "type": "Group Health",
        "max_amount": 300000,
        "covered_conditions": ["Diabetes", "Surgery", "Maternity"]
    },
    "MA-77654": {
        "insurer": "ManipalCigna",
        "type": "Individual",
        "max_amount": 1000000,
        "covered_conditions": ["All conditions", "Critical Illness"]
    }
}

def format_indian_currency(amount):
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return str(amount)
        
    s = f"{amount:.2f}"
    parts = s.split('.')
    integer_part = parts[0]
    decimal_part = parts[1]
    
    reversed_int = integer_part[::-1]
    groups = []
    if len(reversed_int) > 3:
        groups.append(reversed_int[:3])
        remaining = reversed_int[3:]
        for i in range(0, len(remaining), 2):
            groups.append(remaining[i:i+2])
    else:
        groups.append(reversed_int)
        
    formatted_int = ",".join(groups)[::-1]
    if decimal_part == '00':
        return f"₹{formatted_int}"
    return f"₹{formatted_int}.{decimal_part}"

# ---------------------------------------------------------------------------
# 1. RISK SCORING ENGINE (AI-powered)
# ---------------------------------------------------------------------------
class RiskScoringEngine:
    def __init__(self):
        # Maps patient_name.lower().strip() -> list of service dates (date objects)
        self._history = defaultdict(list)

    def score(self, patient_name: str, discharge_summary: str) -> dict:
        import json
        
        # 1. AI evaluation & extraction using Gemini
        system_instruction = """You are the Core Medical Audit Engine for MediClaim AI.
Analyze the discharge summary text. You MUST extract critical fields and evaluate claim risk across three factors:

1. Code/Clinical Compatibility (0-100 risk score):
   - Evaluate whether the diagnoses clinically justify the procedures/treatments described.
   - 0-20: Standard procedure, highly relevant and expected for the diagnosis.
   - 21-50: Partially relevant, or minor mismatch.
   - 51-100: Completely unrelated, unjustified, or highly suspicious (e.g. Chest X-ray for isolated knee pain).

2. Billing Anomaly (0-100 risk score):
   - Evaluate if the billed amount mentioned in the summary is reasonable for the treatment performed.
   - 0-20: Expected/normal cost range for this service.
   - 21-50: Slightly elevated, or lacks clear itemization.
   - 51-100: Highly abnormal/inflated compared to standard pricing (e.g. $600 for a routine EKG).

3. Length of Stay (0-100 risk score):
   - Evaluate whether the number of days admitted matches the clinical course and severity of the condition.
   - 0-20: Appropriate length of stay.
   - 21-50: Slightly excessive or insufficient observation stay.
   - 51-100: Highly disproportionate stay (e.g. 5 days for a minor observation, or 0 days for major surgery).

You MUST respond with a valid JSON object matching this schema:
{
  "diagnosis": "Extracted primary diagnosis (e.g., Essential Hypertension)",
  "treatment": "Extracted main treatments/procedures (e.g., 12-lead EKG, routine observation)",
  "billed_amount": 150.0, // Extracted billed amount as a float (numeric only, strip currency symbols)
  "days_admitted": 1, // Extracted length of stay in days as an integer
  "compatibility_score": 10.0, // Risk score 0-100
  "compatibility_rationale": "Brief clinical explanation of the compatibility risk",
  "billing_score": 25.0, // Risk score 0-100
  "billing_rationale": "Brief explanation of the billing risk",
  "length_of_stay_score": 15.0, // Risk score 0-100
  "length_of_stay_rationale": "Brief explanation of the stay duration risk",
  "flags": ["Warning flag 1", "Warning flag 2"] // List of specific warnings or anomalies found. Empty list if none.
}
"""

        prompt = f"Analyze Discharge Summary:\n{discharge_summary}"
        
        extracted = {}
        models_to_try = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest", "gemini-2.0-flash"]
        last_err = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
                )
                response = model.generate_content(
                    contents=prompt,
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json"
                    }
                )
                extracted = json.loads(response.text.strip())
                break
            except Exception as e:
                print(f"[WARNING] Extraction failed on model {model_name}: {str(e)}", file=sys.stderr)
                last_err = e

        if not extracted:
            # Safe default fallback
            extracted = {
                "diagnosis": "Unspecified Diagnosis",
                "treatment": "Unspecified Treatment",
                "billed_amount": 150.0,
                "days_admitted": 1,
                "compatibility_score": 40.0,
                "compatibility_rationale": "AI extraction failed, using safe defaults.",
                "billing_score": 40.0,
                "billing_rationale": "AI extraction failed, using safe defaults.",
                "length_of_stay_score": 40.0,
                "length_of_stay_rationale": "AI extraction failed, using safe defaults.",
                "flags": ["AI Extraction failed: " + str(last_err)]
            }

        # 2. Historical Frequency Pattern Calculation
        patient_key = patient_name.lower().strip()
        past_admissions = self._history[patient_key]
        current_date = date.today()
        
        # Filter past admissions in the last 30 days
        recent_admissions = [d for d in past_admissions if 0 <= (current_date - d).days <= 30]
        occurrence_count = len(recent_admissions) + 1
        
        # Add current admission to history
        past_admissions.append(current_date)
        
        if occurrence_count <= 2:
            freq_score = 0.0
        else:
            # Score climbs to 100 on the 5th admission in 30 days
            progress = (occurrence_count - 2) / 3.0
            freq_score = min(100.0, progress * 100.0)
            extracted.setdefault("flags", []).append(
                f"Patient has been admitted/validated {occurrence_count} times in the last 30 days"
            )

        # 3. Combine scores
        # Compatibility (45%), Billing (30%), Frequency/Length of Stay (25%)
        los_score = float(extracted.get("length_of_stay_score", 0.0))
        billing_score = float(extracted.get("billing_score", 0.0))
        compatibility_score = float(extracted.get("compatibility_score", 0.0))
        
        final_freq_stay_score = max(los_score, freq_score)
        
        final_score = (
            compatibility_score * 0.45
            + billing_score * 0.30
            + final_freq_stay_score * 0.25
        )
        final_score = round(min(100.0, max(0.0, final_score)), 2)
        
        # Assign risk band and triage decision
        # 0-21: LOW, 21-50: MEDIUM, 50-100: HIGH
        if final_score <= 21.0:
            risk_band = "low"
            triage_decision = "Auto Approve"
        elif final_score <= 50.0:
            risk_band = "medium"
            triage_decision = "Send for Review"
        else:
            risk_band = "high"
            triage_decision = "Escalate Claim"
            
        # Collect rationales and reasons
        reasons = []
        if compatibility_score >= 21.0:
            reasons.append(extracted.get("compatibility_rationale", "High compatibility risk."))
        if billing_score >= 21.0:
            reasons.append(extracted.get("billing_rationale", "High billing risk."))
        if final_freq_stay_score >= 21.0:
            if freq_score > los_score:
                reasons.append(f"High admission frequency warning ({occurrence_count} times in 30 days).")
            else:
                reasons.append(extracted.get("length_of_stay_rationale", "High length of stay risk."))
        
        # Add flags to reasons list
        for flag in extracted.get("flags", []):
            if flag not in reasons:
                reasons.append(flag)

        return {
            "diagnosis": extracted.get("diagnosis", "Unspecified"),
            "treatment": extracted.get("treatment", "Unspecified"),
            "billed_amount": float(extracted.get("billed_amount", 0.0)),
            "days_admitted": int(extracted.get("days_admitted", 1)),
            "compatibility_score": round(compatibility_score, 2),
            "billing_score": round(billing_score, 2),
            "frequency_score": round(final_freq_stay_score, 2),
            "final_score": final_score,
            "risk_band": risk_band,
            "triage_decision": triage_decision,
            "reasons": reasons
        }

# Global persistent engine instance
scoring_engine = RiskScoringEngine()

# API configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable is not set. Please set it in your environment or .env file.", file=sys.stderr)
else:
    genai.configure(api_key=GEMINI_API_KEY)

def get_icd10_codes(diagnosis):
    """
    Step 1: ClinicalTables NLM API Lookup
    """
    print(f"[DEBUG] Step 1: Querying NLM ClinicalTables API for diagnosis: '{diagnosis}'")
    url = f"https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?sf=code,name&terms={diagnosis}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse data structure: [total_count, codes, null, [[code, name], [code, name], ...]]
        if len(data) >= 4 and data[3]:
            # Extract top 3 matches
            top_matches = data[3][:3]
            print(f"[DEBUG] Step 1 Success: Found {len(top_matches)} ICD-10 code matches.")
            return top_matches
        else:
            print("[DEBUG] Step 1 Info: NLM ClinicalTables returned no matches.")
            return []
    except Exception as e:
        print(f"[ERROR] Step 1 Failed: NLM ClinicalTables API error: {str(e)}", file=sys.stderr)
        return []

def get_policy_coverage(policy_number, insurer, diagnosis):
    """
    Step 2: Policy Database Lookup / SerpAPI Fallback
    """
    print(f"[DEBUG] Step 2: Checking policies dict for Policy Number: '{policy_number}'")
    
    if policy_number in policies:
        policy_info = policies[policy_number]
        print(f"[DEBUG] Step 2 Success: Policy found in local database. Details: {policy_info}")
        return {
            "source": "Local Database",
            "policy": policy_info,
            "details_text": f"Insurer: {policy_info['insurer']}, Policy Type: {policy_info['type']}, Max Coverage (Sum Insured): Rs. {policy_info['max_amount']}, Covered Conditions: {', '.join(policy_info['covered_conditions'])}"
        }
    
    # If not found in local database, use SerpAPI fallback
    print(f"[DEBUG] Step 2: Policy '{policy_number}' not found in local database.")
    if SERPAPI_KEY:
        query = f"{insurer} insurance coverage {diagnosis} India"
        print(f"[DEBUG] Step 2: Performing SerpAPI fallback search. Query: '{query}'")
        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "api_key": SERPAPI_KEY
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            search_data = response.json()
            
            organic_results = search_data.get("organic_results", [])
            search_snippets = []
            for result in organic_results[:3]:
                title = result.get("title", "No Title")
                snippet = result.get("snippet", "No Snippet")
                link = result.get("link", "")
                search_snippets.append(f"Result Title: {title}\nSummary: {snippet}\nReference Link: {link}\n")
            
            if search_snippets:
                details_text = "\n".join(search_snippets)
                print(f"[DEBUG] Step 2 Success: SerpAPI fallback completed. Extracted {len(search_snippets)} web results.")
                return {
                    "source": "Web Search Fallback (SerpAPI)",
                    "details_text": f"Policy not in local database. Web search results for coverage:\n{details_text}"
                }
            else:
                print("[DEBUG] Step 2 Info: SerpAPI search completed but no organic results found.")
        except Exception as e:
            print(f"[ERROR] Step 2 Failed: SerpAPI search failed: {str(e)}", file=sys.stderr)
            
    else:
        print("[DEBUG] Step 2 Info: SerpAPI fallback skipped (SERPAPI_KEY is not set).")
        
    return {
        "source": "None",
        "details_text": f"Policy details and coverage for insurer '{insurer}' and policy '{policy_number}' could not be retrieved. Proceed with standard medical necessity verification under general Indian IRDAI guidelines."
    }

def extract_risk_level(report_text):
    """
    Helper function to parse the risk level from the generated report.
    """
    match = re.search(r'REJECTION RISK:\s*(LOW|MEDIUM|HIGH)', report_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback heuristics
    lower_report = report_text.lower()
    if "rejection risk: low" in lower_report or "risk: low" in lower_report:
        return "LOW"
    elif "rejection risk: high" in lower_report or "risk: high" in lower_report:
        return "HIGH"
    elif "rejection risk: medium" in lower_report or "risk: medium" in lower_report:
        return "MEDIUM"
        
    return "MEDIUM"  # Default fallback if not found

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/validate', methods=['POST'])
def validate_claim():
    print("\n--- NEW VALIDATION REQUEST RECEIVED ---")
    
    # Check if Gemini API is configured
    if not GEMINI_API_KEY:
        print("[ERROR] Request rejected: GEMINI_API_KEY is not configured on the server.")
        return jsonify({
            "error": "Gemini API key is missing. Please set GEMINI_API_KEY in the server environment or .env file."
        }), 500

    data = request.json or {}
    
    # Extract fields
    name = data.get("name", "Unknown Patient")
    age = data.get("age", "")
    gender = data.get("gender", "")
    discharge_summary = data.get("discharge_summary", "")
    hospital = data.get("hospital", "General Hospital")
    insurer = data.get("insurer", "Unspecified Insurer")
    policy_number = data.get("policy_number", "Unspecified Policy")
    
    gender_suffix = f", {gender}" if gender else ""
    
    print(f"[DEBUG] Patient: {name} | Age: {age} | Gender: {gender} | Hospital: {hospital}")
    print(f"[DEBUG] Insurer: {insurer} | Policy: {policy_number}")
    print(f"[DEBUG] Discharge Summary Length: {len(discharge_summary)} characters")
    
    # STEP 1: Medical Necessity Risk Engine (AI Extraction & Scoring)
    risk_result = scoring_engine.score(name, discharge_summary)
    
    extracted_diagnosis = risk_result["diagnosis"]
    extracted_treatment = risk_result["treatment"]
    billed_amount = risk_result["billed_amount"]
    days_admitted = risk_result["days_admitted"]
    
    print(f"[DEBUG] Extracted Diagnosis: {extracted_diagnosis}")
    print(f"[DEBUG] Extracted Treatment: {extracted_treatment}")
    print(f"[DEBUG] Extracted Billed Amount: ${billed_amount:.2f} | Days: {days_admitted}")

    # STEP 2: ICD-10 Registry Lookup
    icd10_matches = get_icd10_codes(extracted_diagnosis)
    if icd10_matches:
        best_code, best_desc = icd10_matches[0]
        other_codes_list = [f"{code} ({desc})" for code, desc in icd10_matches[1:]]
        other_codes = ", ".join(other_codes_list) if other_codes_list else "None"
    else:
        best_code = "U07.1"  # fallback
        best_desc = "Clinical diagnosis requiring confirmation"
        other_codes = "None"
        
    icd10_context = f"Main Code: {best_code} - {best_desc}\nAlternatives: {other_codes}"
    print(f"[DEBUG] Compiled ICD-10 Context:\n{icd10_context}")

    # STEP 3: Policy & Coverage Lookup
    coverage_info = get_policy_coverage(policy_number, insurer, extracted_diagnosis)
    policy_details = coverage_info["details_text"]
    coverage_source = coverage_info["source"]
    
    sum_insured_amount = "Unknown"
    if "policy" in coverage_info:
        sum_insured_amount = f"{coverage_info['policy'].get('max_amount', 500000)}"

    # STEP 4: Gemini Report Generation
    print("[DEBUG] Step 4: Compiling data and prompt for Gemini report...")
    
    necessity_rationale = "; ".join(risk_result["reasons"]) if risk_result["reasons"] else "All engine factors within normal limits."
    triage_decision = risk_result["triage_decision"]
    
    system_prompt = f"""You are MediClaim AI, an expert medical insurance claim validator
for Indian hospitals. Given patient details, ICD-10 codes, policy coverage data, and risk metrics, output EXACTLY in this format:

🏥 MEDICLAIM AI — CLAIM VALIDATION REPORT
==========================================
👤 Patient: {name}, {age}{gender_suffix}
🔍 Diagnosis: {extracted_diagnosis}
💊 Treatment: {extracted_treatment}
🏨 Hospital: {hospital}
📋 Insurer: {insurer} | Policy: {policy_number}

🔢 ICD-10 CODE: {best_code} — {best_desc}
   Alternatives: {other_codes}

⚖️ MEDICAL NECESSITY TRIAGE:
- Billed Amount: {format_indian_currency(billed_amount)}
- Compatibility Score: {risk_result['compatibility_score']:.1f}/100
- Billing Anomaly Score: {risk_result['billing_score']:.1f}/100
- Frequency Pattern Score: {risk_result['frequency_score']:.1f}/100
- Final Risk Score: {risk_result['final_score']:.1f}/100
- Risk Band: {risk_result['risk_band'].upper()}
- Triage Decision: {triage_decision}
- Engine Flags: {necessity_rationale}

✅ COVERAGE CHECK:
- Condition covered: YES/NO
- Sum insured: {format_indian_currency(sum_insured_amount) if sum_insured_amount != 'Unknown' else '[Determine amount based on policy data]'}
- Billed amount: {format_indian_currency(billed_amount)}
- Days admitted: {days_admitted}

📋 DOCUMENTS REQUIRED:
- Discharge summary
- [Include diagnosis specific document(s) like lab reports, MRI, ECG, biopsy, etc. based on the extracted diagnosis]
- Insurance card copy
- Hospital bills (itemized)

⚠️ POTENTIAL ISSUES:
- [List specific coverage gaps, mismatch in treatment/diagnosis, days mismatch, or missing info. Be highly specific.]

🚦 REJECTION RISK: [LOW or MEDIUM or HIGH]
Reason: [Explain why in 1-2 clear sentences. Mention if policy has coverage mismatch or if treatment matches standard practices]

💡 RECOMMENDATION:
[Provide one clear, actionable advice/instruction for the hospital billing team to avoid claim rejection]

Context: Indian healthcare — PMJAY, Star Health, HDFC Ergo, 
IRDAI standards. Pre-submission validator only.
"""

    prompt = f"""
Please validate the following medical claim for pre-submission:

Patient Information:
- Name: {name}
- Age: {age}
- Gender: {gender}
- Admission Days: {days_admitted}
- Hospital: {hospital}
- Diagnosis Text: {extracted_diagnosis}
- Treatment Administered: {extracted_treatment}
- Billed Amount: {format_indian_currency(billed_amount)}

ICD-10 Lookup Results:
{icd10_context}

Policy Database Information (Source: {coverage_source}):
{policy_details}

Medical Necessity Scoring Engine Results:
- Compatibility Score (45% weight): {risk_result['compatibility_score']}
- Billing Score (30% weight): {risk_result['billing_score']}
- Frequency Score (25% weight): {risk_result['frequency_score']}
- Final Score: {risk_result['final_score']}
- Risk Band: {risk_result['risk_band'].upper()}
- Triage Decision: {triage_decision}
- Reasons/Flags: {necessity_rationale}

Generate the validator report following the exact format requested.
"""
    
    models_to_try = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest", "gemini-2.0-flash"]
    report_text = None
    last_error = None
    
    for model_name in models_to_try:
        try:
            print(f"[DEBUG] Trying model '{model_name}'...")
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )
            response = model.generate_content(
                contents=prompt,
                generation_config={"temperature": 0.2}
            )
            report_text = response.text
            print(f"[DEBUG] Step 4 Success: Gemini successfully generated the validation report using model '{model_name}'.")
            break
        except Exception as e:
            print(f"[WARNING] Model '{model_name}' failed: {str(e)}", file=sys.stderr)
            last_error = e

    if report_text is None:
        print(f"[ERROR] Step 4 Failed: All Gemini models failed. Last error: {str(last_error)}", file=sys.stderr)
        return jsonify({
            "error": f"Failed to run Gemini claim validation reasoning: {str(last_error)}"
        }), 500

    try:
        # STEP 5: Parse Risk Level and Return Response
        risk_level = extract_risk_level(report_text)
        print(f"[DEBUG] Step 5: Parsed Risk Level: '{risk_level}'")
        
        return jsonify({
            "report": report_text,
            "risk_level": risk_level,
            "icd10_code": best_code,
            "icd10_desc": best_desc,
            "diagnosis": extracted_diagnosis,
            "treatment": extracted_treatment,
            "billed_amount": billed_amount,
            "days_admitted": days_admitted,
            "compatibility_score": risk_result["compatibility_score"],
            "billing_score": risk_result["billing_score"],
            "frequency_score": risk_result["frequency_score"],
            "final_score": risk_result["final_score"],
            "risk_band": risk_result["risk_band"],
            "triage_decision": triage_decision,
            "reasons": risk_result["reasons"]
        })
        
    except Exception as e:
        print(f"[ERROR] Step 5 Processing Failed: {str(e)}", file=sys.stderr)
        return jsonify({
            "error": f"Failed to process claim validation report: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)
