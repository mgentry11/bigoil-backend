"""
BigOil.net Backend API
Recruiting tools backend - Interview Analyzer, Email Extractor, etc.
"""

import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app, origins=['https://bigoil.net', 'https://www.bigoil.net', 'http://bigoil.net',
                   'https://d3bqphztmq3l8s.cloudfront.net', 'http://localhost:*'])

# Environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
SUPABASE_REST_URL = f"{SUPABASE_URL}/rest/v1" if SUPABASE_URL else ''


def get_user_id_from_request():
    """Extract user ID from authorization header"""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        # For now, use a default user ID for unauthenticated requests
        # In production, validate JWT token
        return 'anonymous'
    return 'anonymous'


# =============================================
# JOB DESCRIPTIONS API
# =============================================

@app.route('/api/job-descriptions', methods=['GET', 'OPTIONS'])
def list_job_descriptions():
    """List all job descriptions"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        user_id = get_user_id_from_request()

        response = requests.get(
            f"{SUPABASE_REST_URL}/job_descriptions",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}',
                'Content-Type': 'application/json'
            },
            params={'order': 'created_at.desc'}
        )

        if response.ok:
            return jsonify(response.json())
        return jsonify([])

    except Exception as e:
        print(f"Error listing jobs: {e}")
        return jsonify([])


@app.route('/api/job-descriptions', methods=['POST'])
def create_job_description():
    """Create a new job description"""
    try:
        data = request.get_json()

        if not data.get('title') or not data.get('description'):
            return jsonify({'error': 'title and description required'}), 400

        response = requests.post(
            f"{SUPABASE_REST_URL}/job_descriptions",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            },
            json={
                'title': data['title'],
                'company': data.get('company', ''),
                'description': data['description']
            }
        )

        if response.ok:
            return jsonify(response.json()[0])
        return jsonify({'error': 'Failed to create job'}), 500

    except Exception as e:
        print(f"Error creating job: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/job-descriptions/<job_id>', methods=['GET', 'DELETE', 'OPTIONS'])
def job_description_detail(job_id):
    """Get or delete a job description"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if request.method == 'GET':
            response = requests.get(
                f"{SUPABASE_REST_URL}/job_descriptions",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}'
                },
                params={'id': f'eq.{job_id}'}
            )
            if response.ok and response.json():
                return jsonify(response.json()[0])
            return jsonify({'error': 'Not found'}), 404

        elif request.method == 'DELETE':
            response = requests.delete(
                f"{SUPABASE_REST_URL}/job_descriptions",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}'
                },
                params={'id': f'eq.{job_id}'}
            )
            return jsonify({'success': True})

    except Exception as e:
        print(f"Error with job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/job-descriptions/<job_id>/transcripts', methods=['GET', 'OPTIONS'])
def job_transcripts(job_id):
    """Get transcripts for a job"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        response = requests.get(
            f"{SUPABASE_REST_URL}/interview_transcripts",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}'
            },
            params={
                'job_id': f'eq.{job_id}',
                'order': 'created_at.desc'
            }
        )
        return jsonify(response.json() if response.ok else [])

    except Exception as e:
        print(f"Error getting transcripts: {e}")
        return jsonify([])


@app.route('/api/job-descriptions/<job_id>/reports', methods=['GET', 'OPTIONS'])
def job_reports(job_id):
    """Get reports for a job"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        response = requests.get(
            f"{SUPABASE_REST_URL}/interview_reports",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}'
            },
            params={
                'job_id': f'eq.{job_id}',
                'order': 'created_at.desc'
            }
        )
        return jsonify(response.json() if response.ok else [])

    except Exception as e:
        print(f"Error getting reports: {e}")
        return jsonify([])


# =============================================
# TRANSCRIPTS API
# =============================================

@app.route('/api/transcripts', methods=['POST', 'OPTIONS'])
def create_transcript():
    """Create a new transcript"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()

        if not data.get('candidate_name') or not data.get('transcript'):
            return jsonify({'error': 'candidate_name and transcript required'}), 400

        response = requests.post(
            f"{SUPABASE_REST_URL}/interview_transcripts",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            },
            json={
                'job_id': data.get('job_id'),
                'candidate_name': data['candidate_name'],
                'transcript': data['transcript'],
                'resume': data.get('resume'),
                'source': data.get('source', 'paste'),
                'filename': data.get('filename')
            }
        )

        if response.ok:
            return jsonify(response.json()[0])
        return jsonify({'error': 'Failed to create transcript'}), 500

    except Exception as e:
        print(f"Error creating transcript: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/transcripts/<transcript_id>', methods=['DELETE', 'OPTIONS'])
def delete_transcript(transcript_id):
    """Delete a transcript"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        response = requests.delete(
            f"{SUPABASE_REST_URL}/interview_transcripts",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}'
            },
            params={'id': f'eq.{transcript_id}'}
        )
        return jsonify({'success': True})

    except Exception as e:
        print(f"Error deleting transcript: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================
# INTERVIEW REPORTS API
# =============================================

@app.route('/api/interview-reports/<report_id>', methods=['GET', 'PATCH', 'DELETE', 'OPTIONS'])
def interview_report_detail(report_id):
    """Get, update, or delete an interview report"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if request.method == 'GET':
            response = requests.get(
                f"{SUPABASE_REST_URL}/interview_reports",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}'
                },
                params={'id': f'eq.{report_id}'}
            )
            if response.ok and response.json():
                return jsonify(response.json()[0])
            return jsonify({'error': 'Not found'}), 404

        elif request.method == 'PATCH':
            data = request.get_json()
            update_data = {}
            if 'candidate_name' in data:
                update_data['candidate_name'] = data['candidate_name']
            if 'job_title' in data:
                update_data['job_title'] = data['job_title']

            response = requests.patch(
                f"{SUPABASE_REST_URL}/interview_reports",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}',
                    'Content-Type': 'application/json'
                },
                params={'id': f'eq.{report_id}'},
                json=update_data
            )
            return jsonify({'success': True})

        elif request.method == 'DELETE':
            response = requests.delete(
                f"{SUPABASE_REST_URL}/interview_reports",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}'
                },
                params={'id': f'eq.{report_id}'}
            )
            return jsonify({'success': True})

    except Exception as e:
        print(f"Error with report {report_id}: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================
# ANALYZE TRANSCRIPTS API
# =============================================

@app.route('/api/analyze-transcripts', methods=['POST', 'OPTIONS'])
def analyze_transcripts():
    """Analyze selected transcripts with AI"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        transcript_ids = data.get('transcript_ids', [])
        job_id = data.get('job_id')

        if not transcript_ids or not job_id:
            return jsonify({'error': 'transcript_ids and job_id required'}), 400

        # Get job details
        job_response = requests.get(
            f"{SUPABASE_REST_URL}/job_descriptions",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}'
            },
            params={'id': f'eq.{job_id}'}
        )

        if not job_response.ok or not job_response.json():
            return jsonify({'error': 'Job not found'}), 404

        job = job_response.json()[0]
        results = []

        for transcript_id in transcript_ids:
            try:
                # Get transcript
                t_response = requests.get(
                    f"{SUPABASE_REST_URL}/interview_transcripts",
                    headers={
                        'apikey': SUPABASE_KEY,
                        'Authorization': f'Bearer {SUPABASE_KEY}'
                    },
                    params={'id': f'eq.{transcript_id}'}
                )

                if not t_response.ok or not t_response.json():
                    results.append({'transcript_id': transcript_id, 'success': False, 'error': 'Not found'})
                    continue

                transcript = t_response.json()[0]

                # Analyze with AI
                analysis = analyze_interview_with_ai(job, transcript)

                # Save report
                report_data = {
                    'job_id': job_id,
                    'transcript_id': transcript_id,
                    'candidate_name': transcript['candidate_name'],
                    'job_title': job['title'],
                    'fit_score': analysis.get('fit_score', 50),
                    'full_response': analysis
                }

                save_response = requests.post(
                    f"{SUPABASE_REST_URL}/interview_reports",
                    headers={
                        'apikey': SUPABASE_KEY,
                        'Authorization': f'Bearer {SUPABASE_KEY}',
                        'Content-Type': 'application/json',
                        'Prefer': 'return=representation'
                    },
                    json=report_data
                )

                if save_response.ok:
                    results.append({
                        'transcript_id': transcript_id,
                        'success': True,
                        'report_id': save_response.json()[0]['id']
                    })
                else:
                    results.append({'transcript_id': transcript_id, 'success': False, 'error': 'Save failed'})

            except Exception as e:
                print(f"Error analyzing transcript {transcript_id}: {e}")
                results.append({'transcript_id': transcript_id, 'success': False, 'error': str(e)})

        return jsonify({'results': results})

    except Exception as e:
        print(f"Error in analyze_transcripts: {e}")
        return jsonify({'error': str(e)}), 500


def analyze_interview_with_ai(job, transcript):
    """Use OpenAI to analyze interview transcript"""
    import openai

    if not OPENAI_API_KEY:
        return {
            'fit_score': 50,
            'hiring_recommendation': 'maybe',
            'strengths': ['AI not configured'],
            'concerns': ['Set OPENAI_API_KEY'],
            'summary': 'AI analysis unavailable'
        }

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""Analyze this interview transcript for the job position and provide assessment.

JOB TITLE: {job['title']}
COMPANY: {job.get('company', 'Not specified')}
JOB DESCRIPTION:
{job['description']}

CANDIDATE: {transcript['candidate_name']}
INTERVIEW TRANSCRIPT:
{transcript['transcript']}

{f"CANDIDATE RESUME: {transcript['resume']}" if transcript.get('resume') else ""}

Return JSON only:
{{
    "fit_score": <0-100>,
    "hiring_recommendation": "<strong_yes/yes/maybe/no/strong_no>",
    "fit_score_breakdown": {{
        "technical_skills": <0-100>,
        "experience_match": <0-100>,
        "cultural_fit": <0-100>,
        "communication": <0-100>
    }},
    "strengths": [{{"strength": "desc", "evidence": "quote"}}],
    "concerns": [{{"concern": "desc", "evidence": "quote"}}],
    "matching_skills": ["skill1", "skill2"],
    "missing_skills": ["skill1", "skill2"],
    "notable_quotes": [{{"quote": "text", "context": "why significant"}}],
    "follow_up_questions": ["q1", "q2"],
    "summary": "2-3 paragraph summary"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert HR analyst. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )

        result_text = response.choices[0].message.content.strip()

        # Clean markdown
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]

        return json.loads(result_text.strip())

    except Exception as e:
        print(f"AI analysis error: {e}")
        return {
            'fit_score': 50,
            'hiring_recommendation': 'maybe',
            'strengths': [],
            'concerns': [str(e)],
            'summary': f'Analysis error: {str(e)}'
        }


# =============================================
# EMAIL EXTRACTOR API
# =============================================

@app.route('/api/extract-email', methods=['POST', 'OPTIONS'])
def extract_email():
    """Extract email from LinkedIn URL"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        linkedin_url = data.get('linkedin_url', '')

        if not linkedin_url:
            return jsonify({'error': 'LinkedIn URL required'}), 400

        match = re.search(r'linkedin\.com/in/([^/?]+)', linkedin_url)
        if not match:
            return jsonify({'error': 'Invalid LinkedIn URL'}), 400

        # Try ContactOut
        contactout_key = os.getenv('CONTACTOUT_API_KEY')
        if contactout_key:
            try:
                resp = requests.get(
                    'https://api.contactout.com/v2/person',
                    params={'linkedin_url': linkedin_url},
                    headers={'Authorization': f'Bearer {contactout_key}'},
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return jsonify({
                        'email': data.get('email', ''),
                        'work_email': data.get('work_email', ''),
                        'phone': data.get('phone', ''),
                        'source': 'contactout'
                    })
            except Exception as e:
                print(f"ContactOut error: {e}")

        # Try Hunter.io
        hunter_key = os.getenv('HUNTER_API_KEY')
        if hunter_key:
            try:
                resp = requests.get(
                    'https://api.hunter.io/v2/linkedin-email-finder',
                    params={'linkedin_url': linkedin_url, 'api_key': hunter_key},
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json().get('data', {})
                    return jsonify({
                        'email': data.get('email', ''),
                        'work_email': '',
                        'phone': '',
                        'source': 'hunter'
                    })
            except Exception as e:
                print(f"Hunter error: {e}")

        return jsonify({
            'email': '',
            'work_email': '',
            'phone': '',
            'error': 'No email finder API configured',
            'source': 'none'
        })

    except Exception as e:
        print(f"Email extract error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================
# LINKEDIN PDF PARSER API
# =============================================

@app.route('/api/parse-linkedin-pdf', methods=['POST', 'OPTIONS'])
def parse_linkedin_pdf():
    """Parse LinkedIn PDF profile"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename.endswith('.pdf'):
            return jsonify({'error': 'Must be PDF'}), 400

        import pdfplumber
        import io

        pdf_bytes = io.BytesIO(file.read())
        text = ''
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''

        lines = text.split('\n')
        result = {
            'name': lines[0].strip() if lines else file.filename.replace('.pdf', ''),
            'headline': lines[1].strip() if len(lines) > 1 else '',
            'location': lines[2].strip() if len(lines) > 2 else '',
            'skills': []
        }

        return jsonify(result)

    except Exception as e:
        print(f"PDF parse error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================
# RESUME FRAUD DETECTION API
# =============================================

@app.route('/api/analyze-resume-fraud', methods=['POST', 'OPTIONS'])
def analyze_resume_fraud():
    """Analyze resume for potential fraud indicators using AI"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        candidate = data.get('candidate', {})

        if not candidate:
            return jsonify({'error': 'candidate data required'}), 400

        # Use AI for fraud detection if available
        if OPENAI_API_KEY:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            prompt = f"""Analyze this resume/profile for potential fraud indicators.

CANDIDATE DATA:
Name: {candidate.get('name', 'Unknown')}
Headline: {candidate.get('headline', '')}
Location: {candidate.get('location', '')}
About: {candidate.get('about', '')}
Skills: {', '.join(candidate.get('skills', []))}
Experience: {json.dumps(candidate.get('experience', []))}
Education: {json.dumps(candidate.get('education', []))}

Analyze for:
1. Timeline inconsistencies (overlapping jobs, impossible career progression)
2. Exaggerated or unverifiable claims
3. Generic buzzword-heavy content suggesting AI generation
4. Missing critical information
5. Suspicious patterns in experience or education

Return JSON only:
{{
    "status": "<verified/suspicious/flagged>",
    "score": <0-100 confidence score>,
    "flags": ["list of specific concerns"],
    "summary": "brief assessment"
}}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a fraud detection expert. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1000
                )

                result_text = response.choices[0].message.content.strip()
                if result_text.startswith('```'):
                    result_text = result_text.split('```')[1]
                    if result_text.startswith('json'):
                        result_text = result_text[4:]
                if result_text.endswith('```'):
                    result_text = result_text[:-3]

                return jsonify(json.loads(result_text.strip()))

            except Exception as e:
                print(f"AI fraud analysis error: {e}")

        # Fallback to basic rule-based detection
        return jsonify(perform_basic_fraud_check(candidate))

    except Exception as e:
        print(f"Fraud analysis error: {e}")
        return jsonify({'error': str(e)}), 500


def perform_basic_fraud_check(candidate):
    """Basic rule-based fraud detection"""
    flags = []
    risk_score = 0

    skills = candidate.get('skills', [])
    experience = candidate.get('experience', [])
    education = candidate.get('education', [])
    about = candidate.get('about', '')

    if len(skills) > 30:
        flags.append(f"Excessive skills listed ({len(skills)})")
        risk_score += 20

    if len(experience) == 0:
        flags.append("No work experience listed")
        risk_score += 15

    if len(education) == 0:
        flags.append("No education listed")
        risk_score += 5

    if not candidate.get('location'):
        flags.append("No location provided")
        risk_score += 5

    generic_phrases = ['results-driven', 'team player', 'passionate about', 'proven track record']
    generic_count = sum(1 for p in generic_phrases if p in about.lower())
    if generic_count >= 3:
        flags.append("About section uses many generic buzzwords")
        risk_score += 10

    status = 'verified'
    if risk_score >= 30:
        status = 'flagged'
    elif risk_score >= 15:
        status = 'suspicious'

    return {
        'status': status,
        'score': 100 - risk_score,
        'flags': flags
    }


# =============================================
# CANDIDATE FIT ANALYSIS API
# =============================================

@app.route('/api/analyze-candidate-fit', methods=['POST', 'OPTIONS'])
def analyze_candidate_fit():
    """Analyze how well a candidate fits a job description"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        candidate = data.get('candidate', {})
        job = data.get('job', {})

        if not candidate or not job:
            return jsonify({'error': 'candidate and job data required'}), 400

        # Use AI for fit analysis if available
        if OPENAI_API_KEY:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            prompt = f"""Analyze how well this candidate fits the job position.

JOB:
Title: {job.get('title', '')}
Company: {job.get('company', '')}
Description: {job.get('description', '')}
Requirements: {', '.join(job.get('requirements', []))}

CANDIDATE:
Name: {candidate.get('name', '')}
Headline: {candidate.get('headline', '')}
Skills: {', '.join(candidate.get('skills', []))}
Experience: {json.dumps(candidate.get('experience', []))}
Education: {json.dumps(candidate.get('education', []))}
About: {candidate.get('about', '')}

Return JSON only:
{{
    "overallScore": <0-100>,
    "skillsScore": <0-100>,
    "experienceScore": <0-100>,
    "educationScore": <0-100>,
    "matchedSkills": ["skill1", "skill2"],
    "missingSkills": ["skill1", "skill2"],
    "recommendations": ["rec1", "rec2"],
    "analyzedAt": "{__import__('datetime').datetime.now().isoformat()}"
}}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert HR analyst. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )

                result_text = response.choices[0].message.content.strip()
                if result_text.startswith('```'):
                    result_text = result_text.split('```')[1]
                    if result_text.startswith('json'):
                        result_text = result_text[4:]
                if result_text.endswith('```'):
                    result_text = result_text[:-3]

                return jsonify(json.loads(result_text.strip()))

            except Exception as e:
                print(f"AI fit analysis error: {e}")

        # Fallback to basic keyword matching
        return jsonify(perform_basic_fit_analysis(candidate, job))

    except Exception as e:
        print(f"Fit analysis error: {e}")
        return jsonify({'error': str(e)}), 500


def perform_basic_fit_analysis(candidate, job):
    """Basic keyword matching for fit analysis"""
    from datetime import datetime

    job_text = (job.get('description', '') + ' ' + ' '.join(job.get('requirements', []))).lower()
    candidate_text = (
        candidate.get('about', '') + ' ' +
        ' '.join(candidate.get('skills', [])) + ' ' +
        ' '.join([e.get('title', '') + ' ' + e.get('company', '') for e in candidate.get('experience', [])])
    ).lower()

    keywords = [w for w in job_text.split() if len(w) > 3]
    unique_keywords = list(set(keywords))

    matches = 0
    matched_skills = []
    missing_skills = []

    for keyword in unique_keywords:
        if keyword in candidate_text:
            matches += 1
            if any(keyword in s.lower() for s in candidate.get('skills', [])):
                matched_skills.append(keyword)
        else:
            if any(keyword in r.lower() for r in job.get('requirements', [])):
                missing_skills.append(keyword)

    fit_score = min(100, int((matches / max(len(unique_keywords), 1)) * 150))
    skills_score = min(100, int(len(matched_skills) / max(len(candidate.get('skills', [])), 1) * 100))
    exp_score = 70 if candidate.get('experience') else 30
    edu_score = 70 if candidate.get('education') else 40

    recommendations = []
    if fit_score >= 80:
        recommendations.append('Strong candidate - consider for immediate interview')
    elif fit_score >= 60:
        recommendations.append('Good potential - may need skills assessment')
    else:
        recommendations.append('May not be the best fit - review carefully')

    return {
        'overallScore': fit_score,
        'skillsScore': skills_score,
        'experienceScore': exp_score,
        'educationScore': edu_score,
        'matchedSkills': list(set(matched_skills))[:10],
        'missingSkills': list(set(missing_skills))[:5],
        'recommendations': recommendations,
        'analyzedAt': datetime.now().isoformat()
    }


# =============================================
# HEALTH CHECK
# =============================================

# =============================================
# VOICEDAY - NAGGING APP API
# =============================================

@app.route('/api/voiceday/register', methods=['POST', 'OPTIONS'])
def voiceday_register():
    """Register or update a VoiceDay user"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        device_id = data.get('device_id')
        name = data.get('name', 'Anonymous')
        phone = data.get('phone', '')
        push_token = data.get('push_token', '')

        if not device_id:
            return jsonify({'error': 'device_id required'}), 400

        # Check if user exists
        check_response = requests.get(
            f"{SUPABASE_REST_URL}/voiceday_users",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}'
            },
            params={'device_id': f'eq.{device_id}'}
        )

        user_data = {
            'device_id': device_id,
            'name': name,
            'phone': phone,
            'push_token': push_token
        }

        if check_response.ok and check_response.json():
            # Update existing user
            response = requests.patch(
                f"{SUPABASE_REST_URL}/voiceday_users",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}',
                    'Content-Type': 'application/json',
                    'Prefer': 'return=representation'
                },
                params={'device_id': f'eq.{device_id}'},
                json=user_data
            )
        else:
            # Create new user
            response = requests.post(
                f"{SUPABASE_REST_URL}/voiceday_users",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}',
                    'Content-Type': 'application/json',
                    'Prefer': 'return=representation'
                },
                json=user_data
            )

        if response.ok:
            return jsonify(response.json()[0] if isinstance(response.json(), list) else response.json())
        return jsonify({'error': 'Registration failed'}), 500

    except Exception as e:
        print(f"VoiceDay register error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voiceday/connections', methods=['GET', 'POST', 'OPTIONS'])
def voiceday_connections():
    """Manage VoiceDay connections (family/friends)"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if request.method == 'GET':
            device_id = request.args.get('device_id')
            if not device_id:
                return jsonify({'error': 'device_id required'}), 400

            response = requests.get(
                f"{SUPABASE_REST_URL}/voiceday_connections",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}'
                },
                params={
                    'or': f'(owner_device_id.eq.{device_id},connected_device_id.eq.{device_id})',
                    'order': 'created_at.desc'
                }
            )
            return jsonify(response.json() if response.ok else [])

        elif request.method == 'POST':
            data = request.get_json()
            owner_device_id = data.get('owner_device_id')
            connected_phone = data.get('connected_phone')
            relationship = data.get('relationship', 'Friend')
            nickname = data.get('nickname', '')

            if not owner_device_id or not connected_phone:
                return jsonify({'error': 'owner_device_id and connected_phone required'}), 400

            # Look up connected user by phone
            user_response = requests.get(
                f"{SUPABASE_REST_URL}/voiceday_users",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}'
                },
                params={'phone': f'eq.{connected_phone}'}
            )

            connected_device_id = None
            if user_response.ok and user_response.json():
                connected_device_id = user_response.json()[0].get('device_id')

            response = requests.post(
                f"{SUPABASE_REST_URL}/voiceday_connections",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}',
                    'Content-Type': 'application/json',
                    'Prefer': 'return=representation'
                },
                json={
                    'owner_device_id': owner_device_id,
                    'connected_device_id': connected_device_id,
                    'connected_phone': connected_phone,
                    'relationship': relationship,
                    'nickname': nickname
                }
            )

            if response.ok:
                return jsonify(response.json()[0])
            return jsonify({'error': 'Failed to create connection'}), 500

    except Exception as e:
        print(f"VoiceDay connections error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voiceday/connections/<connection_id>', methods=['DELETE', 'OPTIONS'])
def voiceday_delete_connection(connection_id):
    """Delete a connection"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        response = requests.delete(
            f"{SUPABASE_REST_URL}/voiceday_connections",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}'
            },
            params={'id': f'eq.{connection_id}'}
        )
        return jsonify({'success': True})

    except Exception as e:
        print(f"VoiceDay delete connection error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voiceday/shared-tasks', methods=['GET', 'POST', 'OPTIONS'])
def voiceday_shared_tasks():
    """Manage shared tasks between users"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if request.method == 'GET':
            device_id = request.args.get('device_id')
            if not device_id:
                return jsonify({'error': 'device_id required'}), 400

            response = requests.get(
                f"{SUPABASE_REST_URL}/voiceday_shared_tasks",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}'
                },
                params={
                    'or': f'(owner_device_id.eq.{device_id},assigned_device_id.eq.{device_id})',
                    'order': 'created_at.desc'
                }
            )
            return jsonify(response.json() if response.ok else [])

        elif request.method == 'POST':
            data = request.get_json()

            response = requests.post(
                f"{SUPABASE_REST_URL}/voiceday_shared_tasks",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}',
                    'Content-Type': 'application/json',
                    'Prefer': 'return=representation'
                },
                json={
                    'owner_device_id': data.get('owner_device_id'),
                    'assigned_device_id': data.get('assigned_device_id'),
                    'assigned_phone': data.get('assigned_phone'),
                    'title': data.get('title'),
                    'deadline': data.get('deadline'),
                    'priority': data.get('priority', 'medium'),
                    'nag_interval_minutes': data.get('nag_interval_minutes', 15)
                }
            )

            if response.ok:
                return jsonify(response.json()[0])
            return jsonify({'error': 'Failed to create shared task'}), 500

    except Exception as e:
        print(f"VoiceDay shared tasks error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voiceday/shared-tasks/<task_id>', methods=['PATCH', 'DELETE', 'OPTIONS'])
def voiceday_update_shared_task(task_id):
    """Update or delete a shared task"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if request.method == 'PATCH':
            data = request.get_json()
            response = requests.patch(
                f"{SUPABASE_REST_URL}/voiceday_shared_tasks",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}',
                    'Content-Type': 'application/json',
                    'Prefer': 'return=representation'
                },
                params={'id': f'eq.{task_id}'},
                json=data
            )
            if response.ok:
                return jsonify(response.json()[0])
            return jsonify({'error': 'Failed to update task'}), 500

        elif request.method == 'DELETE':
            response = requests.delete(
                f"{SUPABASE_REST_URL}/voiceday_shared_tasks",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}'
                },
                params={'id': f'eq.{task_id}'}
            )
            return jsonify({'success': True})

    except Exception as e:
        print(f"VoiceDay update task error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voiceday/nag', methods=['POST', 'OPTIONS'])
def voiceday_send_nag():
    """Send a nag message to a user"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        from_device_id = data.get('from_device_id')
        to_device_id = data.get('to_device_id')
        to_phone = data.get('to_phone')
        task_id = data.get('task_id')
        message = data.get('message', 'Time to get your task done!')

        if not from_device_id or (not to_device_id and not to_phone):
            return jsonify({'error': 'from_device_id and (to_device_id or to_phone) required'}), 400

        # Store nag in database
        nag_response = requests.post(
            f"{SUPABASE_REST_URL}/voiceday_nags",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            },
            json={
                'from_device_id': from_device_id,
                'to_device_id': to_device_id,
                'to_phone': to_phone,
                'task_id': task_id,
                'message': message
            }
        )

        # If recipient has the app and a push token, send push notification
        if to_device_id:
            user_response = requests.get(
                f"{SUPABASE_REST_URL}/voiceday_users",
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}'
                },
                params={'device_id': f'eq.{to_device_id}'}
            )

            if user_response.ok and user_response.json():
                push_token = user_response.json()[0].get('push_token')
                if push_token:
                    # TODO: Implement push notification via APNs
                    pass

        result = {
            'success': True,
            'nag_id': nag_response.json()[0].get('id') if nag_response.ok else None,
            'delivery_method': 'app' if to_device_id else 'sms_required'
        }

        return jsonify(result)

    except Exception as e:
        print(f"VoiceDay nag error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voiceday/nags', methods=['GET', 'OPTIONS'])
def voiceday_get_nags():
    """Get nags for a user"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        device_id = request.args.get('device_id')
        if not device_id:
            return jsonify({'error': 'device_id required'}), 400

        response = requests.get(
            f"{SUPABASE_REST_URL}/voiceday_nags",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}'
            },
            params={
                'or': f'(from_device_id.eq.{device_id},to_device_id.eq.{device_id})',
                'order': 'created_at.desc',
                'limit': '50'
            }
        )
        return jsonify(response.json() if response.ok else [])

    except Exception as e:
        print(f"VoiceDay get nags error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voiceday/nags/<nag_id>/acknowledge', methods=['POST', 'OPTIONS'])
def voiceday_acknowledge_nag(nag_id):
    """Mark a nag as acknowledged"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        from datetime import datetime

        response = requests.patch(
            f"{SUPABASE_REST_URL}/voiceday_nags",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
            },
            params={'id': f'eq.{nag_id}'},
            json={'acknowledged_at': datetime.utcnow().isoformat()}
        )

        if response.ok:
            return jsonify({'success': True})
        return jsonify({'error': 'Failed to acknowledge nag'}), 500

    except Exception as e:
        print(f"VoiceDay acknowledge error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voiceday/lookup-user', methods=['GET', 'OPTIONS'])
def voiceday_lookup_user():
    """Look up a user by phone number"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        phone = request.args.get('phone')
        if not phone:
            return jsonify({'error': 'phone required'}), 400

        response = requests.get(
            f"{SUPABASE_REST_URL}/voiceday_users",
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}'
            },
            params={'phone': f'eq.{phone}'}
        )

        if response.ok and response.json():
            user = response.json()[0]
            return jsonify({
                'found': True,
                'has_app': True,
                'name': user.get('name', 'Unknown'),
                'device_id': user.get('device_id')
            })
        else:
            return jsonify({
                'found': False,
                'has_app': False
            })

    except Exception as e:
        print(f"VoiceDay lookup error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'bigoil-backend'})


@app.route('/debug-env', methods=['GET'])
def debug_env():
    """Debug endpoint to check environment variables (remove in production)"""
    # Read fresh from os.environ to bypass any caching
    fresh_url = os.environ.get('SUPABASE_URL', '')
    fresh_key = os.environ.get('SUPABASE_KEY', '')

    # Get all env var names (not values) for debugging
    all_env_names = sorted([k for k in os.environ.keys()])

    return jsonify({
        'supabase_url_set': bool(fresh_url),
        'supabase_url_length': len(fresh_url) if fresh_url else 0,
        'supabase_url_preview': fresh_url[:30] + '...' if fresh_url and len(fresh_url) > 30 else fresh_url,
        'supabase_key_set': bool(fresh_key),
        'supabase_key_length': len(fresh_key) if fresh_key else 0,
        'all_env_var_names': all_env_names,
        'total_env_vars': len(all_env_names)
    })


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'BigOil.net API',
        'version': '1.2.0',
        'endpoints': [
            '/api/job-descriptions',
            '/api/transcripts',
            '/api/interview-reports',
            '/api/analyze-transcripts',
            '/api/extract-email',
            '/api/parse-linkedin-pdf',
            '/api/analyze-resume-fraud',
            '/api/analyze-candidate-fit',
            '/api/voiceday/register',
            '/api/voiceday/connections',
            '/api/voiceday/shared-tasks',
            '/api/voiceday/nag',
            '/api/voiceday/nags',
            '/api/voiceday/lookup-user'
        ]
    })


if __name__ == '__main__':
    print("Starting BigOil.net Backend API")
    app.run(debug=True, host='0.0.0.0', port=5000)
