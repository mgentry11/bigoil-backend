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
# HEALTH CHECK
# =============================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'bigoil-backend'})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'BigOil.net API',
        'version': '1.0.0',
        'endpoints': [
            '/api/job-descriptions',
            '/api/transcripts',
            '/api/interview-reports',
            '/api/analyze-transcripts',
            '/api/extract-email',
            '/api/parse-linkedin-pdf'
        ]
    })


if __name__ == '__main__':
    print("Starting BigOil.net Backend API")
    app.run(debug=True, host='0.0.0.0', port=5000)
