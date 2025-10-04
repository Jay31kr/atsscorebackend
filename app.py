from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '/tmp'  # Use '/tmp' for deployment compatibility
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/score', methods=['POST'])
def score_resume():
    resume_file = request.files.get('resume')
    job_description = request.form.get('jobDescription')

    if not resume_file or not job_description:
        return jsonify({'error': 'Missing resume or job description'}), 400

    filename = secure_filename(resume_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    resume_file.save(filepath)

    resume_text = extract_text(filepath)
    score = calculate_score(resume_text, job_description)

    os.remove(filepath)  # Clean up after scoring
    return jsonify({'score': score})

def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif ext == '.pdf':
        import fitz  # PyMuPDF
        doc = fitz.open(filepath)
        return "\n".join([page.get_text() for page in doc])
    elif ext == '.docx':
        import docx
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ''
def calculate_score(resume, jd):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume, jd])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return round(float(similarity[0][0]) * 100, 2)

if __name__ == '__main__':
    app.run(debug=True)

