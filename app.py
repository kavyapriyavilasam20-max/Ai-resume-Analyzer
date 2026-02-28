import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Function to extract text from PDF
# -------------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# -------------------------------
# Function to clean text
# -------------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = text.lower()
    return text

# -------------------------------
# Function to calculate similarity
# -------------------------------
def calculate_similarity(resume_text, job_description):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors)[0][1]
    return round(similarity * 100, 2)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and compare it with a job description.")

# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

# Job Description Input
job_description = st.text_area("Paste Job Description Here")

if uploaded_file is not None and job_description != "":
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_text = clean_text(resume_text)
    job_description = clean_text(job_description)

    score = calculate_similarity(resume_text, job_description)

    st.subheader("🔎 Match Score")
    st.success(f"Your Resume matches {score}% with the Job Description")

    if score > 70:
        st.info("🔥 Excellent match! You have strong alignment.")
    elif score > 40:
        st.warning("⚠ Moderate match. Consider improving keywords.")
    else:
        st.error("❌ Low match. Customize your resume for better results.")
