import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2  # Works on Streamlit Cloud

# ----------------------
# Function to extract text from PDF
# ----------------------
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

# ----------------------
# Function to clean text
# ----------------------
def clean_text(text):
    return text.lower().strip()

# ----------------------
# Function to calculate match %
# ----------------------
def calculate_match(resume_text, job_description_text):
    vectorizer = CountVectorizer().fit_transform([resume_text, job_description_text])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    match_percentage = round(similarity[0][1] * 100, 2)
    return match_percentage

# ----------------------
# Streamlit App
# ----------------------
st.title("📝 AI Resume Analyzer")
st.write("Upload your resume (PDF) and compare it with a Job Description.")

# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Enter Job Description
job_description = st.text_area("Paste Job Description Here", height=200)

if uploaded_file and job_description:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.text_area("📄 Extracted Resume Text (debug)", resume_text, height=200)  # Shows extracted text
    resume_text_clean = clean_text(resume_text)
    job_description_clean = clean_text(job_description)

    match_score = calculate_match(resume_text_clean, job_description_clean)
    st.success(f"✅ Resume matches Job Description by: {match_score}%")
