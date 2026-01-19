import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Course Recommendation System",
    page_icon="🎓",
    layout="centered"
)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;color:#0B5394;'>🎓 AI-Based Smart Course & Platform Recommendation System</h1>",
    unsafe_allow_html=True
)

st.write(
    "This system recommends **multiple courses, platforms, and certification options** based on your **studying year and interest**."
)

st.divider()

# ---------------- COURSE DATASET ----------------
data = [
    # Programming
    ["Python for Everybody", "python programming basics", "Programming", "Beginner",
     "Coursera", "Free", "Yes", "https://www.coursera.org/specializations/python"],

    ["Complete Python Bootcamp", "python full course", "Programming", "Beginner",
     "Udemy", "Paid", "Yes", "https://www.udemy.com/course/complete-python-bootcamp/"],

    ["Python Full Course", "python tutorial coding", "Programming", "Beginner",
     "YouTube", "Free", "No", "https://www.youtube.com/@freecodecamp"],

    # Web Development
    ["Web Development Full Course", "html css javascript web", "Web Development", "Beginner",
     "YouTube", "Free", "No", "https://www.youtube.com/@TraversyMedia"],

    ["React – The Complete Guide", "react javascript frontend", "Web Development", "Intermediate",
     "Udemy", "Paid", "Yes", "https://www.udemy.com/course/react-the-complete-guide/"],

    ["Full Stack Web Development", "mern stack full stack", "Web Development", "Advanced",
     "Coursera", "Paid", "Yes", "https://www.coursera.org/specializations/full-stack-react"],

    # Machine Learning
    ["Machine Learning – Andrew Ng", "machine learning ai models", "Machine Learning", "Intermediate",
     "Coursera", "Free", "Yes", "https://www.coursera.org/learn/machine-learning"],

    ["Machine Learning Full Course", "machine learning tutorial", "Machine Learning", "Beginner",
     "YouTube", "Free", "No", "https://www.youtube.com/@codebasics"],

    ["Deep Learning Specialization", "deep learning neural networks", "Machine Learning", "Advanced",
     "Coursera", "Paid", "Yes", "https://www.coursera.org/specializations/deep-learning"],

    # Data Science
    ["Data Science with Python", "data analysis statistics python", "Data Science", "Intermediate",
     "Coursera", "Paid", "Yes", "https://www.coursera.org/specializations/data-science-python"],

    ["Data Science Full Course", "data science tutorial", "Data Science", "Beginner",
     "YouTube", "Free", "No", "https://www.youtube.com/@simplilearn"],

    # Cyber Security
    ["Cyber Security Fundamentals", "network security linux", "Cyber Security", "Beginner",
     "Udemy", "Paid", "Yes", "https://www.udemy.com/course/cyber-security-course/"],

    ["Cyber Security Full Course", "cyber security basics", "Cyber Security", "Beginner",
     "YouTube", "Free", "No", "https://www.youtube.com/@NetworkChuck"]
]

df = pd.DataFrame(
    data,
    columns=["Course", "Description", "Interest", "Level", "Platform", "Type", "Certificate", "Link"]
)

# ---------------- STUDY YEAR → LEVEL MAP ----------------
year_level_map = {
    "1st Year": ["Beginner"],
    "2nd Year": ["Beginner", "Intermediate"],
    "3rd Year": ["Intermediate"],
    "4th Year": ["Advanced"]
}

# ---------------- USER INPUT ----------------
name = st.text_input("👤 Enter your name")

study_year = st.selectbox(
    "🎓 Select your studying year",
    ["1st Year", "2nd Year", "3rd Year", "4th Year"]
)

interest = st.selectbox(
    "🎯 Select your interest",
    ["Programming", "Web Development", "Machine Learning", "Data Science", "Cyber Security"]
)

skills = st.text_area(
    "🧠 Enter your current skills (optional)",
    placeholder="python, html, statistics"
)

certificate_needed = st.selectbox(
    "📜 Certificate required?",
    ["Any", "Yes", "No"]
)

payment_preference = st.selectbox(
    "💰 Course Payment Preference",
    ["Any", "Free", "Paid"]
)

st.divider()

# ---------------- AI RECOMMENDATION LOGIC ----------------
if st.button("🚀 Get Course Recommendations", use_container_width=True):

    allowed_levels = year_level_map[study_year]

    corpus = df["Description"].tolist()
    corpus.append(skills.lower())

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])[0]
    df["Match Score"] = (similarity * 100).astype(int)

    results = df[
        (df["Interest"] == interest) &
        (df["Level"].isin(allowed_levels))
    ]

    if certificate_needed != "Any":
        results = results[results["Certificate"] == certificate_needed]

    if payment_preference != "Any":
        results = results[results["Type"] == payment_preference]

    results = results.sort_values(by="Match Score", ascending=False).head(8)

    # ---------------- OUTPUT ----------------
    st.success(f"Hello **{name}** 👋")
    st.write(f"🎓 **Year:** {study_year} | 🎯 **Interest:** {interest}")
    st.divider()

    if results.empty:
        st.info("No suitable courses found. Try changing your preferences.")
    else:
        for _, row in results.iterrows():
            st.markdown(f"### 📘 {row['Course']}")
            st.progress(row["Match Score"])

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"🏫 **Platform:** {row['Platform']}")
                st.write(f"🎓 **Level:** {row['Level']}")
                st.write(f"📜 **Certificate:** {row['Certificate']}")

            with col2:
                if row["Type"] == "Free":
                    st.success("💰 FREE COURSE")
                else:
                    st.warning("💳 PAID COURSE")

            st.markdown(f"🔗 **Course Link:** [Open Course]({row['Link']})")
            st.divider()

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center;color:gray;'>AI-Based Course & Platform Recommendation Project</p>",
    unsafe_allow_html=True
)
