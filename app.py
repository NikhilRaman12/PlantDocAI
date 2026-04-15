import streamlit as st
import sys
import os

# Ensure generator.py is in same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from generator import ResponseGenerator
except ImportError:
    ResponseGenerator = None

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Krishi Seva AI – Bharat",
    page_icon="🌾",
    layout="centered"
)

# ── Custom CSS (Dark + Green Premium UI) ────────────────────
st.markdown("""
<style>
.stApp { background-color: #0b1410; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
h1 { color: #e8f5e9; }
.subtitle { color: #a5d6a7; font-size: 15px; margin-top: -10px; margin-bottom: 5px; }
.sanskrit { color: #8d6e63; font-size: 12px; font-style: italic; margin-bottom: 20px; }
.lang-badge { background-color: #1b2b1f; color: #81c784; padding: 4px 10px; border-radius: 12px; font-size: 11px; display: inline-block; margin: 3px 2px; border: 1px solid #2e7d32; }
textarea { border-radius: 10px !important; border: 1px solid #2e7d32 !important; background-color: #111c15 !important; color: #e8f5e9 !important; }
.stButton button[kind="primary"] { width: 100%; border-radius: 10px; height: 48px; font-size: 16px; background-color: #2e7d32; color: white; font-weight: 500; }
.stButton button[kind="primary"]:hover { background-color: #1b5e20; }
.stButton button[kind="secondary"] { background-color: #111c15; color: #81c784; border: 1px solid #2e7d32; border-radius: 8px; font-size: 13px; }
.answer-card { background: linear-gradient(145deg, #111c15, #0e1a13); padding: 22px; border-radius: 14px; border: 1px solid #2e7d32; margin-top: 20px; color: #e8f5e9; font-size: 15px; line-height: 1.7; }
.answer-label { font-weight: 600; font-size: 12px; color: #81c784; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.6px; }
.footer { text-align: center; margin-top: 60px; padding: 20px 10px; font-size: 12px; color: #9e9e9e; border-top: 1px solid #1b2b1f; }
.footer strong { color: #81c784; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────
st.title("🌾 Krishi Seva AI – Bharat")
st.markdown('<p class="subtitle">AI-powered agricultural guidance for Bharat</p>', unsafe_allow_html=True)
st.markdown('<p class="sanskrit">सस्यरक्षणाय भारत कृषक क्षेत्रपतिः</p>', unsafe_allow_html=True)

# ── Language Support ───────────────────────────────────────
st.markdown("""
<div>
    <span class="lang-badge">हिंदी</span>
    <span class="lang-badge">English</span>
    <span class="lang-badge">தமிழ்</span>
    <span class="lang-badge">తెలుగు</span>
    <span class="lang-badge">ಕನ್ನಡ</span>
    <span class="lang-badge">मराठी</span>
    <span class="lang-badge">ગુજરાતી</span>
    <span class="lang-badge">ਪੰਜਾਬੀ</span>
    <span class="lang-badge">বাংলা</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Example Queries ───────────────────────────────────────
examples = [
    "Best pesticide for aphids on wheat?",
    "गेहूं में कीट नियंत्रण कैसे करें?",
    "Symptoms of Fusarium wilt in tomato",
    "How to manage rice blast disease?"
]

st.markdown("**Try example queries:**")
cols = st.columns(2)
selected_example = None
for i, example in enumerate(examples):
    with cols[i % 2]:
        if st.button(example, key=f"ex_{i}", type="secondary"):
            selected_example = example

# ── Input ────────────────────────────────────────────────
query = st.text_area(
    "Your question",
    value=selected_example if selected_example else "",
    placeholder="Ask about crops, pests, diseases, or farming practices...",
    height=100,
    label_visibility="collapsed"
)

ask_clicked = st.button("Ask", type="primary")

# ── Response ─────────────────────────────────────────────
if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question")
    else:
        if ResponseGenerator is None:
            st.error("generator.py not found")
        else:
            try:
                with st.spinner("Processing your query..."):
                    generator = ResponseGenerator()
                    result = generator.run(query)
                    answer = getattr(result, "content", str(result))
                    mode = getattr(result, "mode", None)

                # Detection logic for mode
                if mode == "RAG":
                    st.success("Answer from knowledge base")
                elif mode == "HYBRID":
                    st.info("Answer from hybrid knowledge")
                elif mode == "LLM_FALLBACK":
                    st.warning("Answer from general knowledge")
                else:
                    st.info("Response generated")

                # Display answer
                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-label">Response</div>
                    {answer}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ── Footer ───────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>Krishi Seva AI – Bharat</strong><br>
    Intelligence rooted in soil of Bharat • Powered by AI for every farmer
</div>
""", unsafe_allow_html=True)
