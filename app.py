"""Streamlit app: upload policy PDFs, ask questions, get cited answers.

Run:  streamlit run app.py
"""
import os

import streamlit as st

from src.config import ANTHROPIC_MODEL, DATA_DIR, TOP_K
from src.generator import answer
from src.ingest import build_index
from src.retriever import index_size, retrieve

DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="University Policy Assistant", page_icon="🎓", layout="wide")
st.title("🎓 University Policy Q&A Assistant")
st.caption(
    "Answers come only from the documents you provide, with page-level citations. "
    "This is not official advice. Always confirm with your university."
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is not set. Add it to a .env file and restart the app.")
    st.stop()


def show_sources(sources):
    with st.expander(f"Retrieved passages ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}**, page {s['page']} (distance {s['distance']:.3f})")
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


with st.sidebar:
    st.header("Documents")
    st.write(f"Indexed chunks: **{index_size()}**")
    for name in sorted(p.name for p in DATA_DIR.glob("*.pdf")):
        st.write(f"• {name}")

    uploads = st.file_uploader("Add PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Index documents", use_container_width=True):
        for f in uploads or []:
            (DATA_DIR / f.name).write_bytes(f.getbuffer())
        with st.spinner("Reading, chunking and embedding documents..."):
            n_chunks = build_index(verbose=False)
        if n_chunks:
            st.success(f"Indexed {n_chunks} chunks.")
        else:
            st.warning("No readable PDFs found. Upload text-based PDFs and try again.")
        st.rerun()

    st.header("Settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, TOP_K)
    st.caption(f"Model: {ANTHROPIC_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

question = st.chat_input("Ask about a policy, for example: How do I appeal a grade?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if index_size() == 0:
            reply = "No documents are indexed yet. Upload PDFs in the sidebar and click **Index documents**."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            with st.spinner("Searching documents..."):
                chunks = retrieve(question, k=top_k)
                try:
                    result = answer(question, chunks)
                except Exception as exc:
                    result = {"answer": f"Sorry, the model request failed: {exc}", "sources": []}
            st.markdown(result["answer"])
            if result["sources"]:
                show_sources(result["sources"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
