import streamlit as st
import boto3
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from huggingface_hub import InferenceClient

# ---------------- AWS CONFIG ---------------- #

bucket_name = "documindai-bucket-isha"
s3 = boto3.client("s3")

# ---------------- HF CONFIG ---------------- #

client = InferenceClient(
    model="google/flan-t5-small",
    token=os.getenv("HF_TOKEN")
)

# ---------------- UI ---------------- #

st.title("📄 DocuMind AI")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    file_name = uploaded_file.name

    # Save locally
    with open(file_name, "wb") as f:
        f.write(uploaded_file.read())

    # Upload to S3
    s3.upload_file(file_name, bucket_name, file_name)
    st.success(f"{file_name} uploaded to S3 successfully!")

    # ----------- RAG PIPELINE ----------- #

    loader = PyPDFLoader(file_name)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)

    st.success("Document processed!")

    question = st.text_input("Ask a question:")

    if question:

        results = db.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in results])

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question: {question}
Answer:
"""

        try:
            answer = client.text2text_generation(
                prompt,
                max_new_tokens=200
            )
        except Exception as e:
            answer = f"HF Error: {str(e)}"

        st.subheader("Answer")
        st.write(answer)
