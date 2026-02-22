import streamlit as st
import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# AWS S3 Config
bucket_name = "documindai-bucket-isha"

s3 = boto3.client("s3")

st.title("📄 DocuMind AI")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    # Save locally
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Upload to S3
    s3.upload_file("temp.pdf", bucket_name, "temp.pdf")

    st.success("Uploaded to S3 Successfully!")

    # RAG Processing
    loader = PyPDFLoader("temp.pdf")
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

        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model.generate(
            **inputs,
            max_new_tokens=200
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Answer")
        st.write(answer)