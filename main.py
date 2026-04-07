import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
)

# Load PDF
loader = PyPDFLoader("YourResume.pdf")
documents = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Store
vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever()

# Prompt
prompt_template = """
You are a resume assistant for XXX.

Answer ONLY from the resume context.
If answer is not present, say:

"I don't have that information in my resume."

Resume Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)
while True:
    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})

    print("\nAnswer:", result["result"])
