from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# 1. Load PDFs
loader = PyPDFDirectoryLoader("data")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# 3. Embeddings (local, free, no API key)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector store
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store ready")

# 5. Local LLM (Mistral via Ollama)
llm = Ollama(
    model="mistral",
    base_url="http://127.0.0.1:11434"
)

print("\nPDF assistant ready. Type a question or 'exit'.\n")

# 6. Question loop
while True:
    query = input("Question: ")
    if query.lower() == "exit":
        break

    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Réponds uniquement avec les informations du document.
Si la réponse n’est pas dans le document, dis :
"Je ne trouve pas la réponse dans le document."

DOCUMENT :
{context}

QUESTION :
{query}
"""

    answer = llm.invoke(prompt)
    print("\nRéponse:\n", answer, "\n")
