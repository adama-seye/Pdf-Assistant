# Pdf-Assistant

This project is a simple AI-powered study assistant that allows users to ask questions about PDF documents and receive answers based only on the document content.

It was built as a personal learning project to explore how AI tools can be used to improve studying and information retrieval.

## What this project does
- Loads PDF files from a folder
- Splits the text into smaller chunks
- Converts text into vector embeddings
- Stores them in a vector database
- Retrieves the most relevant parts of the document
- Uses an AI model to generate answers based on those parts

If the answer is not found in the document, the assistant says so.

## Technologies used
- Python
- LangChain
- FAISS (vector database)
- HuggingFace embeddings
- OpenAI API (for the language model)
- PyPDF for document loading

## Project structure
