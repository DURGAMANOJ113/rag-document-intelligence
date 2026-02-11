\# 🧠 RAG-Based Document Intelligence System



A Retrieval-Augmented Generation (RAG) system that allows users to upload PDFs and ask grounded questions based on document content.



---



\## 🚀 Features



\- Upload PDF documents

\- Automatic text chunking

\- Embedding generation using sentence-transformers

\- FAISS similarity search

\- Grounded response generation using Ollama (LLaMA3)

\- Clean modern UI

\- Transparent retrieved chunks display



---



\## 🏗 Architecture



User → Upload PDF  

→ Text Splitter  

→ Generate Embeddings  

→ Store in FAISS  

→ Similarity Search (Top K chunks)  

→ LLM (Ollama - LLaMA3)  

→ Grounded Answer  



---



\## 📦 Installation



```bash

git clone https://github.com/DURGAMANOJ113/rag-document-intelligence.git

cd rag-document-intelligence

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```



---



\## ▶ Run Application



```bash

python app.py

```



Open in browser:



```

http://127.0.0.1:5000

```



---



\## 🧠 Tech Stack



\- Python

\- Flask

\- LangChain

\- FAISS

\- HuggingFace Embeddings

\- Ollama (LLaMA3)

\- Custom HTML/CSS UI



---



\## 🎯 Use Cases



\- Research paper Q\&A

\- Legal document analysis

\- Internal documentation assistant

\- Knowledge base retrieval systems



---



\## 🚀 Future Improvements



\- Persistent vector database

\- Multi-document indexing

\- Streaming responses

\- Cloud deployment

\- User authentication



---



\## 👨‍💻 Author



Durga Manoj  

GitHub: https://github.com/DURGAMANOJ113



