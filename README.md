## RAG Architecture Overview

This project implements a Retrieval-Augmented Generation (RAG) system using `Streamlit`, `LangChain`, and `Hugging Face` for interactive question-answering based on PDF documents uploaded by the user. The architecture integrates document embedding, retrieval, and generative AI for comprehensive and context-aware responses.

### Key Components

1. **Document Handling and Embedding:**
   - The application allows users to upload PDF documents through a file uploader widget.
   - Once the files are uploaded, the documents are processed using `PyPDFLoader` to extract text data.
   - The text is split into manageable chunks using `RecursiveCharacterTextSplitter` for efficient handling during the embedding and retrieval process.
   - These text chunks are then embedded using a `HuggingFaceBgeEmbeddings` model, which generates vector representations for the text using the `sentence-transformers/all-MiniLM-L6-v2` model.
   - A `FAISS` index is created from these embeddings, enabling efficient vector-based document retrieval.

2. **User Interaction:**
   - Users interact with the system through a chat interface powered by `Streamlit`. They can input questions and view responses directly within the app.
   - The sidebar allows users to upload documents, configure the Hugging Face API token, adjust the temperature for generation, and select or input a custom language model repository ID.

3. **Language Model Integration:**
   - The system utilizes `HuggingFaceEndpoint` to interface with various Hugging Face models. Users can choose between pre-configured models like `meta-llama/Meta-Llama-3-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.2`, or input their own model repository ID.
   - The selected model generates responses to user queries based on the most relevant document chunks retrieved from the FAISS index.

4. **Retrieval and Generation Workflow:**
   - The core of the RAG system is the retrieval process, where the top 3 most relevant document chunks are fetched from the FAISS index based on the user's query.
   - These chunks are passed to a `create_stuff_documents_chain` along with the language model, forming the input for response generation.
   - The system processes the query through this chain and returns a detailed, contextually relevant response, which is displayed in the chat interface.

### Additional Features

- **Chat History:** The application maintains a session-based chat history, displaying past interactions for reference.
- **Dynamic Model Selection:** Users can dynamically switch between different language models or provide custom model endpoints, making the system flexible and adaptable.
- **Customizable Settings:** The application provides options for adjusting generation temperature, enabling fine-tuning of the response creativity and precision.

### Future Enhancements

- **Improved Document Handling:** Implement support for additional document formats and enhanced text processing capabilities.
- **Model Expansion:** Add more pre-configured models or advanced features like model ensemble to improve answer quality.
- **Scalability:** Introduce database-backed storage for embeddings and queries, enabling the system to scale and support larger document sets.
