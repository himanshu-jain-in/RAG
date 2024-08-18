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

To add a "How to Use" section to your README, you can include step-by-step instructions on how to set up, configure, and run the project on a user's local machine. Here's an example of how to structure it:

---

## How to Use

Follow these steps to set up and run the project on your local machine:

### 1. Clone the Repository
First, clone the repository to your local machine using the following command:
```bash
git clone https://github.com/himanshu-jain-in/RAG
cd RAG
```

### 2. Install Dependencies
Make sure you have Python installed (version 3.8+ recommended). Then, install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Set Up Hugging Face API Token
To access the Hugging Face models, you'll need an API token. Create an account on [Hugging Face](https://huggingface.co/) if you don’t have one, then generate an API token from your account settings.

You can either set the token in the application settings (using the sidebar) or export it as an environment variable:
```bash
export HUGGINGFACE_API_KEY='your_huggingface_api_key'
```

### 4. Run the Application
To start the application, run the following command:
```bash
streamlit run app.py
```

### 5. Upload Documents and Interact
1. Once the application is running, open the web interface in your browser.
2. Use the sidebar to upload PDF documents.
3. Click "Documents Embedding" to process the uploaded files.
4. Enter your Hugging Face API token or use the default one provided.
5. Adjust the temperature and select a model (or input your own model ID).
6. Start asking questions in the chat input box, and the system will generate answers based on the documents.

### 6. (Optional) Customize the Model
If you want to use a different model, select "other" from the model dropdown and enter the custom model repository ID. The system will use your specified model for generating responses.

---

### Example Usage

Here’s an example of how to interact with the system:

1. Upload your academic papers, reports, or any PDF documents.
2. Ask a question like: "What are the key findings from the document on CO2 sequestration?"
3. The system will retrieve relevant document sections and provide a detailed response.
