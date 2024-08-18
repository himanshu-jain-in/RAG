import streamlit as st
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub

class Document:
    def __init__(self):
        pass

    def upload_files(self):
        uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=["pdf"])
        return uploaded_files

    def embed_documents(self,uploaded_files):
        if "vectors" not in st.session_state:
            st.session_state.embeddings=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={"device":"cpu"},
                                        encode_kwargs={"normalize_embeddings":True})
            st.session_state.docs = []

            if uploaded_files: # check if path is not None
                for uploaded_file in uploaded_files:
                    with open(uploaded_file.name, mode='wb') as w:
                        w.write(uploaded_file.getvalue())
                    st.session_state.loader=PyPDFLoader(uploaded_file.name) # Data Ingestion
                    st.session_state.docs.extend(st.session_state.loader.load()) # Documents Loading

            st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
            st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
            # st.write(st.session_state.final_documents)
            st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

        return st.session_state.vectors

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

doc = Document()

with st.sidebar:
    st.title("Settings")
    st.markdown("""
    <style>
        .css-13sdm1b.e16nr0p33 {
        margin-top: -75px;
        }
    </style>
    """, unsafe_allow_html=True)

    uploaded_files = doc.upload_files() # upload file
    if st.button("Documents Embedding"): # Embed documents
        st.session_state.vectors = doc.embed_documents(uploaded_files) # Get embeded DB
        st.write("Documents are ready to talk!")

    #Get user token
    HF_TOKEN = st.text_input("Enter your Hugging Face Token")
    if HF_TOKEN:
        os.environ['HUGGINGFACE_API_KEY']=HF_TOKEN
        st.write("Hugging Face Token is set")
    else: # Set default token
        st.write("Using default API key.")
        os.environ['HUGGINGFACE_API_KEY']="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"
    # Change temperature settings
    temperature = st.slider("Temperature",min_value=0.01,max_value=1.0,step=0.1)
    # Choose model to use
    repo_id = st.selectbox("Select model",['meta-llama/Meta-Llama-3-8B-Instruct',
    "mistralai/Mistral-7B-Instruct-v0.2",'other'])
    if repo_id=='other':
        repo_id = st.text_input("Enter repo id:")

llm = HuggingFaceEndpoint(repo_id=repo_id,temperature = temperature,max_new_tokens=100)

def process_input(inp):
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    document_chain=create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
    retriever=st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    # start=time.process_time()

    response=retrieval_chain.invoke({'input':inp})

    return response

    # print("Response time :",time.process_time()-start)
    # st.write(response['answer'])
    # return response['answer']

    # # With a streamlit expander
    # with st.expander("Document Similarity Search"):
    #     # Find the relevant chunks
    #     for i, doc in enumerate(response["context"]):
    #         st.write(doc.page_content)
    #         st.write("--------------------------------")

if inp := st.chat_input("Ask your question!"):
    # Display user message in chat message container
    st.chat_message("user").markdown(inp)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": inp})

    rep = process_input(inp)
    if repo_id=="mistralai/Mistral-7B-Instruct-v0.2":
        try:
            rep = rep['answer'].split("AI:")[1].strip()
        except:
            rep = rep['answer'].strip()
    elif repo_id =='meta-llama/Meta-Llama-3-8B-Instruct':
        try:
            rep = rep['answer'].split('System:')[1].split('Human:')[0].strip()
        except:
            rep = rep['answer'].strip()
    response = f"Moni: {rep}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})