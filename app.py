import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Set up the Web Page
st.set_page_config(page_title="Local Agentic RAG", page_icon="🤖")
st.title("🤖 Private Document Assistant")
st.caption("100% Free & Local AI powered by Llama 3")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 2. Load the AI Pipeline (Cached so it only loads once)
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = Ollama(model="llama3")
    
    template = """You are an expert research assistant. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer based strictly on the context, just say that you don't know. DO NOT guess or hallucinate.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Load the AI
chain = load_rag_pipeline()

# 3. Create the Chat UI State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle User Input
if prompt := st.chat_input("Ask a question about your document..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show AI response with a loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            response = chain.invoke(prompt)
            st.markdown(response)
            
    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response})