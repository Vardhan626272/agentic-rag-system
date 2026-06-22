from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    print("🤖 Waking up Local AI Agent...")
    
    # 1. Load the database we just built
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 2. Set up the retriever (the tool the agent uses to search)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 3. Load the Local LLM (Llama 3 via Ollama)
    print("🦙 Connecting to Ollama (Llama 3)...")
    llm = Ollama(model="llama3")

    # 4. The System Prompt (This prevents hallucinations)
    template = """You are an expert research assistant. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer based strictly on the context, just say that you don't know. DO NOT guess or hallucinate.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    custom_rag_prompt = PromptTemplate.from_template(template)

    # 5. Build the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ Agent is ready! Type 'quit' to exit.\n")
    
    # 6. The Chat Loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        print("Agent is thinking (searching database)...")
        response = rag_chain.invoke(user_input)
        print(f"\nAgent: {response}\n")

if __name__ == "__main__":
    main()