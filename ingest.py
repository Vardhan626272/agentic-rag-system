from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    try:
        print("📚 Loading knowledge.pdf...")
        loader = PyPDFLoader("knowledge.pdf")
        docs = loader.load()

        print("✂️ Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        print("🧠 Creating FREE local embeddings...")
        # Using HuggingFace's free all-MiniLM-L6-v2 model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
        
        print("💾 Saving database...")
        Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")

        print("🎉 DONE! Local database successfully created in 'chroma_db'.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()