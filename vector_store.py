from langchain_community.vectorstores import FAISS

def create_vectorstore(
    chunks,
    embeddings
):

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    db.save_local(
        "faiss_index"
    )

    return db