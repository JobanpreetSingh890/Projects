from langchain_community.document_loaders import RecursiveUrlLoader

def load_website(url):

    loader = RecursiveUrlLoader(
        url = url,
        max_depth=3
    )

    docs = loader.load()

    return docs