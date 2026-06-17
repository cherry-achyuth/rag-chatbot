from retriever import retrieve_documents

docs = retrieve_documents(
    "What is inheritance?"
)

for doc in docs:
    print(doc.page_content)
    print("="*50)