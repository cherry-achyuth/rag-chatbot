from retriever import retrieve_documents
from generator import generate_answer

def ask_question(query):

    docs = retrieve_documents(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    answer = generate_answer(
        query,
        context
    )

    return answer