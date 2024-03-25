import argparse
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

def load_llm():
    llm = CTransformers(
        model="res\llama-2-7b-chat.ggmlv3.q4_1.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.9
    )
    return llm

def cricbot(query, qa_chain):
    answer = qa_chain({"query": query})
    return answer["result"]

def main(csv_path, query):
    # Load CSV data
    loader = CSVLoader(file_path=csv_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    # Load language model
    llm = load_llm()

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='thenlper/gte-large', model_kwargs={'device': 'cpu'})

    # Create vector store
    db = FAISS.from_documents(data, embeddings)
    db.save_local('faiss/cricket')

    # Define prompt template
    prompt_temp = '''
    With the information provided try to answer the question. 
    If you cant answer the question based on the information either say you cant find an answer or unable to find an answer.
    This is related to cricket domain. So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

    Context: {context}
    Question: {question}
    Do provide only correct answers

    Correct answer:
    '''
    custom_prompt_temp = PromptTemplate(template=prompt_temp, input_variables=['context', 'question'])

    # Create QA chain
    retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(search_kwargs={'k': 1}),
                                                     chain_type="stuff", return_source_documents=True,
                                                     chain_type_kwargs={"prompt": custom_prompt_temp})

    # Run query
    answer = cricbot(query, retrieval_qa_chain)
    print(f"Query: {query}\nAnswer: {answer}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cricket data query.')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file containing cricket data')
    parser.add_argument('query', type=str, help='Query to run')
    args = parser.parse_args()
    main(args.csv_path, args.query)
