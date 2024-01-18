from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader 
from langchain import hub 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import time

load_dotenv()

def PDFQA(text: str):
    loader = PyPDFLoader(
        file_path='tublian-challenge/llm-ebook.pdf'
    )

    docs = loader.load()

    text_spitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    splits = text_spitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents = splits,
        embedding = OpenAIEmbeddings()
    )

    retriever = vectorstore.as_retriever()
    prompt = hub.pull('rlm/rag-prompt')
    llm = ChatOpenAI(
        model = 'gpt-3.5-turbo',
        temperature = 0
    )

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(text)

    vectorstore.delete_collection()

    return result

def test_cases():
    print('Test Case 1 Executing', end='')
    question1 = 'What is Large Language Model?'
    result1 = PDFQA(text = question1)
    for _ in range(10):
        print('.', end='')
        time.sleep(1)
    print()
    print('------------------------------')
    print()
    print(f"Question :- {question1}")
    print(f"Answer :- {result1}")
    print()
    time.sleep(2)

    print('Test Case 2 Executing', end='')
    question2 = 'Why are large language model are useful?'
    result2 = PDFQA(text = question2)
    for _ in range(10):
        print('.', end='')
        time.sleep(1)
    print()
    print('------------------------------')
    print()
    print(f"Question :- {question2}")
    print(f"Answer :- {result2}")
    print()

if __name__=='__main__':
    test_cases()

