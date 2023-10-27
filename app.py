# Import necessary libraries
import gradio as gr
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Define a function to display "Loading..." when loading a PDF
def loading_pdf():
    return "Loading..."

# Define a function to process PDF changes
def pdf_changes(pdf_doc, repo_id):
    # Initialize the OnlinePDFLoader to load the PDF document
    loader = OnlinePDFLoader(pdf_doc.name)
    documents = loader.load()

    # Split the loaded documents into chunks using CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Initialize HuggingFaceHubEmbeddings for embeddings
    embeddings = HuggingFaceHubEmbeddings()

    # Create a Chroma vector store from the text chunks and embeddings
    db = Chroma.from_documents(texts, embeddings)

    # Convert the vector store to a retriever
    retriever = db.as_retriever()

    # Initialize an HuggingFaceHub language model (LLM)
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.25, "max_new_tokens": 1000})

    # Create a RetrievalQA chain with the LLM, retriever, and return_source_documents option
    global qa
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

    return "Ready"

# Define a function to add text to a history
def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

# Define a bot function to generate responses
def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response['result']
    return history

# Define an inference function to query the LLM
def infer(query):
    result = qa({"query": query})
    return result

# Define custom CSS styles
css = """
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

# Define a title HTML for the interface
title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF</h1>
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the PDF ;)</p>
"""

# Create the Gradio interface
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)

        with gr.Column():
            # Create a file input for loading PDF
            pdf_doc = gr.File(label="Load a PDF", file_types=['.pdf'], type="file", value="AhmedS_Resume.pdf")

            # Create a dropdown for selecting the LLM
            repo_id = gr.Dropdown(label="LLM", choices=["HuggingFaceH4/zephyr-7b-alpha", "adept/fuyu-8b", "adept/fuyu-8b"], value="HuggingFaceH4/zephyr-7b-alpha")

            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="Waiting...", interactive=False)
                load_pdf = gr.Button("Load PDF to LangChain")

        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        query = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send message")

    # Set up actions for UI elements
    repo_id.change(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    load_pdf.click(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    query.submit(add_text, [chatbot, query], [chatbot, query]).then(bot, chatbot, chatbot)
    submit_btn.click(add_text, [chatbot, query], [chatbot, query]).then(bot, chatbot, chatbot)

# Launch the Gradio interface
demo.launch()
