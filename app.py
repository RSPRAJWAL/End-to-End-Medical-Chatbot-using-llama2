from flask import Flask, render_template, request, jsonify 
from dotenv import load_dotenv
import os
import logging
import time

from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import prompt_template  # ensure prompt_template is defined
from src.image_gen import generate_image_url

load_dotenv()

logging.basicConfig(level=logging.INFO)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

embeddings = download_hugging_face_embeddings()

docsearch = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 256,
        'temperature': 0.7
    }
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 1}),
    return_source_documents=False,
    chain_type_kwargs={"prompt": PROMPT}
)

app = Flask(__name__)

cache = {}  # simple cache for queries

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def get_bot_response():
    try:
        user_input = request.json.get("msg", "")
        if not user_input:
            return jsonify({"response": "⚠️ Please provide a message."})

        logging.info(f"User Query: {user_input}")
        result = qa.invoke({"query": user_input})
        text_response = result["result"]

        # ✅ Generate image
        image_url = generate_image_url(user_input)

        return jsonify({
            "response": text_response,
            "image": image_url  # send image URL too
        })

    except Exception as e:
        logging.error(f"Error in chat response: {e}")
        return jsonify({"response": "⚠️ Sorry, something went wrong. Please try again."})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=True)  # Enable multi-threading here
