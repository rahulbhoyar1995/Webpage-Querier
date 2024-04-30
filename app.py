import os
import streamlit as st
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,download_loader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Gemini, HuggingFaceInferenceAPI, OpenAI

# Create Streamlit web app
def main():
    st.title("Webpage Querier by Rahul Bhoyar")
    # Sidebar for customizations
    with st.sidebar:
        st.subheader("Customize Settings")
        loader = download_loader("BeautifulSoupWebReader")()
        hf_token = st.text_input("Enter your Hugging Face token:")
        llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-alpha", token=hf_token)
      
    # Main content area
    st.markdown("Query your Web page data with using this chatbot")

    # User input: Web page link
    url = st.text_input("Enter the URL of the web page:")

    # Create Service Context
    embed_model_uae = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1")
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=800, chunk_overlap=20, embed_model=embed_model_uae)

    # Load documents
    if url:
        documents = loader.load_data(urls=[url])
        st.success("Documents loaded successfully!")
        with st.spinner('Creating Vector Embeddings...'):
            # Create Vector Store Index
            index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)

            # Persist Storage Context
            index.storage_context.persist()

            # Create Query Engine
            query_engine = index.as_query_engine()

        # User input: Query
        query = st.text_input("Ask a question:")
        if query:
            # Run Query
            response = query_engine.query(query)

            # Display Result
            st.markdown(f"**Response:** {response}")
    else:
        st.warning("Please enter a valid URL.")

if __name__ == "__main__":
    main()
