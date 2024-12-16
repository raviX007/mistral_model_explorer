import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def initialize_mistral_endpoint(api_key, model_version, max_length=128, temperature=0.7):
    """
    Initialize Mistral LLM endpoint from Hugging Face.
    
    Args:
        api_key (str): Hugging Face API token
        model_version (str): Model repository ID
        max_length (int): Maximum response length
        temperature (float): Sampling temperature
    
    Returns:
        HuggingFaceEndpoint: Initialized language model
    """
    try:
        return HuggingFaceEndpoint(
            repo_id=model_version, 
            max_length=max_length, 
            temperature=temperature, 
            token=api_key
        )
    except Exception as e:
        st.error(f"Error initializing model {model_version}: {e}")
        return None

def initialize_local_model(model_id="gpt2", max_new_tokens=100):
    """
    Initialize a local model pipeline.
    
    Args:
        model_id (str): Model identifier
        max_new_tokens (int): Maximum new tokens to generate
    
    Returns:
        HuggingFacePipeline: Initialized local model pipeline
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error initializing local model {model_id}: {e}")
        return None

def main():
    """
    Streamlit main application for Hugging Face model interactions.
    """
    st.set_page_config(page_title="Hugging Face Model Explorer", page_icon="🤖")
    
    # Title and description
    st.title("🤖 Hugging Face Model Playground")
    st.markdown("Explore different language models from Hugging Face!")
    
    # Sidebar for API Key and Model Selection
    st.sidebar.header("🔑 Configuration")
    
    # API Key Input
    api_key = st.sidebar.text_input("Hugging Face API Token", type="password")
    
    # Model Selection
    model_options = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "gpt2"
    ]
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    
    # Temperature and Max Length Sliders
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    max_length = st.sidebar.slider("Max Response Length", 50, 500, 128)
    
    # Query Input
    query = st.text_input("Enter your query:")
    
    # Process and Display Results
    if api_key and query:
        st.subheader("Model Response")
        
        try:
            # Different model handling based on selection
            if selected_model.startswith("mistralai"):
                # Mistral Models (Endpoint)
                llm = initialize_mistral_endpoint(
                    api_key, 
                    selected_model, 
                    max_length=max_length, 
                    temperature=temperature
                )
                
                if llm:
                    # Create a prompt template for step-by-step reasoning
                    template = """Question: {question}
                    Answer: Let's think step by step."""
                    prompt = PromptTemplate(template=template, input_variables=["question"])
                    
                    # Create LLM Chain
                    llm_chain = LLMChain(llm=llm, prompt=prompt)
                    
                    # Invoke and display response
                    response = llm_chain.invoke(query)
                    st.write(response['text'])
            
            elif selected_model == "gpt2":
                # Local GPT-2 Model
                hf_pipeline = initialize_local_model(max_new_tokens=max_length)
                
                if hf_pipeline:
                    # Create a prompt template
                    template = """Question: {question}
                    Answer: Let's think step by step."""
                    prompt = PromptTemplate.from_template(template)
                    
                    # Create Chain
                    chain = prompt | hf_pipeline
                    
                    # Invoke and display response
                    response = chain.invoke({"question": query})
                    st.write(response)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    elif query and not api_key:
        st.warning("Please enter your Hugging Face API Token in the sidebar.")

if __name__ == "__main__":
    main()