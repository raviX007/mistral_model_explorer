import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain

AVAILABLE_MODELS = {
    "Available Models": [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "gpt2"
    ]
}

# Rest of your imports and other constants remain the same
EXAMPLE_PROMPTS = {
    "General": [
        "Explain the concept of quantum entanglement",
        "Write a short story about a future society",
        "Compare different programming paradigms"
    ],
    "Coding": [
        "Write a Python function for binary search",
        "Explain the concept of dependency injection",
        "Create a simple API endpoint using FastAPI",
        "Implement a sorting algorithm"
    ],
    "Analysis": [
        "Analyze the impact of AI on healthcare",
        "Compare different machine learning algorithms",
        "Explain the blockchain technology stack",
        "Discuss modern software architecture patterns"
    ],
    "Writing": [
        "Write a blog post about sustainable technology",
        "Create a technical documentation template",
        "Draft a project proposal outline",
        "Write a system design document"
    ]
}

def clean_response(text):
    """Clean and format the response text"""
    # Remove any XML-like tags
    import re
    text = re.sub(r'<[^>]+>', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def initialize_llm(api_key, model_version, max_length=128, temperature=0.7):
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

def main():
    st.set_page_config(page_title="LLM Playground", page_icon="🤖", layout="wide")
   
    st.title("🤖 Open Source LLM Playground")
    st.markdown("Experiment with Mistral and GPT-2 models using Hugging Face's endpoints!")
   
    col1, col2 = st.columns([2, 3])
   
    with col1:
        st.sidebar.header("🔑 Configuration")
        api_key = st.sidebar.text_input("Hugging Face API Token", type="password")
       
        # Simplified model selection since we only have one category
        selected_model = st.sidebar.selectbox("Select Model", AVAILABLE_MODELS["Available Models"])
       
        st.sidebar.header("📝 Model Parameters")
        temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.7)
        max_length = st.sidebar.slider("Max Response Length", 50, 2000, 500)
       
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🌟 Model Information")
        st.sidebar.markdown(f"**Selected Model:** {selected_model}")
       
    with col2:
        example_category = st.selectbox("Example Categories", list(EXAMPLE_PROMPTS.keys()))
        example_prompt = st.selectbox("Example Prompts", EXAMPLE_PROMPTS[example_category])
       
        query = st.text_area("Enter your prompt:", value=example_prompt, height=150)
       
        if st.button("Generate Response", use_container_width=True):
            if not api_key:
                st.warning("Please enter your Hugging Face API Token in the sidebar.")
                return
               
            try:
                with st.spinner(f"Generating response using {selected_model}..."):
                    llm = initialize_llm(
                        api_key,
                        selected_model,
                        max_length=max_length,
                        temperature=temperature
                    )
                   
                    if llm:
                        # Enhanced prompt template for better formatting
                        template = """Task: {question}

Please provide a clear and structured response. For coding tasks:
1. Start with a brief explanation
2. Provide the implementation
3. Include example usage and output

Response:"""
                        
                        prompt = PromptTemplate(template=template, input_variables=["question"])
                        llm_chain = LLMChain(llm=llm, prompt=prompt)
                        response = llm_chain.invoke(query)
                        
                        # Clean and format the response
                        cleaned_response = clean_response(response['text'])
                        
                        # Display response in a structured way
                        st.markdown("### 📝 Generated Response:")
                        
                        # Try to detect if response contains code
                        if "```" in cleaned_response:
                            # Split into explanation and code sections
                            parts = cleaned_response.split("```")
                            
                            # Display explanation
                            if parts[0].strip():
                                st.write(parts[0].strip())
                            
                            # Display code in proper code block
                            for i in range(1, len(parts), 2):
                                if i < len(parts):
                                    code = parts[i].strip()
                                    if code.startswith("python"):
                                        code = code[6:]
                                    st.code(code, language="python")
                                    
                                    # Display any explanation after the code block
                                    if i + 1 < len(parts) and parts[i + 1].strip():
                                        st.write(parts[i + 1].strip())
                        else:
                            # Display as regular text if no code blocks detected
                            st.write(cleaned_response)
                            
                        st.success(f"Response generated successfully using {selected_model}!")
                   
            except Exception as e:
                st.error(f"""
                An error occurred: {str(e)}
                
                Troubleshooting steps:
                1. Check your API token
                2. Try reducing the response length
                3. Try a different model
                4. Check your internet connection
                """)

if __name__ == "__main__":
    main()