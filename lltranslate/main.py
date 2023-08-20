import streamlit as st
from langchain_extras.llms.together import TogetherLLM
import together
from langchain.prompts import PromptTemplate
from langchain_extras.llms.together import TogetherLLM

TOGETHER_MODEL_NAME = "togethercomputer/llama-2-70b-chat"

@st.cache_data
def get_models():
    return [m['name'] for m in together.Models.list()]

def change_model(previous_model_name: str):
    if is_model_running(previous_model_name):
        stop_model(previous_model_name)

    new_model_name = st.session_state.requested_model_name

    start_model(new_model_name)



def is_model_running(model_name: str):
    running_models = together.Models.instances()
    return model_name in running_models

def start_model(model_name: str | None):
    together.Models.start(model_name)
    st.toast(f"Started model: {model_name}")

def stop_model(model_name: str | None):
    together.Models.stop(model_name or TOGETHER_MODEL_NAME)
    st.toast(f"Stopped model: {model_name}")

st.session_state.setdefault("engilsh_word", "")

@st.cache_data
def translation(english_word):
    if len(english_word)==0:
        return

    prompt = PromptTemplate.from_template("""
        Translate the following english words from english to spanish. Keep in mind colloquialisms and context when performing the translation.)

        {english_word}
    """)

    model = TogetherLLM(  model= TOGETHER_MODEL_NAME,
    temperature = 0.1,
    max_tokens = 1024)

    chain = prompt | LLMChain(llm=TogetherLLM)


st.selectbox("Model", get_models(), on_change=lambda: change_model(st.session_state.requested_model_name), key="requested_model_name")

st.button("Start Model", on_click=lambda: start_model(st.session_state.requested_model_name))
st.button("Stop Model", on_click=lambda: stop_model(st.session_state.requested_model_name))
english_word = st.text_input("English word:")

st.markdown("### Translation:")

st.write(translation(english_word))
