from typing import List

import streamlit as st
import together
from langchain.prompts import PromptTemplate
from langchain_extras.llms.together import TogetherLLM

st.session_state.setdefault("requested_model_name", "togethercomputer/llama-2-70b-chat")


@st.cache_data
def get_models():
    return [m["name"] for m in together.Models.list()]


def is_more_than_one_model_running():
    all_instances = together.Models.instances()
    print(all_instances)
    running_models = [k for (k, m) in all_instances.items() if all_instances[k] is True]
    return (len(running_models) > 1, running_models)


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
    together.Models.stop(model_name)
    st.toast(f"Stopped model: {model_name}")


@st.cache_data(show_spinner=True)
def translation(model_name, english_word, language):
    if len(english_word) == 0 or len(language) == 0:
        return

    prompt = PromptTemplate.from_template(
        """
        You are a professional linguist who works as a language interpreter. You
        have many years of experience speaking many different languages. You have an
        expert understanding of cultural differences in phrasing, word choice, 
        honorifics, colloquialisms. You are now working with a client who is only
        able to communicate over chat. They are a native resident of the state of Ohio
        and have little experience with foreign cultures. They wish to say something in
        another language, but want to be sure they are using correct pronunciation, 
        local vernacular, level of politeness & showing proper respect. Their request may
        include an audience; if it does you must ensure they say it in a way they can make
        a connection with their audience.
        
        You need to tell them how to say it in their chosen language. Include
        an explanation of cultural variations of the phrase and situations in which you
        might use each. Also include a phonetic spelling to help an English speaker 
        learn to recite the word correctly.

        Keep in mind cultural rules around respect and politeness.
        The client should be able to converse without accidentally insulting anyone.
        Keep answers as succinct as possible, and always remain friendly,
        helpful and casual.
        
        [INST]
        
        Client has requested the following:
        
        {english_word}

        and would like to understand how to express their request in:
        
        {language}
        
        [/INST]
    """
    )

    model = TogetherLLM(model=model_name, temperature=0.1, max_tokens=1024)

    chain = prompt | model
    return chain.invoke({"english_word": english_word, "language": language})


models = get_models()


def get_selected_model_index(models: List[str]):
    if len(st.session_state.requested_model_name) == 0:
        return 1
    else:
        return models.index(st.session_state.requested_model_name)


st.selectbox(
    "Model",
    models,
    args=(st.session_state.requested_model_name,),
    on_change=change_model,
    index=get_selected_model_index(models),
    key="requested_model_name",
)
st.text(st.session_state.requested_model_name)

(show_error, running_models) = is_more_than_one_model_running()
if show_error:
    st.error(
        f"""
        Hey boy, you have too many together instances running. Go shut some down!

        {', '.join(running_models)}
        """
    )

args = (st.session_state.requested_model_name,)
st.button("Start Model", args=args, on_click=start_model)
st.button(
    "Stop Model",
    args=args,
    on_click=stop_model,
)
english_word = st.text_input("What do you want to say?")
language = st.text_input("Language")

st.markdown("### Translation:")

st.write(translation(st.session_state.requested_model_name, english_word, language))
