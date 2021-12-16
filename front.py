import numpy as np
import pandas as pd
import streamlit as st
from statistics import mean
import requests
import lettria

# Configuration
lettria_key = st.secrets['lettria_key']
model_auth = st.secrets['model_auth']

default_text = "Entrez une phrase pour voir l'état actuel de notre compréhension...#compréhension"
lang = 'en'
data_path = 'test_news.csv'
threshold = 2

def get_predictions(df):
    df['émotion'] = get_predictions_lettria(df['phrase'].to_list())
    df['ressenti'] = get_predictions_model(df)
    return df

def get_predictions_model(df):
    preds_obj = {"sentences": df['phrase'].to_list(),
                 "targets" : df['cible'].to_list()}
    header = {'Authorization': model_auth,}
    preds = requests.post("https://target.sentiment.lettria.com/predict", data=preds_obj, headers=header)
    return preds


def get_predictions_lettria(documents):
    nlp = lettria.NLP(lettria_key)
    for id, document in enumerate(documents):
        nlp.add_document(document, id=id)

    preds = list(map(lambda x: x[0], nlp.emotion_ml_flat))
    return preds


def benchmark_test():
    df_eval = pd.DataFrame(st.session_state['df_eval'])
    if "Modèle générique" not in df_eval.columns:
        df_eval = get_predictions(df_eval.copy())
        st.session_state['df_eval'] = df_eval.to_dict(orient='list')

    st.write("## Phrases d'exemple")

    st.write(df_eval[['phrase','cible','ressenti','émotion']])


def sentence_user():
    df_user = pd.DataFrame(st.session_state['df_user'])

    st.write("## Phrases utilisateurs")
    st.write("Entrez une phrase quelconque.")
    st.session_state['user_text'] = st.text_area(
        "La phrase et la cible doivent être séparés d'un #. Exemple : Malgré la crise, Casino a réussi à maintenir les ventes.#Casino", value=default_text)

    if st.session_state['user_text'] != default_text and len(st.session_state['user_text']) > 0 and (not st.session_state['user_text'] in df_user['sentence'].to_list()):
        df_user_text = pd.DataFrame({"phrase": [st.session_state['user_text'].split("#")[0]], "cible":[st.session_state['user_text'].split("#")[1]], "ressenti": [
                                    None],  "émotion": [None]})
        df_user = pd.concat([df_user, df_user_text])
        st.session_state['df_user'] = df_user.to_dict(orient='list')
        st.session_state['user_text'] = default_text


def benchmark_user():
    df_user = pd.DataFrame(st.session_state['df_user'])

    if st.button("Analyser"):
        df_pred = df_user[['phrase', 'cible']].copy()
        df_pred['y'] = 0
        df_user = get_predictions(
            df_pred)[['phrase', 'cible', 'émotion', 'ressenti']]
        st.session_state['df_user'] = df_user.to_dict(orient='list')

    st.write(df_user)


# FRONT
st.set_page_config(page_title="Lettria", page_icon=":brain:",
                   layout='wide', initial_sidebar_state='auto')


if 'df_eval' not in st.session_state:
    st.session_state['df_eval'] = pd.read_csv(
        data_path, sep=',', encoding="utf-8")[['phrase','cible','y']].sample(frac=1).iloc[:threshold].to_dict(orient='list')


if 'df_user' not in st.session_state:
    st.session_state['df_user'] = pd.DataFrame({"phrase": [],
                                                "cible": [],
                                                "ressenti": [],
                                                "émotion": []}).to_dict(orient='list')

if 'user_text' not in st.session_state:
    st.session_state['user_text'] = default_text

st.write("# Démonstration des possibilités Lettria")

benchmark_test()

sentence_user()

benchmark_user()
