import spacy
import re
import pandas as pd
import streamlit as st
import torch
import numpy as np
from transformers import BertJapaneseTokenizer, BertModel
from tqdm import tqdm
import fugashi

st.title('規程マッピングアプリ')

uploaded_file = st.file_uploader('CSVを選択', type='csv')
model = st.selectbox('モデルを選択してください',['Doc2Vec','BERT'])

if uploaded_file is not None:
    df_jis = pd.read_csv(uploaded_file, encoding='cp932')
    jis_col = st.selectbox('対象列選択', df_jis.columns)
    jis_key = st.selectbox('キー選択', df_jis.columns)
    if jis_col is not None:
        
        # GiNZAの日本語モデルをロード
        nlp = spacy.load("ja_ginza")
        
        # 前処理の関数
        def cleansing(text):
            text = text.rstrip()
            text = re.sub(r'\n|\s|\t|\u3000|\xa0', '', text)
            text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', '', text)
            text = re.sub(r'^.\）|^.\)|[0-9].[0-9]|[0-9]月', '', text)
            text = re.sub(r',|，|、|。|「|」|『|』|\（|\）|\(|\)|：|:|＝|=|／|/|～|~|・', '', text)
            text = text.upper()
            text = re.sub(r'\d+', '', text)
            return text
        
        if model == 'Doc2Vec':
            word_list = list(df_jis[jis_col].apply(cleansing))
            no_list = list(df_jis[jis_key])
            company_security_policy_text = st.text_area('規程を入力してください')
            doc_company = nlp(cleansing(company_security_policy_text))
            
            def calculate_similarity(doc_company, jis_text):
                doc_jis = nlp(jis_text)
                score = doc_company.similarity(doc_jis)
                return score
            
            max_similarity = 0
            best_match_jis = ''
            best_match_index = ''
            
            for jis_text, no in zip(word_list, no_list):
                score = calculate_similarity(doc_company, jis_text)
                if score > max_similarity:
                    max_similarity = score
                    best_match_jis = jis_text
                    best_match_index = no
            
            df_output = df_jis.loc[df_jis[jis_key] == best_match_index, [jis_key, jis_col]]
            st.table(df_output)
        
        if model == 'BERT':
            tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
            model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
            BATCH_SIZE = 16
            
            def preprocess_text(text):
                tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
                return tokens
            
            def get_sentence_vector_batch(tokens_batch):
                with torch.no_grad():
                    outputs = model(**tokens_batch)
                sentence_vectors = outputs.last_hidden_state.mean(dim=1).numpy()
                return sentence_vectors
            
            word_list = list(df_jis[jis_col].apply(cleansing))
            no_list = list(df_jis[jis_key])
            company_security_policy_text = st.text_area('規程を入力してください')
            company_sent = nlp(cleansing(company_security_policy_text))
            company_vector = get_sentence_vector_batch(preprocess_text(company_sent.text))
            
            progress_text = '処理中...'
            my_bar = st.progress(0, text=progress_text)
            
            def calculate_vectors(texts):
                vectors = []
                for i in tqdm(range(0, len(texts), BATCH_SIZE)):
                    batch_texts = texts[i:i + BATCH_SIZE]
                    my_bar.progress(i / len(texts), text=progress_text)
                    tokens_batch = tokenizer(batch_texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
                    vectors_batch = get_sentence_vector_batch(tokens_batch)
                    vectors.extend(vectors_batch)
                return np.array(vectors)
            
            jis_vectors = calculate_vectors([nlp(text).text for text in word_list])
            my_bar.empty()
            
            def cosine_sim(a, b):
                return (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            max_similarity = 0
            best_match_jis = ''
            best_match_index = ''
            
            for jis_vector, jis_sent, index in zip(jis_vectors, word_list, no_list):
                similarity = cosine_sim(jis_vector, company_vector)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_jis = jis_sent
                    best_match_index = index
            
            df_output = df_jis.loc[df_jis[jis_key] == best_match_index, [jis_key, jis_col]]
            st.table(df_output)
