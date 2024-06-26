import spacy
import re
import pandas as pd
import streamlit as st
import torch
import numpy as np
from transformers import BertJapaneseTokenizer, BertModel
from tqdm import tqdm
import fugashi

st.title('類似文章検索アプリ')

uploaded_file = st.file_uploader('CSVを選択', type='csv')
model = st.selectbox('モデルを選択してください',['Doc2Vec','BERT'])

if uploaded_file is not None:
    df_jis = pd.read_csv(uploaded_file, encoding='cp932', dtype='object')
    jis_col = st.selectbox('対象列選択', df_jis.columns)
    jis_key = st.selectbox('キー選択', df_jis.columns)
    if jis_col != jis_key:
        
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
            text = text.replace(",", "")
            text = text.replace("注記", "")
            text = text.replace("例えば", "")
            text = text.replace("その", "")
            text = re.sub(r'\d+', '', text)
            return text
        
        if model == 'Doc2Vec':
            word_list = list(df_jis[jis_col].astype('str').apply(cleansing))
            no_list = list(df_jis[jis_key])
            company_security_policy_text = st.text_area('検索したい文章を入力してください',height=150)
            doc_company = nlp(cleansing(company_security_policy_text))
            
            def calculate_similarity(doc_company, jis_text):
                doc_jis = nlp(jis_text)
                if not doc_company.has_vector or not doc_jis.has_vector:
                    score = 0
                else:
                    score = doc_company.similarity(doc_jis)
                return score
            
            # max_similarity = 0
            # best_match_jis = ''
            # best_match_index = ''
            
            # for jis_text, no in zip(word_list, no_list):
            #     score = calculate_similarity(doc_company, jis_text)
            #     if score > max_similarity:
            #         max_similarity = score
            #         best_match_jis = jis_text
            #         best_match_index = no

            # st.text(f'類似度：{max_similarity:.4f}')
            # df_output = df_jis.loc[df_jis[jis_key] == best_match_index, [jis_key, jis_col]]

            df_output = df_jis.copy()
            df_output['類似度']=0
            for jis_text, no in zip(word_list, no_list):
                score = calculate_similarity(doc_company, jis_text)
                df_output.loc[df_jis[jis_key] == no,['類似度']]=score
            df_output = df_output.sort_values(by=['類似度'],ascending=False)
            rows = st.number_input('表示行数を入力してください',value=3)
            st.table(df_output.head(rows))
        
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
                if jis_vector.size>0 and company_vector.size > 0:
                    similarity = cosine_sim(jis_vector, company_vector)
                else:
                    similarity = 0
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_jis = jis_sent
                    best_match_index = index

            st.text(f'類似度：{max_similarity[0]:.4f}')
            df_output = df_jis.loc[df_jis[jis_key] == best_match_index, [jis_key, jis_col]]
            st.table(df_output)
