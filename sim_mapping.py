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
            # 不要なスペース・改行
            text = text.rstrip() #改行･スペース削除
            text = text.replace("", "")
            text = re.sub(r'\n|\s|\t|\u3000|\xa0', '', text) #スペース削除
        
            # text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text) #URL削除
            # text = neologdn.normalize(text) #アルファベット･数字：半角、カタカナ：全角
            text = re.sub(r'^.）', '', text) ##項番（0-9）を削除
            text = re.sub(r'[a-z]', '', text) ##項番（0-9）を削除
            text = re.sub(r'^.\)', '', text) ##項番（0-9）を削除
            text = re.sub(r'[0-9].[0-9]', '', text) #項番を削除
            text = re.sub(r'[0-9]月', '', text) #日付を削除（何月）
            text = re.sub(r'，|、|。', '', text) #句読点を削除
            text = re.sub(r'「|」|『|』|\（|\）|\(|\)', '', text) #カッコを削除
            text = re.sub(r'：|:|＝|=|／|/|～|~|・', '', text) #記号を削除
        
            # ニュースソース
            text = text.replace(",", "")
            text = text.replace("\nb）", "")
            text = text.replace("注記", "")
            text = text.replace("例えば", "")
            text = text.replace("その", "")
            text = text.upper() #アルファベット：大文字
            text = re.sub(r'\d+', '', text) ##アラビア数字を削除
        
            return text
        
        if model == 'Doc2Vec':
            # JIS27001データのインポート
            # df_jis = pd.read_csv('/content/drive/MyDrive/JIS27001/JIS27001.csv', encoding='cp932', index_col=['小項目'])
            word_list = list(df_jis[jis_col].apply(cleansing))
            no_list = list(df_jis[jis_key])
            
            # 会社のセキュリティ規定テキスト
            company_security_policy_text = st.text_area('規程を入力してください')
            
            # 会社のセキュリティ規定テキストを処理
            doc_company = nlp(cleansing(company_security_policy_text))
            
            # 類似度計算の関数
            def calculate_similarity(doc_company, jis_text):
                doc_jis = nlp(jis_text)
                score = doc_company.similarity(doc_jis)
                return score
            
            # JISのテキストと会社の規定テキストの類似度を計算
            max_similarity = 0
            best_match_jis = ''
            best_match_index = ''
            
            for jis_text, no in zip(word_list, no_list):
                score = calculate_similarity(doc_company, jis_text)
                if score > max_similarity:
                    max_similarity = score
                    best_match_jis = jis_text
                    best_match_index = no
            # dict_temp = {'Index':[best_match_index],'内容':[best_match_jis],'類似度':[f'{max_similarity:.4f}']}
            # df_output = pd.DataFrame(dict_temp).set_index(['Index'],drop=True)
            df_output = df_jis.loc[df_jis[jis_key]==best_match_index,[jis_key,jis_col]]
            st.table(df_output)
        
        if model == 'BERT':
            # モデルとトークナイザーのロード
            tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
            model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
            
            # バッチ処理のサイズ
            BATCH_SIZE = 16
            
            # テキストの前処理
            def preprocess_text(text):
                tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
                return tokens
            
            # 文書ベクトルの取得（バッチ処理対応）
            def get_sentence_vector_batch(tokens_batch):
                with torch.no_grad():
                    outputs = model(**tokens_batch)
                # バッチ全体のベクトルとして各文の平均を取る
                sentence_vectors = outputs.last_hidden_state.mean(dim=1).numpy()
                return sentence_vectors
            
            word_list = list(df_jis[jis_col].apply(cleansing))
            no_list = list(df_jis[jis_key])
            
            # 会社のセキュリティ規定テキスト
            company_security_policy_text = st.text_area('規程を入力してください')
            
            # 会社のセキュリティ規定テキストをベクトル化
            company_sent = nlp(cleansing(company_security_policy_text))
            company_vector = get_sentence_vector_batch(preprocess_text(company_sent.text))
            
            progress_text = '処理中...'
            my_bar = st.progress(0, text=progress_text)
            # JISのベクトルを事前計算してキャッシュ
            def calculate_vectors(texts):
                vectors = []
                for i in tqdm(range(0, len(texts), BATCH_SIZE)):
                    batch_texts = texts[i:i + BATCH_SIZE]
                    my_bar.progress(i, text=progress_text)
                    tokens_batch = tokenizer(batch_texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
                    vectors_batch = get_sentence_vector_batch(tokens_batch)
                    vectors.extend(vectors_batch)
                return np.array(vectors)
            
            # JISのベクトルをキャッシュ
            jis_vectors = calculate_vectors([nlp(text).text for text in word_list])
            my_bar.empty()
            
            # 類似度計算関数
            def cosine_sim(a, b):
                return (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            # 類似度計算と最大類似度の文を特定
            max_similarity = 0
            best_match_jis = ''
            best_match_index = ''
            
            for jis_vector, jis_sent, index in zip(jis_vectors, word_list, no_list):
                similarity = cosine_sim(jis_vector, company_vector)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_jis = jis_sent
                    best_match_index = index
            
            # dict_temp = {'Index':[best_match_index],'内容':[best_match_jis],'類似度':[f'{max_similarity[0]:.4f}']}
            # df_output = pd.DataFrame(dict_temp).set_index(['Index'],drop=True)
            df_output = df_jis.loc[df_jis[jis_key]==best_match_index,[jis_key,jis_col]]
            st.table(df_output)
