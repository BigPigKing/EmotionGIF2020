import os
import copy
import string
import numpy as np
import pandas as pd

from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from bert_serving.client import BertClient

TRAINING_PATH = os.path.join("data", "train_gold.json")
CATEGROY_PATH = os.path.join("data", "categories.json")
DEV_PATH = os.path.join("data", "dev_unlabeled.json")

ADDITION_STOPWORDS = ['—', '‘', '’', '“', '”', '\'️'] + list(string.punctuation) + list(string.ascii_lowercase)


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        print('Epoch: {}'.format(self.epoch))
        self.epoch += 1


class Data_preproceesor:
    def __init__(self, embedder_list, category_list, split_ratio=0.9,
                 drop_list=["categories", "idx", "mp4"], token_list=["text", "reply"],
                 Y_name="categories", stopwords=set(stopwords.words('english')),):
        self.embedder_list = embedder_list
        self.category_list = category_list
        self.split_ratio = split_ratio
        self.drop_list = drop_list
        self.token_list = token_list
        self.Y_name = Y_name
        self.stopwords = stopwords

    def _extract_gold_XY(self, gold_path):
        gold_df = pd.read_json(gold_path, lines=True)
        gold_df_Y = pd.DataFrame(gold_df[self.Y_name], columns=[self.Y_name])
        gold_df_X = gold_df.drop(columns=self.drop_list)

        return gold_df_X, gold_df_Y

    def _tokenize(self, input_df):
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

        for token_col in self.token_list:
            for row_idx, row in enumerate(input_df[token_col]):
                # give empty row default value
                if row == "":
                    row = "empty_doc"
                else:
                    pass

                doc = sent_tokenize(row)
                doc_tokenize_result = []

                for sentence in doc:
                    tokenize_result = tokenizer.tokenize(row)
                    if(tokenize_result == []):
                        tokenize_result = ["empty_token"]
                    else:
                        pass
                    doc_tokenize_result.append(tokenize_result)

                input_df.at[row_idx, token_col] = doc_tokenize_result

        return input_df

    def _embed_transform(self, tokenize_df):
        new_df_dict = {}
        for embedder_idx, embedder in enumerate(self.embedder_list):
            for token_col in self.token_list:
                new_df_dict[str(embedder_idx) + token_col] = embedder.custom_transform(tokenize_df[token_col])

        new_df = pd.DataFrame(new_df_dict)
        return new_df

    def _embed_train(self, tokenize_df):
        all_text = []

        for token_col in self.token_list:
            all_text += tokenize_df[token_col].tolist()

        for embedder in self.embedder_list:
            embedder.custom_fit(all_text)

    def _preprocess_and_train(self, input_df):
        # tokenize
        tokenize_df = self._tokenize(input_df)

        # use embedder to embedding
        self._embed_train(tokenize_df)
        embeded_df = self._embed_transform(tokenize_df)

        return embeded_df

    def _preprocess(self, input_df):
        # tokenize
        tokenize_df = self._tokenize(input_df)

        # use embedder to embedding
        embeded_df = self._embed_transform(tokenize_df)

        return embeded_df

    def _get_one_hot_label(self, input_df):
        category_dict = {}
        for value, item in enumerate(self.category_list):
            category_dict[item] = value

        for row_idx, categories in enumerate(input_df[self.Y_name]):
            one_hot = np.zeros((len(self.category_list)), dtype=float)

            for category in categories:
                one_hot[category_dict[category]] = 1.0

            input_df.at[row_idx, self.Y_name] = one_hot

        return input_df

    def _get_pure_np_format(self, input_df):
        return input_df.to_numpy()

    def preprocess_gold(self, gold_path):
        gold_df_X, gold_df_Y = self._extract_gold_XY(gold_path)
        gold_embed_df_X = self._preprocess_and_train(gold_df_X)
        gold_onehot_df_Y = self._get_one_hot_label(gold_df_Y)

        gold_X = self._get_pure_np_format(gold_embed_df_X)
        gold_Y = self._get_pure_np_format(gold_onehot_df_Y)

        gold_train_X, gold_test_X, gold_train_Y, gold_test_Y = train_test_split(gold_X, gold_Y, train_size=self.split_ratio)

        return gold_train_X, gold_test_X, gold_train_Y, gold_test_Y

    def preprocess_dev(self, dev_path):
        dev_df = pd.read_json(dev_path, lines=True)
        dev_embed_df = self._preprocess(dev_df)

        dev_X = self._get_pure_np_format(dev_embed_df)

        return dev_X


class Custom_TfidfVectorizer(TfidfVectorizer):
    def __init__(self, tokenizer=(lambda text: text), lowercase=False, max_features=256,
                 stopwords=set(stopwords.words('english'))):
        super(Custom_TfidfVectorizer, self).__init__(tokenizer=tokenizer, lowercase=lowercase,
                                                     max_features=max_features, stop_words=stopwords)

    def _custom_prerprocess(self, corpus):
        train_corpus = copy.deepcopy(corpus)

        for doc_id, doc in enumerate(train_corpus):
            combine_doc = []
            for sentence in doc:
                combine_doc += sentence
            train_corpus[doc_id] = combine_doc

        return train_corpus

    def custom_fit(self, corpus):
        train_corpus = self._custom_prerprocess(corpus)

        self.fit(train_corpus)

    def custom_transform(self, corpus):
        doc_vector_list = []
        train_corpus = self._custom_prerprocess(corpus)

        for doc in tqdm(train_corpus):
            doc_vector = self.transform([doc]).toarray()
            doc_vector = doc_vector.ravel()
            doc_vector_list.append(doc_vector)

        return doc_vector_list


class Custom_Word2Vectorizer(Word2Vec):
    def __init__(self, epochs, size, window=5, min_count=1, workers=4, sg=0):
        super(Custom_Word2Vectorizer, self).__init__(size=size, window=window, min_count=min_count,
                                                     workers=workers, sg=sg)
        self.epochs = epochs
        self.size = size

    def custom_fit_preprocess(self, corpus):
        train_corpus = []

        for doc in corpus:
            train_corpus += doc

        return train_corpus

    def custom_fit(self, corpus):
        train_corpus = self.custom_fit_preprocess(corpus)
        self.build_vocab(train_corpus)
        self.train(train_corpus, total_examples=len(train_corpus), epochs=self.epochs, callbacks=[callback()])
        print(self.most_similar(positive=["china"]))

    def custom_transform(self, corpus):
        doc_vector_list = []

        for doc in tqdm(corpus):
            doc_vector = np.zeros((self.size, ), dtype=np.float32)

            for sentence in doc:
                sentence_vector = np.zeros((self.size, ), dtype=np.float32)

                for word in sentence:
                    try:
                        word_vector = self.wv[word]
                    except(KeyError):
                        word_vector = self.wv["empty_token"]

                    sentence_vector += word_vector

                sentence_vector /= len(sentence)
                doc_vector += sentence_vector

            doc_vector /= len(doc)
            doc_vector_list.append(doc_vector)

        return doc_vector_list


class Custom_Doc2Vectorizer(Doc2Vec):
    def __init__(self, epochs, vector_size, window=5, min_count=10, workers=2, dm=1):
        super(Custom_Doc2Vectorizer, self).__init__(vector_size=vector_size, epochs=epochs,
                                                    window=window, min_count=min_count, workers=workers,
                                                    dm=dm)

    def custom_preprocess(self, corpus):
        new_corpus = copy.deepcopy(corpus)

        for doc_idx, doc in enumerate(new_corpus):
            new_doc = []
            for sentence in doc:
                new_doc += sentence
            new_corpus[doc_idx] = new_doc

        return new_corpus

    def custom_fit(self, corpus):
        train_corpus = self.custom_preprocess(corpus)
        tagged_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_corpus)]
        self.build_vocab(tagged_corpus)
        self.train(tagged_corpus, total_examples=self.corpus_count, epochs=self.epochs, callbacks=[callback()])
        print(self.most_similar(positive=["wuhan"]))

    def custom_transform(self, corpus):
        doc_vector_list = []

        new_corpus = self.custom_preprocess(corpus)

        for doc in tqdm(new_corpus):
            doc_vector = self.infer_vector(doc)
            doc_vector_list.append(doc_vector)

        return doc_vector_list


class Custom_BertVectorizer:
    def __init__(self):
        self.bert_client = BertClient()

    def custom_fit(self, corpus):
        pass

    def custom_transform(self, corpus):
        doc_vector_list = []

        for doc in tqdm(corpus):
            sentence_vectors = self.bert_client.encode(doc, is_tokenized=True)
            doc_vector = np.sum(sentence_vectors, axis=0)
            doc_vector_list.append(doc_vector)

        return doc_vector_list


def save_process_result(train_X, test_X, train_Y, test_Y):
    np.save("train_X", train_X)
    np.save("test_X", test_X)
    np.save("train_Y", train_Y)
    np.save("test_Y", test_Y)


def main():
    # preprocess relative setting
    custom_stopwords = stopwords.words('english') + ADDITION_STOPWORDS
    category_list = pd.read_json(CATEGROY_PATH)[0].tolist()

    # embedder setting
    tfidf_embedder = Custom_TfidfVectorizer(max_features=1024, stopwords=custom_stopwords)
    w2v_embedder = Custom_Word2Vectorizer(100, 300)
    d2v_embedder = Custom_Doc2Vectorizer(666, 300)
    # bert_embedder = Custom_BertVectorizer()

    # data preprocessor setting
    data_preproceesor = Data_preproceesor([w2v_embedder, tfidf_embedder], category_list)

    # get preprocess data
    gold_train_X, gold_test_X, gold_train_Y, gold_test_Y = data_preproceesor.preprocess_gold(TRAINING_PATH)
    dev_X = data_preproceesor.preprocess_dev(DEV_PATH)

    # save the preprocess data
    save_process_result(gold_train_X, gold_test_X, gold_train_Y, gold_test_Y)
    np.save("dev_X", dev_X)


if __name__ == '__main__':
    main()
