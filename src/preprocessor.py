from __future__ import annotations

import nltk
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
import textract
import spacy
import re


class DataPreprocessor:
    @staticmethod
    def _load_data_from_file(filepath: str) -> str:
        """Read different types of source files.

        Args:
            filepath (str): Multiple file types like DOCX, PDF, TXT.

        Returns:
            data (str): Text string.
        """
        data = str(textract.process(filepath), encoding="UTF-8")
        return data

    @staticmethod
    def _clean_text(text: str, lang: str = "ru") -> str:
        """Clean non-ASCII special characters from input text data.

        Args:
            text (str): Text string.
            lang (str): Text language ["ru", "en"].

        Returns:
            text (str): Text string.
        """
        if lang == "ru":
            text = re.sub(r"[^а-яА-Я0-9\s\/]", "", text)
        if lang == "en":
            text = re.sub(r"[^a-zA-Z0-9\s\/]", "", text)
        text = text.replace("/", " ")
        return text

    @staticmethod
    def _filter_token_tag(tagged_token_list: list, filter_tag_list: list) -> list:
        """Filter the tagged token list present in the filter tag list.

        Args:
            tagged_token_list (list): Tagged token list.
            filter_tag_list (list): Filter tag list.

        Returns:
            filtered_token_list (list): List containing tokens corresponding to tags present in the filter tag list.
        """
        filtered_token_list = [t[0] for t in tagged_token_list if t[1] in filter_tag_list]
        filtered_token_list = [str(item) for item in filtered_token_list]
        return filtered_token_list

    @staticmethod
    def _unique_tokens(token_list: list) -> list:
        """Remove duplicate tokens from the input token list.

        Args:
            token_list (list): Token list.

        Returns:
            unique_token_list (list): Unique token list.
        """
        unique_token_list = []
        for x in token_list:
            x = x.lower()
            if x not in unique_token_list:
                unique_token_list.append(x)
        return unique_token_list

    @staticmethod
    def _nltk_tokenizer(text: str) -> list:
        """Use the NLTK tokeniser to tokenise the input text.

        Args:
            text (str): Text string.

        Returns:
            tokens (list): Tokens.
        """
        tokens = word_tokenize(text)
        return tokens

    @staticmethod
    def _nltk_pos_tag(token_list: list) -> list:
        """Use the NLTK parts of speech tagger to apply tags to the input token list.

        Args:
            token_list (list): Token List.

        Returns:
            tagged_list (list): Tagged token list.
        """
        tagged_list = pos_tag(token_list)
        return tagged_list

    @staticmethod
    def _nltk_stopwords_removal(token_list: list, lang: str = "ru") -> list:
        """Use the NLTK parts of speech tagger to apply tags to the input token list.

        Args:
            token_list (list): Token List.
            lang (str): Text language ["ru", "en"].

        Returns:
            stopwords_filtered_list (list): Stopwords filtered list.
        """
        if lang == "ru":
            stop_words = set(stopwords.words("russian"))
        if lang == "en":
            stop_words = set(stopwords.words("english"))
        stopwords_filtered_list = [w for w in token_list if w not in stop_words]
        return stopwords_filtered_list

    @classmethod
    def nltk_keywords(self, data: str, lang: str = "ru") -> list:
        """Use the NLTK pipeline to detect keywords from input text data.

        Args:
            data (str): Text data.
            lang (str): Text language ["ru", "en"].

        Returns:
            keywords (list): Keywords.
        """
        data = self._clean_text(data, lang=lang)
        tokens = self._nltk_tokenizer(data)
        pos_tagged_tokens = self._nltk_pos_tag(tokens)
        keywords = self._filter_token_tag(pos_tagged_tokens, filter_tag_list=["NNP", "NN", "VBP", "JJ"])
        keywords = self._nltk_stopwords_removal(keywords, lang=lang)
        keywords = self._unique_tokens(keywords)
        return keywords

    @staticmethod
    def _spacy_tokenizer(text: str, nlp) -> list:
        """Use the spacy tokeniser to tokenise the input text.

        Args:
            data (str): Text string.
            nlp: Spacy language module.

        Returns:
            tokens (list): Tokens.
        """
        tokens = nlp(text)
        return tokens

    @staticmethod
    def _spacy_pos_tag(token_list: list) -> list:
        """Use the spacy parts of speech tagger to apply tags to the input token list.

        Args:
            token_list (list): Token List.

        Returns:
            tagged_list (list): Tagged token list.
        """
        tagged_list = []
        for tok in token_list:
            tagged_list.append((tok,tok.tag_))
        return tagged_list

    @staticmethod
    def _spacy_stopwords_removal(token_list: list, nlp) -> list:
        """Remove stopwords from the input token list using the spacy stopwords dictionary.

        Args:
            token_list (list): Token List.
            nlp: Spacy language module.

        Returns:
            stopwords_filtered_list (list): Stopwords filtered list.
        """
        stop_words = nlp.Defaults.stop_words
        stopwords_filtered_list = [w for w in token_list if w not in stop_words] 
        return stopwords_filtered_list

    @classmethod
    def spacy_keywords(self, data: str, lang: str = "ru") -> list:
        """Use the spacy pipeline to detect keywords from input text data.

        Args:
            data (str): Text data.
            lang (str): Text language ["ru", "en"].

        Returns:
            keywords (list): Keywords.
        """
        if lang == "ru":
            nlp = spacy.load("ru_core_news_sm")
        if lang == "en":
            nlp = spacy.load("en_core_web_sm")
        data = self._clean_text(data, lang=lang)
        tokens = self._spacy_tokenizer(data, nlp=nlp)
        pos_tagged_tokens = self._spacy_pos_tag(tokens)
        keywords = self._filter_token_tag(pos_tagged_tokens, filter_tag_list=["NNP"])
        keywords = self._spacy_stopwords_removal(keywords, nlp=nlp)
        keywords = self._unique_tokens(keywords)
        return keywords
