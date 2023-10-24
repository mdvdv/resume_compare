from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessor import DataPreprocessor


class DataComparator:
    def __init__(self, resume_path: str, vacancy_path: str, lang: str = "ru") -> None:
        self.preprocessor = DataPreprocessor()
        self.vectorizer = CountVectorizer()
        self.resume_data = self.preprocessor._load_data_from_file(resume_path)
        self.vacancy_data = self.preprocessor._load_data_from_file(vacancy_path)
        self.lang = lang

    def matchKeywords(self):
        # keywords_resume = self.preprocessor.spacy_keywords(self.resume_data, lang=self.lang)
        # keywords_vacancy = self.preprocessor.spacy_keywords(self.vacancy_data, lang=self.lang)
        keywords_resume = self.preprocessor.nltk_keywords(self.resume_data, lang=self.lang)
        keywords_vacancy = self.preprocessor.nltk_keywords(self.vacancy_data, lang=self.lang)
        vacancy_keywords_in_resume_list = [w for w in keywords_vacancy if w in keywords_resume]
        vacancy_keywords_in_resume_count = len(vacancy_keywords_in_resume_list)
        vacancy_keywords_count = len(keywords_vacancy)
        matchPercentage = (vacancy_keywords_in_resume_count / vacancy_keywords_count) * 100
        matchPercentage = round(matchPercentage, 2)
        return matchPercentage

    def matchSimilarity(self):
        data = [self.resume_data, self.vacancy_data]
        count_matrix = self.vectorizer.fit_transform(data)
        matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
        matchPercentage = round(matchPercentage, 2)
        return matchPercentage