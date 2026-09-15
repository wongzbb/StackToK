# SPDX-License-Identifier: Apache-2.0
"""
VQA text processor: extract question/keywords for text token embedding (coverage over text).
"""
import re

# Stopwords with little visual relevance; keep descriptive words for coverage.
NON_VISUAL_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "it",
    "they",
    "he",
    "she",
    "we",
    "you",
    "i",
    "what",
    "where",
    "when",
    "why",
    "how",
    "who",
    "which",
    "and",
    "or",
    "but",
    "so",
    "if",
    "then",
    "question",
    "answer",
    "to",
    "for",
    "with",
    "by",
    "from",
    "at",
    "into",
    "onto",
    "upon",
}

WORD_REGEX = re.compile(r"\b\w+\b")
PROMPT_1 = "Answer the question using a single word or phrase."
PROMPT_2 = "Answer with the option's letter from the given choices directly."


class QuestionTextProcessor:
    """
    Text processor for StackTok: extract question/keywords used as text tokens
    in the coverage criterion (cover text tokens + vision token set).
    """

    def extract_keywords_simple(self, text: str) -> str:
        """
        Filter stopwords and return descriptive text for text token embedding (coverage).
        """
        if not text:
            return ""
        text = text.replace(PROMPT_1, "").replace(PROMPT_2, "")
        words = WORD_REGEX.findall(text)
        filtered_words = [w for w in words if w.lower() not in NON_VISUAL_WORDS]
        filtered_str = " ".join(filtered_words) if filtered_words else text
        return filtered_str
