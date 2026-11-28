import unittest

from spam_classifier.dataset.email_message import Email
from spam_classifier.filters.word_score_filter import WordScoreFilter
from spam_classifier.preprocessing.preprocessor import Preprocessor
from tests.worked_example import make_new_email, make_train_emails


class WordScoreFilterTest(unittest.TestCase):
    def test_worked_example(self):
        spam_filter = WordScoreFilter(Preprocessor([]))
        spam_filter.fit(make_train_emails())

        self.assertAlmostEqual(spam_filter.score(make_new_email()), 0.625)
        self.assertTrue(spam_filter.predict(make_new_email()))

    def test_email_with_only_unknown_words_is_not_spam(self):
        spam_filter = WordScoreFilter(Preprocessor([]))
        spam_filter.fit(make_train_emails())

        self.assertFalse(spam_filter.predict(Email("", "zebra")))

    def test_top_spam_words(self):
        spam_filter = WordScoreFilter(Preprocessor([]))
        spam_filter.fit(make_train_emails())

        top_words = spam_filter.top_spam_words(make_new_email(), k=2)

        # "free" and "money" both score 1.0, so either order is fine.
        self.assertEqual(sorted(top_words), [("free", 1.0), ("money", 1.0)])
