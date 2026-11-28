import unittest
import datetime

from spam_classifier.dataset.email_message import Email
from spam_classifier.dataset.loading import load_emails, split_emails

class DatasetLoadingTest(unittest.TestCase):
    def test_load_emails(self):
        emails = load_emails("tests/test_data/emails.csv")
        self.assertEqual(len(emails), 8)
        self.assertEqual(emails[0].subject, "Hello")
        self.assertEqual(emails[0].message, "How are you?")
        self.assertFalse(emails[0].is_spam)
        self.assertEqual(emails[0].date, datetime.datetime(2026, 11, 20))
        self.assertEqual(emails[1].subject, "Win a free iPhone!")
        self.assertEqual(emails[1].message, "Click here to claim your prize.")
        self.assertTrue(emails[1].is_spam)
        self.assertEqual(emails[1].date, datetime.datetime(2026, 11, 21))
        self.assertEqual(emails[2].subject, "Meeting Reminder")
        self.assertEqual(emails[2].message, "Don't forget our meeting tomorrow.")
        self.assertFalse(emails[2].is_spam)
        self.assertEqual(emails[2].date, datetime.datetime(2026, 11, 22))

    def test_split_emails(self):
        emails = load_emails("tests/test_data/emails.csv")
        train_emails, test_emails = split_emails(emails, test_ratio=0.5, seed=42)
        self.assertEqual(len(train_emails), 4)
        self.assertEqual(len(test_emails), 4)

    def test_enron_set_loading(self):
        emails = load_emails("data/raw/enron_spam_data.csv")
        self.assertEqual(len(emails), 30494)