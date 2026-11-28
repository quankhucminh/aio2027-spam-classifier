from spam_classifier.dataset.email_message import Email
import numpy as np

class Evaluator:
    """Scores every detector on the same labelled test emails."""

    def __init__(self, test_emails: list[Email]):
        self.test_emails = test_emails
        # Extract the actual labels (True for spam, False for not spam)
        self.actual = np.array([e.is_spam for e in test_emails], dtype=bool)

    def evaluate(self, spam_filter) -> dict:
        """Run spam_filter.predict on every test email and return compute_metrics(...)."""
        # Check if the spam_filter has a predict method
        if not hasattr(spam_filter, 'predict') or not callable(getattr(spam_filter, 'predict')):
            raise ValueError("spam_filter must have a callable 'predict' method")

        # Get predictions from the spam filter
        predicted = [spam_filter.predict(email) for email in self.test_emails]
        # Compute and return the evaluation metrics
        return self.compute_metrics(self.actual, predicted)

    def compute_metrics(self, actual: list, predicted: list) -> dict:
        """Return a dict with accuracy, precision, recall, f1, tp, fp, fn and tn.

        Spam (True) is the positive class. If a division would be by 0, use 0.0.
        """
        # Convert actual and predicted to numpy arrays of boolean type
        actual = np.array(actual, dtype=bool)
        predicted = np.array(predicted, dtype=bool)

        # Calculate true positives, false positives, false negatives, and true negatives
        tp = np.sum(actual & predicted)
        fp = np.sum(~actual & predicted)
        fn = np.sum(actual & ~predicted)
        tn = np.sum(~actual & ~predicted)

        # Calculate precision, recall, accuracy, and F1 score with checks to avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }
