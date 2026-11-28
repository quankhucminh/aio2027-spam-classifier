import datetime
from dataclasses import dataclass

@dataclass(repr=True, init=True, eq=True)
class Email:
    """One email: a subject, a message and a spam label.

    For an email pasted into the demo the label is unknown, so leave is_spam as False.
    """

    subject: str
    message: str
    is_spam: bool = False
    date: str = ""
    id: int = 0  # optional, for debugging only

    def __post_init__(self):
        self.date = datetime.datetime.strptime(self.date, "%Y-%m-%d") if self.date else None

    def get_text(self) -> str:
        """Return the subject and message joined by a space."""
        return f"{self.subject} {self.message}"
