import pandas as pd
from .email_message import Email

def load_emails(path: str) -> list[Email]:
    # Load emails from a CSV file and return a list of Email objects.
    df = pd.read_csv(path)
    # Convert NaN values to None for consistency
    df = df.astype(object).where(df.notna(), None)
    # Create Email objects from the DataFrame rows
    emails = [
        Email(subject=row["Subject"], message=row["Message"], is_spam=row["Spam/Ham"] == "spam", date=row.get("Date") or "", id=row.get("Message ID") or 0)
        for row in df.to_dict("records")
    ]

    # deduplicate emails based on subject and message, keeping the first occurrence
    seen = set()
    unique_emails = []
    for email in emails:
        identifier = email.get_text()  # Use the combined subject and message as the identifier
        if identifier not in seen:
            seen.add(identifier)
            unique_emails.append(email)
    return unique_emails


def save_emails(emails: list[Email], path: str):
    """Write emails to a CSV file that load_emails can read."""
    raise NotImplementedError


def split_emails(emails: list[Email], test_ratio: float = 0.2, seed: int = 42) -> tuple[list[Email], list[Email]]:
    """Shuffle with the seed, then return (train_emails, test_emails).

    The same seed always gives the same split.
    """
    # Create a DataFrame from the list of Email objects
    df = pd.DataFrame({"is_spam": [e.is_spam for e in emails]})
    # Shuffle the DataFrame and split it into training and testing sets
    train_idx = (df.groupby("is_spam",group_keys=False)
                .sample(frac=1 - test_ratio, random_state=seed)
                .sample(frac=1, random_state=seed) # Shuffle again to ensure randomness
                .index
                )
    # Convert the indices back to lists of Email objects
    train_emails = [emails[i] for i in train_idx]
    test_emails = [emails[i] for i in df.index if i not in train_idx]
    return train_emails, test_emails
    

