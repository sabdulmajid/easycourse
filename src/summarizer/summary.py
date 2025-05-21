from transformers import pipeline

class SummaryGenerator:
    """Generate summaries of text using a transformer model."""

    def __init__(self, model: str = "facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model)

    def summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """Return a summary of the provided text."""
        result = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]["summary_text"]
