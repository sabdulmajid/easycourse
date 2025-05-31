from transformers import pipeline

class QuestionAnswerer:
    """Answer questions based on provided context using a transformer model."""

    def __init__(self, model: str = "deepset/roberta-base-squad2"):
        self.qa_pipeline = pipeline("question-answering", model=model)

    def answer(self, question: str, context: str) -> str:
        """Return an answer for the question given the context."""
        result = self.qa_pipeline(question=question, context=context)
        return result["answer"]
