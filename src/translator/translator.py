from transformers import pipeline

class TextTranslator:
    """Translate English text to another language using a transformer model."""

    def __init__(self, target_lang: str = "es"):
        self.target_lang = target_lang
        model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
        task = f"translation_en_to_{target_lang}"
        self.translator = pipeline(task, model=model_name)

    def translate(self, text: str) -> str:
        """Translate the given text and return the result."""
        result = self.translator(text)
        return result[0]["translation_text"]
