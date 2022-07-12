from transformers import pipeline


class MonoSummarizer():
    def __init__(self):
        self.model = pipeline("summarization", device=0)
        #self.model = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")

    def predict(self, text):
        summarized = self.model(text, min_length=25, max_length=50)

        return summarized
