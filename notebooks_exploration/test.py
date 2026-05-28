# from transformers import pipeline

# classifier = pipeline(
#     "sentiment-analysis",
#     model="nlptown/bert-base-multilingual-uncased-sentiment"
# )

# print(classifier("J'adore cette IA"))

from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print(classifier("I love this product"))