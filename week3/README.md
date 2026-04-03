# Notes on Week3

HuggingFace is hub for open-source models, datasets (it is the choice hub for transformers, still Keggle is for the other types of models), and Apps (including many built in Gradio)


## Diffusion vs. Transformer

Diffusion models are used for generating images. During the training images are transformed into white noise in several steps and the model learns how to create them (in backward steps). During the inference:
Text → numbers (embeddings) → guide the denoising process → image


## Sentiment Analysis

You can do sentiment analysis with transformers:

```python
from transformers import pipeline
my_simple_sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device="cuda")
result = my_simple_sentiment_analyzer("I'm super excited to be on the way to LLM mastery!")
print(result)

# Result
# [{'label': 'POSITIVE', 'score': 0.9993460774421692}]
```

## Specialized models

There are models specifically trained for specific tasks, e.g. summarization, translation, entity recognition, sentiment analysis, classification, image generation, audio generation, etc. and they are open-source, so an option is using them.
