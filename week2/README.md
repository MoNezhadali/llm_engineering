# Notes on week 2

You can setup your API keys with several different providers (OpenAI, Anthropic, Google, Deepseek, etc.), but you can also set one API key in `openrouter.ai` and then use that key for connecting to everything.

## OpenAI library arguments

You can add an argument called `reasoning_effort`:

```python
response = openai.chat.completions.create(model="gpt-5-nano", messages=easy_puzzle, reasoning_effort="minimal")
```

You have more than one way to improve the results:

- Improve the model, e.g. use `mini` instead of `nano`.
- Improve the reasoning effort, e.g. from `minimal` to `low`.

## Groq

**Groq**, which is different from **Grok** of **XAI**, is a platform which runs open-source models which are too large to run on premise. They have optimized their hardware such that inferences are extremely fast.

## LiteLLM

**Langchain** is a heavy abstraction layer on top of LLM libraries. One lightweight alternative is **LiteLLM**. **LiteLLM** gives tools to analyze every single call, e.g. input tokens, output tokens, total tokens, cached tokens, total cost, etc.
