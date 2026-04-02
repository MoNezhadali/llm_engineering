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

## Caching

It is a very cost-saving and important feature of LLMs to use, and it is quite vendor-specific. For example for OpenAI, cache hits only for exact prefix matches, i.e. if you have several questions about some text, it has to go at the beginning of the text. Hence, always provide the static context at the beginning and variables, e.g. questions, at the end.

With Anthropic you have tell you want to prime the cache, then you will pay `25%` more, but when you use those cache you will pay ``10x` less.

Gemini supports both, having both **implicit** and **explicti**.

## User Interface

You can use **Gradio** (or **Streamlit**) to create a user interface for your apps including LLM apps. It is as easy as doing:

```python
import gradio as gr
gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch()
# where shout is the callback function which takes the input and returns output"
```
