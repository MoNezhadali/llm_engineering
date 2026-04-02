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

## Some python trick

```python
# Given:
def stream_model(prompt, model):
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    # These two are equivalent
    # version 1
    yield from result
    # version 2
    for chunk in result:
        yield chunk
```

## Adding an interface with several different vendors

You can get the input in a textbox, choose model from a dropdown menu, and get the output as markdown like this:

```python
message_input = gr.Textbox(label="Your message:", info="Enter a message for the LLM", lines=7)
model_selector = gr.Dropdown(["GPT", "Shout"], label="Select model", value="GPT")
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_model,
    title="LLMs",
    inputs=[message_input, model_selector],
    outputs=[message_output],
    examples=[
            ["Explain the Transformer architecture to a layperson", "GPT"],
            ["Explain the Transformer architecture to an aspiring AI engineer", "Claude"]
        ],
    flagging_mode="never"
    )
view.launch()
```
