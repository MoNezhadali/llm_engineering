# Notes

You do not want AI to read your secrets. So remember if you set your variables in a `.env` file or something similar, block AI from reading it.
You can do so by creating a `.vscode/settings.json` file and copying these stuff to it:

```json
{
  "files.associations": {
    ".env": "plaintext",
    ".env.*": "plaintext"
  },
  "github.copilot.enable": {
    "*": true,
    "plaintext": false,
    "dotenv": false
  },
  "[plaintext]": {
    "editor.inlineSuggest.enabled": false
  },
  "[dotenv]": {
    "editor.inlineSuggest.enabled": false
  },
}
```

## Tools

`beautifulsoup` is an HTML-parser, but it cannot run Javascript. So it is not the best option for the webpages which have JS. In order to do so you can use `selenium` (some browser automation framework), or `playwright` which is a more modern browser automation framework (by `Microsoft`).

## OpenAI Library

You can use `openai` library not only for accessing **OpenAI** endpoints but also **Google**, and also **Ollama** (e.g. running on your local machine!).

## Tokenization

You can use the `tiktoken` package for tokenization of text and converting tokens back to text.

## Stateless vs Stateful

Every single prompt we send to an LLM is stateless. We send all the previous messages as well (or possibly a summary of them) to create an illusion of statefullness.

## Context window

that is the maximum number of tokens that a model can consider when generating the next token. It includes the original input prompt, subsequent conversation, the latest input prompt, and almost all the output prompt.

You can check the context siez, input/output cost, latency, etc. in websites like: [venujm.ai](https://vellum.ai/llm-leaderboard).

## Cost optimization

One of the things which is worth knowing is that when you send the same chunk twice to OpenAI GPT models, it may be cheaper, since OpenAI does some caching as well.

## Response format

You can choose the response format to be of type JSON, in the OpenAI APIs:

```python
def select_relevant_links(url):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    links = json.loads(result)
    return links
```

This constrains the response generation at inference time, makes sure the `choices[0]` will not be choses such that it results in an invalid JSON object.
