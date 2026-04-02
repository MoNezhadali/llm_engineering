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

## Enforcing JSON outpu

By use of `response_format={"type": "json_object"}` you can make sure the response format is **JSON**:


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
