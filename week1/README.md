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

# Tools

`beautifulsoup` is an HTML-parser, but it cannot run Javascript. So it is not the best option for the webpages which have JS. In order to do so you can use `selenium` (some browser automation framework), or `playwright` which is a more modern browser automation framework.
