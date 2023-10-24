# sigh

Background voice detection program that listens for a wake word and activates transcription mode.
After the transcription is finished, the GPT model responds to the transcribed text.

Repository is in active development.

## Setup:

```
set OPENAI_API_KEY=sk-...

git clone https://github.com/eryk-mazus/sigh.git
cd sigh
pip install -e .
# run:
python ./sigh/main.py --help

# no wake word detection:
python ./sigh/main.py --detect_wake_word=False

# running in background (soon):
# https://janakiev.com/blog/python-background/?ref=python-shell-commands
```

## TODOs:
- [x] add automatic transcription stopping criteria
- [x] better GPT responses (system prompt, chat, memory)
- [ ] faster wake phrase detection
- [ ] implement different modes (wake word, always transcribe, save to file, parallel transcription and LLM commentary)
- [ ] talk with local models, e.g. llama2, mistral, etc.

## Contributing:
