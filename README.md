# sigh

Seamless Voice Interactions with LLMs

**Key Features:**
* **Unlimited Real-time Transcription:** Continuously capture audio directly from your microphone.
* **Customizable Wake Word:** Choose a wake word or phrase to trigger transcription mode.
* **Automatic Speech Termination:** Detects when you've finished speaking, with an option for manual control.

**Note:** This repository is under active development. Contributions are welcome!

**Demo:**

## Setup:

```
set OPENAI_API_KEY=sk-...

git clone https://github.com/eryk-mazus/sigh.git
cd sigh
pip install -e .

# run:
python ./sigh/main.py --help

# run without wake word detection (by default):
python ./sigh/main.py

# run with wake phrase detection:
python ./sigh/main.py --detect_wake_phrase=True --wake_phrase="""Hey GPT"""
```

## Backlog:

Near-term:
- [x] Add automatic transcription stopping
- [x] Better GPT responses (system prompt, chat mode, sliding memory buffer)
- [ ] Talk with local models, e.g. llama2, mistral, etc.
- [ ] Improve code coherence and composition (refactoring)

Medium-term:
- [ ] Add second mode: parallel transcription and LLM commentary
- [ ] Docker


## Contributing:
Issues, new ideas, suggestions, and PRs are all welcome!
