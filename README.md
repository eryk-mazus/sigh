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
```

## TODOs:
- [ ] add automatic transcription stopping criteria
- [ ] implement different modes (wake word, always transcribe, save to file)
- [ ] talk with llama 2 (chat)

Lower priority:
- [ ] add video example
- [ ] colors, bold
- [ ] measure execution time
