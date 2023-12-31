#!/usr/bin/env python3
import os
import re
import sys
import threading
import time
import warnings
from typing import Dict

import fire
import numpy as np
import torch
from faster_whisper import WhisperModel
from loguru import logger
from rapidfuzz.distance.Levenshtein import normalized_distance

from sigh import AudioAsync, LLMInteractor, OpenAILLMFactory, vad

# avoiding the scipy exit error:
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)

# consts:
WHISPER_SAMPLE_RATE = 16_000

# configure logger:
logger.remove()
logger.add(lambda msg: print(msg, end=""), format="{function}: {message}")


def transcribe(
    model,
    pcmf32: np.array,
    **kwargs: Dict,
) -> str:
    segments, _ = model.transcribe(pcmf32, **kwargs)
    segments = list(segments)
    return "".join([s.text for s in segments])


def listen_for_enter_key(stop_event: threading.Event) -> None:
    try:
        input("Press Enter to stop\n")  # Block until Enter is pressed
        stop_event.set()
    except (KeyboardInterrupt, EOFError):
        pass


def simple_transcription_postprocessing(s: str) -> str:
    s = s.replace("\n", " ")
    s = re.sub(" +", " ", s)
    return s.strip()


def main(
    # AudioAsync params:
    length_ms: int = 5000,
    capture_device_id: int = 1,
    # VAD params:
    vad_thold: float = 0.6,
    # Whisper params:
    whisper_model_name: str = "large",
    compute_type: str = "float16",
    language: str = "en",
    log_prob_threshold: float = -1.0,
    no_speech_threshold: float = 0.1,
    beam_size: int = 5,
    # Wake phrase params:
    detect_wake_phrase: bool = False,
    wake_phrase_cutoff: float = 0.6,
    wake_ms: int = 2000,
    wake_phrase: str = "gpt",
    # Transcription params:
    keep_ms: int = 100,
    step_ms: int = 1000,
    silent_chunks_stop_condition: int = 8,
    # LLM params:
    llm_name: str = "gpt-4",
    system_prompt: str = "You are a general purpose assistant.",
    respond_with_gpt: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"{'loading model':.<25}{whisper_model_name:.>25}")
    logger.info(f"{'device':.<25}{device:.>25}")
    logger.info(f"{'compute_type':.<25}{compute_type:.>25}")
    logger.info(f"{'language':.<25}{language:.>25}")
    logger.info(f"{'beam_size':.<25}{beam_size:.>25}")
    logger.info(f"{'no_speech_threshold':.<25}{no_speech_threshold:.>25}")
    logger.info(f"{'log_prob_threshold':.<25}{log_prob_threshold:.>25}")

    whisper_gen_kwargs = {
        "language": language,
        "beam_size": beam_size,
        # If the no_speech probability is higher than this value
        # AND the average log probability over sampled tokens
        # is below `logprob_threshold`, consider the segment as silent:
        "no_speech_threshold": no_speech_threshold,
        "log_prob_threshold": log_prob_threshold,
    }

    model = WhisperModel(whisper_model_name, device=device, compute_type=compute_type)
    audio = AudioAsync(
        len_ms=length_ms,
        sample_rate=WHISPER_SAMPLE_RATE,
        capture_id=capture_device_id,
    )
    llm_interactor = LLMInteractor(
        llm=OpenAILLMFactory.create(model_name=llm_name), system_prompt=system_prompt
    )

    # streaming params:
    keep_ms = min(keep_ms, step_ms)
    length_ms = max(length_ms, step_ms)

    n_samples_len = int((1e-3 * length_ms) * WHISPER_SAMPLE_RATE)
    n_samples_keep = int((1e-3 * keep_ms) * WHISPER_SAMPLE_RATE)

    n_new_line = max(1, length_ms // step_ms - 1)

    if detect_wake_phrase:
        logger.info(
            f"wake phrase = \033[96m{wake_phrase}\033[0m | "
            f"wake phrase duration = {wake_ms/1000:.2f} sec |"
            f"similarity cutoff = {wake_phrase_cutoff:.2f}"
        )
        have_prompt = False
    else:
        logger.info("Detection of wake word is turned off.")
        have_prompt = True

    logger.info(
        f"step = {step_ms/1000:.2f} sec | "
        f"buffer = {length_ms/1000:.2f} sec | "
        f"segments = {n_new_line}"
    )

    # auxiliary variables:
    pcmf32_old = np.empty(0)
    n_iter = 0

    # wait for 1 second to avoid any buffered noise
    audio.resume()
    time.sleep(1)
    audio.clear()

    if not have_prompt:
        logger.info("Waiting for wake phrase...")

    try:
        while True:
            if not have_prompt:
                # time.sleep(100 / 1000)
                data = audio.get(2000)

                if vad(data, sampling_rate=WHISPER_SAMPLE_RATE, threshold=vad_thold):
                    logger.info("Speech detected! Processing ...")

                    pcmf32_cur = audio.get(wake_ms)

                    txt = transcribe(model, pcmf32_cur, **whisper_gen_kwargs)
                    txt = txt.strip().strip(".")

                    if not txt:
                        print("WARNING: prompt not recognized, try again")
                        continue

                    print(f"Heard: `{txt}`")
                    similarity = 1 - normalized_distance(
                        txt.lower(), wake_phrase.lower()
                    )
                    print(
                        f"Wake Phrase = `{wake_phrase}`, Similarity= {similarity:.3f}"
                    )

                    if similarity >= wake_phrase_cutoff:
                        have_prompt = True
                        audio.clear()
                    else:
                        continue
            else:
                print(
                    "\033[35m"
                    + "[Start speaking | Press `Enter` to manually send to LLM]"
                    + "\033[0m"
                )
                sys.stdout.flush()

                stop_event = threading.Event()
                listener_thread = threading.Thread(
                    target=listen_for_enter_key, args=(stop_event,), daemon=True
                )
                listener_thread.start()

                llm_hash_table = {}
                llm_iter = 0
                silence_counter = 0
                started_speaking = False

                while not stop_event.is_set():
                    # process new audio
                    pcmf32_new = audio.get(step_ms)
                    audio.clear()

                    n_samples_new = pcmf32_new.size
                    n_samples_take = min(
                        pcmf32_old.size,
                        max(0, n_samples_keep + n_samples_len - n_samples_new),
                    )

                    pcmf32 = np.hstack((pcmf32_old[-n_samples_take:], pcmf32_new))

                    pcmf32_old = pcmf32

                    if not vad(
                        pcmf32, sampling_rate=WHISPER_SAMPLE_RATE, threshold=vad_thold
                    ):
                        silence_counter += 1
                        is_silent = True
                    else:
                        silence_counter = 0
                        is_silent = False

                    if (
                        started_speaking
                        and silence_counter >= silent_chunks_stop_condition
                    ):
                        stop_event.set()
                        break

                    # run inference
                    txt = transcribe(model, pcmf32, **whisper_gen_kwargs)

                    # print results
                    sys.stdout.write("\33[2K\r")
                    sys.stdout.flush()

                    print(" " * 50, end="")

                    sys.stdout.write("\33[2K\r")
                    sys.stdout.flush()

                    if txt and not is_silent:
                        print(txt, end="", flush=True)
                        llm_hash_table[llm_iter] = txt

                    if not started_speaking and txt:
                        started_speaking = True

                    n_iter += 1

                    if n_iter % n_new_line == 0:
                        print("\n", end="", flush=True)
                        llm_iter += 1

                        # keep part of the audio for next iteration
                        # to try to mitigate word boundary issues
                        pcmf32_old = pcmf32[-n_samples_keep:]

                        # TODO:
                        # add parameter
                        # whisper_prompt_tokens = ""
                        # # Add tokens of the last full length segment as the prompt
                        # # if not no_context (keep context between audio chunks)
                        # whisper_prompt_tokens += txt

                print("\n", end="", flush=True)

                if respond_with_gpt:
                    llm_prompt = " ".join(chunk for chunk in llm_hash_table.values())
                    llm_prompt = simple_transcription_postprocessing(llm_prompt)

                    print("\033[35m" + "[GPT Response]" + "\033[0m")
                    _ = llm_interactor.on_user_message(
                        llm_prompt, k=5, min_new_tokens=256
                    )
                    print("\n", end="", flush=True)

                stop_event.clear()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        audio.terminate()


if __name__ == "__main__":
    fire.Fire(main)
