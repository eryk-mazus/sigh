import os

import openai
import pytest

from sigh.llm import OpenAILLM, OpenAILLMFactory

# Check if 'TEST_OPENAI' is not set or if it's set to 'False'
skip_test = os.environ.get("TEST_OPENAI") != "True"

example_messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful, pattern-following assistant that "
            "translates corporate jargon into plain English."
        ),
    },
    {
        "role": "user",
        "content": "New synergies will help drive top-line growth.",
    },
    {
        "role": "assistant",
        "content": "Things working well together will increase revenue.",
    },
    {
        "role": "user",
        "content": (
            "Let's circle back when we have more bandwidth to touch base "
            "on opportunities for increased leverage."
        ),
    },
    {
        "role": "assistant",
        "content": "Let's talk later when we're less busy about how to do better.",
    },
    {
        "role": "user",
        "content": (
            "This late pivot means we don't have time to boil the ocean "
            "for the client deliverable."
        ),
    },
]


@pytest.mark.parametrize(
    "model_name, context_length, tokens_per_message",
    [
        ("gpt-4", 1024, 4),
        ("gpt-4", 2048, 4),
        ("gpt-3.5-turbo", 4097, 4),
    ],
)
def test_initialization(model_name, context_length, tokens_per_message):
    llm_instance = OpenAILLM(
        model_name=model_name,
        context_length=context_length,
        tokens_per_message=tokens_per_message,
    )
    assert llm_instance.context_length == context_length
    assert llm_instance.tokens_per_message == tokens_per_message


@pytest.mark.skipif(skip_test, reason="Test only runs if TEST_OPENAI is set to True")
def test_count_tokens():
    supported_models = (
        "gpt-4",
        "gpt-4-0613",
        # "gpt-4-32k",
        # "gpt-4-32k-0613",
        # "gpt-4-0314",
        # "gpt-4-32k-0314",
        # "gpt-3.5-turbo",
        # "gpt-3.5-turbo-16k",
        # "gpt-3.5-turbo-instruct",
        # "gpt-3.5-turbo-0613",
        # "gpt-3.5-turbo-16k-0613",
        # "gpt-3.5-turbo-0301",
    )

    for model in supported_models:
        llm = OpenAILLMFactory.create(model)

        estimation = sum([llm.count_tokens(m["content"]) for m in example_messages])

        response = openai.ChatCompletion.create(
            model=model,
            messages=example_messages,
            temperature=0,
            max_tokens=1,
        )
        assert (
            estimation + 3 == response["usage"]["prompt_tokens"]
        ), f"incorrect estimation for {model}"
