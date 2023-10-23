import pytest

from sigh.llm import OpenAILLM


@pytest.mark.parametrize(
    "model_name, context_length",
    [
        ("gpt-4", 1024),
        ("gpt-4", 2048),
        ("gpt-3.5-turbo", 4097),
    ],
)
def test_initialization(model_name, context_length):
    llm_instance = OpenAILLM(model_name=model_name, context_length=context_length)
    assert llm_instance.context_length == context_length
