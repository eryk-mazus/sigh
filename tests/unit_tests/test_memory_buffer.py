import pytest

from sigh.llm import MemoryBuffer, SighMessage


def test_initialization():
    mb = MemoryBuffer(
        system_message=SighMessage(content="Hello", role="system", length=1)
    )
    assert mb.system_message.content == "Hello"
    assert len(mb.buffer) == 0


def test_initialization_wwo_system():
    mb = MemoryBuffer()
    assert mb.system_message is None


def test_add_message():
    mb = MemoryBuffer(
        system_message=SighMessage(content="Hello", role="system", length=1)
    )
    test_message = SighMessage(content="Hello GPT", role="user", length=3)
    mb.add_message(test_message)
    assert mb.buffer[0].content == "Hello GPT"


@pytest.mark.parametrize("k", [-1, 5])
def test_retrieval_within_context_length_wwo_system(k):
    mb = MemoryBuffer(system_message=None)
    for i in range(5):
        mb.add_message(SighMessage(f"Msg {i}", "user", 5))

    result = mb.retrieve_up_to_k_messages(k=k, context_length=35, min_new_tokens=2)
    assert len(result.messages) == 5
    assert result.remaining_tokens == 10


@pytest.mark.parametrize("k", [-1, 5])
def test_retrieval_within_context_length_with_system(k):
    system_message = SighMessage(content="System", role="system", length=4)

    mb = MemoryBuffer(system_message=system_message)
    for i in range(5):
        mb.add_message(SighMessage(f"Msg {i}", "user", 5))

    result = mb.retrieve_up_to_k_messages(k=k, context_length=35, min_new_tokens=2)
    assert len(result.messages) == 6  # the result includes the system prompt
    assert result.remaining_tokens == 6
