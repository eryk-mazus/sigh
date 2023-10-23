from sigh.llm import MemoryBuffer, SighMessage


def test_initialization():
    mb = MemoryBuffer(
        system_message=SighMessage(content="Hello", role="system", length=5)
    )
    assert mb.system_message.content == "Hello"
    assert len(mb.buffer) == 0
