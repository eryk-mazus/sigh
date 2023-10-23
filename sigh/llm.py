from abc import ABC, abstractmethod
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import openai
import tiktoken

Role = Literal["assistant", "user", "system"]


@dataclass(slots=True)
class SighMessage:
    content: str
    role: Role
    length: int

    def __post_init__(self):
        if self.role not in ("assistant", "user", "system"):
            raise ValueError(
                f"Invalid role: {self.role}. "
                "Expected one of ['assistant', 'user', 'system']"
            )


class LLM(ABC):
    @property
    @abstractmethod
    def context_length(self) -> int:
        ...

    @abstractmethod
    def get_reponse(self, messages: List[SighMessage], max_tokens: int) -> SighMessage:
        ...

    @abstractmethod
    def count_tokens(self, content: str) -> int:
        ...


class OpenAILLM(LLM):
    def __init__(self, model_name: str, context_length: int) -> None:
        self.model_name = model_name
        self.context_length = context_length
        self.encoding = tiktoken.encoding_for_model(model_name)

    @property
    def context_length(self) -> int:
        return self._context_length

    def count_tokens(self, content: str) -> int:
        return len(self.encoding.encode(content))

    def get_reponse(self, messages: List[SighMessage], max_tokens: int) -> SighMessage:
        gpt_messages = self.convert_sigh_messages_to_openai(messages=messages)

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=gpt_messages,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=max_tokens,
            stream=True,
        )

        collected_messages = []

        for chunk in response:
            chunk_message = chunk["choices"][0]["delta"].get("content", "")
            # The actual streaming to the console occurs here:
            print(chunk_message, end="", flush=True)
            collected_messages.append(chunk_message)

        # TODO:
        # number of generated tokens can be retrieved from api

        full_reply_content = "".join(collected_messages)
        return SighMessage(
            content=full_reply_content,
            role="assistant",
            length=self.count_tokens(full_reply_content),
        )

    def convert_sigh_messages_to_openai(
        self, messages: List[SighMessage]
    ) -> List[Dict[str, str]]:
        output = []
        for sm in messages:
            output.append({"role": sm.role, "content": sm.content})
        return output


class OpenAILLMFactory:
    @staticmethod
    def create(model_name: str) -> OpenAILLM:
        model_to_context_length = {
            "gpt-4": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-32k-0613": 32768,
            "gpt-4-0314": 8192,
            "gpt-4-32k-0314": 32768,
            "gpt-3.5-turbo": 4097,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-3.5-turbo-instruct": 4097,
            "gpt-3.5-turbo-0613": 4097,
            "gpt-3.5-turbo-16k-0613": 16385,
            "gpt-3.5-turbo-0301": 4097,
        }
        return OpenAILLM(
            model_name=model_name, context_length=model_to_context_length[model_name]
        )


RetrievalResult = namedtuple("RetrievalResult", ["messages", "remaining_tokens"])
RetrievalResult.__annotations__ = {
    "messages": List[SighMessage],
    "remaining_budget": int,
}


class NegativeTokenBudgetError(ValueError):
    """Raised when the token budget becomes negative."""

    pass


class MemoryBuffer:
    def __init__(self, system_message: Optional[SighMessage] = None) -> None:
        self.buffer: deque[SighMessage] = []
        self.system_message = system_message

    def add_message(self, message: SighMessage) -> None:
        self.buffer.appendleft(message)

    def retrieve_up_to_k_messages(
        self,
        k: int,
        context_length: int,
        min_new_tokens: int,
    ) -> RetrievalResult:
        """
        Retrieve up to k messages from the buffer, considering context length.

        If there's a system message, it will always be included as the first message.
        The method ensures that the total number of tokens from the retrieved
        messages does not exceed the given context_length minus min_new_tokens.

        Args:
            k (int): The maximum number of messages to retrieve.
                If set to -1, attempts to retrieve all messages.
            context_length (int): The maximum token count of all retrieved messages
                combined.
            min_new_tokens (int): The minimum number of tokens that should remain
                available after retrieving the messages.

        Returns:
            RetrievalResult: A named tuple containing the list of retrieved messages
                            (including the system message, if any) and the remaining
                            token budget after deduction.

        Raises:
            NegativeTokenBudgetError: If the token budget after deducting
                the system message length becomes negative.

        Note:
            The method will prioritize recent messages (i.e., those added
            later to the buffer).
        """
        token_budget = context_length - min_new_tokens
        if self.system_message:
            token_budget -= self.system_message.length
            if token_budget < 0:
                raise NegativeTokenBudgetError(
                    f"Token budget is negative: {token_budget}"
                )

        k = len(self.buffer) if k == -1 else min(k, len(self.buffer))

        output = deque()

        for i in range(k):
            if (deduction := self.buffer[i].length) <= token_budget:
                output.appendleft(self.buffer[i])
                token_budget -= deduction
            else:
                break

        output.appendleft(self.system_message)
        return RetrievalResult(
            messages=list(output), remaining_tokens=min_new_tokens + token_budget
        )


class LLMInteractor:
    def __init__(self, llm: LLM, system_prompt: str) -> None:
        self.llm = llm

        system_message = SighMessage(
            content=system_prompt, role="system", length=llm.count_tokens(system_prompt)
        )
        self.memory_buffer = MemoryBuffer(system_message=system_message)

    def on_user_message(
        self,
        message: str,
        k: Optional[int] = -1,
        min_new_tokens: Optional[int] = 256,
    ) -> SighMessage:
        user_message = SighMessage(
            content=message, role="user", length=self.llm.count_tokens(message)
        )

        self.memory_buffer.add_message(user_message)
        try:
            result = self.memory_buffer.retrieve_up_to_k_messages(
                k=k,
                context_length=self.llm.context_length,
                min_new_tokens=min_new_tokens,
            )
        except NegativeTokenBudgetError:
            raise

        ai_message = self.llm.get_response(result.messages, result.remaining_tokens)
        self.memory_buffer.add_message(ai_message)

        return ai_message
