from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import openai
import tiktoken


class RoleEnum(str, Enum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"


@dataclass(slots=True)
class SighMessage:
    content: str
    role: RoleEnum
    length: int

    def __post_init__(self):
        if not isinstance(self.role, RoleEnum):
            raise ValueError(
                f"Invalid role: {self.role}. Expected one of {list(RoleEnum)}"
            )


class LLM(ABC):
    @property
    @abstractmethod
    def context_length(self) -> int:
        ...

    @abstractmethod
    def get_reponse(self, messages: List[SighMessage]) -> SighMessage:
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

    def get_reponse(self, messages: List[SighMessage]) -> SighMessage:
        gpt_messages = self.convert_sigh_messages_to_openai(messages=messages)

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=gpt_messages,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
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


class MemoryBuffer:
    def __init__(self) -> None:
        self.buffer: List[SighMessage] = []
