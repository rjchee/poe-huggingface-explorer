"""

Bot that lets you talk to Conversational models hosted on HuggingFace.

"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import os
from typing import AsyncIterable

import aiohttp
from fastapi_poe import PoeBot, run
from fastapi_poe.types import (
    QueryRequest,
    SettingsRequest,
    SettingsResponse,
)
from sse_starlette.sse import ServerSentEvent


bot_name_pattern = re.compile("^[a-zA-Z0-9\.\-]+/[a-zA-Z0-9\.\-]+$")

prog_name = "huggingface_poe.py"
parser = argparse.ArgumentParser(prog=prog_name, add_help=False)
parser.add_argument("bot_name", help="Name of the bot in HuggingFace. For example, microsoft/DialoGPT-large.")
parser.add_argument("--min_length", type=int, help="Integer to define the minimum length **in tokens** of the output summary.")
parser.add_argument("--max_length", type=int, help="Integer to define the maximum length **in tokens** of the output summary.")
parser.add_argument("--top_k", type=int, help="Integer to define the top tokens considered within the `sample` operation to create new text.")
parser.add_argument("--top_p", type=float, help="Float to define the tokens that are within the `sample` operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than `top_p`.")
parser.add_argument("--temperature", type=float, help="Float (0.0-100.0). The temperature of the sampling operation. `1` means regular sampling, `0` means always take the highest score, `100.0` is getting closer to uniform probability.")
parser.add_argument("--repetition_penalty", type=float, help="Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in successive generation passes.")

help_message = f"""Unable to parse the HuggingFace bot you want to talk to.
```
{parser.format_help().replace(prog_name + ' ', '')}
```"""

def parse_bot_args(message: str) -> Optional[argparse.Namespace]:
    try:
        split = shlex.split(message)
        print(split)
        args = parser.parse_args(split)
    except SystemExit as e:
        print(f"error parsing args: {e}")
        print(f"original args: {repr(message)}")
        return None

    if bot_name_pattern.match(args.bot_name):
        return args
    print(f"{args.bot_name} does not match the name of a huggingface model")
    return None


class HFBot(PoeBot):
    def __init__(self, hf_api_key: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.client_session = None
        self.hf_api_key = hf_api_key

    def get_or_create_session(self):
        if self.client_session is None:
            self.client_session = aiohttp.ClientSession(
                "https://api-inference.huggingface.co",
                headers={"Authorization": f"Bearer {self.hf_api_key}"},
            )
        return self.client_session

    async def query_huggingface(self, bot_messages: list[str], user_messages: list[str], args: argparse.Namespace) -> tuple[int, dict]:
        url = f"/models/{args.bot_name}"
        latest_message = user_messages.pop()
        assert len(bot_messages) == len(user_messages), f"Number of bot and user messages should be the same, got {len(bot_messages)} bot messages and {len(user_messages)} user messages."
        inputs = {"text": latest_message}
        if bot_messages:
            inputs["generated_responses"] = bot_messages
            inputs["past_user_inputs"] = user_messages
        data = {
            "inputs": inputs,
            "parameters": {
                # Poe times out at 5 seconds, so set the max time a little lower
                "max_time": 4.95,
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k != "bot_name" and v is not None
                },
            },
        }
        print(data)
        async with self.get_or_create_session().post(url, json=data) as resp:
            response_data = await resp.json()
            return resp.status, response_data

    async def get_response(self, query: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        args = None
        conversation_start_index = None
        for i, message in enumerate(query.query):
            if message.role == "user" and (args := parse_bot_args(message.content)) != None:
                conversation_start_index = i + 2
                break

        if conversation_start_index is None:
            yield self.text_event(help_message)
            yield self.suggested_reply_event("microsoft/DialoGPT-large")
            return
        if conversation_start_index >= len(query.query):
            status, response_data = await self.query_huggingface([], ["Hi, how are you?"], args)
            if status == 200:
                yield self.text_event(f"Configured to talk to bot {args.bot_name} with {args=}. You can start the conversation now.")
            else:
                yield self.text_event(f"Error calling the model with these arguments: `{response_data}`")
            return

        user_messages = []
        bot_messages = []
        for message in query.query[conversation_start_index:]:
            if message.role == "user":
                user_messages.append(message.content)
            elif message.role == "bot":
                bot_messages.append(message.content)
            else:
                assert False, f"unknown role {message.role}"

        assert query.query[-1].role == "user", "last message role should be from the user"
        status, response_data = await self.query_huggingface(bot_messages, user_messages, args)
        print(status, response_data)
        if status != 200:
            yield self.text_event(f"Error calling the model: `{response_data}`")
            return
        yield self.text_event(response_data["generated_text"])

    async def get_settings(self, setting: SettingsRequest) -> SettingsResponse:
        print(f"got settings request: {setting}")
        resp = SettingsResponse(
            introduction_message=(
                "Hi, I am the HuggingFaceExplorer. Please provide me the name "
                "of a model on HuggingFace with Hosted Inference API support "
                "get started. For example, microsoft/DialoGPT-large"
            ),
        )
        print(f"responding with {resp.dict()}")
        return resp

    async def on_error(self, error_request: ReportErrorRequest) -> None:
        print(f"received error from server: {error_request}")
        super().on_error()


if __name__ == "__main__":
    hf_key = os.environ["HUGGINGFACE_API_KEY"]
    run(HFBot(hf_key))
