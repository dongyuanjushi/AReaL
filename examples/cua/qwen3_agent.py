import datetime
import json
import os
import sys
import time
from pathlib import Path

import json5
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

from areal.experimental.openai import ArealOpenAI
from areal.utils import logging
from .prompt import SYSTEM_PROMPT_QWEN_3

import requests

logger = logging.getLogger("Qwen3 CUA agent")

from .utils import encode_image

def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

def parse_action_and_pyautogui_code(raw_response):
    pass

class MultiTurnQwen3CUAgent():
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_tokens_per_turn: int = 500,
        max_llm_calls_per_run: int = 50,
        max_total_tokens: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.max_total_tokens = max_total_tokens
        self.max_total_tokens_before_finishing = int(max_total_tokens * 0.8)
        self.os_env_ip = "127.0.0.1"

    def reset_os_env(self, task_config: dict):
        self.task_config = task_config
        response = requests.post(f"http://{self.os_env_ip}:20000/reset", json={"task_config": self.task_config})
        self.vm_id = response.json()["vm_id"]
        return response.json()["screenshot"]

    def step(self, pyautogui_code: str):
        response = requests.post(f"http://{self.os_env_ip}:20000/step", json={"action": pyautogui_code, "vm_id": self.vm_id})
        return response.json()

    async def call_server(
        self, client: ArealOpenAI, messages: list[dict], max_attempts: int = 100
    ) -> str:
        attempts = 0
        while attempts < max_attempts:
            try:
                completion = await client.chat.completions.create(
                    messages=messages,
                    temperature=1.0,
                    max_completion_tokens=self.max_tokens_per_turn,
                )
                message = completion.choices[0].message
                assert message, "Error: LLM response is empty."
                return completion, message
            except RuntimeError as e:
                logger.warning(
                    f"RuntimeError during LLM call_server at attempt {attempts}: {e}"
                )
                continue
        raise RuntimeError(
            f"Failed to get response from LLM after {max_attempts} attempts."
        )

    async def run_agent(
        self, data, client: ArealOpenAI, save_path: str | None = None
    ) -> None:
        start_time = time.time()
        data["qid"]
        instruction = data["instruction"]
        self.user_prompt = instruction

        history_images = []
        action_history = []
        completions = []

        initial_screenshot = self.reset_os_env(data["task_config"])

        history_images.append(initial_screenshot)
        system_prompt = SYSTEM_PROMPT_QWEN_3
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": encode_image(initial_screenshot)}}]}
        ]
        stats = dict(
            turns=0,
            num_search=0,
            num_access=0,
        )
        num_llm_calls_available = self.max_llm_calls_per_run
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                break

            completion, message = await self.call_server(client, messages)
            action, pyautogui_code = parse_action_and_pyautogui_code(completion)
            round += 1
            stats["turns"] += 1
            num_llm_calls_available -= 1
            step_response = self.step(pyautogui_code)
            screenshot = step_response["screenshot"]
            is_finish = step_response["is_finish"]
            reward = step_response["reward"]
            history_images.append(screenshot)
            action_history.append(action)
            completions.append(completion)
            if is_finish:
                break
            messages.append({"role": "assistant", "content": action})

        if save_path:
            for i, image in enumerate(history_images):
                with open(os.path.join(save_path, f"step_{i}.png"), "wb") as f:
                    f.write(image)
            logger.debug(f"Result dumped to {save_path}")
            with open(os.path.join(save_path, "action_history.jsonl"), "w") as f:
                for action in action_history:
                    f.write(json.dumps(action) + "\n")

        return completions, reward

    async def make_trajectory(
        self,
        data: dict[str, str],
        client: ArealOpenAI,
        save_path: str | None = None,
    ) -> dict:
        completions, reward = await self.run_agent(data, client, save_path=save_path)
        last_completion = completions[-1]
        client.set_reward(last_completion.id, reward)
        return completions
