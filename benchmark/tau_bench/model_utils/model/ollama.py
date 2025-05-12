import os
import requests
import json as pyjson

from benchmark.tau_bench.model_utils.api.datapoint import Datapoint
from benchmark.tau_bench.model_utils.model.chat import ChatModel, Message
from benchmark.tau_bench.model_utils.model.completion import approx_cost_for_datapoint, approx_prompt_str
from benchmark.tau_bench.model_utils.model.general_model import wrap_temperature
from benchmark.tau_bench.model_utils.model.utils import approx_num_tokens


# Placeholder: update with actual Ollama pricing if available
PRICE_PER_INPUT_TOKEN = 0.0
CAPABILITY_SCORE = 0.5
LATENCY_MS_PER_OUTPUT_TOKEN = 0.0
MAX_CONTEXT_LENGTH = 8192

class OllamaModel(ChatModel):
    def __init__(
        self,
        model: str | None = None,
        api_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        if model is None:
            self.model = os.getenv("DEFAULT_OLLAMA_MODEL","gemma3")
        else:
            self.model = model
        if api_url is None:
            api_url = os.getenv("OLLAMA_API_URL_ENV_VAR", "http://localhost:8434")
        self.api_url = api_url
        self.temperature = temperature

    def generate_message(
        self,
        messages: list[Message],
        force_json: bool,
        temperature: float | None = None,
    ) -> Message:
        """
        Calls the Ollama API to generate a message based on the conversation history.
        """
        api_url = self.api_url.rstrip("/") + "/api/chat"
        temp = self.temperature if temperature is None else temperature
        # Convert Message objects to dicts as expected by Ollama
        ollama_messages = []
        for m in messages:
            role = m.role if hasattr(m, "role") else "user"
            ollama_messages.append({
                "role": role,
                "content": m.content,
            })
        # If force_json, add a system message to instruct the model
        if force_json:
            ollama_messages.insert(0, {
                "role": "system",
                "content": "Respond ONLY in valid JSON."
            })
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "temperature": temp,
        }
        try:
            response = requests.post(api_url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            # If force_json, try to parse the response as JSON
            if force_json:
                try:
                    _ = pyjson.loads(content)
                except Exception:
                    # Optionally, you could raise or log here
                    pass
            return Message(role="assistant", content=content)
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")

    def get_approx_cost(self, dp: Datapoint) -> float:
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=PRICE_PER_INPUT_TOKEN)

    def get_latency(self, dp: Datapoint) -> float:
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=LATENCY_MS_PER_OUTPUT_TOKEN)

    def get_capability(self) -> float:
        return CAPABILITY_SCORE

    def supports_dp(self, dp: Datapoint) -> bool:
        prompt = approx_prompt_str(dp)
        return approx_num_tokens(prompt) <= MAX_CONTEXT_LENGTH
