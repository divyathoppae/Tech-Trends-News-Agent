import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOAD_8BIT = False
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

T_PATTERN = re.compile(r"Thought:\s*(.+)")
A_PATTERN = re.compile(r"Action:\s*(.+)")

def _postprocess_to_two_lines(text: str) -> str:
    text = text.split("\nObservation:")[0]
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    thought, action = None, None
    for ln in lines:
        if thought is None:
            m = T_PATTERN.match(ln)
            if m:
                thought = m.group(1).strip()
                continue
        if action is None:
            m = A_PATTERN.match(ln)
            if m:
                action = m.group(1).strip()
                continue

    if thought is None:
        thought = "I should search for key facts related to the question."
    if action is None:
        action = 'search[query="(auto) refine the user question", k=3]'

    return f"Thought: {thought}\nAction: {action}"

class HF_LLM:
    def __init__(self, model_name=MODEL_NAME, load_8bit=LOAD_8BIT,
                 dtype=DTYPE, max_new_tokens=160, generation_kwargs=None):
        self.model_name = model_name
        self.load_8bit = load_8bit
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = generation_kwargs or {}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            dtype=self.dtype,
            trust_remote_code=True
        )

        self.gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.generation_kwargs.get("temperature", 0.3),
            do_sample=self.generation_kwargs.get("do_sample", True)
        )

        self.format_guard = (
            "You are a helpful ReAct agent. Respond with EXACTLY two lines:\n"
            "Thought: <one concise sentence>\n"
            "Action: <ONE tool call ONLY, either search[...] OR finish[...]>\n"
            "Do NOT combine search and finish in the same line.\n"
        )

    def __call__(self, prompt: str) -> str:
        full_prompt = prompt + "\n\n" + self.format_guard
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=self.gen_cfg)
        completion = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
        print(_postprocess_to_two_lines(completion))
        return _postprocess_to_two_lines(completion)

if __name__ == "__main__":
    llm = HF_LLM()
    print(llm("Who painted The Starry Night?"))
