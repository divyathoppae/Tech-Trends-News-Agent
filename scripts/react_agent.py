import re
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from llm_interface import HF_LLM
from search_articles import load_processed, search_corpus

# ----------------------------
# Step + Config dataclasses
# ----------------------------
@dataclass
class Step:
    thought: str
    action: str
    observation: str

@dataclass
class AgentConfig:
    max_steps: int = 6
    allow_tools: Tuple[str, ...] = ("search", "finish")
    verbose: bool = True

# ----------------------------
# Helper functions
# ----------------------------
def parse_action(line: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse an Action line into (tool_name, args)."""
    if not line.startswith("Action:"):
        return None
    s = line[len("Action:"):].strip()
    lb, rb = s.find("["), s.rfind("]")
    if lb == -1 or rb == -1 or rb < lb:
        return None
    name = s[:lb].strip()
    inner = s[lb+1:rb].strip()

    args = {}
    if inner:
        for field in inner.split(","):
            if "=" in field:
                key, val = field.split("=", 1)
                args[key.strip()] = val.strip().strip('"')
    return name, args

def format_history(trajectory: List[Step]) -> str:
    lines = []
    for step in trajectory:
        lines.append(f"Thought: {step.thought}")
        lines.append(f"Action: {step.action}")
        lines.append(f"Observation: {step.observation}")
    return "\n".join(lines)

def make_prompt(user_query: str, trajectory: List[Step]) -> str:
    SYSTEM_PREAMBLE = (
        "You are a helpful ReAct agent. You may use tools to answer factual questions.\n\n"
        "Available tools:\n"
        "- search[query=\"<text>\", k=<int>] # searches the tech corpus\n"
        "- finish[answer=\"<final answer>\"] # ends the task\n\n"
        "Follow the exact step format:\n"
        "Thought: <your reasoning>\n"
        "Action: <one of the tool calls above>\n\n"
        "IMPORTANT:\n"
        "- When you use finish[answer=...], provide a clear, well-structured paragraph that fully answers the user question.\n"
        "- The paragraph should be natural language, not just keywords.\n"
    )
    history_block = format_history(trajectory)
    return f"{SYSTEM_PREAMBLE}\n\nUser Question: {user_query}\n\n{history_block}\nNext step:\nThought:"

# ----------------------------
# ReActAgent class
# ----------------------------
class ReActAgent:
    def __init__(self, llm=None, config=None):
        self.llm = llm or HF_LLM()
        self.config = config or AgentConfig()
        self.trajectory: List[Step] = []
        self.corpus = load_processed()  # load latest processed corpus (list of dicts)

    def run(self, user_query: str) -> Dict[str, Any]:
        self.trajectory.clear()
        for step_idx in range(self.config.max_steps):
            prompt = make_prompt(user_query, self.trajectory)
            out = self.llm(prompt)

            # Extract Thought + Action
            t_match = re.search(r"Thought:\s*(.*)", out)
            a_match = re.search(r"Action:\s*(.*)", out)
            thought = t_match.group(1).strip() if t_match else "(no thought)"
            action_line = a_match.group(1).strip() if a_match else "finish[answer=\"(no answer)\"]"
            action_line = "Action: " + action_line

            parsed = parse_action(action_line)
            if not parsed:
                obs = "Invalid action."
                self.trajectory.append(Step(thought, action_line, obs))
                continue

            name, args = parsed
            obs = ""

            if name == "search":
                query = args.get("query", "")
                try:
                    k = int(args.get("k", 3))
                except ValueError:
                    k = 3
                results = search_corpus(query, self.corpus, k=k)
                obs = json.dumps({"results": results}, indent=2)
            elif name == "finish":
                obs = "done"
                self.trajectory.append(Step(thought, action_line, obs))
                final = {"answer": args.get("answer", ""), "trajectory": [asdict(s) for s in self.trajectory]}
                self.save_run(user_query, final)
                return final

            self.trajectory.append(Step(thought, action_line, obs))

        final = {"answer": "(max steps reached, no final answer)", "trajectory": [asdict(s) for s in self.trajectory]}
        self.save_run(user_query, final)
        return final

    def save_run(self, user_query: str, result: Dict[str, Any]):
        base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "agent_runs")
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(base_dir, f"run_{timestamp}.json")
        payload = {"query": user_query, "result": result}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if self.config.verbose:
            print(f"💾 Agent run saved to {out_path}")

# ----------------------------
# Quick test
# ----------------------------
if __name__ == "__main__":
    agent = ReActAgent()
    result = agent.run("What are the latest technology trends?")
    print("\nFinal Answer:", result["answer"])
    print("\nTrajectory:")
    for step in result["trajectory"]:
        print(step)
