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
    inner = s[lb + 1:rb].strip()

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
        "- You MUST call finish[] once you have enough information to answer.\n"
        "- Do not search more than 2-3 times before finishing.\n"
    )
    history_block = format_history(trajectory)
    
    # Don't append "Thought:" here - let the LLM generate the full response
    if history_block:
        return f"{SYSTEM_PREAMBLE}\n\nUser Question: {user_query}\n\n{history_block}\n\nNext step:"
    else:
        return f"{SYSTEM_PREAMBLE}\n\nUser Question: {user_query}\n\nBegin:"


# ----------------------------
# ReActAgent class
# ----------------------------
class ReActAgent:
    def __init__(self, llm=None, config=None):
        self.llm = llm or HF_LLM()
        self.config = config or AgentConfig()
        self.trajectory: List[Step] = []
        self.corpus = load_processed()

    def _parse_llm_output(self, out: str) -> Tuple[str, str]:
        """Parse LLM output to extract thought and action."""
        thought = "(no thought)"
        action_line = "Action: finish[answer=\"(no answer)\"]"

        # Try to find Thought and Action in the output
        # Handle case where LLM includes "Thought:" or not
        
        # First, check if output contains both markers
        if "Action:" in out:
            if "Thought:" in out:
                # Standard format: "Thought: ... Action: ..."
                t_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", out, re.DOTALL)
                thought = t_match.group(1).strip() if t_match else "(no thought)"
            else:
                # LLM didn't include "Thought:" - everything before "Action:" is the thought
                parts = out.split("Action:", 1)
                thought = parts[0].strip()
            
            # Extract the action
            a_match = re.search(r"Action:\s*(.+?)(?:\n|$)", out)
            if a_match:
                action_line = "Action: " + a_match.group(1).strip()
        else:
            # No Action found - treat entire output as thought and force finish
            thought = out.strip()
            if self.config.verbose:
                print("⚠️ No Action found in LLM output, forcing finish")

        return thought, action_line

    def run(self, user_query: str) -> Dict[str, Any]:
        self.trajectory.clear()
        
        for step_idx in range(self.config.max_steps):
            prompt = make_prompt(user_query, self.trajectory)
            out = self.llm(prompt)

            if self.config.verbose:
                print(f"\n{'='*50}")
                print(f"Step {step_idx + 1}/{self.config.max_steps}")
                print(f"{'='*50}")
                print(f"LLM Output:\n{out}")

            # Parse the output
            thought, action_line = self._parse_llm_output(out)

            if self.config.verbose:
                print(f"\nParsed Thought: {thought}")
                print(f"Parsed Action: {action_line}")

            parsed = parse_action(action_line)
            if not parsed:
                obs = "Invalid action format. Use: search[query=\"...\", k=3] or finish[answer=\"...\"]"
                self.trajectory.append(Step(thought, action_line, obs))
                if self.config.verbose:
                    print(f"Observation: {obs}")
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
                if self.config.verbose:
                    print(f"🔍 Search query: '{query}', k={k}")
                    print(f"Observation: {obs[:500]}..." if len(obs) > 500 else f"Observation: {obs}")
                    
            elif name == "finish":
                obs = "done"
                self.trajectory.append(Step(thought, action_line, obs))
                final = {
                    "answer": args.get("answer", ""),
                    "trajectory": [asdict(s) for s in self.trajectory]
                }
                if self.config.verbose:
                    print(f"✅ Finished!")
                self.save_run(user_query, final)
                return final
            else:
                obs = f"Unknown tool: {name}. Available tools: search, finish"
                if self.config.verbose:
                    print(f"Observation: {obs}")

            self.trajectory.append(Step(thought, action_line, obs))

        # Max steps reached - compile what we have
        if self.config.verbose:
            print(f"\n⚠️ Max steps ({self.config.max_steps}) reached without finish")
        
        # Try to generate a final answer from the trajectory
        final_answer = self._generate_fallback_answer(user_query)
        final = {
            "answer": final_answer,
            "trajectory": [asdict(s) for s in self.trajectory]
        }
        self.save_run(user_query, final)
        return final

    def _generate_fallback_answer(self, user_query: str) -> str:
        """Generate a fallback answer if max steps reached."""
        # Collect all search results from observations
        all_results = []
        for step in self.trajectory:
            if step.observation and step.observation != "done":
                try:
                    data = json.loads(step.observation)
                    if "results" in data:
                        all_results.extend(data["results"])
                except (json.JSONDecodeError, TypeError):
                    pass
        
        if all_results:
            return f"Based on search results: {json.dumps(all_results[:3], indent=2)}"
        return "(max steps reached, no final answer)"

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
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(f"\nFinal Answer: {result['answer']}")
    print(f"\nTotal Steps: {len(result['trajectory'])}")