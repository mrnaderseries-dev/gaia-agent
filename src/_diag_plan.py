"""Quick check: does the planner produce a valid PlanSchema with qwen2.5:3b?"""
import asyncio
import sys

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from gaia_agent.llm.model import LLMModel
from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.planner.planner import Planner
from gaia_agent.reliability.loop_detector import LoopDetector
from gaia_agent.tools.registry import ToolRegistry


async def main():
    llm_client = OllamaClient(base_url="http://localhost:11434")
    model = LLMModel(provider="ollama", model="qwen2.5:3b",
                     max_tokens=2048, temperature=0.2)

    reg = ToolRegistry(base_dir=".")
    available_tools = {s.name: s for s in reg.get_tool_specs()}

    planner = Planner(
        client=llm_client, model=model,
        available_tools=available_tools,
        loop_detector=LoopDetector(),
    )

    question = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?"
    print("Generating plan...", flush=True)
    try:
        plan = await planner.create_plan(user_question=question, context=[])
        print("PLAN_OK, steps:", len(plan.steps))
        for s in plan.steps:
            print(f"  #{s.step_id} {s.step_type.value} tool={s.tool_name} args={s.arguments} final={s.is_final_answer}")
    except Exception as exc:
        print("PLAN_FAILED:", type(exc).__name__)
        print(str(exc)[:1000])


if __name__ == "__main__":
    asyncio.run(main())
