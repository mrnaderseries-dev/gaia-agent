import sys, types

sys.path.insert(0, r"c:\Users\user\gaia-agent\src")

from gaia_agent.planner.planner import Planner
from gaia_agent.planner.plan_schema import PlanStep, StepType


def spec(name, arguments_schema):
    return types.SimpleNamespace(name=name, arguments_schema=arguments_schema)


tools = {
    "web_search": spec("web_search", {"query": {"type": "string"}}),
    "visit_webpage": spec("visit_webpage", {"url": {"type": "string"}}),
}


class FakeLoop:
    pass


pl = Planner(client=object(), model="m", available_tools=tools, loop_detector=FakeLoop())

assert pl._repair_tool_name("extract_information_from_webpage") == "visit_webpage"
assert pl._repair_tool_name("web search") == "web_search"
assert pl._repair_tool_name("totally_unrelated_thing") is None

step = PlanStep(
    step_id=0,
    action="a",
    step_type=StepType.TOOL,
    tool_name="extract_information_from_webpage",
    arguments={"link": "https://x", "junk": 1},
)
pl._validate_step(step)
assert step.tool_name == "visit_webpage", step.tool_name
assert step.arguments == {"url": "https://x"}, step.arguments

print("REPAIR_OK")
