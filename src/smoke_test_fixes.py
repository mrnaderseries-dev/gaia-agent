import asyncio
import sys

from gaia_agent.core.evidence import ToolResultRecord
from gaia_agent.core.agent_execution import is_tool_error_result
from gaia_agent.tools.path_utils import is_placeholder_path, list_available_files, resolve_file
from gaia_agent.planner.task_classifier import TaskClassifier, detect_factorial_ratio, detect_simple_operation
from gaia_agent.agents.verifier import deterministic_verification, VerificationStatus


def test_placeholders():
    assert is_placeholder_path("<file_path_to_sales_file>")
    assert is_placeholder_path("[FILE_PATH]")
    assert is_placeholder_path("[/path/to/sales_file]")
    assert is_placeholder_path("<your_actual_path_here>")
    assert is_placeholder_path("[the file path here]")
    assert is_placeholder_path("your_file_path")
    assert not is_placeholder_path("data.csv")
    print("placeholders OK")


def test_resolve_existing():
    p = resolve_file(".", "config.py")
    print("resolve config.py ->", p)
    assert p is not None and p.name == "config.py"


def test_tool_error_detection():
    assert is_tool_error_result("Error: Excel file not found: abc.xlsx (searched...)")
    assert is_tool_error_result("Error fetching the webpage: 403 Client Error: Forbidden")
    assert is_tool_error_result("Traceback (most recent call last)")
    assert is_tool_error_result("Image not found: img.png (placeholder or invalid path).")
    assert is_tool_error_result("Excel file not found: f.xlsx")
    assert is_tool_error_result("Error: File 'x' not found in base_dir 'y' or any working directory.")
    assert is_tool_error_result("name 'file_reader' is not defined")
    assert is_tool_error_result("Timeout while fetching")
    assert is_tool_error_result("Rate Limit Hit. Too Many Requests")
    assert is_tool_error_result("Tool 'foo' is not registered")
    assert not is_tool_error_result("## Search Results\n\n[Ocean - Wikipedia](...) The Pacific is the largest.")
    assert not is_tool_error_result("Result variable: 210")
    assert not is_tool_error_result("42")
    print("tool_error detection OK")


def test_classifier():
    c = TaskClassifier()
    tools = ["python_interpreter", "web_search", "visit_webpage", "file_reader", "analyze_excel", "analyze_image"]
    cases = [
        ("What is the value of 15 factorial divided by 13 factorial?", "arithmetic"),
        ("Rewrite the sentence backwards if you can understand it", "text_transformation"),
        ("If you can understand this sentence write the opposite of the word left as the answer", "text_transformation"),
        ("What is the chemical symbol for the element with atomic number 79?", "factual_search"),
        ("What is 12 multiplied by 3?", "arithmetic"),
        ("Read the attached sales CSV and compute total revenue", "local_file"),
        ("https://www.youtube.com/watch?v=abc What did he say?", "audio_video"),
        ("Count the number of letters in the word conspicuously", "text_transformation"),
        ("Capital of Australia", "factual_search"),
    ]
    for q, expected in cases:
        a = c.classify(q, available_files=[], available_tools=tools)
        print(f"{q[:45]!r:47} -> {a.intent.value} (expected {expected})")
        assert a.intent.value == expected, q
    print("classifier OK")


def test_factorial_detection():
    assert detect_factorial_ratio("What is the value of 15 factorial divided by 13 factorial?") == (15, 13)
    assert detect_factorial_ratio("What is 10! / 8! ?") == (10, 8)
    assert detect_simple_operation("what is 12 + 3?") == "12 + 3"
    print("factorial/op detection OK")


def test_deterministic_verification():
    ev_3 = ToolResultRecord(step_id=0, tool_name="python_interpreter", arguments={}, result="3", succeeded=True)
    ev_4 = ToolResultRecord(step_id=0, tool_name="python_interpreter", arguments={}, result="4", succeeded=True)
    ev_210 = ToolResultRecord(step_id=0, tool_name="python_interpreter", arguments={}, result="210", succeeded=True)
    ev_bad = ToolResultRecord(step_id=0, tool_name="web_search", arguments={}, result="Error: file not found", succeeded=True)
    ev_fail = ToolResultRecord(step_id=0, tool_name="python_interpreter", arguments={}, result="Traceback ...", succeeded=True)

    status, reason = deterministic_verification("4", [ev_3])
    assert status == VerificationStatus.FAIL, (status, reason)
    print("FAIL detect:", reason)

    status, reason = deterministic_verification("3", [ev_3])
    assert status == VerificationStatus.PASS, (status, reason)

    status, reason = deterministic_verification("210", [ev_210])
    assert status == VerificationStatus.PASS, (status, reason)

    status, reason = deterministic_verification("210", [])
    assert status == VerificationStatus.UNCERTAIN, (status, reason)

    # No strong evidence -> uncertain
    status, reason = deterministic_verification("4", [ev_bad])
    assert status == VerificationStatus.UNCERTAIN or status == VerificationStatus.FAIL, (status, reason)

    # Failed tool record should not be treated as strong evidence
    status, reason = deterministic_verification("something", [ev_fail])
    print("status for failed record:", status)
    print("deterministic_verification OK")


def test_list_files():
    files = list_available_files(".")
    print("available files:", files[:10])
    print("list_files OK")


def main():
    test_placeholders()
    test_resolve_existing()
    test_tool_error_detection()
    test_classifier()
    test_factorial_detection()
    test_deterministic_verification()
    test_list_files()
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()