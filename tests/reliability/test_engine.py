import pytest

from gaia_agent.reliability.engine import ReliabilityEngine
from gaia_agent.reliability.error_handler import ErrorHandler
from gaia_agent.reliability.failure_classifier import FailureClassifier
from gaia_agent.reliability.policies.recovery_policy import RecoveryPolicy
from gaia_agent.reliability.policies.retry_policy import RetryPolicy
from gaia_agent.reliability.recovery import Recovery
from gaia_agent.reliability.retry import Retry
from gaia_agent.reliability.exception import ToolExecutionError


def make_engine(
    *,
    max_attempts=3,
    max_recoveries=2,
    max_total_executions=10,
):
    return ReliabilityEngine(
        error_handler=ErrorHandler(),
        failure_classifier=FailureClassifier(),
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            base_delay=0,
            max_delay=0,
        ),
        recovery_policy=RecoveryPolicy(
            allow_replan=True,
        ),
        retry=Retry(),
        recovery=Recovery(),
        max_recoveries=max_recoveries,
        max_total_executions=max_total_executions,
    )


@pytest.mark.asyncio
async def test_1_success_no_retry_no_recovery():
    engine = make_engine()

    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        return "success"

    result = await engine.execute(
        operation,
        operation_name="test_operation",
        source="test",
    )

    assert result.success is True
    assert result.result == "success"
    assert result.attempts == 1
    assert result.recovery_count == 0
    assert calls == 1


@pytest.mark.asyncio
async def test_2_transient_failure_then_success():
    engine = make_engine(
        max_attempts=3,
    )

    calls = 0

    async def operation():
        nonlocal calls
        calls += 1

        if calls == 1:
            raise TimeoutError("temporary timeout")

        return "success"

    result = await engine.execute(
        operation,
        operation_name="test_operation",
        source="test",
    )

    assert result.success is True
    assert result.result == "success"

    # First execution failed, second succeeded.
    assert calls == 2
    assert result.attempts == 2
    assert result.recovery_count == 0


@pytest.mark.asyncio
async def test_3_recoverable_failure_replans_and_executes_new_plan():
    engine = make_engine(
        max_attempts=1,
        max_recoveries=1,
        max_total_executions=3,
    )

    executions = 0
    recovery_calls = 0

    async def operation():
        nonlocal executions
        executions += 1

        if executions == 1:
            raise ToolExecutionError("recoverable failure")

        return "new-plan-success"

    async def recovery_operation(error):
        nonlocal recovery_calls
        recovery_calls += 1

        # Simulate creating a genuinely different plan.
        return {
            "plan_id": "plan-2",
            "strategy": "different",
        }

    def change_detector(new_plan):
        return new_plan["plan_id"] == "plan-2"

    result = await engine.execute(
        operation,
        operation_name="test_operation",
        source="test",
        recovery_operation=recovery_operation,
        recovery_change_detector=change_detector,
    )

    assert result.success is True
    assert result.result == "new-plan-success"

    assert executions == 2
    assert recovery_calls == 1
    assert result.recovery_count == 1


@pytest.mark.asyncio
async def test_4_same_replan_is_rejected():
    engine = make_engine(
        max_attempts=1,
        max_recoveries=2,
        max_total_executions=5,
    )

    executions = 0
    recovery_calls = 0

    async def operation():
        nonlocal executions
        executions += 1
        raise ToolExecutionError("same failure")

    async def recovery_operation(error):
        nonlocal recovery_calls
        recovery_calls += 1

        # Simulate planner returning the SAME execution.
        return {
            "plan_id": "plan-1",
            "tool": "web_search",
            "query": "Malko",
        }

    def change_detector(new_plan):
        # False means:
        # the replan did not actually change the execution.
        return False

    result = await engine.execute(
        operation,
        operation_name="web_search",
        source="test",
        recovery_operation=recovery_operation,
        recovery_change_detector=change_detector,
    )

    assert result.success is False

    # Only one original execution.
    assert executions == 1

    # Recovery was attempted once.
    assert recovery_calls == 1

    assert result.recovery_count == 0

    assert "change" in result.reason.lower()