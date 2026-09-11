from pathlib import Path

from gaia_agent.context.attachments import Attachment
from gaia_agent.context.request_builder import ContextRequestBuilder
from gaia_agent.core.agent_state import AgentState


def test_attachment_survives_state_to_context_boundary():
    path = str(Path("evaluation_files/task-123/report.pdf").resolve())

    attachment = Attachment(
        attachment_id="task-123:report.pdf",
        filename="report.pdf",
        path=path,
    )

    state = AgentState(
        user_request="Analyze the attached report.",
        attachments=[attachment],
    )

    context = ContextRequestBuilder.from_state(state)

    assert context.user_request == "Analyze the attached report."
    assert len(context.attachments) == 1

    forwarded = context.attachments[0]

    assert forwarded.attachment_id == "task-123:report.pdf"
    assert forwarded.filename == "report.pdf"
    assert forwarded.path == path


def test_attachment_does_not_modify_user_request():
    attachment = Attachment(
        attachment_id="task-123:report.pdf",
        filename="report.pdf",
        path=r"C:\evaluation_files\task-123\report.pdf",
    )

    state = AgentState(
        user_request="Analyze the attached report.",
        attachments=[attachment],
    )

    context = ContextRequestBuilder.from_state(state)

    assert context.user_request == "Analyze the attached report."
    assert "C:\\" not in context.user_request
    assert "report.pdf" not in context.user_request

def test_attachment_is_available_to_planner_context():
    path = str(Path("evaluation_files/task-123/report.pdf").resolve())

    attachment = Attachment(
        attachment_id="task-123:report.pdf",
        filename="report.pdf",
        path=path,
    )

    state = AgentState(
        user_request="Analyze the attached report.",
        attachments=[attachment],
    )

    context = ContextRequestBuilder.from_state(state)

    assert len(context.attachments) == 1
    assert context.attachments[0].attachment_id == "task-123:report.pdf"
    assert context.attachments[0].filename == "report.pdf"
    assert context.attachments[0].path == path