from __future__ import annotations

from gaia_agent.core.risk.models import (
    RiskContext,
    RiskFactor,
)
from gaia_agent.planner.tool_spec import (
    TOOL_CAPABILITIES,
    ToolCapability,
)


class RiskRules:

    def analyze(
        self,
        context: RiskContext,
    ) -> set[RiskFactor]:

        action = context.action.lower().strip()
        tool = (context.tool_name or "").lower().strip()
        arguments = {
            str(key).lower()
            for key in context.arguments
        }

        factors: set[RiskFactor] = set()

        if self._is_destructive(action, tool):
            factors.add(RiskFactor.DESTRUCTIVE)

        if self._modifies_data(action):
            factors.add(RiskFactor.DATA_MODIFICATION)

        if self._is_financial(action, arguments):
            factors.add(RiskFactor.FINANCIAL)

        if self._has_external_side_effect(action, tool):
            factors.add(RiskFactor.EXTERNAL_SIDE_EFFECT)

        if self._involves_privacy(action, arguments):
            factors.add(RiskFactor.PRIVACY)

        if self._involves_security(action, tool):
            factors.add(RiskFactor.SECURITY)

        if self._involves_legal_risk(action):
            factors.add(RiskFactor.LEGAL)

        if self._involves_reputational_risk(action):
            factors.add(RiskFactor.REPUTATIONAL)

        return factors

    def _is_destructive(
        self,
        action: str,
        tool: str,
    ) -> bool:

        destructive_actions = {
            "delete",
            "destroy",
            "drop",
            "remove",
            "terminate",
            "wipe",
            "erase",
            "purge",
        }

        destructive_tools = {
            "delete_tool",
            "destructive_tool",
            "drop_database",
            "wipe_data",
        }

        return (
            any(
                word in action
                for word in destructive_actions
            )
            or tool in destructive_tools
        )

    def _modifies_data(
        self,
        action: str,
    ) -> bool:

        modification_actions = {
            "update",
            "modify",
            "change",
            "edit",
            "insert",
            "create",
            "write",
            "overwrite",
        }

        return any(
            word in action
            for word in modification_actions
        )

    def _is_financial(
        self,
        action: str,
        arguments: set[str],
    ) -> bool:

        financial_keywords = {
            "payment",
            "pay",
            "transfer",
            "send money",
            "refund",
            "purchase",
            "withdraw",
            "deposit",
            "charge",
            "transaction",
        }

        if any(
            word in action
            for word in financial_keywords
        ):
            return True

        financial_keys = {
            "amount",
            "payment",
            "currency",
            "account",
            "bank_account",
            "card",
        }

        return bool(
            financial_keys & arguments
        )

    def _has_external_side_effect(
        self,
        action: str,
        tool: str,
    ) -> bool:

        # Phase 8: capability-aware. Read-only / computation /
        # network-read tools (web_search, file_reader, analyze_image,
        # analyze_excel, python_interpreter, youtube_transcript) never
        # create external side effects and must NOT be blocked.
        capability = TOOL_CAPABILITIES.get(tool)

        if capability in {
            ToolCapability.READ_ONLY,
            ToolCapability.COMPUTATION,
            ToolCapability.NETWORK_READ,
        }:
            return False

        side_effect_actions = {
            "send",
            "publish",
            "post",
            "email",
            "deploy",
            "submit",
        }

        side_effect_tools = {
            "email_tool",
            "http_tool",
            "deployment_tool",
            "webhook_tool",
            "database_write",
            "database_modify",
        }

        return (
            any(
                word in action
                for word in side_effect_actions
            )
            or tool in side_effect_tools
        )

    def _involves_privacy(
        self,
        action: str,
        arguments: set[str],
    ) -> bool:

        privacy_keywords = {
            "password",
            "email",
            "phone",
            "address",
            "personal",
            "private",
            "user data",
            "identity",
            "profile",
        }

        if any(
            word in action
            for word in privacy_keywords
        ):
            return True

        sensitive_keys = {
            "password",
            "email",
            "phone",
            "address",
            "user_id",
            "personal_data",
            "identity",
        }

        return bool(
            sensitive_keys & arguments
        )

    def _involves_security(
        self,
        action: str,
        tool: str,
    ) -> bool:

        security_keywords = {
            "permission",
            "permissions",
            "credential",
            "credentials",
            "authentication",
            "authorization",
            "access control",
            "api key",
            "secret",
        }

        security_tools = {
            "admin_tool",
            "permission_tool",
            "credential_tool",
        }

        return (
            any(
                word in action
                for word in security_keywords
            )
            or tool in security_tools
        )

    def _involves_legal_risk(
        self,
        action: str,
    ) -> bool:

        legal_keywords = {
            "legal",
            "contract",
            "agreement",
            "compliance",
            "regulation",
            "tax",
            "law",
        }

        return any(
            word in action
            for word in legal_keywords
        )

    def _involves_reputational_risk(
        self,
        action: str,
    ) -> bool:

        reputational_keywords = {
            "public statement",
            "official statement",
            "publish announcement",
            "public announcement",
            "represent the company",
            "post publicly",
        }

        return any(
            word in action
            for word in reputational_keywords
        )