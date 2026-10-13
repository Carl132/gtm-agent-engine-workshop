import json
from uuid import UUID

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from gtm_agent import gtm_agent as agent_module


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def test_agent_exposes_only_app_tools_and_no_deep_agent_nodes():
    tool_node = agent_module.gtm_agent.nodes["tools"].bound
    expected_tools = {
        "lookup_offering",
        "build_prospect_profile",
        "get_prospect",
        "get_current_rep",
        "send_prospect_email",
        "score_prospect",
        "update_prospect_info",
    }

    assert set(tool_node._tools_by_name) == expected_tools
    assert set(agent_module.gtm_agent.get_graph().nodes) == {
        "__start__",
        "model",
        "tools",
        "__end__",
    }
    assert not any(
        unused in agent_module.SYSTEM_PROMPT.lower()
        for unused in ("write_todos", "filesystem", "subagent", "summarization")
    )


def test_app_tools_run_with_runtime_and_representative_email_flow(monkeypatch):
    tool_node = agent_module.gtm_agent.nodes["tools"].bound
    test_agent = create_agent(
        model=ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "get_current_rep", "args": {}, "id": "call1", "type": "tool_call"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_prospect",
                            "args": {"prospect_id": "LEAD-12853"},
                            "id": "call2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "send_prospect_email",
                            "args": {
                                "prospect": {"name": "Taylor", "email": "taylor@example.com"},
                                "subject": "Following up",
                                "body": "Thanks for your time.",
                            },
                            "id": "call3",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Email sent to Taylor."),
            ]
        ),
        tools=list(tool_node._tools_by_name.values()),
        system_prompt=agent_module.SYSTEM_PROMPT,
    )

    monkeypatch.setattr(agent_module, "gtm_agent", test_agent)
    result = agent_module.run_agent(
        "Email Taylor a follow-up.", user_id="rep_amills", thread_id="thread-email"
    )

    tool_messages = [message for message in result["messages"] if message.type == "tool"]
    assert [message.name for message in tool_messages] == [
        "get_current_rep",
        "get_prospect",
        "send_prospect_email",
    ]
    rep_result = json.loads(tool_messages[0].content)
    email_result = json.loads(tool_messages[2].content)
    assert rep_result["rep"]["rep_id"] == "rep_amills"
    assert email_result["from"] == rep_result["rep"]["email"]
    assert email_result["to"] == "taylor@example.com"
    assert result["reply"] == "Email sent to Taylor."


def test_run_agent_preserves_invocation_configuration(monkeypatch):
    captured = {}

    class AgentStub:
        def invoke(self, inputs, config):
            captured["inputs"] = inputs
            captured["config"] = config
            return {"messages": [type("Message", (), {"content": "Done."})()]}

    monkeypatch.setattr(agent_module, "gtm_agent", AgentStub())

    result = agent_module.run_agent(
        "Email this prospect", user_id="rep_amills", environment="staging", thread_id="thread-1"
    )

    assert captured["inputs"] == {"messages": [{"role": "user", "content": "Email this prospect"}]}
    assert captured["config"]["run_name"] == "GTM Assistant"
    assert isinstance(captured["config"]["run_id"], UUID)
    assert captured["config"]["metadata"] == {
        "thread_id": "thread-1",
        "user_id": "rep_amills",
        "environment": "staging",
        "request_intent": "send_email",
    }
    assert result["reply"] == "Done."
    assert result["run_id"] == str(captured["config"]["run_id"])
