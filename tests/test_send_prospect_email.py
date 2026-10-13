import unittest
from unittest.mock import patch


class StubChatModel:
    def __init__(self, *args, **kwargs):
        pass

    def with_structured_output(self, *args, **kwargs):
        return self


with (
    patch("langchain_openai.ChatOpenAI", StubChatModel),
    patch("deepagents.create_deep_agent", return_value=object()),
):
    from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def test_disqualified_prospect_is_blocked_without_message_id(self):
        prospect = {
            "prospect_id": "LEAD-50001",
            "name": "Taylor Morgan",
            "email": "taylor@example.com",
        }
        with (
            patch(
                "gtm_agent.gtm_agent.data_service.get_prospect_record",
                return_value={"disqualified": True},
            ) as get_prospect_record,
            patch("gtm_agent.gtm_agent.uuid.uuid4") as uuid4,
        ):
            result = send_prospect_email.func(
                prospect,
                "Hello",
                "Checking in",
                runtime=None,
                from_rep={},
            )

        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
        })
        get_prospect_record.assert_called_once_with("LEAD-50001")
        uuid4.assert_not_called()

    def test_qualified_prospect_is_sent(self):
        prospect = {
            "prospect_id": "LEAD-50006",
            "name": "Jordan Lee",
            "email": "jordan@example.com",
        }
        with patch(
            "gtm_agent.gtm_agent.data_service.get_prospect_record",
            return_value={"disqualified": False},
        ):
            result = send_prospect_email.func(
                prospect,
                "Hello",
                "Checking in",
                runtime=None,
                from_rep={},
            )

        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], prospect["email"])

    def test_prompt_explains_blocked_disqualified_send(self):
        self.assertIn("Never email a prospect marked disqualified", SYSTEM_PROMPT)
        self.assertIn("if a send is blocked, tell the rep why", SYSTEM_PROMPT)
        self.assertIn("accurately report the outcome", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
