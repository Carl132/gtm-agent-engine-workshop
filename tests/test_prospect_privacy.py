import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent.gtm_agent import (
    ProspectScore,
    RubricBreakdown,
    build_prospect_profile,
    get_prospect,
    score_prospect,
)


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()

    def assert_no_billing_data(self, value):
        serialized = json.dumps(value)
        for field in ("billing_qualification", "tax_id", "card_on_file"):
            self.assertNotIn(field, serialized)

    def test_get_prospect_returns_only_contact_fields(self):
        result = get_prospect.invoke({"prospect_id": "LEAD-12853"})

        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": "LEAD-12853",
                "name": "Omar Okafor",
                "email": "omar.okafor@lakesideanalytics.com",
                "disqualified": False,
            },
        )
        self.assert_no_billing_data(result)

    def test_get_prospect_defaults_disqualified_to_false(self):
        with patch.object(data_service, "get_prospect_record", return_value={
            "name": "Prospect",
            "email": "prospect@example.com",
        }):
            result = get_prospect.invoke({"prospect_id": "LEAD-TEST"})

        self.assertFalse(result["prospect"]["disqualified"])

    def test_build_profile_sanitizes_new_and_cached_profiles(self):
        with patch.object(data_service, "get_prospect_record") as get_record:
            get_record.return_value = {
                **data_service.get_prospect_record("LEAD-12853"),
                "billing_qualification": {
                    "tax_id": "123-45-6789",
                    "card_on_file": "4111111111111111",
                },
            }
            built = build_prospect_profile.invoke({"prospect_id": "LEAD-12853"})

        self.assert_no_billing_data(built)
        self.assert_no_billing_data(data_service._PROFILES["LEAD-12853"])

        data_service._PROFILES["LEAD-12853"]["billing_qualification"] = {
            "tax_id": "123-45-6789",
            "card_on_file": "4111111111111111",
        }
        cached = build_prospect_profile.invoke({"prospect_id": "LEAD-12853"})

        self.assert_no_billing_data(cached)
        self.assert_no_billing_data(data_service._PROFILES["LEAD-12853"])

    def test_scoring_receives_sanitized_profile(self):
        profile = build_prospect_profile.invoke({
            "prospect_id": "LEAD-12853",
        })["prospect_profile"]
        score = ProspectScore(
            score=80,
            justification="Good fit",
            rubric_breakdown=RubricBreakdown(
                revenue_fit=80,
                tech_stack_match=80,
                segment_fit=80,
            ),
        )
        with patch("gtm_agent.gtm_agent._scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = score
            result = score_prospect.invoke({
                "prospect_profile": profile,
                "offering": {
                    "required_tech_stack": ["Kafka"],
                    "min_annual_revenue": 50000000,
                    "description": "Test offering",
                },
            })

        self.assertEqual(result["score"], 80)
        self.assert_no_billing_data(profile)
        self.assert_no_billing_data(
            scoring_llm.invoke.call_args.args[0][1]["content"]
        )


if __name__ == "__main__":
    unittest.main()
