import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from qmagent.conversation import parse_reply, validate_message, validate_proposal
from qmagent.policies import CHAT_PROMPT, HuggingFacePolicy, MINIMAL_POLICY_PROMPT, RulePolicy
from qmagent.service import AgentService
from qmagent.session import CalibrationSession
from qmagent.settings import Settings
from qmagent.terminal import Terminal


class ChatPolicy(RulePolicy):
    def __init__(self):
        self.reply = "你好，我可以解释 IQ 分割和校准步骤。"
        self.calls = []

    def chat(self, messages, context):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(context)))
        return self.reply


class ChatPromptTests(unittest.TestCase):
    def test_calibration_keeps_reference_decoder(self):
        policy = HuggingFacePolicy.__new__(HuggingFacePolicy)
        policy.max_new_tokens = 128
        policy.prompt_profile = "skill"
        policy.fit_update_tool = False
        from qmagent.protocol import TOOLS_SCHEMA
        policy._generate = Mock(return_value='<tool_call>{"name":"calibration.step","arguments":{"next_tool":"sq.s21","parameter_action":{"updates":{},"scan":{}},"reason":"Start"}}</tool_call>')
        policy.decide({"state": {}})
        self.assertEqual(policy._generate.call_args.kwargs, {"allow_graph": False, "tools": TOOLS_SCHEMA})

    def test_minimal_ablation_changes_only_instruction_profile(self):
        from qmagent.protocol import policy_messages
        context = {"state": {}, "observation": None}
        minimal = policy_messages(context, prompt_profile="minimal")
        skill = policy_messages(context, prompt_profile="skill")
        self.assertEqual(minimal[1], skill[1])
        self.assertEqual(minimal[0]["content"].splitlines()[0], MINIMAL_POLICY_PROMPT.splitlines()[0])
        self.assertNotIn("Ramsey needs correction", minimal[0]["content"])
        self.assertIn("Ramsey needs correction", skill[0]["content"])

    def test_language_rule_and_general_capabilities_are_explicit(self):
        self.assertIn("must be in English", CHAT_PROMPT)
        self.assertIn("requests in any language", CHAT_PROMPT)
        self.assertIn("general-purpose language model", CHAT_PROMPT)
        self.assertIn("Do not redirect unrelated questions to calibration", CHAT_PROMPT)
        self.assertNotIn("用中文自然", CHAT_PROMPT)

    def test_mixed_language_history_is_forwarded_without_forced_translation(self):
        policy = HuggingFacePolicy.__new__(HuggingFacePolicy)
        policy._generate = Mock(return_value="A list is mutable; a tuple is immutable.")
        messages = [{"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                    {"role": "user", "content": "How do Python lists and tuples differ?"}]
        reply = policy.chat(messages, {"note": "中文记录不决定回答语言"}, max_new_tokens=64)
        generated_messages, budget = policy._generate.call_args.args
        self.assertEqual(generated_messages[1:], messages)
        self.assertEqual(budget, 64)
        self.assertEqual(generated_messages[0]["role"], "system")
        self.assertIn("not response-language instructions", generated_messages[0]["content"])
        self.assertTrue(reply.startswith("A list"))

    def test_language_change_does_not_relax_execution_contract(self):
        self.assertIn("terminal still asks for confirmation", CHAT_PROMPT)
        self.assertIn("does not execute arbitrary code or shell commands", CHAT_PROMPT)
        self.assertIn("Do not add tags to ordinary Q&A", CHAT_PROMPT)
        self.assertIn("protocol keys and command names unchanged", CHAT_PROMPT)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.policy = ChatPolicy()
        self.session = CalibrationSession.create(self.home, Settings(), policy=self.policy)
        self.service = AgentService(self.home, self.session)
        self.addCleanup(lambda: self.service.close())

    def test_chat_does_not_change_calibration(self):
        before = self.service.status()
        answer = self.service.ask("IQ 分割是什么？")
        self.assertEqual(answer["source"], "model")
        self.assertEqual(self.service.status(), before)
        self.assertEqual(len(self.policy.calls), 1)

    def test_multiple_turns_and_context_restore(self):
        self.service.ask("问题一")
        self.service.ask("刚才我问了什么？")
        self.assertEqual(len(self.policy.calls[-1][0]), 3)
        session_id = self.session.metadata["session_id"]
        self.service.close()
        self.service = AgentService(self.home, CalibrationSession.resume(self.home, session_id))
        self.assertEqual(self.service.total_turns, 2)
        self.assertEqual(self.service.messages[0]["content"], "问题一")

    def test_clear_preserves_state_and_audit(self):
        self.service.ask("你好")
        self.session.step()
        before = self.service.status()
        self.service.clear_chat()
        self.assertEqual(self.service.status(), before)
        self.assertEqual(self.service.messages, [])
        self.assertIn('"event":"chat"', "".join(p.read_text() for p in (self.session.directory / "journal").glob("*.json")).replace(" ", ""))

    def test_failed_chat_can_retry_without_terminal_failure(self):
        for error in (TimeoutError("slow"), RuntimeError("load failed"), KeyboardInterrupt()):
            with patch.object(self.policy, "chat", side_effect=error):
                with self.assertRaises(type(error)):
                    self.service.ask("问题")
            self.assertEqual(self.service.status()["status"], "active")
            self.assertEqual(self.service.status()["experiment_count"], 0)
        self.assertEqual(self.service.ask("重新问")["source"], "model")

    def test_explicit_intent_never_calls_model_or_executes(self):
        answer = self.service.ask("帮我开始校准这个比特")
        self.assertEqual(answer["proposal"], {"command": "run"})
        self.assertEqual(self.policy.calls, [])
        self.assertEqual(self.service.status()["experiment_count"], 0)

    def test_capability_required_bounded_and_not_replayable(self):
        with self.assertRaises(ValueError):
            self.service.execute("guessed")
        token = self.service.prepare("run", 2)["token"]
        for _ in range(4):
            result = self.service.execute(token)
            if result["done"]:
                break
        self.assertTrue(result["done"])
        self.assertEqual(result["status"]["experiment_count"], 2)
        with self.assertRaises(ValueError):
            self.service.execute(token)

    def test_chat_and_state_change_invalidate_confirmation(self):
        token = self.service.prepare()["token"]
        self.service.ask("你好")
        with self.assertRaises(ValueError):
            self.service.execute(token)
        token = self.service.prepare()["token"]
        self.session.step()
        with self.assertRaises(ValueError):
            self.service.execute(token)

    def test_full_rule_loop(self):
        token = self.service.prepare("run")["token"]
        for _ in range(40):
            result = self.service.execute(token)
            if result["done"]:
                break
        self.assertTrue(result["done"])
        self.assertEqual(result["status"]["status"], "accepted")

    def test_ui_proposal_needs_confirmation(self):
        self.policy.reply = '可以开始。<qm_action>{"command":"run","limit":1}</qm_action>'
        answers = iter(["n", "y"])
        terminal = Terminal(self.home, service=self.service, input_fn=lambda _: next(answers), output_fn=lambda _: None)
        terminal.dispatch("Please calibrate")
        self.assertEqual(self.service.status()["experiment_count"], 0)
        terminal.dispatch("Please calibrate")
        self.assertEqual(self.service.status()["experiment_count"], 1)

    def test_apostrophes_are_not_shell_parsed(self):
        terminal = Terminal(self.home, service=self.service, output_fn=lambda _: None)
        terminal.dispatch("What's IQ separation?")
        self.assertEqual(self.policy.calls[0][0][-1]["content"], "What's IQ separation?")
        self.assertEqual(self.service.status()["experiment_count"], 0)

    def test_secrets_rejected_before_log_or_model(self):
        before = self.session.journal.sequence
        for text in ("password=example", "密码是example", "api_key: example", "验证码：123456"):
            with self.assertRaises(ValueError):
                self.service.ask(text)
        self.assertEqual(self.policy.calls, [])
        self.assertEqual(self.session.journal.sequence, before)

    def test_context_is_bounded(self):
        for i in range(14):
            self.service.ask(f"问题 {i}")
        self.assertEqual(len(self.policy.calls[-1][0]), 25)
        self.assertEqual(self.service.total_turns, 14)
        self.assertEqual(len(self.service.messages), 24)

    def test_status_comes_from_runtime(self):
        self.session.step()
        answer = self.service.ask("现在校准到哪一步了？")
        self.assertEqual(answer["source"], "controller")
        self.assertIn("1 simulated experiments", answer["message"])
        self.assertEqual(self.policy.calls, [])

    def test_rule_mode_clearly_identified(self):
        self.session.policy = RulePolicy()
        answer = self.service.ask("你好")
        self.assertEqual(answer["source"], "rule_notice")
        self.assertIn("uses no language model", answer["message"])


class ReplyTests(unittest.TestCase):
    def test_valid_proposal(self):
        message, proposal, warning = parse_reply('建议查看。<qm_action>{"command":"status"}</qm_action>')
        self.assertEqual(message, "建议查看。")
        self.assertEqual(proposal, {"command": "status"})
        self.assertIsNone(warning)

    def test_invalid_tags_never_propose(self):
        for text in (
            '<qm_action>{"command":"shell"}</qm_action>',
            '<qm_action>{"command":"run","limit":true}</qm_action>',
            '<qm_action>{"command":"run","command":"step"}</qm_action>',
            '<qm_action>{"command":"run","limit":0}</qm_action>',
            '<qm_action>{"command":"report","target":"/tmp/test"}</qm_action>',
            '<qm_action>{"command":"step"}</qm_action> extra',
            '<qm_action>{}</qm_action><qm_action>{}</qm_action>',
            '<qm_action>{',
        ):
            with self.subTest(text=text):
                _, proposal, warning = parse_reply(text)
                self.assertIsNone(proposal)
                self.assertTrue(warning)

    def test_invalid_user_messages(self):
        for text in ("", " ", None, 42, "x" * 8001):
            with self.assertRaises(ValueError):
                validate_message(text)
