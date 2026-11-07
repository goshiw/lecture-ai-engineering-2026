"""app.ask の会話モードのテスト（LLMもAPIキーも使わない）。"""

from google.genai import chats, types

from app import ask


class FakeModels:
    """generate_content に渡された contents を記録し、決まった回答を返す（APIの代わり）。"""

    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(list(contents))
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text=f"回答{len(self.calls)}")]),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        )


def make_chat(models):
    return chats.Chat(modules=models, model="fake", history=[])


def test_chat_loop_sends_previous_turns_as_history(capsys):
    models = FakeModels()
    inputs = iter(["それを短く", ""])
    ask.chat_loop(make_chat(models), "LoRAとは？", read=lambda _: next(inputs))

    assert len(models.calls) == 2
    # 2回目の呼び出しには、1回目の質問と回答が含まれている
    second = [(c.role, c.parts[0].text) for c in models.calls[1]]
    assert second == [("user", "LoRAとは？"), ("model", "回答1"), ("user", "それを短く")]
    out = capsys.readouterr().out
    assert "回答1" in out and "回答2" in out


def test_chat_loop_stops_on_exit_word_and_eof():
    models = FakeModels()
    ask.chat_loop(make_chat(models), "Q1", read=lambda _: "exit")
    assert len(models.calls) == 1

    def eof(_):
        raise EOFError

    ask.chat_loop(make_chat(models), "Q1", read=eof)
    assert len(models.calls) == 2


def test_main_chat_keeps_client_open(monkeypatch):
    """--chat の会話中に genai.Client が閉じられていないこと（閉じると送信時に RuntimeError になる）。"""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setattr("sys.argv", ["ask", "--chat", "Q1"])
    seen = []

    def fake_loop(chat, question, read=input):
        seen.append(chat._modules._api_client._httpx_client.is_closed)

    monkeypatch.setattr(ask, "chat_loop", fake_loop)
    ask.main()
    assert seen == [False]
