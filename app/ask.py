"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat "LoRAとは何ですか？"   # 会話履歴を保ったまま続けて質問する

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）

--chat では、それまでの質問と回答（会話履歴）を毎回モデルに送り直す。
「それをもっと短く」のように、前の回答を指す質問ができる。
"""

import argparse
from collections.abc import Callable

from google import genai
from google.genai import chats, errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# --chat で、この入力（または空行）を受けると終了する
EXIT_WORDS = {"exit", "quit", "q"}


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    # TODO(第1回 演習): response.usage_metadata から
    #   入力トークン数と出力トークン数を表示してみよう（Copilotに頼んでよい）

    return response.text


def start_chat() -> chats.Chat:
    """会話履歴を保持するチャットを作る。send_message するたびに、履歴が Chat の中に溜まる。"""
    client = genai.Client(api_key=get_api_key())
    return client.chats.create(model=MODEL)


def chat_loop(chat: chats.Chat, question: str, read: Callable[[str], str] = input) -> None:
    """最初の質問に答えたあと、空行か exit が入力されるまで続けて質問を受け付ける。"""
    while True:
        try:
            print(chat.send_message(question).text)
        except errors.APIError as e:
            # 失敗したやり取りは履歴に残らないので、同じ質問をもう一度入力すればよい
            print(api_error_message(e.code, e.message))
        print(f"\n[履歴] {len(chat.get_history())} 件\n")

        try:
            question = read("[質問] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question or question.lower() in EXIT_WORDS:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保ったまま続けて質問する（空行か exit で終了）")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    if args.chat:
        chat_loop(start_chat(), args.question)
        return
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
