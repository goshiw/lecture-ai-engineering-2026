"""ツールのテスト（LLMもAPIキーも使わない）。"""

import json

from app import tools


def test_glossary_ids_are_unique_and_related_links_exist():
    terms = tools.load_glossary()
    ids = [t["id"] for t in terms]
    assert len(ids) == len(set(ids))
    for t in terms:
        for r in t["related"]:
            assert r in ids, f"{t['id']} の related に存在しない id: {r}"


def test_search_finds_term_by_name_and_alias():
    for query in ["LoRA", "lora", "Low-Rank Adaptation"]:
        hits = json.loads(tools.search_glossary(query))
        assert hits[0]["term"] == "LoRA", query


def test_search_finds_course_specific_term():
    hits = json.loads(tools.search_glossary("lecNN-start"))
    assert hits[0]["term"] == "lecNN-start タグ"


def test_search_limits_number_of_results():
    hits = json.loads(tools.search_glossary("エージェント"))
    assert 1 <= len(hits) <= tools.MAX_RESULTS


def test_search_reports_not_found():
    assert "見つかりませんでした" in tools.search_glossary("存在しない用語xyz")


def test_lecture_summary():
    assert "エージェント" in tools.get_lecture_summary(3)
    assert "まだありません" in tools.get_lecture_summary(99)


def test_make_quiz_selects_a_term_from_the_requested_lecture(monkeypatch):
    terms = [
        {"lecture": 1, "term": "別の回の用語", "definition": "別の回の定義"},
        {"lecture": 2, "term": "対象回の用語", "definition": "対象回の定義"},
    ]
    monkeypatch.setattr(tools, "load_glossary", lambda: terms)

    def choose_term(candidates):
        assert candidates == [terms[1]]
        return candidates[0]

    monkeypatch.setattr("random.choice", choose_term)

    result = json.loads(tools.make_quiz(2))

    assert result == {
        "question": "第2回の確認問題: 「対象回の用語」とは何ですか？",
        "answer": "対象回の定義",
    }


def test_make_quiz_reports_when_lecture_has_no_terms(monkeypatch):
    monkeypatch.setattr(
        tools,
        "load_glossary",
        lambda: [{"lecture": 1, "term": "第1回の用語", "definition": "定義"}],
    )

    assert tools.make_quiz(2) == "第2回の用語はまだありません。"


def test_save_learning_log_writes_content_to_lecture_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "DATA_DIR", tmp_path)
    content = "学んだこと: 用語と定義"
    expected_path = tmp_path / "learning_log" / "03.txt"

    result = tools.save_learning_log(3, content)

    assert expected_path.read_text(encoding="utf-8") == content
    assert result == f"第3回の学習記録を保存しました: {expected_path}"


def test_default_question_of_lecture1_is_answerable_from_glossary():
    """第1回の既定の質問（app.ask）は、第3回のエージェントが用語集で正しく答えられること。"""
    from app.ask import DEFAULT_QUESTION

    assert "3レベル" in DEFAULT_QUESTION
    hits = json.loads(tools.search_glossary("3レベル"))
    assert hits[0]["term"] == "宿題の3レベル"
