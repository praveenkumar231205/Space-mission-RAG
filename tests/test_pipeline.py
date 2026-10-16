import pytest

from space_rag.config import Config
from space_rag.pipeline import NO_ANSWER, RAGPipeline, build_index, build_messages
from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture(scope="module")
def store():
    cfg = Config()
    return build_index(cfg.docs_dir, FakeEmbedder(), cfg.chunk_words, cfg.chunk_overlap)


@pytest.mark.parametrize(
    "question, expected_source",
    [
        ("Which Apollo mission first landed astronauts on the Moon?", "apollo_program.md"),
        ("How wide is the James Webb telescope primary mirror?", "james_webb_space_telescope.md"),
        ("Which spacecraft visited Uranus and Neptune?", "voyager_missions.md"),
        ("Where did the Chandrayaan-3 lander touch down?", "isro_and_indian_space_missions.md"),
        ("When did Perseverance land on Mars?", "mars_rovers.md"),
        ("How fast does the space station travel and how long is an orbit?", "international_space_station.md"),
        ("Who is the crew of Artemis II?", "artemis_program.md"),
        ("When did SpaceX first land a Falcon 9 booster?", "reusable_rockets.md"),
    ],
)
def test_retrieval_finds_the_right_document(store, question, expected_source):
    pipeline = RAGPipeline(FakeEmbedder(), store, llm=None, top_k=3)
    sources = [h.chunk.source for h in pipeline.retrieve(question)]
    assert expected_source in sources


def test_answer_uses_llm_with_grounded_prompt(store):
    llm = FakeLLM("Apollo 11 [1]")
    pipeline = RAGPipeline(FakeEmbedder(), store, llm, top_k=3)
    ans = pipeline.ask("Which Apollo mission first landed astronauts on the Moon?")

    assert ans.used_llm and ans.text == "Apollo 11 [1]"
    system, user = llm.last_messages
    assert system["role"] == "system" and "ONLY" in system["content"]
    assert "[1]" in user["content"] and "Question: Which Apollo mission" in user["content"]


def test_off_topic_question_is_refused_without_calling_llm(store):
    llm = FakeLLM()
    pipeline = RAGPipeline(FakeEmbedder(), store, llm, top_k=3, min_score=0.2)
    ans = pipeline.ask("What is the best recipe for banana bread?")

    assert ans.text == NO_ANSWER and not ans.used_llm
    assert llm.last_messages is None


def test_retrieval_only_mode(store):
    pipeline = RAGPipeline(FakeEmbedder(), store, llm=None, top_k=2)
    ans = pipeline.ask("Which Apollo mission first landed astronauts on the Moon?")
    assert ans.text == "" and not ans.used_llm and len(ans.hits) == 2


def test_build_messages_numbers_the_passages(store):
    hits = RAGPipeline(FakeEmbedder(), store, top_k=2).retrieve("Voyager Golden Record")
    user = build_messages("q?", hits)[1]["content"]
    assert user.index("[1]") < user.index("[2]") < user.index("Question: q?")
