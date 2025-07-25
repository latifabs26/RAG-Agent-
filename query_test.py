from query_data import query_rag
from langchain_community.llms.ollama import Ollama


EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def test_document_pipeline_stages():
    assert query_and_validate(
        question="How many main stages are mentioned in a typical document analysis pipeline? (Answer with the number only)",
        expected_response="6",
    )

def test_ocr_definition():
    assert query_and_validate(
        question="What does OCR stand for? (Answer with the full expansion only)",
        expected_response="Optical Character Recognition",
    )



def test_commercial_solutions():
    assert query_and_validate(
        question="Which Microsoft service is mentioned as a commercial document analysis solution? (Answer with the service name only)",
        expected_response="Microsoft Azure Form Recognizer",
    )


def test_exaq_project():
    assert query_and_validate(
        question="What is the name of the project that this survey serves as a foundation for? (Answer with the project name only)",
        expected_response="ExaQ",
    )

def test_document_types():
    assert query_and_validate(
        question="What type of documents includes forms, invoices, and receipts according to section 1.3? (Answer with one word only)",
        expected_response="Structured",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
    
#python -m pytest query_test.py -v -s | Out-File -Encoding utf8 first_test_llamaEmbeddings_mistralModel.txt