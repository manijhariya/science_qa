from typing import Dict

import torch
from more_itertools import sort_together
from transformers import (
    AutoTokenizer,
    BartForQuestionAnswering,
    BartTokenizer,
    RagTokenForGeneration,
)

from service.weaviate_db import search_datas

device = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 768

MODEL_STR = "facebook/rag-token-nq"
MODEL_STR_BART = "valhalla/bart-large-finetuned-squadv1"


RAG_tokenizer = AutoTokenizer.from_pretrained(MODEL_STR)
RAG_model = RagTokenForGeneration.from_pretrained(MODEL_STR)

tokenizer = BartTokenizer.from_pretrained(MODEL_STR_BART)
model = BartForQuestionAnswering.from_pretrained(MODEL_STR_BART)


PROMPT = "Consider the given Context. Please answer the question.\n"


def custom_retrieve(question_hidden_states: torch.Tensor) -> Dict:
    """
    The function `custom_retrieve` retrieves search results and organizes them into a dictionary
    containing the retrieved document embeddings, context titles, and context texts.

    :param question_hidden_states: The parameter `question_hidden_states` is a tensor representing the
    hidden states of the question. It is expected to be a torch.Tensor object
    :type question_hidden_states: torch.Tensor
    :return: The function `custom_retrieve` returns a dictionary `docs_dict` containing the following
    keys:
    """
    search_results = search_datas(question_hidden_states)

    docs_dict = {
        "retrieved_doc_embeds": [],
        "context_title": [],
        "context_text": [],
    }

    for idx, search_result in enumerate(search_results.objects):
        properties = search_result.properties
        docs_dict["context_title"].append(properties["title"])
        docs_dict["context_text"].append(properties["text"])
        docs_dict["retrieved_doc_embeds"].append(
            torch.Tensor(search_result.metadata.vector)
        )

    docs_dict["retrieved_doc_embeds"] = torch.stack(
        docs_dict["retrieved_doc_embeds"]
    ).view(1, len(search_results.objects), EMBEDDING_DIM)

    return docs_dict


def inplace_sort_values(docs_dict: Dict) -> None:
    """
    The function `inplace_sort_values` sorts the values in the `docs_dict` dictionary based on the
    `doc_scores` key in descending order.

    :param docs_dict: The `docs_dict` parameter is a dictionary that contains the following keys:
    :type docs_dict: Dict
    """
    (
        docs_dict["doc_scores"],
        docs_dict["context_text"],
        docs_dict["context_title"],
    ) = sort_together(
        [
            docs_dict["doc_scores"].tolist()[0],
            docs_dict["context_text"],
            docs_dict["context_title"],
        ],
        reverse=True,
    )


def generator_model(question: str, docs_dict: Dict) -> str:
    """
    The `generator_model` function takes a question and a dictionary of documents as input and returns a
    string that represents the answer to the question based on the information in the documents.

    :param question: The `question` parameter is a string that represents the question you want to ask
    :type question: str
    :param docs_dict: The `docs_dict` parameter is a dictionary that contains the context information
    for the documents. It has the following structure:
    :type docs_dict: Dict
    :return: The function `generator_model` takes a question and a dictionary of documents as input and
    returns a string. The string represents the answer to the question based on the information in the
    documents.
    """
    ## Do Not use retrieved_doc_embeds after sorting

    text = PROMPT
    inplace_sort_values(docs_dict)
    for idx in range(len(docs_dict["context_title"])):
        text += (
            f"""{docs_dict["context_title"][idx]}\t{docs_dict["context_text"][idx]}\n"""
        )

    encoding = tokenizer(question, text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    start_scores, end_scores = model(
        input_ids, attention_mask=attention_mask, output_attentions=False
    )[:2]

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = " ".join(
        all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores) + 1]
    )
    answer = tokenizer.convert_tokens_to_ids(answer.split())
    return tokenizer.decode(answer, skip_special_tokens=True).strip()


def search_main(question: str) -> str:
    """
    The `search_main` function takes a question as input, encodes it using a RAG model, retrieves
    relevant documents using a custom retrieval function, calculates scores for the retrieved documents,
    and generates an answer using a generator model.

    :param question: The `question` parameter is a string that represents the question you want to
    search for
    :type question: str
    :return: The function `search_main` returns the output of the `generator_model` function, which
    takes the question and the `docs_dict` as input and generates a response.
    """
    input_ids = RAG_tokenizer(question, return_tensors="pt")["input_ids"]
    question_hidden_states = RAG_model.question_encoder(input_ids)[0]
    docs_dict = custom_retrieve(
        question_hidden_states.detach().numpy().squeeze().tolist()
    )

    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1),
        docs_dict["retrieved_doc_embeds"].float().transpose(1, 2),
    ).squeeze(1)

    docs_dict["doc_scores"] = doc_scores

    return generator_model(question, docs_dict)
