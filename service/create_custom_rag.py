"""
1. Create and launch a instance of weaviate db. (https://weaviate.io/developers/weaviate/installation/docker-compose)
2. Download or select the dataset for QA. Here I have selected "KonstantyM/science_qa" from hugging face datasets.
"""

from typing import List

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from weaviate_db import insert_datas

device = "cuda" if torch.cuda.is_available() else "cpu"

CTX_MODEL = "facebook/dpr-ctx_encoder-multiset-base"

ctx_encoder = DPRContextEncoder.from_pretrained(CTX_MODEL).to(device=device)
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(CTX_MODEL)


def get_sentence_embedding(documents: dict) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = get_input_ids_mask(documents)["input_ids"]
    embeddings = ctx_encoder(
        input_ids.to(device=device), return_dict=True
    ).pooler_output
    return {
        **documents,
        "embeddings": embeddings.detach().cpu().numpy().squeeze().tolist(),
    }


def get_input_ids_mask(documents: dict):
    """Compute the input ids for title and text."""
    return ctx_tokenizer(
        documents["title"],
        documents["text"],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    contexts, answers = [], []
    for idx in range(len(documents["answer"])):
        answer = documents["answer"][idx]
        if len(documents["context"]) <= idx:
            context = ""
        else:
            context = documents["context"][idx]
            context = context if context is not None else ""
            context = context.replace("/", " ").strip()
        if answer is not None and context:
            for passage in split_text(answer):
                contexts.append(context)
                answers.append(passage)
    return {"title": contexts, "text": answers}


def insert_main():
    # The dataset needed for RAG must have three columns:
    # - title (string): title of the document
    # - text (string): text of a passage of the document
    # - embeddings (array of dimension d): DPR representation of the passage
    dataset = load_dataset("KonstantyM/science_qa")

    # Then split the documents into passages of 100 words
    insert_data = []
    for d in tqdm(dataset.iter(batch_size=1000), total=dataset.num_rows // 1000):
        data = split_documents(d)
        for title, text in zip(data["title"], data["text"]):
            insert_data.append(get_sentence_embedding({"title": title, "text": text}))

        if insert_data:
            _ = insert_datas(insert_data)
            insert_data = []

    if insert_data:
        _ = insert_datas(insert_data)
        insert_data = []


if __name__ == "__main__":
    insert_main()
