from typing import List, TypedDict

import weaviate
import weaviate.classes as wvc

client = weaviate.connect_to_local(port=8080, grpc_port=50051)
COLLECTION_NAME = "science_qa"
EMBEDDING_DIM = 768


class SearchProperties(TypedDict):
    """
    The SearchProperties class is a TypedDict that defines the properties for searching, including a
    title and text.
    """

    title: str
    text: str


if client.collections.exists(COLLECTION_NAME):
    print(f"Collection already exists {COLLECTION_NAME}")
    QA_COLLECTION = client.collections.get(COLLECTION_NAME)
else:
    COLLECTION_SCHEMA = [
        wvc.Property(name="title", data_type=wvc.DataType.TEXT),
        wvc.Property(name="text", data_type=wvc.DataType.TEXT),
    ]

    QA_COLLECTION = client.collections.create(
        name=COLLECTION_NAME, properties=COLLECTION_SCHEMA
    )


def insert_datas(data_points: List) -> str:
    """
    The function `insert_datas` inserts data points into a collection and returns the number of items
    inserted.

    :param data_points: The `data_points` parameter is a list of dictionaries. Each dictionary
    represents a data point and contains properties and an embedding vector. The properties can be any
    additional information associated with the data point, while the embedding vector is a numerical
    representation of the data point
    :type data_points: List
    :return: The function `insert_datas` returns a string that indicates the number of data items that
    were successfully inserted into the collection.
    """
    if not isinstance(data_points, list):
        raise ValueError(f"Can't insert a {data_points.__class__}")

    data = []
    for data_point in data_points:
        vector = data_point.pop("embeddings", None)
        if vector and len(vector) == EMBEDDING_DIM:
            data.append(wvc.DataObject(properties=data_point, vector=vector))
        else:
            raise ValueError(f"Vector not found in data_point")

    inserted_result = QA_COLLECTION.data.insert_many(data)
    if len(inserted_result.all_responses) != len(data_points):
        raise ValueError("Error while inserting data: {inserted_result.errors}")
    else:
        return f"{len(inserted_result.all_responses)} data items inserted"


def search_datas(data_vector: List[float], limit: int = 5):
    """
    The function `search_datas` takes a data vector and searches for similar vectors in a collection,
    returning a specified number of results.

    :param data_vector: A list of floats representing the vector to search for in the QA_COLLECTION
    :type data_vector: List[float]
    :param limit: The `limit` parameter specifies the maximum number of results to return from the
    search. By default, it is set to 5, but you can change it to any positive integer value, defaults to
    5
    :type limit: int (optional)
    :return: The function `search_datas` returns the result of a query on a QA_COLLECTION. The result is
    a list of items that are near the given data vector. The number of items returned is limited by the
    `limit` parameter, which is set to 5 by default.
    """
    if (
        isinstance(data_vector, List)
        and len(data_vector) == EMBEDDING_DIM
        and isinstance(data_vector[0], float)
    ):
        return QA_COLLECTION.query.near_vector(
            near_vector=data_vector,
            limit=limit,
            return_properties=SearchProperties,
            return_metadata=wvc.MetadataQuery(vector=True, score=True),
        )

    raise ValueError("Invalid data vector to search on")
