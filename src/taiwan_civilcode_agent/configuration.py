import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_postgres import PGVector


class GPT4oMini(AzureChatOpenAI):
    def __init__(self, temperature=0.5, **kwargs):
        super().__init__(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4o-mini",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=temperature,
            **kwargs
        )


class GPT4o(AzureChatOpenAI):
    def __init__(self, temperature=0.5, **kwargs):
        super().__init__(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4o",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=temperature,
            **kwargs
        )

class Gemini20Flash(ChatGoogleGenerativeAI):
    def __init__(self, temperature=1.0, max_tokens=None, timeout=None, max_retries=2, **kwargs):
        super().__init__(
            model="gemini-2.0-flash",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )

class AzureEmbeddings(AzureOpenAIEmbeddings):
    def __init__(self, **kwargs):
         super().__init__(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment = "text-embedding-3-small",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            **kwargs
        )

class OpenAIEmbeddings(OpenAIEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            api_key=os.environ["OPENAI_API_KEY"],
            model="text-embedding-3-small", **kwargs
        )

class GoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(model="models/text-embedding-004", **kwargs)

class VectorStore(PGVector):
    def __init__(self, **kwargs):
        embeddings_instance = OpenAIEmbeddings()
        collection_name = kwargs.pop("collection_name", "civilcode")
        connection = kwargs.pop("connection", os.environ["PGVECTOR_CONNECTION_STRING"])

        super().__init__(
            embeddings=embeddings_instance,
            collection_name=collection_name,
            connection=connection,
            **kwargs
        )
