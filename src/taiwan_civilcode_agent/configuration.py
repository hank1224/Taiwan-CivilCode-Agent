"""Configuration module for Taiwan Civil Code Agent.

This module provides classes for integrating with various AI models and services.
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_postgres import PGVector


class GPT4oMini(AzureChatOpenAI):
    """GPT-4o Mini model for Azure OpenAI."""

    def __init__(self, temperature=0.5, **kwargs):
        """Initialize the GPT-4o-mini model for Azure OpenAI.

        Args:
            temperature (float): Sampling temperature for the model. Default is 0.5.
            **kwargs: Additional keyword arguments for the AzureChatOpenAI class.
        """
        super().__init__(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4o-mini",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=temperature,
            **kwargs
        )


class GPT4o(AzureChatOpenAI):
    """GPT-4o model for Azure OpenAI."""

    def __init__(self, temperature=0.5, **kwargs):
        """Initialize the GPT-4o model for Azure OpenAI.

        Args:
            temperature (float): Sampling temperature for the model. Default is 0.5.
            **kwargs: Additional keyword arguments for the AzureChatOpenAI class.
        """
        super().__init__(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4o",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=temperature,
            **kwargs
        )

class Gemini20Flash(ChatGoogleGenerativeAI):
    """Gemini 2.0 Flash model for Google Generative AI."""

    def __init__(self, temperature=1.0, max_tokens=None, timeout=None, max_retries=2, **kwargs):
        """Initialize the Gemini 2.0 Flash model for Google Generative AI.

        Args:
            temperature (float): Sampling temperature for the model. Default is 1.0.
            max_tokens (int, optional): Maximum number of tokens for the response. Default is None.
            timeout (float, optional): Timeout for the API call in seconds. Default is None.
            max_retries (int): Maximum number of retries for the API call. Default is 2.
            **kwargs: Additional keyword arguments for the ChatGoogleGenerativeAI class.
        """
        super().__init__(
            model="gemini-2.0-flash",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )

class GoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    """Google Generative AI embeddings model.

    This class is used to generate embeddings for text using the Google Generative AI model.
    """
    def __init__(self, **kwargs):
        """Initialize the Google Embeddings model.

        Args:
            **kwargs: Additional keyword arguments for the GoogleGenerativeAIEmbeddings class.
        """
        super().__init__(model="models/text-embedding-004", **kwargs)

class VectorStore(PGVector):
    """VectorStore class for managing embeddings in a PostgreSQL vector store.

    This class integrates with the PGVector library to store and retrieve embeddings
    using Google Generative AI embeddings.
    """

    def __init__(self, **kwargs):
        """Initialize the VectorStore with optional parameters.

        Args:
            **kwargs: Additional keyword arguments for configuring the VectorStore.
        """
        super().__init__(**kwargs)
        self.embeddings = GoogleEmbeddings(**kwargs)
        self.collection_name = kwargs.get("collection_name", "civilcode")
        self.connection = kwargs.get("connection", os.environ["PGVECTOR_CONNECTION_STRING"])
