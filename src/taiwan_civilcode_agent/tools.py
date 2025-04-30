"""This module provides tools for interacting with the Taiwan Civil Code database.
"""

import json
import os
from collections.abc import Callable
from typing import Any

import psycopg
from langchain_core.tools import tool
from pydantic import BaseModel

from taiwan_civilcode_agent.configuration import VectorStore


@tool(response_format="content_and_artifact")
def search_civilcode_by_embedding(query: str):
    """Retrieve information related to a query."""
    vector_store = VectorStore()
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

class SearchCivilcodeByArtcileForm(BaseModel):
    """Index for civilcode: 1. 編(Part), 2. 章(Chapter), 3. 節(Section), 4. 款(Subsection), 5. 目(Item), 6. 條(Article)."""
    article_number: str

@tool(response_format="content_and_artifact")
def search_civilcode_by_articleNumber(civilcode_search_form: SearchCivilcodeByArtcileForm):
    """Retrieve information related to a query."""
    article_number = civilcode_search_form.article_number

    if not article_number:
        return {"error": "article_number is required"}, None

    conn = psycopg.connect(
        dbname="civilcode",
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    query = "SELECT * FROM law_articles WHERE article_number = %s"
    cursor.execute(query, (article_number,))
    result = cursor.fetchone()

    conn.close()

    if result:
        columns = [
            "article_number", "article_title", "article_content", "part_number", "part_title",
            "chapter_number", "chapter_title", "section_number", "section_title",
            "subsection_number", "subsection_title", "item_number", "item_title", "source_url"
        ]
        result_dict = dict(zip(columns, result, strict=False))
        return result_dict["article_content"], result_dict.get("article_number")
    else:
        return {"error": "Article not found"}, None

@tool(response_format="content")
def show_all_civilcode_index():
    """你還可以根據領域來查詢相關法條，這是第一步：先使用show_all_civilcode_index工具，此工具將會給你所有可用查詢的篇章節。."""
    with open("../data-pre-process/civilcode-index-110-01-20.json", encoding="utf-8") as file:
        index_data = json.load(file)
    titles = [item["title"] for item in index_data]
    result = "\n".join(titles)
    return result

@tool(response_format="content")
def search_civilcode_by_index(index_title: str):
    """你還可以根據領域來查詢相關法條，這是第二步：對其輸入你想查詢的篇章節，例如：總則編 - 人章 - 法人節，就會回傳整個指定篇章節的法條。."""
    with open("../data-pre-process/civilcode-index-110-01-20.json", encoding="utf-8") as file:
        index_data = json.load(file)

        part_number, chapter_number, section_number = None, None, None

        for item in index_data:
            if item["title"] == index_title:
                part_number = item["part_number"]
                chapter_number = item["chapter_number"]
                section_number = item["section_number"]
                break

        if part_number is None:
            return {"error": "Index not found"}

    conn = psycopg.connect(
        dbname="civilcode",
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    # Construct SQL query based on the available part_number, chapter_number, section_number
    if part_number is not None and chapter_number is not None and section_number is not None:
        query = """
            SELECT * FROM law_articles
            WHERE part_number = %s AND chapter_number = %s AND section_number = %s
        """
        cursor.execute(query, (part_number, chapter_number, section_number))
    elif part_number is not None and chapter_number is not None:
        query = """
            SELECT * FROM law_articles
            WHERE part_number = %s AND chapter_number = %s
        """
        cursor.execute(query, (part_number, chapter_number))
    elif part_number is not None:
        query = """
            SELECT * FROM law_articles
            WHERE part_number = %s
        """
        cursor.execute(query, (part_number,))

    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results

# TOOLS: List[Callable[..., Any]] = [search_civilcode_by_embedding, search_civilcode_by_articleNumber, show_all_civilcode_index, search_civilcode_by_index]
TOOLS: list[Callable[..., Any]] = [search_civilcode_by_embedding, search_civilcode_by_articleNumber]
