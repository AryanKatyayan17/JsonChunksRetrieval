import json
import os


def build_text(chunk):
    content = chunk.get("content", {})
    classification = chunk.get("classification", {})
    geography = chunk.get("geography", {})

    raw_text = content.get("raw_text", "")
    summary = content.get("chunk_summary", "")

    category = classification.get("category", "")
    sub_category = classification.get("sub_category", "")

    tags = classification.get("tags", [])
    tag_list = [tag.get("tag", "") for tag in tags]

    key_terms = content.get("key_terms", [])

    countries = geography.get("countries", [])

    text = f"""
    Category: {category}
    Sub-category: {sub_category}
    Tags: {', '.join(tag_list)}

    Key Terms: {', '.join(key_terms)}
    Countries: {', '.join(countries)}

    Summary:
    {summary}

    Content:
    {raw_text}
    """

    return text.strip()


def load_all_chunks(folder_path):
    texts = []
    metadatas = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    data = json.load(f)

            for chunk in data:
                text = build_text(chunk)

                metadata = {
                    "doc_id": chunk.get("doc_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "category": chunk.get("classification", {}).get("category"),
                    "source": chunk.get("metadata", {}).get("source"),
                }

                texts.append(text)
                metadatas.append(metadata)

    return texts, metadatas


if __name__ == "__main__":
    texts, metas = load_all_chunks("json data")
    print("Loaded:", len(texts))
    print(texts[0][:500])