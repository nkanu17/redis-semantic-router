from shared.data_types import NewsArticle, NewsArticleWithLabel


def fetch_prompt(
    articles: list[NewsArticleWithLabel | NewsArticle], classes: list[str]
) -> str:
    """Create prompt for batch classification."""
    classes_str = ", ".join(classes)

    # Build articles section
    articles_section = ""
    for i, article in enumerate(articles, 1):
        articles_section += f"""
        **Article {i} (ID: {article.article_id}):**
        {article.text}
        """

    return f"""# Batch News Article Classification Task

        ## Overview
        You are an expert news classifier. Classify multiple news articles into categories based on their content.

        ## Categories
        {classes_str}

        ## Instructions
        1. Read each article carefully
        2. Identify the main topic and focus of each article
        3. Choose the single most appropriate category for each
        4. If an article could fit multiple categories, choose the one that represents the PRIMARY focus, not secondary mentions.

        ## Articles to Classify
        {articles_section}


        ## Output Format
        Output ONLY a JSON object containing classifications for all articles, with the article id as the key and it's corresponding classification:
        {{
            {{"article_id": 1823, "category": "category_name"}},
            {{"article_id": 211, "category": "category_name"}},
            ...
        }}

        Use lowercase for category names. Classify all {len(articles)} articles in order.
    """
