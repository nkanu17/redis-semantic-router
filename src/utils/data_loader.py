"""Simple data loader for BBC News classification task."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from shared.data_types import NewsArticleWithLabel
from utils.logger import get_logger


class NewsDataLoader:
    def __init__(
        self,
        data_dir="../data/bbc-news-articles-labeled",
        train_file="train_data.csv",
        validation_file="validation_data.csv",
    ):
        self.data_dir = Path(data_dir)
        self.train_file = train_file
        self.validation_file = validation_file
        self.logger = get_logger(f"{__name__}.NewsDataLoader")

        # Auto-split if files don't exist
        self._ensure_split_files_exist()

    def load_train(self):
        """Load training data and return texts and labels."""
        df = pd.read_csv(self.data_dir / self.train_file)
        return df["Text"].tolist(), df["Category"].tolist()

    def load_test(self):
        """Load validation data and return texts and article IDs."""
        df = pd.read_csv(self.data_dir / self.validation_file)
        return df["Text"].tolist(), df["ArticleId"].tolist()

    def load_train_samples(self) -> list[NewsArticleWithLabel]:
        """Load training data as NewsArticleWithLabel objects."""
        df = pd.read_csv(self.data_dir / self.train_file)
        return [
            NewsArticleWithLabel(
                article_id=row["ArticleId"],
                text=row["Text"].strip(),
                label=row["Category"],
            )
            for _, row in df.iterrows()
        ]

    def load_test_samples(self) -> list[NewsArticleWithLabel]:
        """Load validation data as NewsArticleWithLabel objects (has ground truth)."""
        df = pd.read_csv(self.data_dir / self.validation_file)
        return [
            NewsArticleWithLabel(
                article_id=row["ArticleId"],
                text=row["Text"].strip(),
                label=row["Category"],
            )
            for _, row in df.iterrows()
        ]

    def _ensure_split_files_exist(self) -> None:
        """Check if split files exist, create them if they don't."""
        train_path = self.data_dir / self.train_file
        validation_path = self.data_dir / self.validation_file
        original_path = self.data_dir / "BBC News Train.csv"

        # If both split files exist, we're good
        if train_path.exists() and validation_path.exists():
            return

        # If original file doesn't exist, we can't split
        if not original_path.exists():
            raise FileNotFoundError(
                f"Cannot create split files: {original_path} not found"
            )

        self.logger.info("Split files not found, creating from BBC News Train.csv")
        self._split_original_data(original_path, train_path, validation_path)

    def _split_original_data(
        self, original_path: Path, train_path: Path, validation_path: Path
    ) -> None:
        """Split the original BBC News Train.csv into train and validation sets."""
        # Load original training data
        df = pd.read_csv(original_path)

        # Remove duplicate texts to ensure clean split
        original_count = len(df)
        df = df.drop_duplicates(subset=["Text"], keep="first")
        duplicates_removed = original_count - len(df)

        if duplicates_removed > 0:
            self.logger.info(
                f"Removed {duplicates_removed} duplicate texts from dataset"
            )

        self.logger.info(
            f"Splitting {len(df)} unique articles (75% train, 25% validation)"
        )

        # Split with stratification to maintain class balance
        train_df, val_df = train_test_split(
            df,
            test_size=0.25,  # 25% for validation
            random_state=42,
            stratify=df["Category"],
        )

        # Save split files
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(validation_path, index=False)

        self.logger.info(f"Created {train_path} with {len(train_df)} articles")
        self.logger.info(f"Created {validation_path} with {len(val_df)} articles")


if __name__ == "__main__":
    logger = get_logger("data_loader")

    # Test the loader
    loader = NewsDataLoader("./bbc-news-articles-labeled")

    train_texts, train_labels = loader.load_train()
    test_texts, test_ids = loader.load_test()
    from shared.data_types import NewsCategory

    classes = NewsCategory.get_valid_classes()

    logger.info(f"Train: {len(train_texts)} texts, {len(train_labels)} labels")
    logger.info(f"Test: {len(test_texts)} texts, {len(test_ids)} ids")
    logger.info(f"Classes: {classes}")
    logger.info(f"First article (50 chars): {train_texts[0][:50]}...")
