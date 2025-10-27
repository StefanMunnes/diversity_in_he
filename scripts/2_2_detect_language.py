import polars as pl
import langdetect
from langdetect.lang_detect_exception import LangDetectException


def detect_language(text: str) -> str:
    """
    Detects the language of a given text.
    Returns 'english', 'german', 'Other', or 'Error'.
    """
    # Handle empty or invalid input
    if not text or not isinstance(text, str) or text.isspace():
        return "Other"

    try:
        lang = langdetect.detect(text)
        if lang == "en":
            return "eng"
        elif lang == "de":
            return "ger"
        else:
            return "Other"
    except LangDetectException:
        # This exception is thrown for texts that are too short or ambiguous
        return "Error"


data_filtered = pl.read_csv("data/data_filtered.csv").with_columns(
    language=pl.col("text").map_elements(detect_language, return_dtype=pl.Utf8)
)


data_filtered.filter(pl.col("language").is_in(["eng", "ger"])).write_csv(
    "data/data_filtered_language.csv"
)
