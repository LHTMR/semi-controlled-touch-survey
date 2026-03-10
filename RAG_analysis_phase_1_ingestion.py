import pandas as pd
import json
import logging

# Import the existing extractor script (must be in the same directory)
import all_words_extractor as awe

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# The specific free-text columns we want to extract from touch_data.txt
TEXT_COLUMNS = [
    "Social_self",
    "Social_body",
    "Social_place",
    "Social_context",
    "Intention&Purpose",
    "Appropriateness",
    "Sensory",
    "Emotional_self",
    "Emotional_touch",
]


def load_transformation_dict(json_path: str) -> dict:
    """
    Loads the word_grouping_dict.json and creates a reverse mapping
    (variant -> main_word) for fast O(1) lookups during preprocessing.
    """
    logger.info(f"Loading transformation dictionary from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reverse_map = {}
    if "transformation_dictionary" in data:
        for main_word, variants in data["transformation_dictionary"].items():
            for variant in variants:
                reverse_map[variant] = main_word

    logger.info(
        f"Loaded {len(reverse_map)} word variants mapping to {len(data.get('transformation_dictionary', {}))} main words."
    )
    return reverse_map


def combine_text_columns(row: pd.Series, columns: list) -> str:
    """
    Safely concatenates all qualitative text responses for a single participant into one string.
    Filters out 'NA' or empty responses.
    """
    texts = []
    for col in columns:
        val = row[col]
        # Check if the value is not null and not a placeholder 'NA' string
        if (
            pd.notna(val)
            and str(val).strip().upper() != "NA"
            and str(val).strip() != ""
        ):
            texts.append(str(val).strip())

    # Join sentences with a period so spaCy can understand boundaries
    return " . ".join(texts)


def run_phase1_pipeline(data_path: str, dict_path: str):
    """
    Executes the Phase 1 pipeline, returning structured Macro and Micro level datasets.
    """
    logger.info(f"Loading touch data from {data_path}...")
    # Read the dataset (touch_data.txt is tab-separated based on the raw file structure)
    df = pd.read_csv(data_path, sep="\t", encoding="utf-8")

    reverse_map = load_transformation_dict(dict_path)

    logger.info("Configuring spaCy and all_words_extractor pipeline...")
    # Initialize awe config and load the spaCy model explicitly so preprocess_text can use it
    config = awe.DEFAULT_CONFIG.copy()
    config["use_spacy_negation"] = True
    config["_spacy_model"] = awe.load_spacy_model(config)

    # Data structures to hold our two scales of analysis
    micro_level = []  # List of dicts (1 dict = 1 Participant's response to 1 Video)
    macro_level = {}  # Dict of dicts (Key = Video ID, Value = Pooled responses)

    logger.info("Processing participant responses... This may take a moment.")
    for idx, row in df.iterrows():
        p_id = str(row["ResponseID"])
        v_id = str(row["Touch No."])
        v_desc = str(row.get("Touch_desc", f"Video {v_id}"))

        # 1. Extract raw text
        raw_text = combine_text_columns(row, TEXT_COLUMNS)
        if not raw_text:
            continue  # Skip if participant provided no text data for this touch

        # 2. Advanced NLP Preprocessing (spaCy Negation Binding)
        # E.g. turns "it was not painful" into ['it', 'was', 'not_painful']
        tokens = awe.preprocess_text(raw_text, config)

        # 3. Apply the manual Word Grouping Dictionary
        # Standardizes typos/morphology (e.g. 'comfy', 'comforting' -> 'comfortable')
        transformed_tokens = [reverse_map.get(token, token) for token in tokens]

        # --- Populate MICRO Level ---
        micro_entry = {
            "participant_id": p_id,
            "video_id": v_id,
            "video_desc": v_desc,
            "raw_text": raw_text,
            "processed_tokens": transformed_tokens,
            "processed_text": " ".join(
                transformed_tokens
            ),  # Space-separated for embedding model
        }
        micro_level.append(micro_entry)

        # --- Populate MACRO Level ---
        if v_id not in macro_level:
            macro_level[v_id] = {
                "video_id": v_id,
                "video_desc": v_desc,
                "participant_ids": [],
                "raw_text_list": [],
                "processed_tokens": [],
            }

        macro_level[v_id]["participant_ids"].append(p_id)
        macro_level[v_id]["raw_text_list"].append(raw_text)
        macro_level[v_id]["processed_tokens"].extend(transformed_tokens)

    logger.info("Finalizing Macro-level pooling...")
    # Finalize Macro level by joining the pooled text
    for v_id in macro_level:
        macro_level[v_id]["raw_text"] = " . ".join(macro_level[v_id]["raw_text_list"])
        macro_level[v_id]["processed_text"] = " ".join(
            macro_level[v_id]["processed_tokens"]
        )
        del macro_level[v_id]["raw_text_list"]  # Cleanup memory

    logger.info(
        f"Phase 1 Complete: Structured {len(micro_level)} Micro-level documents and {len(macro_level)} Macro-level documents."
    )

    return micro_level, macro_level


if __name__ == "__main__":
    # Ensure you are running this in the same directory as the source files
    micro_data, macro_data = run_phase1_pipeline(
        data_path="Processed Data/touch_data.txt",
        dict_path="Analysis/All_words_by_frequency/word_grouping_dict.json",
    )

    # Save the structured outputs to JSON for Phase 2
    with open(
        "Analysis/RAG_analysis/phase1_micro_level.json", "w", encoding="utf-8"
    ) as f:
        json.dump(micro_data, f, indent=2)

    with open(
        "Analysis/RAG_analysis/phase1_macro_level.json", "w", encoding="utf-8"
    ) as f:
        json.dump(macro_data, f, indent=2)

    print(
        "Saved 'phase1_micro_level.json' and 'phase1_macro_level.json'. Ready for Phase 2!"
    )
