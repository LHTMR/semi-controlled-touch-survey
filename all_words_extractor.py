#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All Words Extractor - Scientific Text Analysis Tool

This script extracts and analyzes word frequencies from touch survey data files.
It supports multiple file formats and provides advanced text processing for
linguistic analysis in human-computer interaction research.

Scientific Context:
- Designed for analyzing semi-controlled touch survey responses
- Handles negation patterns common in subjective experience descriptions
- Groups word variations to account for spelling differences and typos
- Preserves linguistic constructs important for qualitative analysis
- Two-pass processing approach for comprehensive analysis

Features:
- Supports multiple file formats (CSV, TSV, TXT, XLS, XLSX, etc.) via pandas
- Automatic format detection with fallbacks
- Configurable text columns for structured data
- JSON configuration file support with priority: command line > config file > defaults
- NLTK stopword removal with scientific stopword customization
- Enhanced negation handling with configurable terms and intensifiers
- spaCy-based negation detection for more accurate parsing (default, requires spaCy installation)
- Word variation grouping (typos, different spellings, morphological variations)
- Multiple output formats (CSV, tree visualization)

Dependencies:
- pandas, numpy (data handling)
- nltk (tokenization, stopwords, lemmatization)
- spacy (enhanced negation detection - default)
- difflib (word similarity comparison)

Installation:
  pip install pandas nltk spacy
  python -m spacy download en_core_web_sm

Usage Examples:
  # Basic usage with automatic detection
  python all_words_extractor.py survey_data.csv

  # Specify text columns for structured data
  python all_words_extractor.py survey_data.csv --text-columns "response,comments"

  # Use custom configuration file
  python all_words_extractor.py survey_data.csv --config my_config.json

  # Disable spaCy for heuristic-based processing
  python all_words_extractor.py survey_data.csv --no-spacy

  # Adjust grouping sensitivity for scientific analysis
  python all_words_extractor.py survey_data.csv --similarity-threshold 0.85 --max-edit-distance 1

Processing Approach:
  1. Raw extraction: Simple word splitting for baseline frequencies
  2. Advanced processing: spaCy-based parsing for negation and compound words
  3. Variation grouping: Accounts for spelling variations and morphological forms
  4. Two-pass analysis: Separate raw and processed outputs for methodological transparency

Scientific Rationale:
- Negation handling preserves semantic meaning in subjective responses
- Word grouping accounts for natural language variation in survey data
- Configurable thresholds allow adaptation to different research contexts
- Stopword customization supports domain-specific terminology

Written by: Yohann OPOLKA, University of Borås, Sweden.
Last updated: 2026.2.20 (with spaCy integration)
"""

import os
import sys
import argparse
import json
import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.metrics import edit_distance
import warnings
import signal
import atexit
import difflib
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# Global variables for interruption handling
# ============================================================================
INTERRUPTED = False
PROCESSED_FILES = []
CURRENT_FILE = ""

# ============================================================================
# Signal and interruption handling
# ============================================================================


def signal_handler(signum, frame):
    """Handle interruption signals (Ctrl+C)."""
    global INTERRUPTED
    INTERRUPTED = True
    print("\n\n⚠️  Interruption requested. Saving current progress...")

    # Try to save current progress
    if PROCESSED_FILES:
        try:
            # Create a summary of processed files
            summary = {
                "interrupted_at": datetime.now().isoformat(),
                "processed_files": PROCESSED_FILES,
                "current_file": CURRENT_FILE,
                "total_files_processed": len(PROCESSED_FILES),
            }

            with open(
                "all_words_extractor_interrupted.json", "w", encoding="utf-8"
            ) as f:
                json.dump(summary, f, indent=2)

            print("✓ Partial progress saved to all_words_extractor_interrupted.json")
        except Exception as e:
            print(f"✗ Could not save partial progress: {e}")

    # Don't exit immediately - let the main loop handle the interruption
    # This allows for cleanup in main()


# ============================================================================
# Default Configuration
# ============================================================================

# Default configuration parameters for scientific text analysis.
# These values are optimized for touch survey analysis but can be adjusted
# for different research contexts through command line arguments or config files.
#
# Scientific Rationale:
# - Parameters balance linguistic accuracy with computational efficiency
# - Defaults reflect common practices in computational linguistics research
# - Configurability enables methodological adaptation without code modification
# - Two-pass processing (raw + advanced) supports methodological transparency

DEFAULT_CONFIG = {
    # File format detection
    "default_format": "txt",
    "csv_delimiters": [",", "\t", ";", "|"],
    "csv_encoding": "utf-8",
    # Text processing
    "text_columns": [],  # Empty means auto-detect or use all columns
    "min_word_length": 2,  # Filters out single-character words and artifacts
    "lowercase": True,  # Standardizes text for frequency counting
    "remove_punctuation": True,  # Removes punctuation for cleaner word analysis
    "remove_numbers": True,  # Filters numeric data not relevant for linguistic analysis
    # Stopword handling
    "remove_stopwords": True,
    "stopword_language": "english",
    "custom_stopwords": [],
    # Enhanced negation handling
    "handle_negations": True,  # Preserves semantic meaning in subjective responses
    "negation_terms": [
        "not",
        "no",
        "never",
        "none",
        "nothing",
        "nobody",
        "nowhere",
        "neither",
        "nor",
    ],  # Common English negation markers
    "negation_intensifiers": [
        "very",
        "really",
        "quite",
        "rather",
        "somewhat",
        "slightly",
        "too",
        "so",
        "extremely",
        "incredibly",
        "absolutely",
        "completely",
        "totally",
        "utterly",
        "fairly",
        "pretty",
        "a bit",
        "a little",
        "kind of",
        "sort of",
    ],  # Intensifiers that modify negation strength (Option B: removed from forms)
    "negation_max_distance": 3,  # Maximum words between negation and target (linguistic constraint)
    "negation_skip_stopwords": True,  # Skip stopwords when looking for target word
    # spaCy-based negation handling (default behavior)
    "use_spacy_negation": True,  # Use spaCy for grammatical accuracy (default)
    "spacy_model": "en_core_web_md",  # Medium English model balances accuracy/speed
    "spacy_negation_terms": [],  # Custom negation terms for spaCy (optional)
    # Word variation grouping
    "group_variations": True,  # Accounts for natural language variation
    "max_edit_distance": 2,  # Maximum Levenshtein distance for typo detection
    "use_lemmatization": True,  # Groups morphological variants (e.g., feel/feeling/felt)
    "similarity_threshold": 0.8,  # Sequence similarity threshold for word grouping
    # Output
    "output_prefix": "word_frequencies",
    "save_raw_counts": True,  # Baseline analysis without linguistic processing
    "save_filtered_counts": True,  # Analysis after stopword removal and negation handling
    "save_grouped_counts": True,  # Detailed grouping information for methodological transparency
    "save_tree_visualization": True,  # Visual representation of word variation patterns
    "tree_max_depth": None,  # Limits tree depth for readability while showing key variations
    # NLTK data
    "nltk_data_path": None,  # Optional custom NLTK data path
    # Debugging
    "verbose": False,  # Enable verbose output for debugging
}

# ============================================================================
# NLTK Data Setup
# ============================================================================


def setup_nltk_data(config: Dict[str, Any]) -> None:
    """
    Download required NLTK data if not already available.

    Args:
        config: Configuration dictionary containing NLTK data path and processing options.

    Raises:
        LookupError: If NLTK data cannot be downloaded and required packages are missing.

    Notes:
        - Downloads punkt tokenizer for sentence/word tokenization
        - Downloads stopwords corpus for stopword removal
        - Downloads WordNet for lemmatization (if enabled)
        - Falls back gracefully if downloads fail, disabling affected features
    """
    print("Checking NLTK data availability...")

    # Set NLTK data path if specified
    if config.get("nltk_data_path"):
        nltk.data.path.append(config["nltk_data_path"])

    # Required NLTK packages
    required_packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    if config.get("use_lemmatization", True):
        required_packages.append(("corpora/wordnet", "wordnet"))

    for find_path, package_name in required_packages:
        try:
            nltk.data.find(find_path)
            print(f"✓ {package_name} is available")
        except LookupError:
            print(f"Downloading {package_name}...")
            try:
                nltk.download(package_name)
                print(f"✓ {package_name} downloaded successfully")
            except Exception as e:
                print(f"✗ Error downloading {package_name}: {e}")
                print("Please check your internet connection and try again.")
                if package_name == "stopwords":
                    print("Stopword removal will be disabled.")
                    config["remove_stopwords"] = False
                elif package_name == "wordnet":
                    print("Lemmatization will be disabled.")
                    config["use_lemmatization"] = False


# ============================================================================
# spaCy Setup and Functions
# ============================================================================


def load_spacy_model(config: Dict[str, Any]) -> Optional[Any]:
    """
    Load spaCy model with error handling for enhanced linguistic analysis.

    Args:
        config: Configuration dictionary containing spaCy model settings.

    Returns:
        Loaded spaCy model object or None if spaCy is not available.

    Raises:
        ImportError: If spaCy package is not installed.
        OSError: If specified spaCy model cannot be loaded.

    Notes:
        - Default model is 'en_core_web_sm' (small English model)
        - Falls back to heuristic negation detection if spaCy fails
        - Provides installation instructions if spaCy is missing
    """
    if not config.get("use_spacy_negation", True):  # Default is True
        return None

    print("Loading spaCy model for enhanced negation detection...")

    try:
        import spacy
        from spacy.tokens import Token
    except ImportError:
        print("✗ spaCy is not installed. Falling back to heuristic negation detection.")
        print("  To use spaCy, install it with: pip install spacy")
        print("  Then download the model with: python -m spacy download en_core_web_sm")
        config["use_spacy_negation"] = False
        return None

    model_name = config.get("spacy_model", "en_core_web_sm")

    try:
        nlp = spacy.load(model_name)
        print(f"✓ Loaded spaCy model: {model_name}")
        return nlp
    except Exception as e:
        print(f"✗ Could not load spaCy model '{model_name}': {e}")
        print(f"  Download the model with: python -m spacy download {model_name}")
        print("  Falling back to heuristic negation detection.")
        config["use_spacy_negation"] = False
        return None


def get_full_compound_spacy(token: "Token") -> list["Token"]:
    """
    Expands a token to include its full compound chain (e.g., 'ice cream').

    Args:
        token: spaCy token to expand.

    Returns:
        List of tokens representing the complete compound phrase.

    Notes:
        - Used for handling multi-word expressions in negation detection
        - Preserves linguistic structure for accurate semantic analysis
        - Essential for scientific text analysis of compound terms
    """
    return [c for c in token.children if c.dep_ == "compound"] + [token]


def find_negated_target_spacy(token: "Token") -> list["Token"] | None:
    """
    Find the actual target of a negation token using spaCy's dependency parser.

    Args:
        token: spaCy token identified as a negation term.

    Returns:
        List of target tokens affected by negation, or None if no clear target.

    Notes:
        - Uses dependency parsing to identify grammatical relationships
        - Handles various negation patterns: AUX verbs, direct objects, adverbials
        - Accounts for compound words and prepositional phrases
        - Essential for accurate semantic analysis in survey responses
    """
    head = token.head

    # Case: Negation modifying an AUX verb (e.g., "is not painful", "was not running")
    if head.pos_ == "AUX":
        # Look for various complements or attributes of the AUX verb
        for child in head.children:
            # Case: "is not painful" (acomp) or "is not a doctor" (attr)
            if child.dep_ in ("acomp", "attr"):
                adv_modifiers = [c for c in child.children if c.dep_ == "advmod"]
                base_tokens = get_full_compound_spacy(child)
                return sorted(adv_modifiers + base_tokens, key=lambda t: t.i)

            # Case: "is not in pain" (prep -> pobj)
            if child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        return get_full_compound_spacy(pobj)

        # Look for a verb being modified by the AUX (e.g., "was not running")
        # In this case, the verb is the head of the AUX
        if head.head.pos_ == "VERB":
            return [head.head]

        # If the AUX is the root, and there's a following verb (e.g. "did not go")
        if head.dep_ == "ROOT":
            for child in head.children:
                if child.pos_ == "VERB":
                    # Check for adverbial modifiers on the verb
                    adv_modifiers = [c for c in child.children if c.dep_ == "advmod"]
                    return sorted(adv_modifiers + [child], key=lambda t: t.i)

    # Case: Direct negation of a verb (e.g. "don't feel")
    if head.pos_ == "VERB":
        # Try to find a direct object first
        for child in head.children:
            if child.dep_ in ("dobj", "attr"):
                return get_full_compound_spacy(child)
        # If no object, the verb itself is negated
        return [head]

    # Case: Determiner "no" (e.g., "no pain")
    if token.dep_ == "det" and token.head.pos_ in ("NOUN", "PROPN"):
        return get_full_compound_spacy(token.head)

    # Case: Adverbial negation (e.g. "never painful", "rarely feel")
    if token.dep_ == "advmod":
        # include other adverbs modifying the same head, e.g. "not very painful"
        adv_modifiers = [
            c for c in token.head.children if c.dep_ == "advmod" and c.i != token.i
        ]
        return sorted([token] + adv_modifiers + [token.head], key=lambda t: t.i)

    # Case: Preposition "without" (e.g., "without pain")
    if token.lemma_ == "without":
        for child in token.children:
            if child.dep_ == "pobj":
                return get_full_compound_spacy(child)

    # Fallback for other cases, like "not a doctor"
    if head.pos_ in ("NOUN", "ADJ", "VERB"):
        return [head]

    return None


def process_text_with_spacy(text: str, nlp: Any, config: Dict[str, Any]) -> list[str]:
    """
    Process text using spaCy for advanced linguistic analysis.

    Args:
        text: Input text string to process.
        nlp: Loaded spaCy model for linguistic analysis.
        config: Configuration dictionary with processing parameters.

    Returns:
        List of processed tokens with negation handling and compound word expansion.

    Notes:
        - Primary method for scientific text analysis using spaCy
        - Handles negation patterns with grammatical accuracy
        - Expands compound words for complete semantic analysis
        - Filters intensifiers from negation forms (Option B approach)
        - Preserves original word forms for frequency counting
    """
    # Use custom negation terms if provided, otherwise use default
    negation_terms = set(config.get("spacy_negation_terms", [])) or {
        "not",
        "n't",
        "no",
        "never",
        "seldom",
        "rarely",
        "without",
    }

    # Pre-process to handle hyphenated words
    text = re.sub(r"(\b\w+)-(\w+\b)", r"\1_\2", text)

    doc = nlp(text.lower())
    output_tokens = []
    tokens_to_skip = set()

    # Handle "absent" pattern
    for token in doc:
        if token.lemma_ == "absent" and token.dep_ == "acomp":
            subj = next((c for c in token.head.children if c.dep_ == "nsubj"), None)
            if subj:
                output_tokens.append(f"absent_{subj.lemma_}")
                tokens_to_skip.update(t.i for t in [subj, token, token.head])

    # Main negation handling
    for token in doc:
        if token.i in tokens_to_skip:
            continue
        if token.text in negation_terms:
            target_tokens = find_negated_target_spacy(token)
            if target_tokens:
                # Filter out intensifiers from target tokens (Option B: remove from negation forms)
                intensifiers = config.get(
                    "negation_intensifiers",
                    [
                        "very",
                        "really",
                        "quite",
                        "rather",
                        "somewhat",
                        "slightly",
                        "too",
                        "so",
                        "extremely",
                        "incredibly",
                        "absolutely",
                        "completely",
                        "totally",
                        "utterly",
                        "fairly",
                        "pretty",
                        "a bit",
                        "a little",
                    ],
                )
                # Filter out intensifiers from target tokens
                filtered_target_tokens = [
                    t
                    for t in target_tokens
                    if t.text.lower() not in intensifiers
                    and t.lemma_.lower() not in intensifiers
                ]

                # If all target tokens were intensifiers, use the original target tokens
                # (this handles cases where the only target is an intensifier)
                if not filtered_target_tokens:
                    filtered_target_tokens = target_tokens

                # Get the lemma of the negation term, but keep "not" for "n't"
                negation_lemma = "not" if token.lemma_ == "n't" else token.lemma_

                # If the negation term is an adverb, we might have already included it
                # in target_tokens. We should avoid duplicating it.
                if token.dep_ == "advmod" and token in filtered_target_tokens:
                    lemmatized_targets = [t.lemma_ for t in filtered_target_tokens]
                    new_token = "_".join(lemmatized_targets)
                else:
                    lemmatized_targets = [t.lemma_ for t in filtered_target_tokens]
                    new_token = f"{negation_lemma}_{'_'.join(lemmatized_targets)}"

                output_tokens.append(new_token)

                tokens_to_skip.add(token.i)
                for t in filtered_target_tokens:
                    tokens_to_skip.add(t.i)
                    # Also skip compounds of the target
                    tokens_to_skip.update(c.i for c in get_full_compound_spacy(t))

    # Add remaining tokens, handling compounds
    for token in doc:
        if token.i in tokens_to_skip:
            continue

        # Don't add compounds of already-skipped tokens
        if token.dep_ == "compound" and token.head.i in tokens_to_skip:
            continue

        # If it's the head of a compound, process the whole thing
        is_head_of_compound = any(c.dep_ == "compound" for c in token.children)
        if is_head_of_compound:
            compound_tokens = get_full_compound_spacy(token)
            # Ensure we haven't processed any part of this compound before
            if all(t.i not in tokens_to_skip for t in compound_tokens):
                # Use text (not lemma) to preserve original forms for frequency counting
                compound_texts = [t.text.lower() for t in compound_tokens]
                output_tokens.append("_".join(compound_texts))
                for t in compound_tokens:
                    tokens_to_skip.add(t.i)
                continue

        # Regular token processing
        if (not token.is_stop or token.pos_ == "AUX") and not token.is_punct:
            # For OOV words, use the lemma if it's a known word in the vocabulary.
            # This is crucial for correctly handling contractions that spaCy splits,
            # for example, "can't" is tokenized into "ca" and "n't". The token "ca"
            # is considered out-of-vocabulary (OOV), but its lemma is "can".
            # The original check using `has_vector` was too strict, especially for
            # models without full word vectors. Using `in nlp.vocab.strings` is a
            # more robust way to check if the lemma is a known word.
            # For frequency counting, we want to preserve the original word forms
            # (not lemmatized) so we can group variations later
            # Use token.text.lower() to preserve original form but in lowercase
            output_tokens.append(token.text.lower())
            tokens_to_skip.add(token.i)  # Mark as processed

    # Remove any stray negation terms that were not handled
    # (but keep all tokens, including duplicates, for accurate frequency counting)
    final_tokens = []
    for t in output_tokens:
        if t not in negation_terms:
            final_tokens.append(t)

    return final_tokens


def handle_negations_heuristic(
    tokens: List[str],
    cleaned_for_negation: List[str],
    is_punctuation: List[bool],
    config: Dict[str, Any],
) -> List[str]:
    """
    Heuristic-based negation handling for fallback when spaCy is unavailable.

    Args:
        tokens: Original tokenized text.
        cleaned_for_negation: Tokens cleaned for negation detection.
        is_punctuation: Boolean list indicating punctuation tokens.
        config: Configuration dictionary with negation parameters.

    Returns:
        List of tokens with heuristic negation handling applied.

    Notes:
        - Fallback method when spaCy is not available
        - Uses configurable distance thresholds for negation-target relationships
        - Implements Option B: intensifiers removed from negation forms
        - Skips certain function words when building negation tokens
        - Scientific rationale: balances accuracy with computational simplicity
    """
    negation_terms = config.get("negation_terms", ["not", "no", "never"])
    intensifiers = config.get(
        "negation_intensifiers",
        [
            "very",
            "really",
            "quite",
            "rather",
            "somewhat",
            "slightly",
            "too",
            "so",
            "extremely",
            "incredibly",
            "absolutely",
            "completely",
            "totally",
            "utterly",
            "fairly",
            "pretty",
            "a bit",
            "a little",
        ],
    )
    max_distance = config.get("negation_max_distance", 3)
    skip_stopwords = config.get("negation_skip_stopwords", True)

    # Get stopwords if needed for skipping
    stopwords_set = set()
    if skip_stopwords:
        stopwords_set = get_stopwords(config)

    processed_tokens = []

    i = 0
    while i < len(tokens):
        # Check if current token (cleaned) is a negation term
        cleaned_token = cleaned_for_negation[i]
        if cleaned_token in negation_terms:
            # Look ahead for potential intensifiers and the target word
            # We'll collect tokens until we find a non-intensifier, non-punctuation word
            collected_tokens = [cleaned_token]

            # Track words we skip so we can add them as separate tokens
            skipped_before_negation = []

            # Start looking from next token
            lookahead_idx = i + 1
            distance_traveled = 0

            # Skip punctuation and optionally stopwords
            while lookahead_idx < len(tokens) and distance_traveled < max_distance:
                if is_punctuation[lookahead_idx]:
                    lookahead_idx += 1
                    continue

                current_cleaned = cleaned_for_negation[lookahead_idx]

                # Check if this is an intensifier - SKIP intensifiers (Option B: remove from negation forms)
                if current_cleaned in intensifiers:
                    # Skip intensifier - don't add to collected_tokens
                    lookahead_idx += 1
                    distance_traveled += 1
                    # Skip any punctuation after intensifier
                    while lookahead_idx < len(tokens) and is_punctuation[lookahead_idx]:
                        lookahead_idx += 1
                    continue

                # Check if this is a word we should skip when looking for negation target
                # These are words that shouldn't be part of negation tokens
                negation_skip_words = [
                    # Determiners that should be skipped (not included in negation tokens)
                    "a",
                    "an",
                    "the",
                    # Auxiliary and common verbs that often come between negation and target
                    "feel",
                    "have",
                    "be",
                    "do",
                    "get",
                    "make",
                    "take",
                    "give",
                    "go",
                    "know",
                    "think",
                    "see",
                    "come",
                    "want",
                    "look",
                    "use",
                    "find",
                    "tell",
                    "ask",
                    "work",
                    "seem",
                    "try",
                    "need",
                    "become",
                    "leave",
                    "put",
                    "mean",
                    "keep",
                    "let",
                    "begin",
                    "show",
                    "hear",
                    "play",
                    "run",
                    "move",
                    "like",
                    "live",
                    "believe",
                    "hold",
                    "bring",
                    "happen",
                    "write",
                    "provide",
                    "sit",
                    "stand",
                    "lose",
                    "pay",
                    "meet",
                    "include",
                    "continue",
                    "set",
                    "learn",
                    "change",
                    "lead",
                    "understand",
                    "watch",
                    "follow",
                    "stop",
                    "create",
                    "speak",
                    "read",
                    "allow",
                    "add",
                    "spend",
                    "grow",
                    "open",
                    "walk",
                    "win",
                    "offer",
                    "remember",
                    "love",
                    "consider",
                    "appear",
                    "buy",
                    "wait",
                    "serve",
                    "die",
                    "send",
                    "expect",
                    "build",
                    "stay",
                    "fall",
                    "cut",
                    "reach",
                    "kill",
                    "remain",
                ]

                # Check if this is a word we should keep in negation tokens
                # These are quantifiers and other words that should be part of negation phrases
                negation_keep_words = [
                    # Quantifiers
                    "any",
                    "some",
                    "no",
                    "every",
                    "each",
                    "all",
                    "both",
                    "many",
                    "much",
                    "few",
                    "little",
                    "several",
                    "enough",
                    "more",
                    "most",
                    "other",
                    "another",
                    # Demonstratives
                    "this",
                    "that",
                    "these",
                    "those",
                    # Possessives
                    "my",
                    "your",
                    "his",
                    "her",
                    "its",
                    "our",
                    "their",
                    # Other function words that might be part of negation phrases
                    "such",
                    "what",
                    "which",
                    "whose",
                ]

                # Skip words that shouldn't be in negation tokens
                if current_cleaned in negation_skip_words:
                    # Add skipped word to separate tokens list
                    skipped_before_negation.append(tokens[lookahead_idx])
                    lookahead_idx += 1
                    distance_traveled += 1
                    # Skip any punctuation
                    while lookahead_idx < len(tokens) and is_punctuation[lookahead_idx]:
                        lookahead_idx += 1
                    continue

                # Keep words that should be part of negation tokens
                if current_cleaned in negation_keep_words:
                    collected_tokens.append(current_cleaned)
                    lookahead_idx += 1
                    distance_traveled += 1
                    # Skip any punctuation
                    while lookahead_idx < len(tokens) and is_punctuation[lookahead_idx]:
                        lookahead_idx += 1
                    continue

                # Skip stopwords if configured (but not if it's an intensifier or keep word)
                if skip_stopwords and current_cleaned in stopwords_set:
                    lookahead_idx += 1
                    distance_traveled += 1
                    continue

                # Not an intensifier, not a keep word, and not a stopword (or stopword skipping is disabled)
                # This should be our main content word
                collected_tokens.append(current_cleaned)
                lookahead_idx += 1
                distance_traveled += 1

                # After finding a content word, check if there's another word that might be part of the phrase
                # (e.g., "real pain" - "real" is adjective, "pain" is noun)
                # We'll include one more word if it's not a word we should skip and we haven't reached max distance
                if lookahead_idx < len(tokens) and distance_traveled < max_distance:
                    next_cleaned = cleaned_for_negation[lookahead_idx]
                    if not is_punctuation[lookahead_idx]:
                        # Check if next word should be skipped
                        should_skip = (
                            skip_stopwords and next_cleaned in stopwords_set
                        ) or next_cleaned in negation_skip_words

                        # Check if next word should be included (keep word, or content word)
                        # Note: intensifiers are NOT included (Option B: remove from negation forms)
                        should_include = (
                            next_cleaned in negation_keep_words or not should_skip
                        )

                        if should_include:
                            # Include this word in the negation token
                            collected_tokens.append(next_cleaned)
                            lookahead_idx += 1
                            distance_traveled += 1

                break

            # If we collected more than just the negation term, create a combined token
            if len(collected_tokens) > 1:
                # First, add any skipped words as separate tokens
                for skipped_token in skipped_before_negation:
                    processed_tokens.append(skipped_token)

                # Then join all collected tokens with underscores
                combined_token = "_".join(collected_tokens)
                processed_tokens.append(combined_token)
                # Skip to after the last token we processed
                i = lookahead_idx
                continue
            else:
                # No target word found within max distance
                # Add any skipped words as separate tokens
                for skipped_token in skipped_before_negation:
                    processed_tokens.append(skipped_token)

                # Don't add standalone negation term - it will be filtered as a stopword
                # Skip to next token
                i += 1
        else:
            # Check if this token was part of a previous negation that got skipped
            # (This handles the case where we're already past tokens that were added as skipped words)
            processed_tokens.append(tokens[i])
            i += 1

    return processed_tokens


# ============================================================================
# File Format Detection and Loading
# ============================================================================


def detect_file_format(
    filepath: str, config: Dict[str, Any], format_override: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Detect file format based on extension and content with robust fallbacks.

    Args:
        filepath: Path to the file.
        config: Configuration dictionary with format detection parameters.
        format_override: Optional format to use (bypasses detection).

    Returns:
        Tuple of (format_type, format_params) where format_params contains
        format-specific parameters like delimiter for CSV files.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the file cannot be read.

    Notes:
        - Supports multiple scientific data formats for survey analysis
        - Uses extension detection with content validation for robustness
        - Delimiter detection for CSV/TSV files with statistical consistency checks
        - Graceful fallbacks for unknown or corrupted file formats
    """
    # Check if format override is specified
    if format_override:
        print(f"  Using specified format: {format_override} (bypassing detection)")

        # Return appropriate format parameters based on the override
        if format_override == "csv":
            # For CSV format override, we should still try to detect the delimiter
            # since the user might be forcing CSV format for a file with non-standard extension
            # but the file could still have a different delimiter (tab, pipe, semicolon, etc.)
            try:
                with open(
                    filepath, "r", encoding=config.get("csv_encoding", "utf-8")
                ) as f:
                    # Read first 10 lines for delimiter detection
                    lines = []
                    for _ in range(10):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line.strip())

                if len(lines) >= 2:
                    # Test different delimiters
                    for delimiter in config.get(
                        "csv_delimiters", [",", "\t", ";", "|"]
                    ):
                        delimiter_counts = []
                        valid_lines = 0

                        for line in lines:
                            if line:  # Skip empty lines
                                delimiter_count = line.count(delimiter)
                                delimiter_counts.append(delimiter_count)
                                valid_lines += 1

                        # Need at least 2 valid lines with delimiter
                        if valid_lines >= 2:
                            # Check if delimiter count is consistent
                            max_count = max(delimiter_counts)
                            min_count = min(delimiter_counts)

                            # Allow some variation (e.g., missing values)
                            if max_count - min_count <= 2 and max_count > 0:
                                # Additional check: try parsing first line
                                parts = lines[0].split(delimiter)
                                if len(parts) >= 2:  # At least 2 columns
                                    print(
                                        f"  Detected delimiter '{repr(delimiter)}' with {max_count + 1} columns"
                                    )
                                    return "csv", {"delimiter": delimiter}

                # If no consistent delimiter found, use default comma
                print("  No consistent delimiter found, using default comma delimiter")
                return "csv", {"delimiter": ","}

            except Exception as e:
                print(f"  Warning: Could not detect delimiter for CSV override: {e}")
                print("  Using default comma delimiter")
                return "csv", {"delimiter": ","}

        elif format_override == "text":
            return "text", {}
        elif format_override == "excel":
            return "excel", {}
        elif format_override == "json":
            return "json", {}
        elif format_override == "parquet":
            return "parquet", {}
        elif format_override == "feather":
            return "feather", {}
        elif format_override == "hdf5":
            return "hdf5", {}
        else:
            print(
                f"  Warning: Unknown format '{format_override}', falling back to detection"
            )

    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    # Check by extension first with robust handling
    if ext in [".csv", ".tsv", ".txt"]:
        # Try to detect delimiter with multiple heuristics
        try:
            with open(filepath, "r", encoding=config.get("csv_encoding", "utf-8")) as f:
                # Read first 10 lines for better detection
                lines = []
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line.strip())

            if len(lines) >= 2:
                # Test different delimiters
                for delimiter in config.get("csv_delimiters", [",", "\t", ";", "|"]):
                    delimiter_counts = []
                    valid_lines = 0

                    for line in lines:
                        if line:  # Skip empty lines
                            delimiter_count = line.count(delimiter)
                            delimiter_counts.append(delimiter_count)
                            valid_lines += 1

                    # Need at least 2 valid lines with delimiter
                    if valid_lines >= 2:
                        # Check if delimiter count is consistent
                        max_count = max(delimiter_counts)
                        min_count = min(delimiter_counts)

                        # Allow some variation (e.g., missing values)
                        if max_count - min_count <= 2 and max_count > 0:
                            # Additional check: try parsing first line
                            parts = lines[0].split(delimiter)
                            if len(parts) >= 2:  # At least 2 columns
                                print(
                                    f"  Detected delimiter '{repr(delimiter)}' with {max_count + 1} columns"
                                )
                                return "csv", {"delimiter": delimiter}

        except Exception as e:
            print(f"  Warning: Could not detect delimiter: {e}")

        # Default based on extension
        if ext == ".tsv":
            print("  Using default tab delimiter for .tsv file")
            return "csv", {"delimiter": "\t"}
        elif ext == ".csv":
            print("  Using default comma delimiter for .csv file")
            return "csv", {"delimiter": ","}
        else:
            # For .txt files, be more conservative
            # Check if it looks like CSV with consistent structure
            try:
                with open(
                    filepath, "r", encoding=config.get("csv_encoding", "utf-8")
                ) as f:
                    content = f.read(2048)  # Read first 2KB

                # Check if it looks like CSV/TSV
                lines = [line.strip() for line in content.split("\n") if line.strip()]
                if len(lines) >= 3:
                    csv_like = False
                    for delimiter in config.get(
                        "csv_delimiters", [",", "\t", ";", "|"]
                    ):
                        # Check if delimiter appears in all first 3 lines
                        delimiter_in_all = all(delimiter in line for line in lines[:3])
                        if delimiter_in_all:
                            # Check consistency
                            counts = [line.count(delimiter) for line in lines[:3]]
                            if max(counts) - min(counts) <= 2:
                                print(
                                    f"  Detected CSV-like .txt file with delimiter '{repr(delimiter)}'"
                                )
                                return "csv", {"delimiter": delimiter}

                    # If we get here, it doesn't look like CSV
                    print("  Detected plain text format for .txt file")
                    return "text", {}
                else:
                    # Not enough lines to determine, default to text
                    print("  Defaulting to plain text format for .txt file")
                    return "text", {}

            except Exception as e:
                print(f"  Warning: Could not analyze .txt file: {e}")
                print("  Defaulting to plain text format")
                return "text", {}

    elif ext in [".xlsx", ".xls", ".xlsm", ".xlsb"]:
        print("  Detected Excel format")
        return "excel", {}

    elif ext in [".json", ".jsonl"]:
        print("  Detected JSON format")
        return "json", {}

    elif ext in [".parquet", ".pq", ".parq"]:
        print("  Detected Parquet format")
        return "parquet", {}

    elif ext in [".feather"]:
        print("  Detected Feather format")
        return "feather", {}

    elif ext in [".h5", ".hdf5", ".hdf"]:
        print("  Detected HDF5 format")
        return "hdf5", {}

    else:
        # Try to read as text and detect format
        try:
            with open(filepath, "r", encoding=config.get("csv_encoding", "utf-8")) as f:
                content = f.read(2048)  # Read first 2KB

            # Check if it looks like CSV/TSV
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            if len(lines) >= 3:
                for delimiter in config.get("csv_delimiters", [",", "\t", ";", "|"]):
                    delimiter_present = all(delimiter in line for line in lines[:3])
                    if delimiter_present:
                        # Check consistency
                        counts = [line.count(delimiter) for line in lines[:3]]
                        if max(counts) - min(counts) <= 2:
                            print(
                                f"  Detected CSV-like format with delimiter '{repr(delimiter)}'"
                            )
                            return "csv", {"delimiter": delimiter}

            # Check for JSON
            if content.strip().startswith("{") or content.strip().startswith("["):
                try:
                    json.loads(content[:500])  # Try to parse a sample
                    print("  Detected JSON format from content")
                    return "json", {}
                except:
                    pass

            # Default to text
            print("  Detected plain text format")
            return "text", {}

        except Exception as e:
            print(f"  Warning: Could not detect format: {e}")
            # Fallback to default format
            default_format = config.get("default_format", "txt")
            print(f"  Using default format: {default_format}")
            return (
                default_format,
                {},
            )  # ============================================================================


# Cleanup Function
# ============================================================================


def cleanup() -> None:
    """
    Cleanup function registered with atexit.
    """
    global INTERRUPTED

    if INTERRUPTED:
        print("\n" + "=" * 60)
        print("Process interrupted by user.")
        print(f"Processed {len(PROCESSED_FILES)} file(s) before interruption.")

        if PROCESSED_FILES:
            print("\nFiles processed:")
            for i, file_info in enumerate(PROCESSED_FILES, 1):
                print(
                    f"  {i}. {file_info['filepath']} ({file_info['text_entries_extracted']} entries)"
                )

        print("=" * 60)


def load_data_file(filepath: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load data file with automatic format detection and error handling.

    Args:
        filepath: Path to the data file.
        config: Configuration dictionary with loading parameters.

    Returns:
        pandas DataFrame containing the loaded data.

    Raises:
        ValueError: If file format cannot be determined or data cannot be loaded.
        pd.errors.EmptyDataError: If the file contains no data.

    Notes:
        - Supports multiple scientific data formats for survey analysis
        - Concatenates multiple sheets for Excel files
        - Provides detailed logging for debugging and reproducibility
        - Robust error handling for corrupted or malformed files
    """
    print(f"Loading file: {filepath}")

    try:
        # Get format override from config if specified
        format_override = config.get("format_override")
        format_type, format_params = detect_file_format(
            filepath, config, format_override
        )
        print(f"  Detected format: {format_type} with params: {format_params}")

        if format_type == "csv":
            df = pd.read_csv(
                filepath,
                delimiter=format_params.get("delimiter", ","),
                encoding=config.get("csv_encoding", "utf-8"),
                on_bad_lines="warn",  # Warn on bad lines but continue
            )

        elif format_type == "excel":
            # Try to read all sheets
            try:
                xls = pd.ExcelFile(filepath)
                # Concatenate all sheets
                dfs = []
                for sheet_name in xls.sheet_names:
                    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                    df_sheet["_sheet"] = sheet_name  # Add sheet name for tracking
                    dfs.append(df_sheet)
                df = pd.concat(dfs, ignore_index=True)
            except Exception as e:
                print(f"  Warning: Could not read all sheets: {e}")
                # Fallback to first sheet
                df = pd.read_excel(filepath)

        elif format_type == "json":
            df = pd.read_json(filepath)

        elif format_type == "parquet":
            df = pd.read_parquet(filepath)

        elif format_type == "feather":
            df = pd.read_feather(filepath)

        elif format_type == "hdf5":
            # Try to read first key
            with pd.HDFStore(filepath, mode="r") as store:
                keys = store.keys()
                if keys:
                    df = store[keys[0]]
                else:
                    raise ValueError("No datasets found in HDF5 file")

        else:  # text format
            with open(filepath, "r", encoding=config.get("csv_encoding", "utf-8")) as f:
                content = f.read()
            # Create a DataFrame with a single column
            df = pd.DataFrame({"text": [content]})

        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"✗ Error loading file {filepath}: {e}")
        raise


def extract_text_from_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    Extract text from DataFrame based on configuration with robust auto-detection.

    Args:
        df: pandas DataFrame containing the data.
        config: Configuration dictionary with text extraction parameters.

    Returns:
        List of text strings extracted from the specified columns.

    Raises:
        ValueError: If no text columns can be found or extracted.

    Notes:
        - Uses multiple heuristics for text column detection in scientific data
        - String ratio and average length analysis for column classification
        - Case-insensitive matching for column name flexibility
        - Detailed statistics reporting for methodological transparency
        - Fallback mechanisms for challenging data structures
    """
    text_columns = config.get("text_columns", [])
    original_text_columns = text_columns.copy() if text_columns else []

    if text_columns:
        # Verify columns exist with robust error handling
        missing_columns = [col for col in text_columns if col not in df.columns]

        if missing_columns:
            print(
                f"⚠️  Warning: {len(missing_columns)} specified column(s) not found: {missing_columns}"
            )
            print(f"   Available columns: {list(df.columns)}")

            # Try case-insensitive matching with detailed reporting
            available_lower = [col.lower() for col in df.columns]
            matched_columns = []
            unmatched_columns = []

            for missing_col in missing_columns:
                # Try case-insensitive match
                if missing_col.lower() in available_lower:
                    idx = available_lower.index(missing_col.lower())
                    matched_col = df.columns[idx]
                    matched_columns.append((missing_col, matched_col))
                    print(
                        f"   Found case-insensitive match: '{missing_col}' -> '{matched_col}'"
                    )
                else:
                    unmatched_columns.append(missing_col)
                    print(f"   No match found for column: '{missing_col}'")

            if matched_columns:
                # Update text_columns with matched names
                text_columns = [
                    matched if orig in missing_columns else col
                    for col in text_columns
                    for orig, matched in matched_columns
                    if col == orig
                ] + [col for col in text_columns if col not in missing_columns]

                print(f"   Using matched columns: {text_columns}")

            if unmatched_columns:
                print(f"   Columns not found (will be skipped): {unmatched_columns}")
                # Use only available columns
                text_columns = [col for col in text_columns if col in df.columns]

                if not text_columns:
                    print(
                        "✗ Error: None of the specified text columns were found in the dataframe."
                    )
                    print(f"   Specified: {original_text_columns}")
                    print(f"   Available: {list(df.columns)}")
                    raise ValueError("No specified text columns found in dataframe")
        else:
            print(f"✓ All specified columns found: {text_columns}")

    if not text_columns:
        # Auto-detect text columns with multiple heuristics
        print("  Auto-detecting text columns...")
        text_columns = []
        column_analysis = []

        for col in df.columns:
            try:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    column_analysis.append((col, 0.0, 0.0, "empty", False))
                    continue

                # Heuristic 1: Check dtype
                is_string_dtype = pd.api.types.is_string_dtype(df[col])
                is_object_dtype = pd.api.types.is_object_dtype(df[col])

                # Heuristic 2: Sample values with error handling
                sample_size = min(20, len(col_data))
                try:
                    sample_values = (
                        col_data.sample(n=sample_size, random_state=42)
                        if len(col_data) > sample_size
                        else col_data
                    )
                except Exception as e:
                    print(f"    ⚠️  Could not sample column '{col}': {e}")
                    sample_values = col_data.head(sample_size)

                # Count string values with robust type checking
                string_count = 0
                total_chars = 0

                for val in sample_values:
                    if isinstance(val, str):
                        string_count += 1
                        total_chars += len(val.strip())
                    elif pd.isna(val):
                        continue
                    else:
                        # Try to convert to string
                        try:
                            str_val = str(val)
                            string_count += 1
                            total_chars += len(str_val.strip())
                        except Exception:
                            pass

                # Heuristic 3: String ratio and average length
                string_ratio = (
                    string_count / len(sample_values) if len(sample_values) > 0 else 0
                )
                avg_length = total_chars / string_count if string_count > 0 else 0

                # Decision criteria with multiple thresholds
                is_text_column = False
                reason = ""

                if is_string_dtype and string_ratio > 0.8:
                    is_text_column = True
                    reason = f"string dtype, {string_ratio:.1%} strings"
                elif is_object_dtype and string_ratio > 0.7 and avg_length > 3:
                    is_text_column = True
                    reason = f"object dtype, {string_ratio:.1%} strings, avg {avg_length:.1f} chars"
                elif string_ratio > 0.9:  # Very high string ratio
                    is_text_column = True
                    reason = f"high string ratio ({string_ratio:.1%})"
                else:
                    reason = f"low string ratio ({string_ratio:.1%}), avg {avg_length:.1f} chars"

                column_analysis.append(
                    (col, string_ratio, avg_length, reason, is_text_column)
                )

                if is_text_column:
                    text_columns.append(col)
                    print(
                        f"    ✓ {col}: {string_ratio:.1%} strings, avg length {avg_length:.1f} chars ({reason})"
                    )
                else:
                    print(
                        f"    ✗ {col}: {string_ratio:.1%} strings, avg length {avg_length:.1f} chars"
                    )

            except Exception as e:
                print(f"    ⚠️  Error analyzing column '{col}': {e}")
                column_analysis.append((col, 0.0, 0.0, f"error: {str(e)[:50]}", False))

        # Summary of auto-detection
        print(f"\n  Auto-detection summary:")
        print(f"    Analyzed {len(column_analysis)} columns")
        print(f"    Detected {len(text_columns)} text columns: {text_columns}")

        if not text_columns:
            print("    ⚠️  No text columns auto-detected. Showing column analysis:")
            for col, ratio, avg_len, reason, is_text in column_analysis:
                status = "✓" if is_text else "✗"
                print(
                    f"      {status} {col}: {ratio:.1%} strings, avg {avg_len:.1f} chars - {reason}"
                )

    if not text_columns:
        # Last resort: use all columns that can be converted to string
        print(
            "  ⚠️  No text columns detected. Attempting to use all columns as fallback..."
        )
        text_columns = []
        conversion_errors = []

        for col_idx, col in enumerate(df.columns):
            try:
                # Try to convert column to string with validation
                test_sample = df[col].head(5).fillna("").astype(str)

                # Check if conversion produced any non-empty strings
                non_empty_test = sum(1 for text in test_sample if text.strip())

                if non_empty_test > 0:
                    text_columns.append(col)
                    print(
                        f"    ✓ Column {col_idx + 1}/{len(df.columns)}: '{col}' - {non_empty_test}/5 non-empty test values"
                    )
                else:
                    print(
                        f"    ✗ Column {col_idx + 1}/{len(df.columns)}: '{col}' - all test values empty"
                    )

            except Exception as e:
                error_msg = f"Column '{col}': {str(e)[:100]}"
                conversion_errors.append(error_msg)
                print(
                    f"    ⚠️  Column {col_idx + 1}/{len(df.columns)}: '{col}' - conversion failed: {str(e)[:50]}..."
                )

        print(f"\n  Fallback summary:")
        print(f"    Attempted {len(df.columns)} columns")
        print(f"    Successfully converted {len(text_columns)} columns")

        if conversion_errors:
            print(f"    Failed to convert {len(conversion_errors)} columns")
            for error in conversion_errors[:3]:  # Show first 3 errors
                print(f"      - {error}")
            if len(conversion_errors) > 3:
                print(f"      ... and {len(conversion_errors) - 3} more errors")

    # Validate we have columns to process
    if not text_columns:
        error_msg = "No text columns could be identified or converted. "
        error_msg += f"DataFrame has {len(df.columns)} columns: {list(df.columns)}"
        print(f"✗ {error_msg}")
        raise ValueError(error_msg)

    print(f"\n  Final text columns to process: {text_columns}")
    print(f"  Total rows in dataframe: {len(df)}")

    # Extract text from selected columns with robust error handling
    all_text = []
    total_rows = len(df)
    extraction_stats = []

    for col_idx, col in enumerate(text_columns):
        print(
            f"\n  Extracting text from column {col_idx + 1}/{len(text_columns)}: '{col}'"
        )

        try:
            # Primary conversion method
            col_text = df[col].fillna("").astype(str).tolist()
            method_used = "primary (fillna + astype(str))"

        except Exception as e:
            print(f"    ⚠️  Primary conversion failed: {str(e)[:50]}...")

            try:
                # Alternative conversion method
                col_text = (
                    df[col].apply(lambda x: str(x) if not pd.isna(x) else "").tolist()
                )
                method_used = "alternative (apply + str conversion)"
                print(f"    ✓ Recovered with alternative conversion method")

            except Exception as e2:
                error_msg = f"Failed to extract text from column '{col}': {e2}"
                print(f"    ✗ {error_msg}")
                extraction_stats.append((col, 0, total_rows, "failed", str(e2)[:50]))
                continue

        # Count non-empty strings
        non_empty = sum(1 for text in col_text if text.strip())
        empty = total_rows - non_empty

        # Calculate statistics
        empty_pct = (empty / total_rows * 100) if total_rows > 0 else 0
        avg_length = (
            sum(len(text.strip()) for text in col_text if text.strip()) / non_empty
            if non_empty > 0
            else 0
        )

        extraction_stats.append((col, non_empty, empty, method_used, "success"))

        print(f"    Conversion method: {method_used}")
        print(
            f"    Non-empty values: {non_empty}/{total_rows} ({empty_pct:.1f}% empty)"
        )
        if non_empty > 0:
            print(f"    Average text length: {avg_length:.1f} characters")

        all_text.extend(col_text)

    # Summary of extraction
    print(f"\n  Extraction summary:")
    total_extracted = len(all_text)
    successful_columns = sum(
        1 for _, _, _, _, status in extraction_stats if status == "success"
    )

    print(
        f"    Successfully extracted from {successful_columns}/{len(text_columns)} columns"
    )
    print(f"    Total text entries: {total_extracted}")

    for col, non_empty, empty, method, status in extraction_stats:
        if status == "success":
            print(f"      ✓ {col}: {non_empty} non-empty, {empty} empty ({method})")
        else:
            print(f"      ✗ {col}: extraction failed")

    # Filter out completely empty strings
    filtered_text = [text for text in all_text if text.strip()]
    empty_count = len(all_text) - len(filtered_text)
    empty_pct_total = (empty_count / len(all_text) * 100) if len(all_text) > 0 else 0

    if empty_count > 0:
        print(
            f"\n  Removed {empty_count} completely empty entries ({empty_pct_total:.1f}% of total)"
        )

    if not filtered_text:
        error_msg = "No non-empty text found in any of the specified columns. "
        error_msg += f"Checked {len(text_columns)} columns: {text_columns}"
        print(f"✗ {error_msg}")
        raise ValueError(error_msg)

    print(f"\n  Final text count: {len(filtered_text)} non-empty text entries")

    return filtered_text  # ============================================================================


# Text Processing
# ============================================================================


def extract_raw_words(text: str, config: Dict[str, Any]) -> List[str]:
    """
    Extract raw words from text with minimal processing for baseline analysis.

    Args:
        text: Input text string to process.
        config: Configuration dictionary with basic processing parameters.

    Returns:
        List of raw words extracted from the text.

    Notes:
        - Used ONLY for creating the raw word frequency file (baseline analysis)
        - Does NOT use spaCy, NLTK tokenization, negation handling, or stopword removal
        - Provides methodological transparency by separating raw from processed analysis
        - Simple regex-based word splitting preserves original word forms
        - Scientific rationale: establishes baseline frequencies before advanced processing

    Features:
        - Simple regex-based word splitting
        - Basic lowercase conversion (if configured)
        - Minimum word length filtering (if configured)
        - Number removal (if configured)
        - No advanced linguistic processing
    """
    # Validate input
    if not isinstance(text, str):
        return []

    if not text or text.strip() == "":
        return []

    original_text = text

    try:
        # Apply lowercase if configured
        if config.get("lowercase", True):
            text = text.lower()

        # Simple regex-based word splitting
        # This captures words with apostrophes and hyphens
        words = re.findall(r"\b\w+(?:['-]\w+)*\b", text)

        # Filter words with configurable rules
        filtered_words = []
        for word in words:
            # Skip if word is empty after stripping
            if not word or not word.strip():
                continue

            cleaned_word = word.strip()

            # Remove punctuation (basic check - regex should already handle this)
            # But we need to ensure no stray punctuation remains
            if config.get("remove_punctuation", True):
                # Remove any non-alphanumeric characters except apostrophes and hyphens
                cleaned_word = re.sub(r"[^a-zA-Z0-9'-]", "", cleaned_word)
                if not cleaned_word:
                    continue

            # Remove numbers (if configured)
            if config.get("remove_numbers", True):
                if cleaned_word.isdigit():
                    continue

            # Minimum length check
            min_length = config.get("min_word_length", 2)
            if len(cleaned_word) < min_length:
                continue

            filtered_words.append(cleaned_word)

        return filtered_words

    except Exception as e:
        print(f"    ⚠️  Unexpected error in extract_raw_words: {str(e)[:50]}...")
        # Return empty list rather than crashing
        return []


def preprocess_text(text: str, config: Dict[str, Any]) -> List[str]:
    """
    Preprocess text with advanced linguistic analysis for scientific research.

    Args:
        text: Input text string to process.
        config: Configuration dictionary with processing parameters.

    Returns:
        List of processed tokens ready for frequency analysis.

    Notes:
        - Uses spaCy as DEFAULT for grammatical accuracy in negation detection
        - Falls back to heuristic methods when spaCy is unavailable
        - Two-phase processing: spaCy first, then configurable filters
        - Scientific rationale: preserves semantic meaning while enabling quantitative analysis

    Features:
        - Configurable negation terms and intensifiers
        - Maximum distance between negation and target word
        - Option to skip stopwords when looking for target word
        - Handles punctuation between negation and target
        - Compound word expansion for complete semantic analysis

    IMPORTANT: spaCy processes text FIRST before other operations to correctly detect
    grammatical groups and negations. This ensures scientific accuracy in semantic analysis.
    """
    # Validate input
    if not isinstance(text, str):
        return []

    if not text or text.strip() == "":
        return []

    original_text = text

    try:
        # ====================================================================
        # PHASE 1: Try spaCy processing FIRST (default behavior)
        # ====================================================================
        use_spacy = config.get("use_spacy_negation", True) and config.get(
            "handle_negations", True
        )

        if use_spacy:
            try:
                # The spaCy model should be loaded in main() and passed through config
                nlp = config.get("_spacy_model")
                if nlp is not None:
                    # Use spaCy-based processing as the PRIMARY processing method
                    # This happens BEFORE any tokenization or other operations
                    spacy_tokens = process_text_with_spacy(original_text, nlp, config)

                    # Apply configurable filters to spaCy output
                    filtered_tokens = []
                    for token in spacy_tokens:
                        # Skip if token is empty after stripping
                        if not token or not token.strip():
                            continue

                        cleaned_token = token.strip()

                        # Remove punctuation (if configured) - but spaCy already removes punctuation
                        # We still need to check for config compliance
                        if config.get("remove_punctuation", True):
                            # Check if token contains only allowed characters
                            # For negation tokens like "not_nice", we need to allow underscores
                            if not re.match(r"^[a-zA-Z0-9_]+$", cleaned_token):
                                # Try to clean the token, but preserve underscores in negation tokens
                                # Replace non-alphanumeric and non-underscore characters
                                cleaned_token = re.sub(
                                    r"[^a-zA-Z0-9_]", "", cleaned_token
                                )
                                if not cleaned_token:
                                    continue

                        # Remove numbers (if configured)
                        if config.get("remove_numbers", True):
                            if cleaned_token.isdigit():
                                continue

                        # Minimum length check
                        min_length = config.get("min_word_length", 2)
                        if len(cleaned_token) < min_length:
                            continue

                        filtered_tokens.append(cleaned_token)

                    # Debug logging for very short texts
                    if (
                        len(original_text) < 50
                        and len(filtered_tokens) == 0
                        and len(spacy_tokens) > 0
                    ):
                        print(
                            f"    ⚠️  Text filtered to empty: '{original_text[:50]}...' -> spaCy tokens: {spacy_tokens[:5]}"
                        )

                    return filtered_tokens
                else:
                    # spaCy model not available, fall back to standard processing
                    print(
                        "  ⚠️  spaCy model not loaded, falling back to standard processing"
                    )
            except Exception as e:
                print(f"  ⚠️  Error in spaCy negation processing: {str(e)[:50]}...")
                print("  Falling back to standard processing")

        # ====================================================================
        # PHASE 2: Fallback processing (used when spaCy is not available or fails)
        # ====================================================================

        # Lowercase
        if config.get("lowercase", True):
            text = text.lower()

        # Tokenize with robust fallbacks
        tokens = []
        tokenization_method = "unknown"

        try:
            # Primary method: NLTK word_tokenize
            tokens = word_tokenize(text)
            tokenization_method = "nltk.word_tokenize"

        except Exception as e1:
            print(f"    ⚠️  NLTK tokenization failed: {str(e1)[:50]}...")

            try:
                # Fallback 1: Simple regex tokenization
                tokens = re.findall(r"\b\w+(?:['-]\w+)*\b", text)
                tokenization_method = "regex (basic word pattern)"

            except Exception as e2:
                print(f"    ⚠️  Regex tokenization failed: {str(e2)[:50]}...")

                try:
                    # Fallback 2: Split on whitespace and clean
                    tokens = text.split()
                    # Clean each token
                    cleaned_tokens = []
                    for token in tokens:
                        # Remove punctuation from start/end
                        cleaned = token.strip(".,!?;:\"'()[]{}<>")
                        if cleaned:
                            cleaned_tokens.append(cleaned)
                    tokens = cleaned_tokens
                    tokenization_method = "whitespace split + cleaning"

                except Exception as e3:
                    print(f"    ⚠️  All tokenization methods failed: {str(e3)[:50]}...")
                    # Last resort: return empty list
                    return []

        # Clean tokens before negation handling to handle punctuation
        # (e.g., "not," should be recognized as "not")
        cleaned_for_negation = []
        is_punctuation = []
        for token in tokens:
            # Basic cleaning: strip common punctuation and lowercase
            cleaned = token.strip(".,!?;:\"'()[]{}<>")
            if cleaned:
                cleaned_for_negation.append(cleaned.lower())
                # Check if token consists entirely of punctuation characters
                # (e.g., "-", "--", "...")
                if re.match(r"^[^a-zA-Z0-9]+$", token):
                    is_punctuation.append(True)
                else:
                    is_punctuation.append(False)
            else:
                # Token is all punctuation (e.g., ",", ".", "!")
                cleaned_for_negation.append(token.lower())
                is_punctuation.append(True)

        # Handle negations if configured (fallback heuristic approach)
        if config.get("handle_negations", True):
            # Use heuristic-based approach as fallback (spaCy was not available or failed)
            tokens = handle_negations_heuristic(
                tokens, cleaned_for_negation, is_punctuation, config
            )

        # Filter tokens with configurable rules
        filtered_tokens = []
        for token in tokens:
            # Skip if token is empty after stripping
            if not token or not token.strip():
                continue

            cleaned_token = token.strip()

            # Remove punctuation (if configured)
            if config.get("remove_punctuation", True):
                # Check if token contains only allowed characters
                # For negation tokens like "not_nice", we need to allow underscores
                if not re.match(r"^[a-zA-Z0-9_]+$", cleaned_token):
                    # Try to clean the token, but preserve underscores in negation tokens
                    # Replace non-alphanumeric and non-underscore characters
                    cleaned_token = re.sub(r"[^a-zA-Z0-9_]", "", cleaned_token)
                    if not cleaned_token:
                        continue

            # Remove numbers (if configured)
            if config.get("remove_numbers", True):
                if cleaned_token.isdigit():
                    continue

            # Minimum length check
            min_length = config.get("min_word_length", 2)
            if len(cleaned_token) < min_length:
                continue

            filtered_tokens.append(cleaned_token)

        # Debug logging for very short texts
        if len(original_text) < 50 and len(filtered_tokens) == 0 and len(tokens) > 0:
            print(
                f"    ⚠️  Text filtered to empty: '{original_text[:50]}...' -> tokens: {tokens[:5]}"
            )

        return filtered_tokens

    except Exception as e:
        print(f"    ⚠️  Unexpected error in preprocess_text: {str(e)[:50]}...")
        # Return empty list rather than crashing
        return []


def get_stopwords(config: Dict[str, Any]) -> Set[str]:
    """
    Get stopwords based on configuration with scientific customization.

    Args:
        config: Configuration dictionary with stopword parameters.

    Returns:
        Set of stopwords for filtering.

    Notes:
        - Removes negation terms from stopwords when negation handling is enabled
        - Adds intensifiers to stopwords (Option B approach)
        - Supports custom stopwords for domain-specific analysis
        - Scientific rationale: preserves negation semantics while filtering common words
        - Multi-word intensifiers are split and added as individual stopwords
    """
    stopwords_set = set()

    if config.get("remove_stopwords", True):
        language = config.get("stopword_language", "english")
        try:
            stopwords_set.update(set(stopwords.words(language)))
        except Exception as e:
            print(f"✗ Warning: Could not load {language} stopwords: {e}")

    # Add custom stopwords
    custom_stopwords = config.get("custom_stopwords", [])
    if custom_stopwords:
        stopwords_set.update(set(custom_stopwords))

    # If negation handling is enabled, remove negation terms from stopwords
    # This ensures negation terms can be used in combined tokens like "not_nice"
    if config.get("handle_negations", True):
        negation_terms = config.get("negation_terms", ["not", "no", "never"])
        for term in negation_terms:
            if term in stopwords_set:
                stopwords_set.remove(term)
                if config.get("verbose", False):
                    print(
                        f"  Removed negation term '{term}' from stopwords to allow negation handling"
                    )

    # Add intensifiers to stopwords (Option B: treat intensifiers as stopwords)
    # This ensures intensifiers are filtered out in positive cases
    intensifiers = config.get(
        "negation_intensifiers",
        [
            "very",
            "really",
            "quite",
            "rather",
            "somewhat",
            "slightly",
            "too",
            "so",
            "extremely",
            "incredibly",
            "absolutely",
            "completely",
            "totally",
            "utterly",
            "fairly",
            "pretty",
            "a bit",
            "a little",
        ],
    )
    for intensifier in intensifiers:
        # Handle multi-word intensifiers like "a bit", "a little"
        if " " in intensifier:
            # Add each word separately
            for word in intensifier.split():
                stopwords_set.add(word)
        else:
            stopwords_set.add(intensifier)
        if config.get("verbose", False):
            print(f"  Added intensifier '{intensifier}' to stopwords")

    return stopwords_set


def is_negative_stopword(
    word: str, stopwords_set: Set[str], config: Dict[str, Any]
) -> bool:
    """
    Check if a word is a negative form of a stopword for scientific filtering.

    Args:
        word: Word to check (may be a negation compound like "not_very_can").
        stopwords_set: Set of stopwords for comparison.
        config: Configuration dictionary with negation parameters.

    Returns:
        True if the word should be removed (is a negative form of a stopword).

    Notes:
        - Negative forms have structure: [negation_term]_[intensifiers?]_[target_word(s)]
        - Handles both heuristic and spaCy negation terms
        - Removes negative forms of stopwords to clean frequency counts
        - Scientific rationale: negative stopwords don't contribute meaningful semantic content
        - Example: "not_can" (negative form of "can") should be filtered
    """
    # Get all negation terms from config (both heuristic and spaCy versions)
    negation_terms = set()

    # Heuristic negation terms from DEFAULT_CONFIG
    heuristic_terms = config.get(
        "negation_terms",
        [
            "not",
            "no",
            "never",
            "none",
            "nothing",
            "nobody",
            "nowhere",
            "neither",
            "nor",
        ],
    )
    negation_terms.update(heuristic_terms)

    # spaCy negation terms (default includes "n't" which becomes "not")
    spacy_terms = config.get("spacy_negation_terms", [])
    if not spacy_terms:
        # Default spaCy negation terms
        spacy_terms = {"not", "n't", "no", "never", "seldom", "rarely", "without"}
    negation_terms.update(spacy_terms)

    # Get intensifiers
    intensifiers = set(
        config.get(
            "negation_intensifiers",
            [
                "very",
                "really",
                "quite",
                "rather",
                "somewhat",
                "slightly",
                "too",
                "so",
                "extremely",
                "incredibly",
                "absolutely",
                "completely",
                "totally",
                "utterly",
                "fairly",
                "pretty",
                "a bit",
                "a little",
                "kind of",
                "sort of",
            ],
        )
    )

    # Split word by underscores and filter out empty strings
    parts = [part for part in word.split("_") if part]

    # Check if first part is a negation term
    if len(parts) < 2 or parts[0] not in negation_terms:
        return False

    # Remove negation terms and intensifiers from the beginning
    remaining_parts = []
    i = 0
    while i < len(parts):
        if i == 0 and parts[i] in negation_terms:
            # Skip initial negation term
            i += 1
            continue
        elif parts[i] in intensifiers:
            # Skip intensifiers
            i += 1
            continue
        else:
            # Keep this and all following parts
            remaining_parts = parts[i:]
            break

    # If no parts left, it's just a negation term (should already be removed)
    if not remaining_parts:
        return False

    # Check different possibilities:
    # 1. The joined remaining parts might be a stopword (e.g., "can" from "not_can")
    joined_remaining = "_".join(remaining_parts)
    if joined_remaining in stopwords_set:
        return True

    # 2. Individual parts might be stopwords (e.g., "can" from "not_very_can")
    for part in remaining_parts:
        if part in stopwords_set:
            return True

    return False


# ============================================================================
# Word Variation Grouping
# ============================================================================


def are_words_similar(word1: str, word2: str, config: Dict[str, Any]) -> bool:
    """
    Check if two words are similar enough to be grouped using dynamic thresholds.

    Args:
        word1: First word to compare.
        word2: Second word to compare.
        config: Configuration dictionary with similarity parameters.

    Returns:
        True if words are similar enough to be grouped.

    Notes:
        - Uses dynamic thresholds based on word length for scientific accuracy
        - Short words (≤3 chars): strict matching (max_dist=0, similarity≥0.95)
        - Medium words (4-5 chars): moderate matching (max_dist=1, similarity≥0.85)
        - Long words (>5 chars): configurable matching (default: max_dist=2, similarity≥0.8)
        - Scientific rationale: accounts for typing errors while preserving distinct meanings
        - Additional checks prevent grouping of semantically distinct short words (e.g., 'do'/'go')
    """
    len1, len2 = len(word1), len(word2)
    min_len = min(len1, len2)

    # Use dynamic thresholds based on word length
    if min_len <= 3:
        max_dist = 0
        min_similarity = 0.95
    elif min_len <= 5:
        max_dist = 1
        min_similarity = 0.85
    else:
        max_dist = config.get("max_edit_distance", 2)
        min_similarity = config.get("similarity_threshold", 0.8)

    # 1. Check edit distance
    dist = edit_distance(word1, word2)
    if dist > max_dist:
        return False

    # 2. Check sequence similarity
    # This helps distinguish between words like 'feel'/'free' (low similarity)
    # and 'test'/'text' (higher similarity)
    similarity = difflib.SequenceMatcher(None, word1, word2).ratio()
    if similarity < min_similarity:
        return False

    # 3. Additional check for short words to avoid 'do'/'go'
    if min_len <= 3 and dist > 0:
        # For very short words, require a very high similarity ratio
        if similarity < 0.9:
            return False

    return True


def group_word_variations(
    word_counts: Dict[str, int], config: Dict[str, Any]
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Group word variations for scientific text analysis.

    Args:
        word_counts: Dictionary of word frequencies.
        config: Configuration dictionary with grouping parameters.

    Returns:
        Dictionary: {main_word: [(variant, count), ...]}

    Notes:
        - Groups typos, spelling variations, and morphological forms
        - Uses frequency-based ordering (most frequent word becomes group key)
        - Applies lemmatization for morphological grouping when enabled
        - Scientific rationale: accounts for natural language variation in survey responses
        - Preserves methodological transparency by showing all grouped variants
    """
    if not config.get("group_variations", True):
        return {word: [(word, count)] for word, count in word_counts.items()}

    print("Grouping word variations...")

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    lemmatizer = None
    if config.get("use_lemmatization", True):
        try:
            lemmatizer = WordNetLemmatizer()
        except Exception:
            print("  Warning: Could not initialize lemmatizer")
            lemmatizer = None

    groups = {}
    processed_words = set()

    for word, count in sorted_words:
        if word in processed_words:
            continue

        group_members = [(word, count)]
        processed_words.add(word)

        for other_word, other_count in sorted_words:
            if other_word in processed_words:
                continue

            is_similar = False

            # 1. Check with dynamic similarity logic
            if are_words_similar(word, other_word, config):
                is_similar = True

            # 2. Lemmatization (for morphological variations)
            elif lemmatizer:
                try:
                    if lemmatizer.lemmatize(word) == lemmatizer.lemmatize(other_word):
                        is_similar = True
                except Exception:
                    pass

            if is_similar:
                group_members.append((other_word, other_count))
                processed_words.add(other_word)

        if group_members:
            group_members.sort(key=lambda x: x[1], reverse=True)
            groups[group_members[0][0]] = group_members

    print(f"  Created {len(groups)} word groups from {len(word_counts)} unique words")
    return groups


def create_tree_visualization(
    groups: Dict[str, List[Tuple[str, int]]], config: Dict[str, Any]
) -> str:
    """
    Create ASCII tree visualization of word groups for scientific reporting.

    Args:
        groups: Dictionary of word groups from group_word_variations().
        config: Configuration dictionary with visualization parameters.

    Returns:
        ASCII tree string showing word group hierarchies.

    Notes:
        - Shows frequency percentages for methodological transparency
        - Configurable depth limit (tree_max_depth) for readability
        - Groups with single members shown without tree structure
        - Scientific rationale: visual representation of word variation patterns
        - Useful for identifying common spelling variations and morphological forms
    """
    print("Creating tree visualization...")

    tree_lines = []
    tree_lines.append("Word Frequency Tree")
    tree_lines.append("=" * 50)
    tree_lines.append("")

    # Sort groups by total frequency
    sorted_groups = sorted(
        groups.items(), key=lambda x: sum(count for _, count in x[1]), reverse=True
    )

    max_depth = config.get("tree_max_depth", 3)

    for main_word, members in sorted_groups:
        total_count = sum(count for _, count in members)

        # Check if this group has only one member (no variations)
        if len(members) == 1:
            # Single form: just show word (count) without tree structure
            tree_lines.append(f"{main_word} ({total_count})")
        else:
            # Multiple forms: create tree structure
            # Main word line (shows total count for the group)
            tree_lines.append(f"{main_word} ({total_count})")

            # Display all members including the main word as branches
            # We need to handle depth limits carefully
            if isinstance(max_depth, int):
                display_members = members[:max_depth]  # Show up to max_depth members
            else:
                display_members = members

            remaining_count = len(members) - len(display_members)

            for i, (word, count) in enumerate(display_members):
                # Calculate percentage of total group frequency
                percentage = (count / total_count) * 100 if total_count > 0 else 0

                # Determine the correct prefix for the tree structure
                # Check if this is the last displayed member (either last in group or at max_depth)
                is_last_displayed = i == len(display_members) - 1

                if is_last_displayed and remaining_count == 0:
                    prefix = "└── "
                else:
                    prefix = "├── "

                tree_lines.append(f"    {prefix}{word} ({count}, {percentage:.1f}%)")

            # If there are more members than max_depth, show a summary line
            if remaining_count > 0:
                tree_lines.append(f"    └── ... and {remaining_count} more variations")

    return "\n".join(
        tree_lines
    )  # ============================================================================


# Output Functions
# ============================================================================


def save_word_outputs(
    word_counts: Dict[str, int], output_path_base: str, file_type: str
) -> None:
    """
    Save word counts to CSV and word lists to TXT files for scientific analysis.

    Args:
        word_counts: Dictionary of word frequencies.
        output_path_base: Base path for output files.
        file_type: Type of analysis (e.g., 'raw', 'filtered').

    Notes:
        - Saves two file types: CSV with frequencies, TXT with word lists
        - Provides detailed statistics for methodological reporting
        - Handles encoding issues for international text data
        - Scientific rationale: separate files for different analysis stages
        - Statistics include: unique words, total occurrences, average frequency, top words
    """
    if not word_counts:
        print(f"⚠️  No word counts to save for type '{file_type}'")
        return

    try:
        sorted_counts = sorted(
            word_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )

        # --- Save CSV File ---
        csv_output_path = f"{output_path_base}_{file_type}.csv"
        df = pd.DataFrame(sorted_counts, columns=["word", "count"])
        try:
            df.to_csv(csv_output_path, index=False, encoding="utf-8")
            print(f"✓ Saved word counts to {csv_output_path}")
        except Exception as e:
            print(f"✗ Error saving CSV to {csv_output_path}: {e}")

        # --- Save TXT File (with "list" instead of "frequency") ---
        txt_output_path_base = output_path_base.replace("frequencies", "list")
        txt_output_path = f"{txt_output_path_base}_{file_type}.txt"
        try:
            with open(txt_output_path, "w", encoding="utf-8") as f:
                for word, _ in sorted_counts:
                    f.write(f"{word}\n")
            print(f"✓ Saved words to {txt_output_path}")
        except Exception as e:
            print(f"✗ Error saving TXT to {txt_output_path}: {e}")

        # --- Statistics Reporting ---
        total_words = len(word_counts)
        total_occurrences = sum(word_counts.values())
        avg_frequency = total_occurrences / total_words if total_words > 0 else 0
        top_n = min(5, len(sorted_counts))
        top_words = [word for word, _ in sorted_counts[:top_n]]

        print(f"  Statistics for '{file_type}':")
        print(f"    Total unique words: {total_words:,}")
        print(f"    Total occurrences: {total_occurrences:,}")
        print(f"    Average frequency: {avg_frequency:.2f}")
        print(f"    Top {top_n} words: {', '.join(top_words)}")

    except Exception as e:
        print(f"✗ Error processing word counts for saving: {e}")


def save_grouped_counts(
    groups: Dict[str, List[Tuple[str, int]]], output_path: str
) -> None:
    """
    Save grouped word counts to CSV file for detailed scientific analysis.

    Args:
        groups: Dictionary of word groups from group_word_variations().
        output_path: Path for the output CSV file.

    Notes:
        - Preserves hierarchical structure of word groups
        - Includes metadata: group key, word, count, total group count, variation rank
        - Scientific rationale: enables detailed analysis of word variation patterns
        - Useful for understanding morphological relationships and spelling variations
    """
    rows = []

    # Sort groups by total frequency
    sorted_groups = sorted(
        groups.items(), key=lambda x: sum(count for _, count in x[1]), reverse=True
    )

    for main_word, members in sorted_groups:
        total_count = sum(count for _, count in members)

        # Add main word row
        rows.append(
            {
                "group_key": main_word,
                "word": main_word,
                "count": members[0][1],  # Count of main word
                "total_group_count": total_count,
                "is_main": True,
                "variation_rank": 1,
            }
        )

        # Add variation rows
        for i, (word, count) in enumerate(members[1:], 2):
            rows.append(
                {
                    "group_key": main_word,
                    "word": word,
                    "count": count,
                    "total_group_count": total_count,
                    "is_main": False,
                    "variation_rank": i,
                }
            )

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved grouped word counts to {output_path}")
    print(f"  Total groups: {len(groups)}")
    print(f"  Total words: {len(df)}")


# ============================================================================
# Configuration Management
# ============================================================================


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file, merging with defaults for scientific reproducibility.

    Args:
        config_file: Optional path to JSON configuration file.

    Returns:
        Configuration dictionary with merged settings.

    Notes:
        - Priority: command line arguments > config file > defaults
        - Provides methodological transparency through configuration logging
        - Graceful fallback to defaults if config file has errors
        - Scientific rationale: enables reproducible analysis with documented parameters
    """
    config = DEFAULT_CONFIG.copy()

    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                file_config = json.load(f)

            # Merge file config with defaults
            config.update(file_config)
            print(f"Loaded configuration from {config_file}")

        except Exception as e:
            print(f"✗ Error loading config file {config_file}: {e}")
            print("Using default configuration.")

    return config


def update_config_from_args(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Update configuration from command line arguments for flexible scientific analysis.

    Args:
        config: Current configuration dictionary.
        args: Parsed command line arguments.

    Returns:
        Updated configuration dictionary.

    Notes:
        - Command line arguments have highest priority in configuration hierarchy
        - Supports multiple input formats for text columns (JSON, comma-separated, space-separated)
        - Enables fine-grained control over analysis parameters for different research contexts
        - Scientific rationale: allows methodological adaptation without code modification
    """
    if args.text_columns:
        # Parse text columns with multiple input formats
        text_columns_str = args.text_columns.strip()

        # Try to parse as JSON first
        if text_columns_str.startswith("[") and text_columns_str.endswith("]"):
            try:
                config["text_columns"] = json.loads(text_columns_str)
                print(f"Parsed text columns as JSON: {config['text_columns']}")
            except json.JSONDecodeError:
                # Fall back to comma/split parsing
                pass

        # If not JSON or JSON parsing failed, try comma or space separation
        if "text_columns" not in config or not config["text_columns"]:
            # Try comma separation first
            if "," in text_columns_str:
                columns = [col.strip() for col in text_columns_str.split(",")]
            else:
                # Try space separation
                columns = [col.strip() for col in text_columns_str.split()]

            config["text_columns"] = columns
            print(f"Parsed text columns: {config['text_columns']}")

    if args.output_prefix:
        config["output_prefix"] = args.output_prefix

    # File format override
    if args.format:
        config["format_override"] = args.format
        print(f"Using specified format: {args.format} (bypassing automatic detection)")

    if args.min_word_length is not None:
        config["min_word_length"] = args.min_word_length

    if args.no_lowercase:
        config["lowercase"] = False

    if args.keep_punctuation:
        config["remove_punctuation"] = False

    if args.keep_numbers:
        config["remove_numbers"] = False

    if args.no_stopwords:
        config["remove_stopwords"] = False

    if args.stopword_language:
        config["stopword_language"] = args.stopword_language

    if args.custom_stopwords:
        config["custom_stopwords"] = args.custom_stopwords

    # Enhanced negation handling
    if args.no_negations:
        config["handle_negations"] = False

    if args.negation_terms:
        config["negation_terms"] = args.negation_terms

    if args.negation_intensifiers:
        config["negation_intensifiers"] = args.negation_intensifiers

    if args.negation_max_distance is not None:
        config["negation_max_distance"] = args.negation_max_distance

    if args.no_negation_skip_stopwords:
        config["negation_skip_stopwords"] = False

    # spaCy-based negation handling
    if args.no_spacy:
        config["use_spacy_negation"] = False

    if args.spacy_model:
        config["spacy_model"] = args.spacy_model

    if args.spacy_negation_terms:
        config["spacy_negation_terms"] = args.spacy_negation_terms

    if args.no_grouping:
        config["group_variations"] = False

    if args.max_edit_distance is not None:
        config["max_edit_distance"] = args.max_edit_distance

    if args.no_lemmatization:
        config["use_lemmatization"] = False

    if args.similarity_threshold is not None:
        config["similarity_threshold"] = args.similarity_threshold

    if args.tree_max_depth is not None:
        config["tree_max_depth"] = args.tree_max_depth

    # Output options
    if args.no_raw_counts:
        config["save_raw_counts"] = False

    if args.no_filtered_counts:
        config["save_filtered_counts"] = False

    if args.no_grouped_counts:
        config["save_grouped_counts"] = False

    if args.no_tree:
        config["save_tree_visualization"] = False

    # Debugging
    if args.verbose:
        config["verbose"] = True

    return config


# ============================================================================
# Main Function
# ============================================================================


def main() -> None:
    """
    Main function to run the scientific word extraction and analysis pipeline.

    Notes:
        - Implements two-pass processing: raw extraction and advanced linguistic analysis
        - Handles graceful interruption with progress saving
        - Provides comprehensive logging for methodological transparency
        - Generates multiple output formats for different analysis needs
        - Scientific rationale: separates baseline analysis from linguistically-informed processing

    Processing Pipeline:
        1. Configuration loading and validation
        2. NLTK and spaCy setup for linguistic analysis
        3. File loading and text extraction with auto-detection
        4. Two-pass word extraction: raw and processed
        5. Stopword removal and negation handling
        6. Word variation grouping for natural language analysis
        7. Output generation with detailed statistics
    """
    global INTERRUPTED, CURRENT_FILE, PROCESSED_FILES

    # Set up signal handling for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Register cleanup function
    atexit.register(cleanup)

    parser = argparse.ArgumentParser(
        description="Extract and analyze word frequencies from data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv
  %(prog)s data1.csv data2.txt data3.xlsx
  %(prog)s data.csv --text-columns "description,comments"
  %(prog)s data.csv --config my_config.json
  %(prog)s data.csv --no-stopwords --no-grouping
  %(prog)s data.csv --output-prefix "my_analysis"
  %(prog)s data.csv --min-word-length 3 --max-edit-distance 1
  %(prog)s data.csv --no-spacy  # Disable spaCy and use heuristic negation detection
  %(prog)s data.txt --format text  # Force text format (bypass detection)
  %(prog)s data.unknown --format csv  # Force CSV format for unknown file extension
        """,
    )

    # Input files
    parser.add_argument(
        "files",
        nargs="+",
        help="Data files to process (CSV, TSV, TXT, XLS, XLSX, JSON, etc.)",
    )

    # Configuration
    parser.add_argument("--config", type=str, help="JSON configuration file")

    # File format override
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "text", "excel", "json", "parquet", "feather", "hdf5"],
        help="Force specific file format (bypasses automatic detection). Options: csv, text, excel, json, parquet, feather, hdf5",
    )

    # Text column selection
    parser.add_argument(
        "--text-columns",
        type=str,
        help="Columns containing text data. Can be: 1) Comma-separated list, 2) Space-separated list, 3) JSON array. Examples: 'description,comments' or 'description comments' or '[\"description\", \"comments\"]'",
    )

    # Output options
    parser.add_argument(
        "--output-prefix",
        type=str,
        help="Prefix for output files (default: word_frequencies)",
    )

    # Text processing options
    parser.add_argument(
        "--min-word-length", type=int, help="Minimum word length (default: 2)"
    )
    parser.add_argument(
        "--no-lowercase", action="store_true", help="Don't convert text to lowercase"
    )
    parser.add_argument(
        "--keep-punctuation", action="store_true", help="Keep punctuation in words"
    )
    parser.add_argument(
        "--keep-numbers", action="store_true", help="Keep numbers as words"
    )

    # Stopword options
    parser.add_argument(
        "--no-stopwords", action="store_true", help="Don't remove stopwords"
    )
    parser.add_argument(
        "--stopword-language",
        type=str,
        default="english",
        help="Stopword language (default: english)",
    )
    parser.add_argument(
        "--custom-stopwords", nargs="+", help="Additional stopwords to remove"
    )

    # Enhanced negation handling options
    parser.add_argument(
        "--no-negations", action="store_true", help="Don't handle negations"
    )
    parser.add_argument(
        "--negation-terms",
        nargs="+",
        help="Negation terms to recognize (default: not, no, never, none, nothing, nobody, nowhere, neither, nor)",
    )
    parser.add_argument(
        "--negation-intensifiers",
        nargs="+",
        help="Intensifiers that can appear between negation and target word",
    )
    parser.add_argument(
        "--negation-max-distance",
        type=int,
        help="Maximum number of words between negation and target (default: 3)",
    )
    parser.add_argument(
        "--no-negation-skip-stopwords",
        action="store_true",
        help="Don't skip stopwords when looking for target word after negation",
    )

    # spaCy-based negation handling options
    parser.add_argument(
        "--no-spacy",
        action="store_true",
        help="Disable spaCy and use heuristic negation detection instead",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to use (default: en_core_web_sm)",
    )
    parser.add_argument(
        "--spacy-negation-terms",
        nargs="+",
        help="Custom negation terms for spaCy (overrides default spaCy terms)",
    )

    # Word grouping options
    parser.add_argument(
        "--no-grouping", action="store_true", help="Don't group word variations"
    )
    parser.add_argument(
        "--max-edit-distance",
        type=int,
        help="Maximum edit distance for typo detection (default: 2)",
    )
    parser.add_argument(
        "--no-lemmatization",
        action="store_true",
        help="Don't use lemmatization for grouping",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Similarity threshold for prefix matching (default: 0.8)",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        help="Maximum depth for tree visualization (default: 3)",
    )

    # Output file options
    parser.add_argument(
        "--no-raw-counts", action="store_true", help="Don't save raw word counts"
    )
    parser.add_argument(
        "--no-filtered-counts",
        action="store_true",
        help="Don't save filtered word counts (without stopwords)",
    )
    parser.add_argument(
        "--no-grouped-counts",
        action="store_true",
        help="Don't save grouped word counts",
    )
    parser.add_argument(
        "--no-tree", action="store_true", help="Don't save tree visualization"
    )

    # Debugging options
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("All Words Extractor")
    print("=" * 60)

    # Load and update configuration
    config = load_config(args.config)
    config = update_config_from_args(config, args)

    # Setup NLTK data
    setup_nltk_data(config)

    # Setup spaCy model (default behavior)
    if config.get("use_spacy_negation", True):  # Default is True
        nlp = load_spacy_model(config)
        if nlp is not None:
            config["_spacy_model"] = nlp
            print("✓ spaCy model loaded and ready for enhanced negation detection")
        else:
            print(
                "⚠️  spaCy model not available, falling back to heuristic negation detection"
            )
            config["use_spacy_negation"] = False
    else:
        print("ℹ️  Using heuristic negation detection (spaCy explicitly disabled)")

    # Process all files
    all_texts = []

    for file_idx, filepath in enumerate(args.files):
        if INTERRUPTED:
            print("\n⚠️  Processing interrupted by user.")
            break

        CURRENT_FILE = filepath

        print(f"\nProcessing file {file_idx + 1}/{len(args.files)}: {filepath}")

        if not os.path.exists(filepath):
            print(f"✗ File not found: {filepath}")
            continue

        try:
            df = load_data_file(filepath, config)
            texts = extract_text_from_dataframe(df, config)
            all_texts.extend(texts)

            # Record successful processing
            PROCESSED_FILES.append(
                {
                    "filepath": filepath,
                    "rows_loaded": len(df),
                    "text_entries_extracted": len(texts),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            print(f"  ✓ Extracted {len(texts)} text entries from {filepath}")

        except Exception as e:
            print(f"✗ Error processing {filepath}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_texts:
        print("✗ No text data found in any files.")
        return

    print(f"\nTotal text entries: {len(all_texts)}")

    # Process text - TWO PASSES:
    # 1. First pass: Extract raw words (simple splitting, no spaCy)
    # 2. Second pass: Process text with spaCy (if available) for advanced processing

    print("\nExtracting raw words (simple splitting)...")
    raw_words = []
    processed_words = []

    for i, text in enumerate(all_texts):
        if INTERRUPTED:
            print("\n⚠️  Text processing interrupted by user.")
            break

        # First pass: Extract raw words (no spaCy, no advanced processing)
        raw_words_batch = extract_raw_words(text, config)
        raw_words.extend(raw_words_batch)

        # Second pass: Process text with spaCy (if available) for advanced processing
        processed_words_batch = preprocess_text(text, config)
        processed_words.extend(processed_words_batch)

        # Show progress for large datasets
        if len(all_texts) > 1000 and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(all_texts)} entries")

            # Check for interruption during long processing
            if INTERRUPTED:
                print("\n⚠️  Text processing interrupted by user.")
                break

    print(f"Extracted {len(raw_words)} raw words (simple splitting)")
    print(f"Extracted {len(processed_words)} processed words (with spaCy if available)")

    # Check if we extracted any words
    if len(raw_words) == 0 and len(processed_words) == 0:
        print("⚠️  No words were extracted from the text.")
        print("   Possible reasons:")
        print("   1. Text columns might be empty or contain no valid words")
        print("   2. Minimum word length setting might be too high")
        print("   3. Text preprocessing might be filtering out all words")
        print("   4. Input files might not contain text data")
        return

    # Count raw word frequencies (for the raw output file)
    raw_word_counts = Counter(raw_words)
    print(f"Unique raw words: {len(raw_word_counts)}")

    # Count processed word frequencies (for the rest of the pipeline)
    processed_word_counts = Counter(processed_words)
    print(f"Unique processed words: {len(processed_word_counts)}")

    # Check if we have any word counts
    if not raw_word_counts and not processed_word_counts:
        print("✗ No word counts generated. Cannot proceed with analysis.")
        return

    # Save raw counts if requested - using ONLY raw words (no spaCy processing)
    if config.get("save_raw_counts", True) and raw_word_counts:
        save_word_outputs(raw_word_counts, config["output_prefix"], "raw")
    elif config.get("save_raw_counts", True):
        print("⚠️  Could not save raw counts - no raw words extracted")

    # Remove stopwords if requested - using PROCESSED words (with spaCy if available)
    filtered_word_counts = processed_word_counts.copy()
    if config.get("remove_stopwords", True):
        print("\nRemoving stopwords from processed words...")
        stopwords_set = get_stopwords(config)

        # Filter out stopwords from processed words, including negative forms of stopwords
        filtered_word_counts = {
            word: count
            for word, count in processed_word_counts.items()
            if word not in stopwords_set
            and not is_negative_stopword(word, stopwords_set, config)
        }

        removed = len(processed_word_counts) - len(filtered_word_counts)
        print(f"  Removed {removed} stopwords (including negative forms)")
        print(f"  Remaining words: {len(filtered_word_counts)}")

    # Group word variations
    groups = group_word_variations(filtered_word_counts, config)

    # Create aggregated counts after grouping
    # This sums frequencies for words in the same group (e.g., "feel" and "felt" both count toward "feel")
    aggregated_filtered_counts = {}
    for main_word, members in groups.items():
        # Sum all counts in the group
        total_count = sum(count for _, count in members)
        aggregated_filtered_counts[main_word] = total_count

    # Save aggregated filtered counts if requested
    if config.get("save_filtered_counts", True) and aggregated_filtered_counts:
        save_word_outputs(
            aggregated_filtered_counts, config["output_prefix"], "filtered"
        )
    elif config.get("save_filtered_counts", True):
        print("⚠️  Could not save filtered counts - no aggregated counts after grouping")

    # Save grouped counts if requested
    if config.get("save_grouped_counts", True) and groups:
        output_path = f"{config['output_prefix']}_grouped.csv"
        save_grouped_counts(groups, output_path)

    # Create and save tree visualization if requested
    if config.get("save_tree_visualization", True) and groups:
        tree_text = create_tree_visualization(groups, config)
        output_path = f"{config['output_prefix']}_tree.txt"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(tree_text)

        print(f"\nSaved tree visualization to {output_path}")
        print("\n" + "=" * 60)
        print("Tree Visualization Preview:")
        print("=" * 60)

        # Show first 20 lines of tree
        tree_lines = tree_text.split("\n")
        for line in tree_lines[:20]:
            print(line)

        if len(tree_lines) > 20:
            print("... (see full tree in output file)")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Processed {len(args.files)} file(s)")
    print(f"Total raw words (simple splitting): {len(raw_words)}")
    print(f"Total processed words (with spaCy): {len(processed_words)}")
    print(f"Unique raw words: {len(raw_word_counts)}")
    print(f"Unique processed words: {len(processed_word_counts)}")

    if config.get("remove_stopwords", True):
        print(
            f"Words after stopword removal (before grouping): {len(filtered_word_counts)}"
        )
        if aggregated_filtered_counts:
            print(
                f"Words after grouping and aggregation: {len(aggregated_filtered_counts)}"
            )
            total_aggregated = sum(aggregated_filtered_counts.values())
            print(f"Total occurrences after aggregation: {total_aggregated}")

    if config.get("group_variations", True):
        print(f"Word groups created: {len(groups)}")

    print("\nOutput files created:")
    output_dir = os.path.dirname(os.path.abspath(args.files[0])) if args.files else "."
    for filename in os.listdir(output_dir):
        if filename.startswith(config["output_prefix"]):
            print(f"  {filename}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
