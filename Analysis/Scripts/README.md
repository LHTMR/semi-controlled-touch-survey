# Analysis Scripts

This directory contains Python scripts designed for comprehensive text analysis, word extraction, and semantic disambiguation of the qualitative touch survey responses. 

## Scripts Overview

### wordnet_disambiguator.py

This Python script is a WordNet disambiguation tool that helps resolve word sense ambiguity by allowing users to select the correct WordNet synset definition for each word in a list. It can also extend synonyms with the closest words in WordNet using semantic similarity measures.

*   **Libraries Used:** It uses `nltk` for WordNet access and natural language processing, along with standard Python libraries for argument parsing, JSON, CSV, and file operations.
*   **Features:**
    *   **Word Sense Disambiguation:** Presents WordNet synsets and definitions for each word, allowing users to select the appropriate sense.
    *   **Synonym Extension:** Optionally extends synonyms with the closest words in WordNet using semantic similarity measures (Wu-Palmer, path similarity, or average of both).
    *   **Robust Interruption Handling:** Automatically saves partial results and supports resume functionality with validation.
    *   **Checkpointing:** Saves progress during time-consuming closest words calculations.
    *   **Safe Mode:** Provides maximum data protection with automatic backup of partial results.
*   **Input:**
    *   Can accept words via command-line arguments (`--words` flag) or from an input file (`--input-file`).
    *   Supports interactive mode when no arguments are provided.
*   **Data Processing:**
    *   For each word, retrieves all WordNet synsets (different senses/meanings).
    *   Presents definitions and examples to the user for selection.
    *   Optionally calculates semantic similarity to find closest words in WordNet.
    *   Handles interruptions gracefully with automatic saving of partial results.
*   **Output:**
    *   `wordnet_selections.json`: Simple mapping of words to selected synsets.
    *   `wordnet_selections_synonyms.json`: Extended mapping with synonyms and closest words.
    *   `wordnet_selections.csv`: CSV file with definitions and synonyms.
    *   `wordnet_selections_extended.csv`: CSV file with closest words (if `--extend-synonyms` used).
    *   `wordnet_selections.txt`: Simple list of synsets, one per line.
    *   Partial/interrupted results saved to `wordnet_selections_partial.json` and `closest_words_checkpoint.json`.
*   **Usage Examples:**
    *   `python wordnet_disambiguator.py --words "word1,word2,word3"`
    *   `python wordnet_disambiguator.py --input-file words.txt`
    *   `python wordnet_disambiguator.py --words "sensation" --extend-synonyms --num-closest 20`
    *   `python wordnet_disambiguator.py --words "apple" --extend-synonyms --similarity-method path`
    *   `python wordnet_disambiguator.py --words "word1,word2" --safe-mode`
    *   `python wordnet_disambiguator.py --resume wordnet_selections_partial.json --force-resume`


### all_words_extractor.py

This Python script is a comprehensive text analysis tool designed for extracting and analyzing word frequencies from the touch survey data files. It provides text processing capabilities specifically tailored for linguistic analysis in human-computer interaction research.

*   **Scientific Context:**
    *   Designed for analyzing semi-controlled touch survey responses
    *   Handles negation patterns common in subjective experience descriptions
    *   Groups word variations to account for spelling differences and typos
    *   Preserves linguistic constructs important for qualitative analysis
    *   Uses a two-pass processing approach for comprehensive analysis

*   **Libraries Used:** It uses `pandas` and `numpy` for data handling, `nltk` for tokenization, stopwords, and lemmatization, `spacy` for enhanced negation detection (default), and `difflib` for word similarity comparison.

*   **Features:**
    *   **Multiple File Format Support:** Supports CSV, TSV, TXT, XLS, XLSX, and other formats via pandas with automatic format detection
    *   **Advanced Negation Handling:** Uses spaCy-based negation detection for grammatical accuracy, with fallback to heuristic methods
    *   **Word Variation Grouping:** Accounts for natural language variation including typos, different spellings, and morphological variations
    *   **Configurable Processing:** JSON configuration file support with priority: command line > config file > defaults
    *   **Scientific Stopword Customization:** NLTK stopword removal with domain-specific customization
    *   **Enhanced Negation Handling:** Configurable negation terms and intensifiers with Option B approach (intensifiers removed from negation forms)
    *   **Multiple Output Formats:** CSV outputs and tree visualization for word variation patterns
    *   **Minimum Frequency Threshold:** The last version of the script now has a frequency threshold to remove from the output words (or groups of words) with low frequency. Turned off by default (min_freq=0).

*   **Input:**
    *   Can accept various file formats containing survey response data
    *   Supports automatic text column detection or manual specification via `--text-columns`
    *   Can use custom configuration files for specialized analysis needs

*   **Data Processing:**
    *   **Two-pass analysis:** Separate raw and processed outputs for methodological transparency
    *   **Raw extraction:** Simple word splitting for baseline frequencies
    *   **Advanced processing:** spaCy-based parsing for negation and compound words
    *   **Variation grouping:** Accounts for spelling variations and morphological forms using configurable similarity thresholds
    *   **Negation handling:** Preserves semantic meaning in subjective responses with configurable distance thresholds (mostly for the fallback analysis without spaCy)

*   **Output:**
    *   `word_frequencies_raw.csv.txt`: Baseline word frequencies without linguistic processing
    *   `word_frequencies_filtered.csv.txt`: Word frequencies after stopword removal and negation handling
    *   `word_frequencies_grouped.csv.txt`: Detailed grouping information showing word variations
    *   `word_frequencies_tree.txt`: Visual tree representation of word variation patterns
    *   `word_list_raw.txt`: Simple list of raw words
    *   `word_list_filtered.txt`: List of filtered words after processing
    *   `word_grouping_dict.json`: JSON dictionary with all the word groups detected by the script, as shown in `word_frequencies_tree.txt`. This file is the interface with human review.

*   **Usage Examples:**
    *   `python all_words_extractor.py survey_data.csv`
    *   `python all_words_extractor.py survey_data.csv --text-columns "response,comments"`
    *   `python all_words_extractor.py survey_data.csv --config my_config.json`
    *   `python all_words_extractor.py survey_data.csv --no-spacy`
    *   `python all_words_extractor.py survey_data.csv --similarity-threshold 0.85 --max-edit-distance 1`


---

Repository README: [README.md](../../README.md)
