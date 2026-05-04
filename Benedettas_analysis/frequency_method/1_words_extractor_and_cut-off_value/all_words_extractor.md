# Script 
The script used is called all_words_extractor_edited.py. This script is the modified version of all_words_extractor.py (found here: https://github.com/LHTMR/semi-controlled-touch-survey/tree/main/Analysis/Scripts), which was modified to read the edited version of the dictionary. 

# Input
 python all_words_extractor_edited.py social_context.txt --custom-dict word_grouping_dict_edited.json

# In terminal 
============================================================
All Words Extractor
============================================================
Checking NLTK data availability...
✓ punkt is available
✓ punkt_tab is available
✓ stopwords is available
Downloading wordnet...
[nltk_data] Downloading package wordnet to
[nltk_data]     C:\Users\bened\AppData\Roaming\nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
✓ wordnet downloaded successfully
Loading spaCy model for enhanced negation detection...
✗ spaCy is not installed. Falling back to heuristic negation detection.
  To use spaCy, install it with: pip install spacy
  Then download the model with: python -m spacy download en_core_web_sm
⚠️  spaCy model not available, falling back to heuristic negation detection

Processing file 1/1: social_context.txt
Loading file: social_context.txt
  Detected plain text format for .txt file
  Detected format: text with params: {}
  Loaded 1 rows, 1 columns
  Columns: ['text']
  Auto-detecting text columns...
    ✓ text: 100.0% strings, avg length 142630.0 chars (string dtype, 100.0% strings)

  Auto-detection summary:
    Analyzed 1 columns
    Detected 1 text columns: ['text']

  Final text columns to process: ['text']
  Total rows in dataframe: 1

  Extracting text from column 1/1: 'text'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 1/1 (0.0% empty)
    Average text length: 142630.0 characters

  Extraction summary:
    Successfully extracted from 1/1 columns
    Total text entries: 1
      ✓ text: 1 non-empty, 0 empty (primary (fillna + astype(str)))

  Final text count: 1 non-empty text entries
  ✓ Extracted 1 text entries from social_context.txt

Total text entries: 1

Extracting raw words (simple splitting)...
Extracted 25497 raw words (simple splitting)
Extracted 25297 processed words (with spaCy if available)
Unique raw words: 2596
Unique processed words: 2610
✓ Saved word counts to word_frequencies_raw.csv
✓ Saved words to word_list_raw.txt
  Statistics for 'raw':
    Total unique words: 2,596
    Total occurrences: 25,497
    Average frequency: 9.82
    Top 5 words: my, me, to, is, and

Removing stopwords from processed words...
  Removed 131 stopwords (including negative forms)
  Remaining words: 2479

Loading custom grouping dictionary from: word_grouping_dict_edited.json
  Loaded 3837 groups from custom dictionary
Grouping words using custom dictionary...
  Groups formed from custom dict: 1879
  Words not in custom dict (kept as-is): 163
✓ Saved word counts to word_frequencies_filtered.csv
✓ Saved words to word_list_filtered.txt
  Statistics for 'filtered':
    Total unique words: 2,042
    Total occurrences: 12,405
    Average frequency: 6.07
    Top 5 words: touching, watching, trying, arm, attention
Saved grouped word counts to word_frequencies_grouped.csv
  Total groups: 2042
  Total words: 2479
✓ Saved grouping dictionary to word_grouping_dict.json
  Total groups: 2042
  Total variants: 2479
  Average variants per group: 1.21

  Example groups (first 5):
    1. 'touching' -> ['touching', 'touuching', 'tuching']
    2. 'watching' -> ['watching', 'whatching', 'wathcing', 'wathing', 'wating']
    3. 'trying' -> ['trying', 'triyng', 'tryng', 'tying']
    4. 'arm' -> ['arm', 'arms']
    5. 'attention' -> ['attention', 'attencion']
Creating tree visualization...

Saved tree visualization to word_frequencies_tree.txt

============================================================
Tree Visualization Preview:
============================================================
Word Frequency Tree
==================================================

touching (226)
    ├── touching (224, 99.1%)
    ├── touuching (1, 0.4%)
    └── tuching (1, 0.4%)
watching (226)
    ├── watching (222, 98.2%)
    ├── whatching (1, 0.4%)
    ├── wathcing (1, 0.4%)
    ├── wathing (1, 0.4%)
    └── wating (1, 0.4%)
trying (218)
    ├── trying (215, 98.6%)
    ├── triyng (1, 0.5%)
    ├── tryng (1, 0.5%)
    └── tying (1, 0.5%)
arm (213)
    ├── arm (208, 97.7%)
... (see full tree in output file)

============================================================
Analysis Complete!
============================================================
Processed 1 file(s)
Total raw words (simple splitting): 25497
Total processed words (with spaCy): 25297
Unique raw words: 2596
Unique processed words: 2610
Words after stopword removal (before grouping): 2479
Words after grouping and aggregation: 2042
Total occurrences after aggregation: 12405
Word groups created: 2042

Output files created:
  word_frequencies_filtered.csv
  word_frequencies_grouped.csv
  word_frequencies_raw.csv
  word_frequencies_tree.txt
