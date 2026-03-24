# Analysis Record

### Aim

- Descriptive statistics about the vocabulary used by participants.
- Create lists of words sorted by frequency to construct a bottom-up thematic analysis


### Command line

Executed from the root of the directory:

```bash
python3 Analysis/Scripts/all_words_extractor.py 'Processed Data/touch_data_fixed.psv.txt' --count-participants --format csv --text-columns "Social_context"
```


### Execution logs

```text
============================================================
All Words Extractor
============================================================
Parsed text columns: ['Social_context']
Using specified format: csv (bypassing automatic detection)
Checking NLTK data availability...
✓ punkt is available
✓ punkt_tab is available
✓ stopwords is available
Downloading wordnet...
[nltk_data] Downloading package wordnet to ${HOME}/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
✓ wordnet downloaded successfully
Loading spaCy model for enhanced negation detection...
✓ Loaded spaCy model: en_core_web_sm
✓ spaCy model loaded and ready for enhanced negation detection

Processing file 1/1: Processed Data/touch_data_fixed.psv.txt
Loading file: Processed Data/touch_data_fixed.psv.txt
  Using specified format: csv (bypassing detection)
  Detected delimiter ''|'' with 23 columns
  Detected format: csv with params: {'delimiter': '|'}
  Loaded 2837 rows, 23 columns
  Columns: ['ResponseID', 'Touch No.', 'Social_self', 'Social_body', 'Social_place', 'Social_context', 'Intention&Purpose', 'Appropriateness', 'Sensory', 'Valence&Arousal_x', 'Valence&Arousal_y', 'Emotional_self', 'Emotional_touch', 'Continue', 'Input', 'Valence', 'Arousal', 'Type', 'Contact', 'Direction', 'Speed (cm/s)', 'Force', 'Touch_desc']
✓ All specified columns found: ['Social_context']

  Final text columns to process: ['Social_context']
  Total rows in dataframe: 2837

  Analyzing text from column 1/1: 'Social_context'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 2828/2837 (0.3% empty)
    Average text length: 49.2 characters

  Extraction summary:
    Successfully extracted from 1/1 columns
    Total text entries: 2837
      ✓ Social_context: 2828 non-empty, 9 empty (primary (fillna + astype(str)))

  Removed 9 completely empty entries (0.3% of total)

  Final text count: 2828 non-empty text entries
  ✓ Extracted 2828 text entries from Processed Data/touch_data_fixed.psv.txt

Total text entries: 2828

Extracting words...
  Processed 1000/2828 entries
  Processed 2000/2828 entries
    ⚠️  Text filtered to empty: 'N/a...' -> spaCy tokens: ['n']
Extracted 25488 raw words (simple splitting)
Extracted 2828 raw words after deduplication per document
Extracted 13566 processed words (with spaCy if available)
Extracted 2828 processed words after deduplication per document
Unique raw words: 2596
Unique processed words: 2419

Calculating participant (document) frequencies...
✓ Saved word counts to word_frequencies_raw.csv
✓ Saved words to word_list_raw.txt
  Statistics for 'raw':
    Total unique words: 2,596
    Total occurrences: 23,989
    Average frequency: 9.24
    Top 5 words: my, is, me, to, and

Removing stopwords from processed words...
  Removed 40 stopwords (including negative forms)
  Remaining words: 2379
Grouping word variations...
  Created 1817 word groups from 2379 unique words
✓ Saved word counts to word_frequencies_filtered.csv
✓ Saved words to word_list_filtered.txt
  Statistics for 'filtered':
    Total unique words: 1,817
    Total occurrences: 11,021
    Average frequency: 6.07
    Top 5 words: watching, trying, touching, touches, attention
Saved grouped word counts to word_frequencies_grouped.csv
  Total groups: 1817
  Total words: 2379
✓ Saved grouping dictionary to word_grouping_dict.json
  Total groups: 1817
  Total variants: 2379
  Average variants per group: 1.31

  Example groups (first 5):
    1. 'watching' -> ['watching', 'catching', 'washing', 'whatching', 'wathcing', 'wathing', 'wating', 'aching', 'itching']
    2. 'trying' -> ['trying', 'crying', 'typing', 'triyng', 'tryng', 'tying']
    3. 'touching' -> ['touching', 'touuching', 'tuching']
    4. 'touches' -> ['touches', 'touch', 'touched', 'toutches', 'touhches', 'touchs']
    5. 'attention' -> ['attention', 'attencion']
Creating tree visualization...

Saved tree visualization to word_frequencies_tree.txt

============================================================
Tree Visualization Preview:
============================================================

Word Frequency Tree
==================================================

watching (239)
    ├── watching (224, 93.7%)
    ├── catching (7, 2.9%)
    ├── washing (2, 0.8%)
    ├── whatching (1, 0.4%)
    ├── wathcing (1, 0.4%)
    ├── wathing (1, 0.4%)
    ├── wating (1, 0.4%)
    ├── aching (1, 0.4%)
    └── itching (1, 0.4%)
trying (235)
    ├── trying (210, 89.4%)
    ├── crying (21, 8.9%)
    ├── typing (1, 0.4%)
    ├── triyng (1, 0.4%)
    ├── tryng (1, 0.4%)
    └── tying (1, 0.4%)
... (see full tree in output file)

============================================================
Analysis Complete!
============================================================

Processed 1 file(s)
Total raw words (simple splitting): 25488
Total processed words (with spaCy): 13566
Unique raw words: 2596
Unique processed words: 2419
Words after stopword removal (before grouping): 2379
Words after grouping and aggregation: 1817
Total occurrences after aggregation: 11021
Word groups created: 1817

Output files created:

```

