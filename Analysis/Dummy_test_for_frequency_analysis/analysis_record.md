# Analysis Record

### Aim

- Test on a short crafter text that the frequency extraction works as intended


### Command line

Executed from the root of the directory:

```bash
python3 Analysis/Scripts/all_words_extractor.py Analysis/Dummy_test_for_frequency_analysis/test_text_frequency_extraction.txt
```

Then second pass after moving `not_go` out of the group `not_good` and into its own group:

```bash
python3 Analysis/Scripts/all_words_extractor.py Analysis/Dummy_test_for_frequency_analysis/test_text_frequency_extraction.txt --use-grouping-dict Analysis/Dummy_test_for_frequency_analysis/word_grouping_dict.edited.json
```


### Execution logs

First run:

```text
============================================================
All Words Extractor
============================================================
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

Processing file 1/1: Analysis/Dummy_test_for_frequency_analysis/test_text_frequency_extraction.txt
Loading file: Analysis/Dummy_test_for_frequency_analysis/test_text_frequency_extraction.txt
  Detected plain text format for .txt file
  Detected format: text with params: {}
  Loaded 1 rows, 1 columns
  Columns: ['text']
  Auto-detecting text columns...
    ✓ text: 100.0% strings, avg length 981.0 chars (string dtype, 100.0% strings)

  Auto-detection summary:
    Analyzed 1 columns
    Detected 1 text columns: ['text']

  Final text columns to process: ['text']
  Total rows in dataframe: 1

  Analyzing text from column 1/1: 'text'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 1/1 (0.0% empty)
    Average text length: 981.0 characters

  Extraction summary:
    Successfully extracted from 1/1 columns
    Total text entries: 1
      ✓ text: 1 non-empty, 0 empty (primary (fillna + astype(str)))

  Final text count: 1 non-empty text entries
  ✓ Extracted 1 text entries from Analysis/Dummy_test_for_frequency_analysis/test_text_frequency_extraction.txt

Total text entries: 1

Extracting words...
Extracted 164 raw words (simple splitting)
Extracted 1 raw words after deduplication per document
Extracted 95 processed words (with spaCy if available)
Extracted 1 processed words after deduplication per document
Unique raw words: 89
Unique processed words: 68
✓ Saved word counts to word_frequencies_raw.csv
✓ Saved words to word_list_raw.txt
  Statistics for 'raw':
    Total unique words: 89
    Total occurrences: 164
    Average frequency: 1.84
    Top 5 words: pain, not, and, is, painful

Removing stopwords from processed words...
  Removed 9 stopwords (including negative forms)
  Remaining words: 59
Grouping word variations...
  Created 50 word groups from 59 unique words
✓ Saved word counts to word_frequencies_filtered.csv
✓ Saved words to word_list_filtered.txt
  Statistics for 'filtered':
    Total unique words: 50
    Total occurrences: 76
    Average frequency: 1.52
    Top 5 words: no_pain, words, feel, not_painful, relevant
Saved grouped word counts to word_frequencies_grouped.csv
  Total groups: 50
  Total words: 59
✓ Saved grouping dictionary to word_grouping_dict.json
  Total groups: 50
  Total variants: 59
  Average variants per group: 1.18

  Example groups (first 5):
    1. 'no_pain' -> ['no_pain', 'not_pain']
    2. 'words' -> ['words', 'word']
    3. 'not_painful' -> ['not_painful']
    4. 'test' -> ['test']
    5. 'feel' -> ['feel']
Creating tree visualization...

Saved tree visualization to word_frequencies_tree.txt

============================================================
Tree Visualization Preview:
============================================================
Word Frequency Tree
==================================================

no_pain (7)
    ├── no_pain (4, 57.1%)
    └── not_pain (3, 42.9%)
words (4)
    ├── words (3, 75.0%)
    └── word (1, 25.0%)
not_painful (3)
test (3)
feel (3)
relevant (3)
    ├── relevant (1, 33.3%)
    ├── relevent (1, 33.3%)
    └── relevnt (1, 33.3%)
never_painful (2)
text (2)
felt (2)
english (2)
... (see full tree in output file)

============================================================
Analysis Complete!
============================================================
Processed 1 file(s)
Total raw words (simple splitting): 164
Total processed words (with spaCy): 95
Unique raw words: 89
Unique processed words: 68
Words after stopword removal (before grouping): 59
Words after grouping and aggregation: 50
Total occurrences after aggregation: 76
Word groups created: 50

Output files created:
  word_frequencies_tree.txt
  word_frequencies_filtered.csv.txt
  word_frequencies_grouped.csv.txt
  word_frequencies_raw.csv.txt
```


Second run:

```text
============================================================
All Words Extractor
============================================================
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

Processing file 1/1: Analysis/Dummy_test_for_frequency_analysis/test_text_frequency_extraction.txt
Loading file: Analysis/Dummy_test_for_frequency_analysis/test_text_frequency_extraction.txt
  Detected plain text format for .txt file
  Detected format: text with params: {}
  Loaded 1 rows, 1 columns
  Columns: ['text']
  Auto-detecting text columns...
    ✓ text: 100.0% strings, avg length 981.0 chars (string dtype, 100.0% strings)

  Auto-detection summary:
    Analyzed 1 columns
    Detected 1 text columns: ['text']

  Final text columns to process: ['text']
  Total rows in dataframe: 1

  Analyzing text from column 1/1: 'text'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 1/1 (0.0% empty)
    Average text length: 981.0 characters

  Extraction summary:
    Successfully extracted from 1/1 columns
    Total text entries: 1
      ✓ text: 1 non-empty, 0 empty (primary (fillna + astype(str)))

  Final text count: 1 non-empty text entries
  ✓ Extracted 1 text entries from Analysis/Dummy_test_for_frequency_analysis/test_text_frequency_extraction.txt

Total text entries: 1

Extracting words...
Extracted 164 raw words (simple splitting)
Extracted 1 raw words after deduplication per document
Extracted 95 processed words (with spaCy if available)
Extracted 1 processed words after deduplication per document
Unique raw words: 89
Unique processed words: 68
✓ Saved word counts to word_frequencies_raw.csv
✓ Saved words to word_list_raw.txt
  Statistics for 'raw':
    Total unique words: 89
    Total occurrences: 164
    Average frequency: 1.84
    Top 5 words: pain, not, and, is, painful

Removing stopwords from processed words...
  Removed 9 stopwords (including negative forms)
  Remaining words: 59
Loading grouping dictionary from Analysis/Dummy_test_for_frequency_analysis/word_grouping_dict.edited.json...
  Loaded 51 word groups
Applying existing grouping dictionary...
  Applied grouping to 59 words, creating 51 groups
✓ Saved word counts to word_frequencies_filtered.csv
✓ Saved words to word_list_filtered.txt
  Statistics for 'filtered':
    Total unique words: 51
    Total occurrences: 76
    Average frequency: 1.49
    Top 5 words: no_pain, words, feel, not_painful, relevant
Saved grouped word counts to word_frequencies_grouped.csv
  Total groups: 51
  Total words: 59
Note: Not saving new grouping dictionary because using existing one from Analysis/Dummy_test_for_frequency_analysis/word_grouping_dict.edited.json
Creating tree visualization...

Saved tree visualization to word_frequencies_tree.txt

============================================================
Tree Visualization Preview:
============================================================
Word Frequency Tree
==================================================

no_pain (7)
    ├── no_pain (4, 57.1%)
    └── not_pain (3, 42.9%)
words (4)
    ├── words (3, 75.0%)
    └── word (1, 25.0%)
not_painful (3)
test (3)
feel (3)
relevant (3)
    ├── relevant (1, 33.3%)
    ├── relevent (1, 33.3%)
    └── relevnt (1, 33.3%)
never_painful (2)
text (2)
negations (2)
    ├── negations (1, 50.0%)
... (see full tree in output file)

============================================================
Analysis Complete!
============================================================
Processed 1 file(s)
Total raw words (simple splitting): 164
Total processed words (with spaCy): 95
Unique raw words: 89
Unique processed words: 68
Words after stopword removal (before grouping): 59
Words after grouping and aggregation: 51
Total occurrences after aggregation: 76
Word groups loaded from: Analysis/Dummy_test_for_frequency_analysis/word_grouping_dict.edited.json
Word groups applied: 51

Output files created:
  word_frequencies_filtered.csv.txt
  word_frequencies_grouped.csv.txt
  word_frequencies_raw.csv.txt
  word_frequencies_tree.txt
```