# Analysis Record

### Aim

- Descriptive statistics about the vocabulary used by participants.
- Create lists of words sorted by frequency to construct a bottom-up thematic analysis


### Command line

Executed from the root of the directory:

```bash
python3 Analysis/Scripts/all_words_extractor.py 'Processed Data/touch_data_fixed.psv.txt' --count-participants --format csv --text-columns "Social_self,Social_body,Social_place,Social_context,Intention&Purpose,Sensory,Emotional_self,Emotional_touch"
```


### Execution logs

```text
============================================================
All Words Extractor
============================================================
Parsed text columns: ['Social_self', 'Social_body', 'Social_place', 'Social_context', 'Intention&Purpose', 'Sensory', 'Emotional_self', 'Emotional_touch']
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
✓ All specified columns found: ['Social_self', 'Social_body', 'Social_place', 'Social_context', 'Intention&Purpose', 'Sensory', 'Emotional_self', 'Emotional_touch']

  Final text columns to process: ['Social_self', 'Social_body', 'Social_place', 'Social_context', 'Intention&Purpose', 'Sensory', 'Emotional_self', 'Emotional_touch']
  Total rows in dataframe: 2837

  Analyzing text from column 1/8: 'Social_self'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 2828/2837 (0.3% empty)
    Average text length: 13.2 characters

  Analyzing text from column 2/8: 'Social_body'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 2827/2837 (0.4% empty)
    Average text length: 10.7 characters

  Analyzing text from column 3/8: 'Social_place'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 2828/2837 (0.3% empty)
    Average text length: 16.1 characters

  Analyzing text from column 4/8: 'Social_context'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 2828/2837 (0.3% empty)
    Average text length: 49.2 characters

  Analyzing text from column 5/8: 'Intention&Purpose'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 2828/2837 (0.3% empty)
    Average text length: 43.4 characters

  Analyzing text from column 6/8: 'Sensory'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 2828/2837 (0.3% empty)
    Average text length: 31.6 characters

  Analyzing text from column 7/8: 'Emotional_self'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 2827/2837 (0.4% empty)
    Average text length: 32.4 characters

  Analyzing text from column 8/8: 'Emotional_touch'
    Conversion method: primary (fillna + astype(str))
    Non-empty values: 2805/2837 (1.1% empty)
    Average text length: 28.1 characters

  Extraction summary:
    Successfully extracted from 8/8 columns
    Total text entries: 2837
      ✓ Social_self: 2828 non-empty, 9 empty (primary (fillna + astype(str)))
      ✓ Social_body: 2827 non-empty, 10 empty (primary (fillna + astype(str)))
      ✓ Social_place: 2828 non-empty, 9 empty (primary (fillna + astype(str)))
      ✓ Social_context: 2828 non-empty, 9 empty (primary (fillna + astype(str)))
      ✓ Intention&Purpose: 2828 non-empty, 9 empty (primary (fillna + astype(str)))
      ✓ Sensory: 2828 non-empty, 9 empty (primary (fillna + astype(str)))
      ✓ Emotional_self: 2827 non-empty, 10 empty (primary (fillna + astype(str)))
      ✓ Emotional_touch: 2805 non-empty, 32 empty (primary (fillna + astype(str)))

  Removed 3 completely empty entries (0.1% of total)

  Final text count: 2834 non-empty text entries
  ✓ Extracted 2834 text entries from Processed Data/touch_data_fixed.psv.txt

Total text entries: 2834

Extracting words...
  Processed 1000/2834 entries
  Processed 2000/2834 entries
Extracted 106765 raw words (simple splitting)
Extracted 2834 raw words after deduplication per document
Extracted 65160 processed words (with spaCy if available)
Extracted 2834 processed words after deduplication per document
Unique raw words: 5454
Unique processed words: 5422

Calculating participant (document) frequencies...
✓ Saved word counts to word_frequencies_raw.csv
✓ Saved words to word_list_raw.txt
  Statistics for 'raw':
    Total unique words: 5,454
    Total occurrences: 84,618
    Average frequency: 15.51
    Top 5 words: my, to, and, me, the

Removing stopwords from processed words...
  Removed 59 stopwords (including negative forms)
  Remaining words: 5363
Grouping word variations...
  Created 3602 word groups from 5363 unique words
✓ Saved word counts to word_frequencies_filtered.csv
✓ Saved words to word_list_filtered.txt
  Statistics for 'filtered':
    Total unique words: 3,602
    Total occurrences: 52,566
    Average frequency: 14.59
    Top 5 words: arm, feel, home, touch, friend
Saved grouped word counts to word_frequencies_grouped.csv
  Total groups: 3602
  Total words: 5363
✓ Saved grouping dictionary to word_grouping_dict.json
  Total groups: 3602
  Total variants: 5363
  Average variants per group: 1.49

  Example groups (first 5):
    1. 'arm' -> ['arm', 'arms']
    2. 'feel' -> ['feel', 'feels']
    3. 'home' -> ['home', 'thome', 'homey', 'homes']
    4. 'touch' -> ['touch', 'touches', 'touchy', 'touchs', 'toutch', 'tousch', 'toch', 'tuch', 'thouch']
    5. 'friend' -> ['friend', 'friendly', 'friends', 'friendy', 'frienfd', 'friennd', 'fried']
Creating tree visualization...

Saved tree visualization to word_frequencies_tree.txt

============================================================
Tree Visualization Preview:
============================================================

Word Frequency Tree
==================================================

arm (1019)
    ├── arm (980, 96.2%)
    └── arms (39, 3.8%)
feel (967)
    ├── feel (910, 94.1%)
    └── feels (57, 5.9%)
home (961)
    ├── home (958, 99.7%)
    ├── thome (1, 0.1%)
    ├── homey (1, 0.1%)
    └── homes (1, 0.1%)
touch (877)
    ├── touch (703, 80.2%)
    ├── touches (157, 17.9%)
    ├── touchy (7, 0.8%)
    ├── touchs (3, 0.3%)
    ├── toutch (2, 0.2%)
... (see full tree in output file)

============================================================
Analysis Complete!
============================================================

Processed 1 file(s)
Total raw words (simple splitting): 106765
Total processed words (with spaCy): 65160
Unique raw words: 5454
Unique processed words: 5422
Words after stopword removal (before grouping): 5363
Words after grouping and aggregation: 3602
Total occurrences after aggregation: 52566
Word groups created: 3602

Output files created:

```
