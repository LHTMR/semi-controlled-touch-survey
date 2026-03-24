
# Semi controlled touch survey

## File Tree
Structure of the dataset, i.e., where files are located.

<details>
<summary>See repository tree</summary>

```text
.
├── LICENSE
├── process_raw_data.R
├── README.md
├── semi-controlled-touch-survey.Rproj
├── Analysis
│   ├── All_words_by_frequency
│   │   ├── analysis_record.md
│   │   ├── word_frequencies_filtered.csv.txt
│   │   ├── word_frequencies_grouped.csv.txt
│   │   ├── word_frequencies_raw.csv.txt
│   │   ├── word_frequencies_tree.txt
│   │   ├── word_grouping_dict.json
│   │   ├── word_list_filtered.txt
│   │   ├── word_list_raw.txt
│   │   └── Social_context_only
│   ├── Dummy_test_for_frequency_analysis
│   │   ├── analysis_record.md
│   │   ├── test_text_frequency_extraction.txt
│   │   ├── word_frequencies_filtered.csv.txt
│   │   ├── word_frequencies_grouped.csv.txt
│   │   ├── word_frequencies_raw.csv.txt
│   │   ├── word_frequencies_tree.txt
│   │   ├── word_grouping_dict.edited.json
│   │   ├── word_grouping_dict.original.json
│   │   ├── word_list_filtered.txt
│   │   └── word_list_raw.txt
│   └── Scripts
│       ├── all_words_extractor.py
│       ├── data_exploration_utils.py
│       ├── generate_setup_stats.py
│       ├── geometric_association_test.py
│       ├── README.md
│       ├── wordnet_cluster.py
│       └── wordnet_disambiguator.py
├── IASAT_poster_Sarah
│   ├── analysis_IASATposter.R
│   ├── descriptor_maps.R
│   ├── Figures
│   │   ├── Appropriateness
│   │   ├── Valence_Arousal
│   │   └── Word_frequencies
│   └── Processed Data
│       ├── descriptor_map_data.tsv
│       ├── Emotional_self_word-freq-plot-data.txt
│       ├── Emotional_self_word-freq.txt
│       ├── Emotional_touch_word-freq-plot-data.txt
│       ├── Emotional_touch_word-freq.txt
│       ├── Intention&Purpose_word-freq-plot-data.txt
│       ├── Intention&Purpose_word-freq.txt
│       ├── Sensory_word-freq-plot-data.txt
│       ├── Social_body_word-freq-plot-data.txt
│       ├── Social_body_word-freq.txt
│       ├── Social_context_word-freq-plot-data.txt
│       ├── Social_context_word-freq.txt
│       ├── Social_place_word-freq-plot-data.txt
│       ├── Social_place_word-freq.txt
│       ├── Social_self_word-freq-plot-data.txt
│       └── Social_self_word-freq.txt
├── Materials
│   ├── emojigrid_qualtrics_precise-key.txt
│   ├── Social_Touch_-_Prolific.qsf
│   ├── video_mapping.csv
│   └── videos.md
├── Metadata
│   ├── data_dictionary.yaml
│   ├── experimental_setup.yaml
│   ├── touch_data_fixed.psv.yaml
│   └── touch_data.yaml
├── Processed Data
│   ├── README.md
│   ├── touch_data_fixed.psv.txt
│   ├── touch_data.txt
│   └── Per_video
│       ├── touch_data_video_1.psv.txt
│       ├── touch_data_video_2.psv.txt
│       ├── touch_data_video_3.psv.txt
│       ├── touch_data_video_4.psv.txt
│       ├── touch_data_video_5.psv.txt
│       ├── touch_data_video_6.psv.txt
│       ├── touch_data_video_7.psv.txt
│       ├── touch_data_video_8.psv.txt
│       ├── touch_data_video_9.psv.txt
│       ├── touch_data_video_10.psv.txt
│       ├── touch_data_video_11.psv.txt
│       ├── touch_data_video_12.psv.txt
│       ├── touch_data_video_13.psv.txt
│       ├── touch_data_video_14.psv.txt
│       ├── touch_data_video_15.psv.txt
│       ├── touch_data_video_16.psv.txt
│       ├── touch_data_video_17.psv.txt
│       ├── touch_data_video_18.psv.txt
│       ├── touch_data_video_19.psv.txt
│       ├── touch_data_video_20.psv.txt
│       ├── touch_data_video_21.psv.txt
│       ├── touch_data_video_22.psv.txt
│       ├── touch_data_video_23.psv.txt
│       └── touch_data_video_24.psv.txt
└── Qualitative_Flavia
    ├── clean_maxqda_codes.R
    ├── code_maps.R
    ├── coded_export_data_processing.R
    ├── coded_preprocessing_quality_control.R
    ├── duplicate_rows.csv
    ├── maxqda_data_preprocessing.R
    ├── missing_Q_remaining38.csv
    ├── missing_Q_situationalwhat60.csv
    ├── output_preprocessed_codes_ilona19_04.csv
    ├── remaining_missing03_05_24.csv
    └── survey_data_preprocessing.R

17 directories, 147 files
```

</details>

## Dataset Content

Here is a brief overview of the different documents contained in this folder.

### process_raw_data.R

This R script processes the raw survey data exported from Qualtrics. It reads the Excel file, cleans and reshapes the data, adds descriptive labels for touch types, and saves the processed touch data as text files.

### Materials/

This directory contains materials used for the survey.

*   **`emojigrid_qualtrics_precise-key.txt`:** Key for the emojigrid used in Qualtrics.
*   **`Social_Touch_-_Prolific.qsf`:** The Qualtrics survey file.
*   **`video_mapping.csv`:** Mapping of videos.
*   **`videos.md`:** Description of videos.

### Metadata/

This directory contains metadata files.

*   **`data_dictionary.yaml`:** Data dictionary.
*   **`experimental_setup.yaml`:** Experimental setup description.
*   **`touch_data.yaml` and `touch_data_fixed.psv.yaml`:** Metadata for touch data.

### Processed Data/

This directory contains the main processed touch data files.

*   **`touch_data.txt`** and **`touch_data_fixed.psv.txt`**: The main processed data files.
*   **`Per_video/`:** Directory containing per-video data files.
*   **`README.md`:** Documentation for the processed data.

See [Processed Data/README.md](Processed%20Data/README.md).


### Analysis/

#### All_words_by_frequency/

This directory contains the output files generated by the `all_words_extractor.py` script using `Processed Data/touch_data.txt` as input, providing comprehensive word frequency analysis for the entire touch survey dataset.

*   **Contents:**
    *   **`word_frequencies_raw.csv.txt`:** Contains baseline word frequencies extracted from the survey data without any linguistic processing. This provides a raw count of all words as they appear in the responses.
    *   **`word_frequencies_filtered.csv.txt`:** Contains word frequencies after applying stopword removal and negation handling. This file represents the cleaned dataset ready for analysis.
    *   **`word_frequencies_grouped.csv.txt`:** Provides detailed grouping information showing how different word variations (typos, morphological forms, spelling variations) were grouped together during processing.
    *   **`word_frequencies_tree.txt`:** A visual tree representation of word variation patterns, showing hierarchical relationships between word forms and their grouped representations.
    *   **`word_list_raw.txt`:** A simple list of all raw words extracted from the survey data, one per line.
    *   **`word_list_filtered.txt`:** A list of filtered words after processing, showing the cleaned vocabulary ready for analysis.
    *   **`word_grouping_dict.json`:** A JSON file containing all the groups of words detected by the script (assumed typos and spelling variations). This can be reviewed and edited by a human to serve as a reference for further analysis.

*   **Usage:**
    *   These files can be used for further statistical analysis, visualization, or as input for other text analysis tools
    *   The grouped frequencies help understand natural language variation in survey responses
    *   The tree visualization aids in understanding word relationships and variation patterns
    *   The raw vs filtered comparison supports methodological decisions in text processing pipelines

#### Scripts/

This directory contains Python scripts designed for comprehensive text analysis, word extraction, and semantic disambiguation of the qualitative touch survey responses.

See [Analysis/Scripts/README.md](Analysis/Scripts/README.md).

### Qualitative_Flavia/

A compact qualitative coding module with MAXQDA export processing, QA, and data-cleaning scripts.

See [Qualitative_Flavia/README.md](Qualitative_Flavia/README.md) for details.

### IASAT_poster_Sarah/

Short summary: analysis scripts and figure outputs for the IASAT poster workflow, including touch rating and descriptor maps.

See [IASAT_poster_Sarah/README.md](IASAT_poster_Sarah/README.md).

### LICENSE

This file contains the MIT License for the software and associated documentation files in this dataset. The copyright holder is S McIntyre (2024). The MIT License is a permissive free software license that allows for the free use, copying, modification, and distribution of the software, with the condition that the original copyright and permission notices are included in all copies. The software is provided "as is" without any warranty.