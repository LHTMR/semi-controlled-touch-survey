
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

### Qualitative_Flavia/

This directory contains R scripts for qualitative data analysis and preprocessing, originally developed by Flavia. These scripts handle MAXQDA data export processing, code mapping, and survey data preprocessing.

#### clean_maxqda_codes.R

This R script is used for cleaning and exploring manually coded data that was exported from the qualitative data analysis software MAXQDA. The main goal of this script is to link the coded segments from MAXQDA with the survey data.

*   **Libraries Used:** It uses `readxl` for reading Excel files, `readr` for reading text files, and `tidyverse` for data manipulation.
*   **Input:**
    *   It reads an Excel file (`.xlsx`) containing coded segments exported from MAXQDA.
    *   It reads the `touch_data.txt` file, which is the processed survey data.
*   **Data Processing and Exploration:**
    *   It attempts to find the coded segments from the MAXQDA data within the `touch_data`.
    *   It extracts and compares variable names and codes from the MAXQDA data to identify inconsistencies or matches.
    *   It filters and views the MAXQDA data based on specific codes (e.g., "Annoy / Tease") to inspect the coded segments.
    *   It tries to identify segments that were coded but not associated with a specific video.
    *   It checks for duplicate coded segments.
*   **Output:** The script is primarily for interactive exploration and cleaning. It does not appear to save any new data files. The main output is the exploration and debugging of the coded data, likely done within an interactive R session.

#### coded_export_data_processing.R

This is a comprehensive R script designed to clean, re-format, and process a coded export file from a qualitative data analysis software. The script's main purpose is to transform the raw exported data into a structured format suitable for further analysis.

*   **Libraries Used:** It uses `readxl` for reading Excel files, `openxlsx` for writing Excel files, `readr` for reading and writing text files, and `tidyverse` for extensive data manipulation.
*   **Input:**
    *   An Excel file (`.xlsx`) which is a coded export from a coding software.
*   **Data Processing:**
    *   **Column Selection and Renaming:** It starts by selecting a subset of important columns and renaming them for clarity.
    *   **Data Filtering:** It removes rows related to demographics and keeps only rows containing specific keywords like "RELATIONAL", "Autocode", or "Video".
    *   **Video ID Extraction:** A significant portion of the script is dedicated to extracting a `VideoID` for each row. It uses a custom function to find the video number from different columns (`Code` and `Other`) based on a list of `video_questions`. It also includes logic to infer missing `VideoID`s from surrounding rows.
    *   **Duplicate Handling:** The script identifies and removes duplicate rows based on combinations of `PID`, `VideoID`, `Segment`, and `Code`.
    *   **Question Extraction:** It extracts the specific question associated with each coded segment. This is a complex process that involves parsing strings from the `Code` and `Other` columns. For rows with multiple questions, it splits them into separate rows.
    *   **Handling Missing Data:** The script identifies rows with missing question information and attempts to resolve them by looking at the "Video" cell which contains all answers in an ordered manner. It also reads a manually created file to fill in missing questions.
    *   **Final Structuring:** The script restructures the data to have one row per coded segment, with columns for `PID`, `VideoID`, `Segment`, `Code`, and `Question`.
*   **Output:**
    *   The script generates several intermediate and final CSV files for debugging and for the final processed data, with `output_preprocessed_codes_ilona07_05_24.csv` as the likely main output file.

#### coded_preprocessing_quality_control.R

This R script is for quality control and debugging of the preprocessed coded data. It aims to identify and address issues like missing data and duplicate rows in the output of the previous processing script.

*   **Libraries Used:** It uses `tidyverse` for data manipulation and `readxl` for reading Excel files.
*   **Input:**
    *   `output_preprocessed_codes_ilona19_04.csv`: The preprocessed coded data from the previous script.
    *   `attempt1_allRecoded.xlsx`: The original coded export file from MAXQDA.
    *   `Copy of missing_Q_situationalwhat60_wQuestionColumn_ADDED.xlsx` and `Copy of missing_Q_remaining38_wQuestionColumn_ADDED.xlsx`: Excel files that contain manually added question information for rows where the question was missing.
*   **Quality Control and Debugging:**
    *   **Missing Question Data:**
        *   It identifies rows with missing `Question` information in the preprocessed data.
        *   It writes out the rows with missing questions to separate CSV files (`missing_Q_situationalwhat60.csv` and `missing_Q_remaining38.csv`) for manual inspection.
        *   It reads in the manually corrected files to solve the missing data issue.
    *   **Duplicate Rows:**
        *   It identifies duplicate rows and writes them to a CSV file (`duplicate_rows.csv`).
        *   It provides a "Solution for duplicates" section that categorizes the reasons for duplicates and suggests how to handle them.
*   **Output:** The script generates several CSV files that highlight issues in the preprocessed data, which are intended for manual inspection and correction:
    *   `missing_Q_situationalwhat60.csv`
    *   `missing_Q_remaining38.csv`
    *   `duplicate_rows.csv`

#### code_maps.R

This R script is for visualizing the frequency of different codes from the coded data. It creates "code maps" that show the distribution of codes across different touch types, speeds, and forces.

*   **Libraries Used:** It uses `tidyverse` for data manipulation, `svglite` for saving plots as SVG files, `readxl` for reading Excel files, and `ggdark` for creating plots with a dark theme.
*   **Input:**
    *   An Excel file (`.xlsx`) that contains code frequencies per video, likely `Frequencies_perVideo.xlsx`.
*   **Data Processing:**
    *   It reads the code frequency data from the Excel file.
    *   It separates the `Video` column into `Contact`, `Direction`, `Speed (cm/s)`, and `Force`.
    *   It cleans and formats these new columns.
    *   It creates a `Type` column by combining `Direction` and `Contact`.
    *   It groups the data by `Code` and the new variables to summarize the frequency of each code.
*   **Visualization:**
    *   It creates a custom dark theme for the plots.
    *   It ranks the codes by their overall frequency and creates a bar plot to show the frequency of each code.
    *   The main visualization is a set of facet plots, where each facet represents a code. Within each facet, it plots the `Type` of touch against the `Speed (cm/s)`, and the size and color of the points represent the frequency and force.
    *   It saves the generated plot as an SVG file in the `Figures/` directory.
*   **Output:**
    *   The script generates a plot of the code maps and saves it as an SVG file in the `Figures/` directory.

#### maxqda_data_preprocessing.R

This R script is for preprocessing and exploring data exported from MAXQDA. It appears to be an initial exploration of the data structure, with a focus on understanding the relationship between codes, questions, and video IDs.

*   **Libraries Used:** It uses `readxl` for reading Excel files, `readr` for reading text files, and `tidyverse` for data manipulation.
*   **Input:**
    *   `MAXQDA 24 Coded Segments_NewProject.xlsx`: An Excel file containing coded segments exported from MAXQDA.
*   **Data Processing and Exploration:**
    *   **Code and Variable Name Comparison:** It extracts code names from both the column headers and the `Code` column and compares them to find differences.
    *   **Video Question Identification:** It identifies the unique video questions from the column names.
    *   **Video ID Extraction:** It creates a `VideoID` column by identifying the start of each video's data.
    *   **Participant Data Subsetting:** It includes an example of how to filter the data for a specific participant and select relevant columns.
    *   **Code Exploration:** It demonstrates how to find which specific codes were applied for a given question and participant.
*   **Output:** The script is primarily for interactive exploration and does not save any new data files.

#### survey_data_preprocessing.R

This R script is designed to clean and re-format the original survey data file. The primary goal is to transform the wide-format survey data into a long-format data frame, which is more suitable for analysis and for import into qualitative data analysis software like MAXQDA.

*   **Libraries Used:** It uses `readxl` for reading Excel files, `openxlsx` for writing Excel files, `readr` for reading CSV files, and `tidyverse` for data manipulation.
*   **Input:**
    *   `SocialTouchProlific_February26_2024_original.xlsx`: The original survey data file, where each row represents a participant and columns contain demographic information and responses to questions for multiple videos.
    *   `video_info.csv`: A CSV file containing metadata for each video shown in the survey, such as the intensity of the touch, speed, body part used, and movement type.
*   **Data Processing:**
    *   **Data Reshaping:** The script iterates through each participant's data (each row) and transforms it from a wide format to a long format. The original format has one row per participant, with answers to video questions in columns prefixed with the video ID. The script creates a new data frame where each row represents a single video watched by a single participant.
    *   **Data Cleaning:** It performs data cleaning by checking each response. If a response is a single character (e.g., "a" or "1"), it is considered an invalid or low-effort response and is replaced with `NA`. If a participant-video combination has too many (`>4`) `NA` values, a warning is printed.
    *   **Merging with Video Info:** For each new row in the long-format data, the script merges the participant's survey data with the corresponding video's metadata from `video_info.csv`. This adds contextual information about the touch stimuli to each response.
    *   **Unique Key Generation:** A unique key is created for each row by concatenating the participant's ID and the video ID, ensuring that every participant-video entry is unique.
*   **Output:**
    *   `output_ilona15_03_24.xlsx`: An Excel file containing the preprocessed survey data in a long format, with clearly defined columns. This file is ready to be imported into MAXQDA for qualitative coding and analysis.


### process_raw_data.R

This R script processes the raw survey data exported from Qualtrics. It reads the Excel file, cleans and reshapes the data, adds descriptive labels for touch types, and saves the processed touch data as text files.

### IASAT_poster_Sarah/

This directory contains R scripts, figures, and processed data related to the analysis for the IASAT poster.

#### analysis_IASATposter.R

This R script is for analyzing data from a social touch survey. It performs data cleaning, processing, and visualization.

*   **Libraries Used:** It uses several R libraries, including `dplyr` for data manipulation, `readxl` for reading Excel files, `tidyr` for tidying data, `ggplot2` for plotting, `Hmisc` for statistical functions, and `quanteda` for text analysis.
*   **Input:** It reads raw survey data from an Excel file (`.xlsx`) located in a specific directory.
*   **Data Processing:**
    *   It separates respondent data and touch data.
    *   It cleans and reshapes the touch data, creating new variables like "Valence" and "Arousal".
    *   It adds descriptive labels to the touch data based on touch number, speed, force, etc.
    *   It saves the processed touch data as a text file (`touch_data.txt`).
*   **Analysis and Visualization:**
    *   It calculates and plots the number of responses per touch and per respondent.
    *   It creates boxplots to visualize the distribution of "Valence" and "Arousal" for different touch types.
    *   It performs direct comparisons of "Valence" and "Arousal" based on speed and force.
    *   It calculates and plots confidence intervals for "Valence" and "Arousal".
    *   It performs word frequency analysis on the text data from the survey.
    *   It analyzes the "Appropriateness" of the touch, creating bar plots to show the distribution of responses.
*   **Output:** The script generates several outputs:
    *   A processed data file: `touch_data.txt`.
    *   Several plots are generated and saved as PDF files in the `Figures/` directory.
    *   Word frequency data is saved as text files.

#### descriptor_maps.R

This R script is for creating "descriptor maps," which are visualizations of word frequencies associated with different types of touch. It groups words into higher-level "descriptors" and then into broader "Groups" (e.g., "emotion", "person", "sensory").

*   **Libraries Used:** It uses `tidyverse` for data manipulation, `svglite` for saving plots as SVG files, and `ggdark` for creating plots with a dark theme.
*   **Input:**
    *   It reads multiple `_word-freq-plot-data.txt` files from the `Processed Data/` directory. These files are the output of the word frequency analysis in `analysis_IASATposter.R`.
*   **Data Processing:**
    *   It reads and combines all the word frequency data files.
    *   **Descriptor Mapping:** It maps the stemmed words to more meaningful `descriptor`s (e.g., "affect" becomes "affection").
    *   **Group Mapping:** It then maps these `descriptor`s to broader `Group`s (e.g., "emotion", "body", "place").
    *   It summarizes the frequency of each `descriptor` and `Group`.
    *   It saves the processed descriptor map data to a file.
*   **Visualization:**
    *   It creates facet plots for different groups of descriptors (e.g., `sensory_descriptors`, `person_descriptors`).
    *   Each facet represents a descriptor, and it plots the `Type` of touch against the `Speed (cm/s)`. The size of the points represents the frequency, and the `Force` is represented by the fill color.
*   **Output:**
    *   `Processed Data/descriptor_map_data.tsv`: A tab-separated file containing the processed descriptor map data.
    *   Several SVG files in the `Figures/` directory, each showing a map for a different group of descriptors (e.g., `sensory_map.svg`, `person_map.svg`).

#### Figures/

This directory contains various plots and visualizations generated by the R scripts. The files are in SVG and PDF formats.

*   **Subdirectories:**
    *   **`Appropriateness/`:** Contains plots related to the appropriateness of touch.
    *   **`Valence_Arousal/`:** Contains plots related to valence and arousal ratings.
    *   **`Word_frequencies/`:** Contains word frequency plots.

#### Processed Data/

This directory contains processed data files from the IASAT poster analysis.

*   **`descriptor_map_data.tsv`:** Processed descriptor map data.
*   **Word Frequency Data:** Various .txt files for word frequencies and plot data.

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

### semi-controlled-touch-survey.Rproj

This is an RStudio project file. It is used by the RStudio IDE to manage project-specific settings. When a user opens this `.Rproj` file in RStudio, the IDE will automatically set the working directory to the directory containing this file and load the project's settings. This makes it easier to work on the project and ensures that the code runs in the correct context. The file contains settings related to workspace management, code indexing, text encoding, and document weaving.


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


### LICENSE

This file contains the MIT License for the software and associated documentation files in this dataset. The copyright holder is S McIntyre (2024). The MIT License is a permissive free software license that allows for the free use, copying, modification, and distribution of the software, with the condition that the original copyright and permission notices are included in all copies. The software is provided "as is" without any warranty.