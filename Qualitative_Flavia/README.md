# Qualitative_Flavia

This folder holds R scripts and intermediate data for qualitative coding and MAXQDA export processing in the semi-controlled social touch project.

## Contents
- `survey_data_preprocessing.R`: Convert wide raw survey data to long format by video and participant.
- `clean_maxqda_codes.R`: Match MAXQDA coded segments to processed `touch_data` and investigate inconsistencies.
- `coded_export_data_processing.R`: Parse and reshape MAXQDA code export (`Video`, `Code`, `Other`) into `PID`, `VideoID`, `Segment`, `Code`, `Question` format.
- `coded_preprocessing_quality_control.R`: Identify duplicates and missing questions; generate QA reports (`missing_Q_*.csv`, `duplicate_rows.csv`).
- `maxqda_data_preprocessing.R`: Exploratory preprocessing of MAXQDA code data.

## Output examples
- `output_preprocessed_codes_ilona19_04.csv`
- `missing_Q_situationalwhat60.csv`
- `missing_Q_remaining38.csv`
- `duplicate_rows.csv`

## Notes
The folder contains manual xlsx files used during missing-question inference (`*_wQuestionColumn*.xlsx`).