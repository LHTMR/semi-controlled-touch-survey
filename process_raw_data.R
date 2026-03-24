# load libraries  ####
library(dplyr)
library(readxl)
library(tidyr)
library(readr)

# input path ####
RAW_DATA_FOLDER <- "~/Library/CloudStorage/OneDrive-Linköpingsuniversitet/projects - in progress/semi-controlled social touch/online survey/Data/"

# output paths ####
PROCESSED_DATA_FOLDER <- "Processed Data/"

# read in raw survey data exported from qualtrics ####
raw_data_file <- paste0(
  RAW_DATA_FOLDER,
  'Social+Touch+-+Prolific_April+26,+2023_17.15(Recorded)_fixed.xlsx'
)

respondent_data <- read_excel(raw_data_file, range = "A1:W1070", col_types = "text")

touch_data <- read_excel(raw_data_file, range = "X1:LW1070", col_types = "text") %>% 
  mutate(ResponseID = respondent_data$ResponseID) %>% 
  filter(ResponseID != "FS_1IN4EwLvt1xrWtL") %>% # something wrong with this person's data, possibly misaligned
  pivot_longer(
    cols = `1_Social_self`:`25_Input`,
    names_to = c("Touch No.", "Question"),
    names_pattern = "([0-9]+)_(.+)"
  ) %>% 
  pivot_wider(
    names_from = Question,
    values_from = value
  ) %>% 
  filter(!if_all(`Social_self`:`Input`, is.na)) %>% 
  mutate(
    `Touch No.` = as.numeric(`Touch No.`),
    # min = 156.5, max = 1244.5
    Valence = as.numeric(`Valence&Arousal_x`), 
    Arousal = as.numeric(`Valence&Arousal_y`) * -1 # reverse so high arousal corresponds to high values
    ) %>% 
  mutate(Touch_desc = case_when(
    `Touch No.` == 1 ~ "finger vertical 3 light",
    `Touch No.` == 2 ~ "finger vertical 3 strong",
    `Touch No.` == 3 ~ "finger vertical 9 light",
    `Touch No.` == 4 ~ "finger vertical 9 strong",
    `Touch No.` == 6 ~ "finger vertical 18 light",
    `Touch No.` == 7 ~ "finger vertical 18 strong",
    `Touch No.` == 8 ~ "hand vertical 3 light",
    `Touch No.` == 9 ~ "hand vertical 3 strong",
    `Touch No.` == 10 ~ "hand vertical 9 light",
    `Touch No.` == 11 ~ "hand vertical 9 strong",
    `Touch No.` == 12 ~ "hand vertical 18 light",
    `Touch No.` == 13 ~ "hand vertical 18 strong",
    `Touch No.` == 14 ~ "finger horizontal 3 light",
    `Touch No.` == 15 ~ "finger horizontal 3 strong",
    `Touch No.` == 16 ~ "finger horizontal 9 light",
    `Touch No.` == 17 ~ "finger horizontal 9 strong",
    `Touch No.` == 18 ~ "finger horizontal 18 light",
    `Touch No.` == 19 ~ "finger horizontal 18 strong",
    `Touch No.` == 20 ~ "hand horizontal 3 light",
    `Touch No.` == 21 ~ "hand horizontal 3 strong",
    `Touch No.` == 22 ~ "hand horizontal 9 light",
    `Touch No.` == 23 ~ "hand horizontal 9 strong",
    `Touch No.` == 24 ~ "hand horizontal 18 light",
    `Touch No.` == 25 ~ "hand horizontal 18 strong"
  )) %>% 
  separate_wider_delim(
    cols = Touch_desc, 
    delim = " ",
    names = c("Contact", "Direction", "Speed (cm/s)", "Force"),
    cols_remove = FALSE
    ) %>% 
  unite("Type", c(Direction, Contact), sep = " ", remove = FALSE) %>% 
  mutate(
    `Touch No.` = if_else(`Touch No.` > 5, `Touch No.` - 1, `Touch No.`),
    `Speed (cm/s)` = factor(`Speed (cm/s)`, levels = c("3", "9", "18")),
    Force = factor(Force, levels = c("light", "strong"))
    ) 

touch_data %>% 
  write_tsv(paste0(PROCESSED_DATA_FOLDER,"touch_data.txt"))