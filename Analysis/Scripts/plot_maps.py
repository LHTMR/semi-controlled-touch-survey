import argparse
import pandas as pd
import numpy as np
import yaml
import re
import math
from plotnine import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Faceted Bubble Maps from Touch Data"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to aggregated data"
    )
    parser.add_argument(
        "--setup", type=str, required=True, help="Path to experimental setup"
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["dark", "light"],
        default="dark",
        help="Color theme",
    )
    parser.add_argument("--words", type=str, nargs="+", help="Specific words to plot")
    parser.add_argument(
        "--out", type=str, default="output_map.svg", help="Output file path"
    )
    return parser.parse_args()


def load_and_merge_data(data_path, setup_path):
    df_data = pd.read_csv(data_path)

    video_cols = [
        c for c in df_data.columns if re.match(r"^video_\d+$", c, re.IGNORECASE)
    ]

    df_melt = df_data.melt(
        id_vars=["Word"],
        value_vars=video_cols,
        var_name="video_id",
        value_name="affinity_score",
    )
    df_melt.rename(columns={"Word": "descriptor"}, inplace=True)
    df_melt["video_id"] = df_melt["video_id"].str.capitalize()

    with open(setup_path, "r") as file:
        setup_raw = yaml.safe_load(file)

    video_details = setup_raw.get("video_details", {})
    setup_list = []

    for vid, details in video_details.items():
        params = details.get("parameters", {})
        touch_type = (
            f"{params.get('direction', '')} {params.get('contact', '')}".strip()
        )
        setup_list.append(
            {
                "video_id": vid,
                "Type": touch_type,
                "Speed (cm/s)": str(params.get("speed_cm_s", "")),
                "Force": params.get("force", ""),
            }
        )

    df_setup = pd.DataFrame(setup_list)
    merged_df = pd.merge(df_melt, df_setup, on="video_id", how="left")
    merged_df = merged_df.dropna(subset=["Type", "Speed (cm/s)", "Force"])

    # 1. Ensure the column is numeric
    merged_df["affinity_score"] = pd.to_numeric(
        merged_df["affinity_score"], errors="coerce"
    )

    # 2. Blank out the sizes for negative/zero scores
    merged_df.loc[merged_df["affinity_score"] <= 0, "affinity_score"] = np.nan

    # --- THE NEW FIX: Z-ORDER SORTING ---
    # Sort descending: largest affinity plotted first (bottom), smallest plotted last (top)
    # Using na_position="first" draws the invisible NaN rows first so they don't interfere
    merged_df = merged_df.sort_values(
        by="affinity_score", ascending=False, na_position="first"
    )

    return merged_df


def calculate_grid_layout(words):
    """
    Implements the custom square-first layout rule.
    Returns: The ordered list of facets (including invisible dummies), Rows, Columns
    """
    N = len(words)
    if N == 0:
        return [], 1, 1

    k = int(math.floor(math.sqrt(N)))

    if k * k == N:
        R, C = k, k
    elif N <= k * (k + 1):
        R, C = k, k + 1
    else:
        R, C = k + 1, k + 1

    empty_count = R * C - N
    layout = [None] * (R * C)

    # Place empty spaces at the top of the last column
    for r in range(empty_count):
        # Using increasing spaces ensures categories are unique but visually blank
        layout[r * C + (C - 1)] = " " * (r + 1)

    # Fill the remaining slots sequentially with actual words
    word_idx = 0
    for i in range(len(layout)):
        if layout[i] is None:
            layout[i] = words[word_idx]
            word_idx += 1

    return layout, R, C


def create_plot(df, layout, R, C, theme_mode="dark"):
    if theme_mode == "dark":
        bg_color = "#191919"
        text_color = "white"
        border_color = "#555555"
        grid_color = "#2b2b2b"
        fill_colors = {"strong": "#EDC948", "light": "#17becf"}
    else:
        bg_color = "white"
        text_color = "black"
        border_color = "#cccccc"
        grid_color = "#eeeeee"
        fill_colors = {"strong": "#D47171", "light": "#89B4D9"}

    fig_width = (C * 2.2) + 2.5
    fig_height = (R * 1.8) + 2.0

    p = ggplot(df, aes(x="Type", y="Speed (cm/s)")) + facet_wrap(
        "~ descriptor", ncol=C, drop=False
    )

    # --- THE SPINE FIX ---
    # 1. Create a clean dataframe with exactly one row per panel
    # to prevent drawing hundreds of overlapping borders.
    unique_descriptors = df["descriptor"].unique()
    panel_df = pd.DataFrame({"descriptor": unique_descriptors})

    dummy_panels = panel_df[panel_df["descriptor"].str.strip() == ""]
    valid_panels = panel_df[panel_df["descriptor"].str.strip() != ""]

    if not dummy_panels.empty:
        # Patch dummy panels (np.inf works fine for fills)
        p += geom_rect(
            data=dummy_panels,
            xmin=-np.inf,
            xmax=np.inf,
            ymin=-np.inf,
            ymax=np.inf,
            fill=bg_color,
            color="none",
            inherit_aes=False,
        )

    # 2. Draw manual borders using exact categorical boundary math
    # X has 4 categories (centers at 1,2,3,4) -> boundaries at 0.5 and 4.5
    # Y has 3 categories (centers at 1,2,3) -> boundaries at 0.5 and 3.5
    p += geom_rect(
        data=valid_panels,
        xmin=0.5,
        xmax=4.5,
        ymin=0.5,
        ymax=3.5,
        fill="none",
        color=border_color,
        size=0.6,
        inherit_aes=False,
    )
    # ---------------------

    p += geom_point(
        aes(size="affinity_score", fill="Force"),
        shape="o",
        color="white",
        stroke=0.3,
        alpha=0.8,
        # --- THE JITTER ADJUSTMENT ---
        # Increased width and height from 0.05 to 0.1
        position=position_jitter(width=0.1, height=0.1, random_state=42),
    )

    p += scale_fill_manual(values=fill_colors)
    p += scale_size_area(max_size=10)  # Area scales beautifully with [0, 1] data

    # Update the labels
    p += labs(x="", y="Speed (cm/s)", size="Affinity", fill="Force")

    p += guides(
        size=guide_legend(
            title="\n\nAffinity",  # <-- Changed title
            override_aes={"fill": "none", "color": text_color, "stroke": 0.8},
        ),
        fill=guide_legend(override_aes={"size": 4, "color": "white", "stroke": 0.3}),
    )

    p += theme(
        plot_background=element_rect(fill=bg_color, color="none"),
        panel_background=element_rect(fill=bg_color, color="none"),
        legend_background=element_rect(fill=bg_color, color="none"),
        legend_key=element_rect(fill=bg_color, color="none"),
        # THE FIX: Turn off global panel borders since we draw them manually now
        panel_border=element_blank(),
        panel_grid_major=element_line(color=grid_color, size=0.5),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.025,
        panel_spacing_y=0.025,
        strip_background=element_blank(),
        strip_text=element_text(color=text_color, size=13),
        axis_text_x=element_text(angle=45, hjust=1, color=text_color, size=11),
        axis_text_y=element_text(color=text_color, size=11),
        axis_title=element_text(color=text_color),
        legend_text=element_text(color=text_color, size=11),
        legend_title=element_text(color=text_color, size=13),
        text=element_text(color=text_color),
        plot_margin=0.05,
        figure_size=(fig_width, fig_height),
    )

    return p


if __name__ == "__main__":
    args = parse_args()
    df = load_and_merge_data(args.data, args.setup)

    # 1. Filter / Sort Words
    if args.words:
        # Keep only the words that exist in the dataframe
        words = [w for w in args.words if w in df["descriptor"].values]
        df = df[df["descriptor"].isin(words)].copy()
    else:
        # Default: Sort by total frequency if no words provided
        words = (
            df.groupby("descriptor")["frequency"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

    if not words:
        print("Error: No words to plot.")
        exit(1)

    # 2. Compute Target Grid Layout
    layout, R, C = calculate_grid_layout(words)

    # 3. Inject dummy rows so plotnine natively draws the empty grid boxes
    dummy_rows = []
    base_type = df["Type"].iloc[0]
    base_speed = df["Speed (cm/s)"].iloc[0]
    base_force = df["Force"].iloc[0]

    for l in layout:
        if l.strip() == "":
            dummy_rows.append(
                {
                    "descriptor": l,
                    "Type": base_type,
                    "Speed (cm/s)": base_speed,
                    "Force": base_force,
                    "frequency": np.nan,  # NaN ensures no point is drawn
                }
            )

    if dummy_rows:
        df = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)

    # 4. Enforce exact Categorical Ordering
    df["Type"] = pd.Categorical(
        df["Type"],
        categories=[
            "horizontal hand",
            "vertical hand",
            "horizontal finger",
            "vertical finger",
        ],
        ordered=True,
    )
    df["Force"] = pd.Categorical(
        df["Force"], categories=["strong", "light"], ordered=True
    )
    df["Speed (cm/s)"] = pd.Categorical(
        df["Speed (cm/s)"], categories=["3", "9", "18"], ordered=True
    )

    # Crucial: Order the descriptors perfectly matching our computed matrix
    df["descriptor"] = pd.Categorical(df["descriptor"], categories=layout, ordered=True)

    print(f"Generating {R}x{C} layout for {len(words)} words...")
    plot = create_plot(df, layout, R, C, theme_mode=args.theme)
    plot.save(args.out, dpi=300)
