"""
Functions for easier generation of plots.
"""

import altair as alt
from matplotlib import font_manager
import matplotlib.pyplot as plt
from wordcloud import WordCloud

ChartType = alt.vegalite.v4.api.Chart

# Fonts and colours
FONT = "Averta"
TITLE_FONT = "Averta"
FONTSIZE_TITLE = 20
FONTSIZE_SUBTITLE = 16
FONTSIZE_NORMAL = 13

NESTA_COLOURS = [
    "#0000FF",
    "#18A48C",
    "#FDB633",
    "#9A1BBE",
    "#EB003B",
    "#FF6E47",
    "#646363",
    "#0F294A",
    "#97D9E3",
    "#A59BEE",
    "#F6A4B7",
    "#D2C9C0",
    "#FFFFFF",
    "#000000",
]


def nestafont():
    """Define Nesta fonts"""
    return {
        "config": {
            "title": {"font": TITLE_FONT, "anchor": "start"},
            "axis": {"labelFont": FONT, "titleFont": FONT},
            "header": {"labelFont": FONT, "titleFont": FONT},
            "legend": {"labelFont": FONT, "titleFont": FONT},
            "range": {
                "category": NESTA_COLOURS,
                "ordinal": {
                    "scheme": NESTA_COLOURS
                },  # this will interpolate the colors
            },
        }
    }


alt.themes.register("nestafont", nestafont)
alt.themes.enable("nestafont")


def configure_plots(fig, chart_title: str = "", chart_subtitle: str = ""):
    """
    Adds titles, subtitles, configures font sizes andadjusts gridlines
    """
    return (
        fig.properties(
            title={
                "anchor": "start",
                "text": chart_title,
                "fontSize": FONTSIZE_TITLE,
                "subtitle": chart_subtitle,
                "subtitleFont": FONT,
                "subtitleFontSize": FONTSIZE_SUBTITLE,
            },
        )
        .configure_axis(
            gridDash=[1, 7],
            gridColor="grey",
            labelFontSize=FONTSIZE_NORMAL,
            titleFontSize=FONTSIZE_NORMAL,
        )
        .configure_legend(
            titleFontSize=FONTSIZE_NORMAL,
            labelFontSize=FONTSIZE_NORMAL,
        )
        .configure_view(strokeWidth=0)
    )


def finding_path_to_font(font_name: str):
    """
    Finds path to specific font.
    Args:
        font_name: name of font
    """

    all_font_files = font_manager.findSystemFonts()
    font_files = [f for f in all_font_files if font_name in f]
    if len(font_files) == 0:
        font_files = [f for f in all_font_files if "DejaVuSans.ttf" in f]
    return font_files[0]


def create_wordcloud(frequencies, max_words, stopwords):
    """
    Creates word cloud based on frequencies.
    Args:
        frequencies: frequencies of words or n-grams
        max_words: maximum number of words or n-grams to be displayed
        stopwords: stopwords that should be removed from the wordcloud
    """
    font_path_ttf = finding_path_to_font("Averta-Regular")

    wordcloud = WordCloud(
        font_path=font_path_ttf,
        width=1000,
        height=1000,
        margin=0,
        collocations=True,
        stopwords=stopwords,
        background_color="white",
        max_words=max_words,
    )

    wordcloud = wordcloud.generate_from_frequencies(frequencies=frequencies)

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
