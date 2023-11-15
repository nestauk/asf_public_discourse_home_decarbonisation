"""
Functions for easier generation of plots.
"""

import altair as alt
from matplotlib import font_manager
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib as mpl
import seaborn as sns
from asf_public_discourse_home_decarbonisation import PROJECT_DIR

MSE_FIG_PATH = PROJECT_DIR / "outputs/figures/mse"

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


def create_wordcloud(frequencies, max_words, stopwords, fig_name):
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

    plt.savefig(
        f"{MSE_FIG_PATH}/{fig_name}.png",
        dpi=600,
    )


def set_spines():
    """
    Function to add or remove spines from plots.
    """
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.bottom"] = True


def set_plotting_styles():
    """
    Function that sets plotting styles.
    """

    sns.set_context("talk")

    set_spines()

    # Had trouble making it find the font I set so this was the only way to do it
    # without specifying the local filepath
    all_font_files = font_manager.findSystemFonts()

    try:
        mpl.rcParams["font.family"] = "sans-serif"
        font_files = [f for f in all_font_files if "Averta-Regular" in f]
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        mpl.rcParams["font.sans-serif"] = "Averta"
    except:
        print("Averta" + " font could not be located. Using 'DejaVu Sans' instead")
        font_files = [f for f in all_font_files if "DejaVuSans.ttf" in f][0]
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = "DejaVu Sans"

    mpl.rcParams["xtick.labelsize"] = 20
    mpl.rcParams["ytick.labelsize"] = 20
    mpl.rcParams["axes.titlesize"] = 20
    mpl.rcParams["axes.labelsize"] = 16
    mpl.rcParams["legend.fontsize"] = 16
    mpl.rcParams["figure.titlesize"] = 20


set_plotting_styles()
