"""
Plotting configurations
"""

import altair as alt
from matplotlib import font_manager
import matplotlib as mpl
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

ChartType = alt.vegalite.v4.api.Chart

# Fonts and colours
FONT_NAME = "Averta-Regular"
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

NESTA_FLOURISH_COLOURS = [
    "#0000ff",
    "#f6a4b7",
    "#18a48c",
    "#eb003b",
    "#9a1bbe",
    "#fdb633",
    "#97d9e3",
    "#d2c9c0",
    "#0f294a",
    "#ff6e47",
    "#a59bee",
    "#646363",
]

figsize_x = 8
figsize_y = 4


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
    Adds titles, subtitles, configures font sizes and adjusts gridlines
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
        logger.info(
            "Averta" + " font could not be located. Using 'DejaVu Sans' instead"
        )
        font_files = [f for f in all_font_files if "DejaVuSans.ttf" in f][0]
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = "DejaVu Sans"

    mpl.rcParams["xtick.labelsize"] = FONTSIZE_TITLE
    mpl.rcParams["ytick.labelsize"] = FONTSIZE_TITLE
    mpl.rcParams["axes.titlesize"] = FONTSIZE_TITLE
    mpl.rcParams["axes.labelsize"] = FONTSIZE_SUBTITLE
    mpl.rcParams["legend.fontsize"] = FONTSIZE_SUBTITLE
    mpl.rcParams["figure.titlesize"] = FONTSIZE_TITLE


set_plotting_styles()
