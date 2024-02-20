from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    set_plotting_styles,
    NESTA_COLOURS,
)
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    finding_path_to_font,
)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

set_plotting_styles()
font_path_ttf = finding_path_to_font("Averta-Regular")

import pandas as pd
import textwrap

# Replace 'your_file.csv' with the path to your actual file
# Replace 'question_column' with the actual name of the column containing questions
# df = pd.read_csv('/Users/aidan.kelly/nesta/ASF/asf_public_discourse_home_decarbonisation/outputs/extracted_questions/bh/forum_119_air_source_heat_pumps_ashp/extracted_questions_119_air_source_heat_pumps_ashp.csv')
df = pd.read_csv(
    "/Users/aidan.kelly/nesta/ASF/asf_public_discourse_home_decarbonisation/outputs/extracted_questions/bh/forum_combined_data/extracted_questions_combined_data.csv"
)
# df = pd.read_csv('/Users/aidan.kelly/nesta/ASF/asf_public_discourse_home_decarbonisation/outputs/extracted_questions/bh/forum_combined_data/idk_phrases_combined_data.csv')
questions = df["Question"]
question_counts = questions.value_counts()


# Select the top N questions to display
top_n = 5
top_questions = question_counts.head(top_n)


# Function to wrap text and add an ellipsis after the first couple of lines
def wrap_text_with_ellipsis(text, line_width, max_lines):
    wrapped_text = textwrap.wrap(text, width=line_width)
    if len(wrapped_text) > max_lines:
        return "\n".join(wrapped_text[:max_lines]) + "..."
    else:
        return "\n".join(wrapped_text)


# Wrap text for each question and add ellipsis if it's longer than 2 lines
wrapped_questions = [wrap_text_with_ellipsis(q, 40, 2) for q in top_questions.index]

# Increase figure size for better visibility

plt.figure(figsize=(12, 8))  # Adjust the width as necessary

# Plotting
plt.barh(wrapped_questions, top_questions.values, color=NESTA_COLOURS[0])
# Increase left margin to fit the wrapped questions
plt.gcf().subplots_adjust(left=0.5)
plt.xlabel("Frequency")
plt.ylabel("Questions")
plt.title(f"Top {top_n} Most Frequent Questions")
plt.gca().invert_yaxis()  # To display the highest value at the top

# Set x-axis major locator to integer values
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# Show plot with wrapped text for y-axis labels
# plt.tight_layout()  # Adjust layout to make room for the wrapped question text
plt.show()
