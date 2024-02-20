import pandas as pd
from bertopic import BERTopic
import matplotlib.pyplot as plt
from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    set_plotting_styles,
    NESTA_COLOURS,
)
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    finding_path_to_font,
)

set_plotting_styles()
font_path_ttf = finding_path_to_font("Averta-Regular")

# Load the CSV file
# file_path = '../forum_120_ground_source_heat_pumps_gshp/extracted_questions_120_ground_source_heat_pumps_gshp.csv'  # Make sure to replace this with your actual file path
file_path = "/Users/aidan.kelly/nesta/ASF/asf_public_discourse_home_decarbonisation/outputs/extracted_questions/bh/forum_119_air_source_heat_pumps_ashp/extracted_questions_119_air_source_heat_pumps_ashp.csv"
df = pd.read_csv(file_path)

# Assuming your questions are in a column named 'questions'
questions = df["Question"].tolist()

# Initialize BERTopic
topic_model = BERTopic()


# Fit the model on your questions
topics, probabilities = topic_model.fit_transform(questions)

fig = topic_model.visualize_topics()
# Save the figure as a png file
fig.write_image("topic_visualization.png")

fig_barchart = topic_model.visualize_barchart(top_n_topics=16, n_words=10)
"""
for trace in fig_barchart.data:
    trace.marker.color = NESTA_COLOURS[0]  # Set each bar's color to NESTAS[0]
fig_barchart.show()

plt.show()
"""
fig_barchart.write_image("topic_visualization_barchart.png")

# fig_hierarchy = topic_model.visualize_hierarchy()

# fig_hierarchy.write_image("topic_visualization_hierarchy.png")

# Print the topics found
print(topic_model.get_topic_info())  # This prints the basic info about the topics found

topics_info = topic_model.get_topic_info()
topics_info["%"] = topics_info["Count"] / len(questions) * 100
# topics_info
# doc_info = topic_model.get_document_info(questions)
# doc_info

"""

for i in range(10):  # Adjust the range as needed
    topic = topics_info.iloc[i]
    print(f"Topic {topic['Topic']} - Count: {topic['Count']}")
    print(f"Name: {topic['Name']}")
    print(f"Representation: {topic['Representation']}")
    print(f"Representative Docs: {topic['Representative_Docs']}")
    print()
"""
# Topic Distribution Bar Chart
topic_counts = topic_model.get_topic_info()["Count"][1:17]
topic_labels = topic_model.get_topic_info()["Name"][1:17].str.replace("_", " ")
plt.figure(figsize=(14, 8))
plt.barh(topic_labels, topic_counts, color=NESTA_COLOURS[0])
# plt.barh(topic_labels, topic_counts)
# plt.xticks(rotation=90)
plt.ylabel("Topics")
plt.xlabel("Count")
plt.title("Topic Distribution")
plt.tight_layout()
# Save the figure before showing it
plt.savefig("topic_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# new_topics, new_probs = topic_model.reduce_topics(questions, nr_topics=30)

# Now visualize the hierarchy with the reduced number of topics
fig_hierarchy = topic_model.visualize_hierarchy()

fig_hierarchy.write_image("topic_visualization_hierarchy.png")
