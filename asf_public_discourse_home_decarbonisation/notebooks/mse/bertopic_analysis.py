# %%
from bertopic import BERTopic
import pandas as pd
import altair as alt

# %%
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data

# %%
mse_data = get_mse_data(
    category="all", collection_date="2023_11_15", processing_level="processed"
)

# %%
mse_data = mse_data[
    mse_data["text"].str.contains("heat pump", case=True)
    | mse_data["title"].str.contains("heat pump", case=True)
]

# %%
# mse_data["text_no_stopwords"] = mse_data["tokens_text_no_stopwords"].apply(lambda x: " ".join(x))

# %%
model = BERTopic()  # Change 'num_topics' to the desired number of topics|

# %%
topics, _ = model.fit_transform(mse_data[mse_data["is_original_post"] == 1]["title"])

# %%
topics = model.get_topics()

# %%
# Convert the data to a DataFrame
topics_df = pd.DataFrame(
    [
        (topic, keyword, weight)
        for topic, keywords in topics.items()
        for keyword, weight in keywords
    ],
    columns=["Topic", "Keyword", "Weight"],
)


# %%
print("Number of topics:", topics_df["Topic"].nunique())

# %%
from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    NESTA_COLOURS,
)

# %%
import seaborn as sns

# %%
# Removing the colour white
nesta = NESTA_COLOURS
nesta = [NESTA_COLOURS[i] for i in range(len(NESTA_COLOURS)) if i != 12]
print(len(nesta))

# %%
color_list = (
    nesta
    if topics_df["Topic"].nunique() <= 13
    else list(sns.color_palette("hls", 30).as_hex())
)

# %%
# Create a bar chart for each topic
charts = []
for i, topic in enumerate(list(topics.keys())[:30]):
    topic_data = topics_df[topics_df["Topic"] == topic]
    chart = (
        alt.Chart(topic_data)
        .mark_bar()
        .encode(
            x=alt.X("Weight:Q", title="Weight"),
            y=alt.Y(
                "Keyword:N",
                title="Keyword",
                sort=alt.SortField(field="Weight", order="descending"),
            ),
            color=alt.ColorValue(color_list[i]),
        )
        .properties(title=f"Topic {topic} Keywords")
    )
    charts.append(chart)

# Combine and display the charts
alt.vconcat(*charts)

# %%


# %%
