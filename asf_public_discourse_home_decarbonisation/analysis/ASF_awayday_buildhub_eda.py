import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import nltk
from nltk import FreqDist
from nltk import bigrams, trigrams
from nltk.corpus import stopwords
import pandas as pd
import s3fs
import os
from asf_public_discourse_home_decarbonisation import PROJECT_DIR
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    set_plotting_styles,
    finding_path_to_font,
    NESTA_COLOURS,
)

set_plotting_styles()

font_path_ttf = finding_path_to_font("Averta-Regular")

# BUILDHUB_OUTPUTS_DATA_PATH = os.path.join(PROJECT_DIR, "outputs/figures/buildhub/forum_119_air_source_heat_pumps_ashp/")
# BUILDHUB_OUTPUTS_DATA_PATH = os.path.join(PROJECT_DIR, "outputs/figures/buildhub/forum_120_ground_source_heat_pumps_gshp/")
# BUILDHUB_OUTPUTS_DATA_PATH = os.path.join(PROJECT_DIR, "outputs/figures/buildhub/forum_125_general_alternative_energy_issues/")
# BUILDHUB_OUTPUTS_DATA_PATH = os.path.join(PROJECT_DIR, "outputs/figures/buildhub/forum_136_underfloor_heating/")
# BUILDHUB_OUTPUTS_DATA_PATH = os.path.join(PROJECT_DIR, "outputs/figures/buildhub/forum_137_central_heating_radiators/")
# BUILDHUB_OUTPUTS_DATA_PATH = os.path.join(PROJECT_DIR, "outputs/figures/buildhub/forum_139_boilers_hot_water_tanks/")
BUILDHUB_OUTPUTS_DATA_PATH = os.path.join(
    PROJECT_DIR, "outputs/figures/buildhub/forum_140_other_heating_systems/"
)
# Ensure the output directory exists
os.makedirs(BUILDHUB_OUTPUTS_DATA_PATH, exist_ok=True)

# S3 URI
# buildhub_ashp_s3_uri = "s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/buildhub_forum_119_air_source_heat_pumps_ashp__231101.csv"
# buildhub_ashp_s3_uri = "s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/buildhub_forum_120_ground_source_heat_pumps_gshp__231114.csv"
# buildhub_ashp_s3_uri = "s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/buildhub_forum_125_general_alternative_energy_issues__231114.csv"
# buildhub_ashp_s3_uri = "s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/buildhub_forum_136_underfloor_heating__231114.csv"
# buildhub_ashp_s3_uri = "s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/buildhub_forum_137_central_heating_radiators__231114.csv"
# buildhub_ashp_s3_uri = "s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/buildhub_forum_139_boilers_hot_water_tanks__231114.csv"
buildhub_ashp_s3_uri = "s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/buildhub_forum_140_other_heating_systems__231114.csv"

# Read CSV file from S3
buildhub_ashp_dataframe = pd.read_csv(buildhub_ashp_s3_uri, parse_dates=["date"])
sample_size = buildhub_ashp_dataframe.shape[0]  # shape[0] gives the number of rows

######### 1. Distribution of posts over time #########

# Assuming buildhub_ashp_dataframe['date'] is already in datetime format
buildhub_ashp_dataframe["year"] = buildhub_ashp_dataframe["date"].dt.year

# Group the data by year and count the number of posts
post_count_by_year = (
    buildhub_ashp_dataframe.groupby("year").size().reset_index(name="post_count")
)

# Identify the minimum and maximum years in the data
min_year = post_count_by_year["year"].min()
max_year = post_count_by_year["year"].max()

# Create a DataFrame that includes all years between min_year and max_year
all_years = pd.DataFrame({"year": list(range(min_year, max_year + 1))})

# Merge this DataFrame with your original DataFrame to include missing years
post_count_by_year = pd.merge(
    all_years, post_count_by_year, on="year", how="left"
).fillna(0)

# Plot the bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x="year", y="post_count", data=post_count_by_year, color=NESTA_COLOURS[0])
plt.title(
    f"Distribution of Posts Over Time (By Year) (total sample size n = {sample_size})"
)
plt.xlabel("Year")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.savefig(
    os.path.join(BUILDHUB_OUTPUTS_DATA_PATH, "Distribution_of_Posts_By_Year.png")
)
plt.show()


########## 2. Number of posts per user ###########
user_post_counts = buildhub_ashp_dataframe["username"].value_counts().reset_index()
user_post_counts.columns = ["username", "post_count"]

plt.figure(figsize=(12, 8))
sns.barplot(
    x="username", y="post_count", data=user_post_counts.head(10), palette="coolwarm"
)
plt.title("Top 10 Users by Number of Posts")
plt.xlabel("Username")
plt.ylabel("Number of Posts")
plt.xticks(rotation=24)
plt.tight_layout()
plt.savefig(os.path.join(BUILDHUB_OUTPUTS_DATA_PATH, "Number_of_Posts_Username.png"))
plt.show()

######### 3. Word cloud of frequently used words in posts ############

text_data = " ".join(buildhub_ashp_dataframe["text"].astype(str)).lower()
stop_words = set(stopwords.words("english"))
new_stopwords = [
    "would",
    "hours",
    "hour",
    "minute",
    "minutes",
    "ago",
    "ago,",
    "dan",
    "(presumably",
    "looks",
    "like",
    "need",
    "ap50",
    ".3page",
    "fraser",
    "lamont",
    "got",
    "bit",
    "sure",
    "steamytea",
    "could",
    "get",
    "still",
    "October",
    "october",
    "6",
    "2013",
    "January",
    "january",
    "2016",
    "moderator",
    "thisreport",
]
stop_words.update(new_stopwords)

# Tokenize the text into words
tokens = text_data.split()

# Remove manual stopwords
filtered_tokens = [word for word in tokens if word not in stop_words]

# Create a frequency distribution of the filtered tokens
freq_dist = FreqDist(filtered_tokens)

# Apply frequency filter to remove words with a count less than 10
total_words = sum(freq_dist.values())
# threshold = round(total_words * 0.002)
# threshold = max(10, total_words * 0.0002)
threshold = 100
max_words_in_wordcloud = 30
filtered_freq_dist = {
    word: freq for word, freq in freq_dist.items() if freq >= threshold
}

# Generate the word cloud
wordcloud = WordCloud(
    font_path=font_path_ttf,
    background_color="white",
    width=800,
    height=400,
    max_words=max_words_in_wordcloud,
).generate_from_frequencies(filtered_freq_dist)

# Show the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title(f"Enhanced Word Cloud (with Frequency Filter > {threshold})")
plt.tight_layout()
plt.savefig(os.path.join(BUILDHUB_OUTPUTS_DATA_PATH, "Word_Cloud_freq_filter.png"))
plt.show()

######## 4.Generate bigram and trigram frequency distributions ########
raw_bigram_freq_dist = FreqDist(bigrams(filtered_tokens))
raw_trigram_freq_dist = FreqDist(trigrams(filtered_tokens))

total_bigrams = sum(raw_bigram_freq_dist.values())
# bigram_threshold = round(total_bigrams * 0.0002)
bigram_threshold = 100
bigram_freq_dist = {
    bigram: freq
    for bigram, freq in raw_bigram_freq_dist.items()
    if freq >= bigram_threshold
}
total_trigrams = sum(raw_trigram_freq_dist.values())
# trigram_threshold = round(max(3, total_trigrams * 0.00005))
trigram_threshold = 20
trigram_freq_dist = {
    trigram: freq
    for trigram, freq in raw_trigram_freq_dist.items()
    if freq >= trigram_threshold
}

# Get the top 10 most common bigrams and trigrams
top_10_bigrams = raw_bigram_freq_dist.most_common(10)
top_10_trigrams = raw_trigram_freq_dist.most_common(10)

# Separate bigrams/trigrams and frequencies
bigram_labels, bigram_freqs = zip(*top_10_bigrams)
trigram_labels, trigram_freqs = zip(*top_10_trigrams)

# Convert bigram and trigram tuples to strings for labeling
bigram_labels = ["_".join(label).replace("_", " ") for label in bigram_labels]
trigram_labels = ["_".join(label).replace("_", " ") for label in trigram_labels]

# # Plot bigrams
plt.figure(figsize=(12, 8))
plt.bar(bigram_labels, bigram_freqs, color=NESTA_COLOURS[0])
plt.ylabel("Frequency")
plt.xlabel("Bigrams")
plt.xticks(rotation=24)
plt.title("Top 10 Most Common Bigrams")
plt.tight_layout()
plt.savefig(os.path.join(BUILDHUB_OUTPUTS_DATA_PATH, "Top_10_Most_Common_Bigrams.png"))
plt.show()

# Plot bigrams - barh
plt.figure(figsize=(12, 8))
plt.barh(bigram_labels, bigram_freqs, color=NESTA_COLOURS[0])
plt.xlabel("Frequency")
plt.ylabel("Bigrams")
plt.title("Top 10 Most Common Bigrams")
plt.tight_layout()
plt.savefig(
    os.path.join(
        BUILDHUB_OUTPUTS_DATA_PATH, "Top_10_Most_Common_Bigrams_horizontal.png"
    )
)
plt.show()

# Plot trigrams
plt.figure(figsize=(12, 8))
plt.bar(trigram_labels, trigram_freqs, color=NESTA_COLOURS[1])
plt.ylabel("Frequency")
plt.xlabel("Trigrams")
plt.xticks(rotation=24)
plt.title("Top 10 Most Common Trigrams")
plt.tight_layout()
plt.savefig(os.path.join(BUILDHUB_OUTPUTS_DATA_PATH, "Top_10_Most_Common_Trigrams.png"))
plt.show()

# Plot trigrams - barh
plt.figure(figsize=(12, 8))
plt.barh(trigram_labels, trigram_freqs, color=NESTA_COLOURS[1])
plt.xlabel("Frequency")
plt.ylabel("Trigrams")
plt.title("Top 10 Most Common Trigrams")
plt.tight_layout()
plt.savefig(
    os.path.join(
        BUILDHUB_OUTPUTS_DATA_PATH, "Top_10_Most_Common_Trigrams_horizontal.png"
    )
)
plt.show()

######## 5.Generate bigram and trigram word clouds ########
# Convert bigrams and trigrams to strings
bigram_strings = [" ".join(bigram) for bigram in bigram_freq_dist.keys()]
trigram_strings = [" ".join(trigram) for trigram in trigram_freq_dist.keys()]

# Create frequency distributions for bigram and trigram strings
bigram_string_freq_dist = FreqDist(
    [" ".join(bigram) for bigram in bigram_freq_dist.keys()]
)
trigram_string_freq_dist = FreqDist(
    [" ".join(trigram) for trigram in trigram_freq_dist.keys()]
)

# Update frequency counts
for bigram, freq in bigram_freq_dist.items():
    bigram_string = " ".join(bigram)
    bigram_string_freq_dist[bigram_string] = freq

for trigram, freq in trigram_freq_dist.items():
    trigram_string = " ".join(trigram)
    trigram_string_freq_dist[trigram_string] = freq

# Generate the word cloud for bigrams
wordcloud_bigrams = Wwordcloud = WordCloud(
    font_path=font_path_ttf,
    background_color="white",
    width=800,
    height=400,
    max_words=max_words_in_wordcloud,
).generate_from_frequencies(bigram_string_freq_dist)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_bigrams, interpolation="bilinear")
plt.axis("off")
plt.title(f"Word Cloud of Bigrams (with Frequency > {bigram_threshold})")
plt.tight_layout()
plt.savefig(os.path.join(BUILDHUB_OUTPUTS_DATA_PATH, "word_cloud_common_bigrams.png"))
plt.show()

# Generate the word cloud for trigrams
wordcloud_trigrams = WordCloud(
    font_path=font_path_ttf,
    background_color="white",
    width=800,
    height=400,
    max_words=max_words_in_wordcloud,
).generate_from_frequencies(trigram_string_freq_dist)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_trigrams, interpolation="bilinear")
plt.axis("off")
plt.title(f"Word Cloud of Trigrams (with Frequency > {trigram_threshold})")
plt.tight_layout()
plt.savefig(os.path.join(BUILDHUB_OUTPUTS_DATA_PATH, "word_cloud_common_trigrams.png"))
plt.show()


############ 6. Distribution of post lengths #############
buildhub_ashp_dataframe["post_length"] = buildhub_ashp_dataframe["text"].apply(
    lambda x: len(str(x).split())
)

plt.figure(figsize=(12, 8))
sns.histplot(
    buildhub_ashp_dataframe["post_length"], bins=50, kde=False, color=NESTA_COLOURS[2]
)
plt.title("Distribution of Post Lengths")
plt.xlabel("Post Length (Number of Words)")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.savefig(
    os.path.join(BUILDHUB_OUTPUTS_DATA_PATH, "distribution_of_post_lengths.png")
)
plt.show()


# same as above but y-axis with log scale
plt.figure(figsize=(12, 8))
sns.histplot(
    buildhub_ashp_dataframe["post_length"], bins=50, kde=False, color=NESTA_COLOURS[2]
)
plt.title("Distribution of Post Lengths")
plt.xlabel("Post Length (Number of Words)")
plt.ylabel("Number of Posts (log scale")
plt.yscale("log")
plt.tight_layout()
plt.savefig(
    os.path.join(
        BUILDHUB_OUTPUTS_DATA_PATH, "distribution_of_post_lengths_logscale.png"
    )
)
plt.show()


############ 7. Frequency of selected keywords in posts using the dictionary ###########
custom_keyword_counter = Counter()


heating_technologies_ruleset_twitter = [
    {
        "value": "heat pump|heat pumps",
        "tag": "general_heat_pumps",
    },
    {
        "value": "air source hp|air source hps|air-source hp|air-source hps",
        "tag": "ashp",
    },
    {
        "value": "ground source hp|ground source hps|ground-source hp|ground-source hps",
        "tag": "gshp",
    },
    {
        "value": '"water source hp|"water source hps|water-source hp|water-source hps',
        "tag": "wshp",
    },
    {
        "value": "air to air hp|air to air hps|air-to-air hp|air-to-air hps|air2air hp|air2air hps|a2a hp|a2a hps",
        "tag": "atahp",
    },
    {
        "value": 'air to water hp|air to water hps|air-to-water hp|air-to-water hps|"air2water hps|air2water hp|a2w hp|a2w hps',
        "tag": "atwhp",
    },
    {
        "value": "hybrid hp|hybrid hps|bivalent hp|bivalent hps|warm air hp|warm air hps",
        "tag": "heat_pump_others",
    },
    {
        "value": "boiler|boilers|#boiler|#boilers",
        "tag": "general_boilers",
    },
    {
        "value": "combi boiler | combi boilers | combi-boiler | combi-boilers ",
        "tag": "combi_boilers",
    },
    {
        "value": " gas boiler | gas boilers ",
        "tag": "gas_boilers",
    },
    {
        "value": " oil boiler | oil boilers ",
        "tag": "oil_boilers",
    },
    {
        "value": "hydrogen boiler | hydrogen boilers ",
        "tag": "hydrogen_boilers",
    },
    {
        "value": "hydrogen ready boiler|hydrogen ready boilers|hydrogen-ready boiler|hydrogen-ready boilers",
        "tag": "hydrogen_ready_boilers",
    },
    {
        "value": "electric boiler|electric boilers",
        "tag": "electric_boilers",
    },
    {
        "value": "biomass boiler|biomass boilers",
        "tag": "biomass_boilers",
    },
    {
        "value": "electric heating|electric radiator|electric radiators|central heating",
        "tag": "heating_others",
    },
    {"value": "solar thermal|solar water heating", "tag": "solar_thermal"},
    {"value": "heating system|heating systems", "tag": "heating_system"},
    {
        "value": "district heating|heat network|heat networks",
        "tag": "district_heating",
    },
    {"value": "hybrid heating", "tag": "hybrid_heating"},
    {
        "value": "hp installation|hp installations|hybrid installation|hybrid installations|hybrid heating installation|hybrid heating installations",
        "tag": "installations",
    },
    {
        "value": "hp installer|hp installers|hp engineer|hp engineers|heating engineer|heating engineers|boiler engineer|boiler engineers",
        "tag": "installers_and_engineers",
    },
    {
        "value": "retrofit installer|retrofit installers|renewables installer|renewables installers",
        "tag": "retrofit_or_renewable_installers",
    },
    {
        "value": "low carbon heating|low-carbon heating|#lowcarbonheating|home decarbonisation",
        "tag": "low_carbon_heating_and_home_decarbonisation",
    },
    {
        "value": "bus scheme|boiler upgrade scheme|renewable heat incentive|domestic rhi|clean heat grant|home energy scotland grant|home energy scotland loan|home energy scotland scheme",
        "tag": "government_grants",
    },
    {
        "value": "microgeneration certification scheme|mcs certified|mcs certification|mcs installation|mcs installations",
        "tag": "mcs",
    },
    {
        "value": "hp cost estimator|hp cost calculator",
        "tag": "nesta_cost_estimator_tool",
    },
]


def update_keyword_frequencies_custom(text, ruleset):
    """
    Updates frequencies of keywords based on a custom ruleset within the provided text.

    This function iterates through a ruleset, which contains regex patterns and corresponding tags.
    If a pattern is found within the text, it increments the count for the associated tag in a global
    counter.

    Parameters:
        text (str): The text in which to search for keywords.
        ruleset (list): A list of dictionaries where each dictionary contains a 'value' key with a
                        regex pattern and a 'tag' key with the corresponding tag to be updated.

    Returns:
        None: This function updates the global counter in place and does not return anything.
    """
    for rule in ruleset:
        value = rule["value"]
        tag = rule["tag"]
        if re.search(value, text, re.IGNORECASE):
            custom_keyword_counter[tag] += 1


buildhub_ashp_dataframe["text"].apply(
    lambda x: update_keyword_frequencies_custom(
        str(x), heating_technologies_ruleset_twitter
    )
)

custom_keyword_buildhub_ashp_dataframe = pd.DataFrame.from_dict(
    custom_keyword_counter, orient="index", columns=["Frequency"]
).reset_index()
custom_keyword_buildhub_ashp_dataframe.columns = ["Tag", "Frequency"]
custom_keyword_buildhub_ashp_dataframe = (
    custom_keyword_buildhub_ashp_dataframe.sort_values(by="Frequency", ascending=False)
)
total_rows = len(custom_keyword_buildhub_ashp_dataframe)
df_threshold = round(max(5, total_rows * 0.0001))
custom_keyword_buildhub_ashp_dataframe = custom_keyword_buildhub_ashp_dataframe[
    custom_keyword_buildhub_ashp_dataframe["Frequency"] > df_threshold
]


plt.figure(figsize=(12, 8))
custom_keyword_buildhub_ashp_dataframe["Tag"] = custom_keyword_buildhub_ashp_dataframe[
    "Tag"
].str.replace("_", " ")
sns.barplot(
    x="Tag",
    y="Frequency",
    data=custom_keyword_buildhub_ashp_dataframe,
    palette="winter",
)
plt.title(f"Frequency of Selected Keywords in Posts (Frequency > {df_threshold})")
plt.xlabel("Tag")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(BUILDHUB_OUTPUTS_DATA_PATH, "freq_of_selected_keywords_in_posts.png")
)
plt.show()

# same as above but with barh
plt.figure(figsize=(12, 8))
plt.barh(
    custom_keyword_buildhub_ashp_dataframe["Tag"],
    custom_keyword_buildhub_ashp_dataframe["Frequency"],
    color=NESTA_COLOURS[0],
)
plt.title(f"Frequency of Selected Keywords in Posts (Frequency > {df_threshold})")
plt.ylabel("Tag")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig(
    os.path.join(
        BUILDHUB_OUTPUTS_DATA_PATH, "freq_of_selected_keywords_in_posts_barh.png"
    )
)
plt.show()
