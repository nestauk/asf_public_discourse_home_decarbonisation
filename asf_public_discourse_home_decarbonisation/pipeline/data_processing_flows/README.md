# 🛠 Data processing

This folder contains scripts to perform data processing tasks on large amounts of MSE text data making use of Metaflow.

The processing pipeline includes:

- reading in the `DataFrame` with text data;
- Pre-processing the text columns, e.g.:
  - replacing &amp/& with "and"
  - replacing ">" and "=" with space
  - removing URLs
  - transforming text to lower case
  - removing username patterns
  - removing emojis
- lemmating and tokenise text data;
- remove stopwords from text data;

Processed data is stored on S3.

To run the pipeline:

```
python asf_public_discourse_home_decarbonisation/pipeline/data_processing_flows/processing_text_data.py --datastore=s3 --package-suffixes=.txt run --max-num-splits 2000 --max-workers 100 --category <category>
```

where `<category>`` is one of the sub-forums:

- "green-ethical-moneysaving": Green and Ethical Money Saving sub-forum.
- "lpg-heating-oil-solid-other-fuels": LPG, heating, oil, solid and other fuels sub-forum.
- "energy": Energy sub-forum.
- "is-this-quote-fair": Is this quote fair? sub-forum.
- "heat-pumps": heat pumps sub-forum.

Running a Metaflow script requires scripts called by the flow to be in the same folder, so `text_processing_utils.py` and `flow_requirements.txt` are located in this folder. `text_processing_utils.py` contains utils for processing text and `flow_requirements.txt` contain the required packages to run the flow.
