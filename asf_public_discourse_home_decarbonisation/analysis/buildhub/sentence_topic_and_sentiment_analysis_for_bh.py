#!/usr/bin/env python
# coding: utf-8

# In[1]:


from asf_public_discourse_home_decarbonisation.getters.bh_getters import get_bh_data
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    plot_post_distribution_over_time,
    plot_mentions_line_chart,
)
import pandas as pd
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    process_abbreviations,
)
from asf_public_discourse_home_decarbonisation.utils.preprocessing_utils import (
    preprocess_data_for_linechart_over_time,
    resample_and_calculate_averages_for_linechart,
)


# In[2]:


bh_data = get_bh_data(category="all")
print("Date Range:", bh_data["date"].min(), "to", bh_data["date"].max())


# In[3]:


key_terms = ["heat pump", "boiler"]
# Preprocess data
bh_data = preprocess_data_for_linechart_over_time(bh_data, key_terms=key_terms)
# Resample and calculate averages
bh_data_monthly = resample_and_calculate_averages_for_linechart(
    bh_data, key_terms=key_terms, cadence_of_aggregation="M", window=12
)


# In[4]:


key_terms_colour = {"heat pump": "#97D9E3", "boiler": "#0F294A"}
plot_mentions_line_chart(bh_data_monthly, key_terms_colour, plot_type="both")


# In[4]:


# In[ ]:


# In[ ]:
