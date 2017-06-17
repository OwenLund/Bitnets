from pytrends.request import TrendReq
import numpy as np
import os
import matplotlib.pyplot as plt

# enter your own credentials
google_username = "owen.p.lund@gmail.com"

google_password = open(os.path.join(os.path.expanduser('~'),'.googlepass'), "r").read() 

path = ""

# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq(google_username, google_password)

# Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()


pytrend.build_payload(kw_list=['bitcoin', 'btc'],timeframe='now 7-d')

# Interest Over Time
interest_over_time_df = pytrend.interest_over_time()

interest_over_time_df.plot()
plt.show()

"""
# Interest by Region
interest_by_region_df = pytrend.interest_by_region()
print interest_by_region_df

# Related Queries, returns a dictionary of dataframes
related_queries_dict = pytrend.related_queries()

# Get Google Hot Trends data
trending_searches_df = pytrend.trending_searches()

# Get Google Top Charts
top_charts_df = pytrend.top_charts(cid='actors', date=201611)

# Get Google Keyword Suggestions
suggestions_dict = pytrend.suggestions(keyword='pizza')

#"""