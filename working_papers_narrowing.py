# %%
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time
import logging
from tqdm import tqdm
import re
import os

# %%
username = os.getlogin()
if username =="lucyjhampton":
    local_path = r"/Users/lucyjhampton/Projects/AI Landscape"
    os.chdir(local_path)

#%%
import gdown

# File ID and destination path for the data
url = r"https://drive.google.com/uc?id=10zgN1i8qajRDkNcLvc6gwd8zbCaYTXHV"
output = f'{local_path}/Data/NBER_no_duplicates.csv'

# File ID and destination path for the data
url = r"https://drive.google.com/uc?id=1e1Q3mwg3iN3plM6S4m_TjRM5JAI7FrJu"
output = f'{local_path}/Data/SSRN_no_duplicates.csv'

# Ensure the data is downloaded before proceeding
gdown.download(url, output, quiet=False)

# %%

NBER_no_duplicates = pd.read_csv(f'{local_path}/Data/NBER_no_duplicates.csv')
NBER_no_duplicates['Source'] = "NBER"

SSRN_no_duplicates = pd.read_csv(f'{local_path}/Data/SSRN_no_duplicates.csv')
SSRN_no_duplicates['Source'] = "SSRN"

working_papers = pd.concat([NBER_no_duplicates, SSRN_no_duplicates])

working_papers = working_papers.drop(columns = {"type", "journal", "M1", "M2", "AB", "endref", "cond", "nber_is_published", "nber_in_ssrn", "keywords", "searchterm", "source", "ssrn_is_published", "tow", "vol", "access", "note", "url", "date"})
working_papers
# %%
