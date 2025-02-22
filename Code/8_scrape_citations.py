import pandas as pd
from scholarly import scholarly, ProxyGenerator
import os
import sys
sys.path.insert(1, 'ssrn_scraper/')

import importlib

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait

import pickle
import time
import random

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tqdm import tqdm

os.chdir('/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data')

# load papers csv
papers = pd.read_csv('constructed/wp_topic_modeling/papers_topics_wps.csv')

# merge with urls
orig = pd.read_csv('original/working_papers/ssrn/ssrn_results.csv')
ssrn = pd.merge(papers, orig, on='doi', how = 'left')

# scrape links
importlib.reload(scraper)

# Load progress from the last run (if any)
try:
    ssrn_results = pd.read_csv("constructed/ssrn_cits.csv")
    processed_urls = set(ssrn_results['Link'])
    data_dict = ssrn_results.to_dict('list')
except FileNotFoundError:
    ssrn_results = pd.DataFrame(columns=['doi', 'Link', 'Citations'])
    processed_urls = set()
    data_dict = {
        'doi': [],
        'Link': [],
        'Citations': []
    }

# load the driver
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_service = Service('/usr/local/bin/chromedriver')  # Update path if needed
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
ssrn_url = "https://papers.ssrn.com/sol3/DisplayAbstractSearch.cfm"
driver.get(ssrn_url)
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# load ssrn home page to accept cookies
driver.get('https://www.ssrn.com/index.cfm/en/')
time.sleep(1)
button = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div[2]/div/div[1]/div/div[2]/div/button[2]"))
)
button.click()

# Start run from last url
with open('unique_urls.pickle', 'rb') as handle:
    unique_urls = pickle.load(handle)
    
for url in tqdm(unique_urls, desc="Processing URLs"):
    if url in processed_urls:
        continue  # Skip URLs that have already been processed
    
    # change user agent
    ua = UserAgent()
    user_agent = ua.random
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": user_agent})

    # load
    driver.get(url)

    # random respectful window movement
    time.sleep(3)
    x = random.randint(0, driver.get_window_size()['width'])
    y = random.randint(0, driver.get_window_size()['height'])
    driver.execute_script(f"window.scrollTo({x}, {y});")

    # get the soup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Append to dict
    try:
        data_dict['Title'].append(get_title(soup))
        data_dict['Authors'].append(get_authors(soup))
        data_dictdoi
    except IndexError:
        time.sleep(1)
        page_text = soup.get_text()
        if 'This paper has been removed from SSRN at the request of the author' in page_text:
            print('paper removed')
            data_dict['Title'].append('')
            data_dict['Authors'].append('')
            data_dict['Date'].append('')
            data_dict['Abstract'].append('')
            data_dict['Keywords'].append('')
            data_dict['doi'].append('')
            data_dict['Search-term'].append('')
            data_dict['Source'].append("SSRN"),
            data_dict['Link'].append(url)
            continue
        else:
            print('paper present')
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            data_dict['Title'].append(get_title(soup))
            data_dict['Authors'].append(get_authors(soup))
            data_dict['Date'].append(get_date(soup))
            data_dict['Abstract'].append(get_abstract(soup))
            data_dict['Keywords'].append(get_keywords(soup))
            data_dict['doi'].append(get_doi(soup))
            data_dict['Search-term'].append('')
            data_dict['Source'].append("SSRN"),
            data_dict['Link'].append(url)

    # create df out of dict
    ssrn_results = pd.DataFrame(data_dict)
    
    # save progress, add url to processed url list
    if len(ssrn_results) % 10 == 0:
        try:
            if len(ssrn_results) - len(pd.read_csv("ssrn_results.csv")) == 10:
                ssrn_results.to_csv("ssrn_results.csv", index=False)
            else:
                print('warning!')
                break
        except FileNotFoundError:
            ssrn_results.to_csv("ssrn_results.csv", index=False)
    processed_urls.add(url)

# Display the DataFrame
ssrn_results = pd.DataFrame(data_dict)
ssrn_results