from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import pandas as pd
import requests

# Change directory to your data folder
os.chdir('/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data')

# Load papers CSV
papers = pd.read_csv('constructed/wp_topic_modeling/papers_topics_wps.csv', index_col=0)

# Set username and password for the proxy service
USERNAME = "ljh92_rkqiU"
PASSWORD = "qJfR_Qdm+s7CQEeb"

proxies = {
  'http': f'http://{USERNAME}:{PASSWORD}@unblock.oxylabs.io:60000',
  'https': f'https://{USERNAME}:{PASSWORD}@unblock.oxylabs.io:60000',
}

headers = {
    'html' : 'x-oxylabs-render: html',
    'x-oxylabs-geo-location' : 'United States'
}

response = requests.request(
    'GET',
    'https://ip.oxylabs.io/headers',
    verify=False,  # Ignore the SSL certificate
    proxies=proxies
)

def get_html_for_page(url):
    print(f"DEBUG: Fetching URL: {url}")
    try:
        response = requests.get(url, proxies=proxies, headers=headers, verify=False)
        response.raise_for_status()
    except Exception as e:
        print(f"DEBUG: Error fetching URL: {url} - {e}")
        return ""
    
    html_content = response.text
    return html_content

def get_citation_count_from_soup(soup):
    citation_link = soup.find("a", string=lambda x: x and "Cited by" in x)
    print(citation_link)
    if citation_link:
        citation_text = citation_link.get_text()
        citation_count = citation_text.split("Cited by ")[1].strip()
    else:
        citation_count = 0
    return citation_count

def get_abstract_from_soup(soup):
    abstract_div = soup.find("div", class_="gs_rs gs_fma_s")
    if abstract_div:
        abstract_text = abstract_div.get_text()
    else:
        abstract_text = ""
    return abstract_text

# define the list of DOIs
paper_rows = papers[:2].iterrows()
citation_counts = []
processed_doi_list = []

# Iterate over the list of DOIs
for index, paper in tqdm(paper_rows, desc="Processing papers"):
    doi = paper['doi']
    if doi:
        search_url = f"https://scholar.google.com/scholar?q={doi}"
        print(f"DEBUG: Searching for DOI: {doi}")
        
        html = get_html_for_page(search_url)
        soup = BeautifulSoup(html, 'html.parser')

        # if no abstract, try again
        abstract = get_abstract_from_soup(soup)
        if not abstract:
            attempts = 0
            while attempts < 10 and not abstract:
                print(f"DEBUG: No abstract found for DOI: {doi}, retrying... (attempt {attempts+1})")
                html = get_html_for_page(search_url)
                soup = BeautifulSoup(html, 'html.parser')
                abstract = get_abstract_from_soup(soup)
                attempts += 1
            if not abstract:
                print(f"DEBUG: Abstract still not found for DOI: {doi} after retries, skipping...")
                continue

        # proceed
        if abstract:
            citation_count = get_citation_count_from_soup(soup)
            citation_counts.append(citation_count)
            processed_doi_list.append(doi)


citations_df = pd.DataFrame({
    'doi': processed_doi_list,
    'citation_count': citation_counts
})
print(citations_df)