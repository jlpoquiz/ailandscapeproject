from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import pandas as pd
import requests
import urllib
from langdetect import detect, LangDetectException

# Change directory to your data folder
os.chdir('/Users/lhampton/Library/CloudStorage/GoogleDrive-lucyjhampton5@gmail.com/.shortcut-targets-by-id/1EznD-uaTblXtaIVx04-toZMQh7cQf9hA/AI Landscape Large Files/Data')

# Load papers CSV
papers = pd.read_csv('constructed/wp_topic_modeling/papers_topics_nocits.csv', index_col=0)
papers = papers.drop_duplicates('abstract')

# Set username and password for the proxy service
# USERNAME = "ljh92_rkqiU"
# PASSWORD = "qJfR_Qdm+s7CQEeb"

USERNAME = "lucyh_OvgeW"
PASSWORD = "qJfR_Qdm+s7CQEeb"

proxies = {
  'http': f'http://{USERNAME}:{PASSWORD}@unblock.oxylabs.io:60000',
  'https': f'https://{USERNAME}:{PASSWORD}@unblock.oxylabs.io:60000',
}

headers = {
    'x-oxylabs-force-headers': '1',
    'html' : 'x-oxylabs-render: html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
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
    abstract_div = soup.find("div", class_="gs_rs")
    if abstract_div:
        abstract_text = abstract_div.get_text()
    else:
        abstract_text = ""
    return abstract_text

# Function to detect if text is in English
def is_english(text):
    try:
        # Skip empty abstracts
        if pd.isna(text) or text.strip() == '':
            return False
        # Detect language
        return detect(text) == 'en'
    except LangDetectException:
        # If there's an error in detection, assume it's not English
        return False

# define the list of DOIs
papers_nodois = papers[papers['doi'].isnull()].reset_index(drop=True)
print(papers_nodois)
print(f"Originally: {len(papers_nodois)}")
papers_nodois = papers_nodois[papers_nodois['abstract'].apply(is_english)]
print(f"After removing non-english: {len(papers_nodois)}")
papers_nodois = papers_nodois.drop_duplicates('abstract')
print(f"After removing duplicates: {len(papers_nodois)}")
paper_rows = papers_nodois.iterrows()
citation_counts = []
processed_doi_list = []
processed_abstract_list = []
missing_papers = []

# Iterate over the list of DOIs
for index, paper in tqdm(paper_rows, desc="Processing papers"):
    # doi = paper['doi']
    # if :
    #     search_url = f"https://scholar.google.com/scholar?q={doi}"
    #     print(f"DEBUG: Searching for DOI: {doi}")
        
    #     html = get_html_for_page(search_url)
    #     soup = BeautifulSoup(html, 'html.parser')

    #     # if no abstract, try again
    #     abstract = get_abstract_from_soup(soup)
    #     if not abstract:
    #         attempts = 0
    #         while attempts < 20 and not abstract:
    #             print(f"DEBUG: No abstract found for DOI: {doi}, retrying... (attempt {attempts+1})")
    #             html = get_html_for_page(search_url)
    #             soup = BeautifulSoup(html, 'html.parser')
    #             abstract = get_abstract_from_soup(soup)
    #             attempts += 1
    #         if not abstract:
    #             print(f"DEBUG: Abstract still not found for DOI: {doi} after retries, skipping...")
    #             missing_papers.append(doi)
    #             continue
    #     # proceed
    #     if abstract:
    #         citation_count = get_citation_count_from_soup(soup)
    #         citation_counts.append(citation_count)
    #         processed_doi_list.append(doi)
    abstract_search = paper['abstract'][:500]
    encoded_abstract = urllib.parse.quote(abstract_search)
    search_url = f"https://scholar.google.com/scholar?as_q=%22{encoded_abstract}%22"
    print(f"DEBUG: Searching for abstract: {abstract_search}")
    html = get_html_for_page(search_url)
    soup = BeautifulSoup(html, 'html.parser')

    
    abstract = get_abstract_from_soup(soup)
    # get citation count, if abstract found
    if abstract:
        citation_count = get_citation_count_from_soup(soup)
        citation_counts.append(citation_count)
        processed_abstract_list.append(abstract_search)
    # if no abstract, try again
    if not abstract:
        attempts = 0
        while attempts < 15 and not abstract:
            print(f"DEBUG: No abstract found for abstract: {abstract_search}, retrying... (attempt {attempts+1})")
            html = get_html_for_page(search_url)
            soup = BeautifulSoup(html, 'html.parser')
            abstract = get_abstract_from_soup(soup)
            attempts += 1
        if not abstract:
            print(f"DEBUG: Abstract still not found for abstract: {abstract_search} after retries, skipping...")
            missing_papers.append(abstract_search)
            continue
        # proceed
        if abstract:
            citation_count = get_citation_count_from_soup(soup)
            citation_counts.append(citation_count)
            processed_abstract_list.append(abstract_search)

citations_df = pd.DataFrame({
    'abstract': processed_abstract_list,
    'citation_count': citation_counts
})
print(citations_df)
citations_df.to_csv('constructed/wp_topic_modeling/citations_abstracts.csv')