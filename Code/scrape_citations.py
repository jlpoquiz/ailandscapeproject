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
USERNAME = "ljh92_P3sZg"
PASSWORD = "qJfR_Qdm+s7CQEeb"

def get_html_for_page(url):
    print(f"DEBUG: Fetching URL: {url}")
    payload = {
        "url": url,
        "source": "google",
    }
    try:
        response = requests.post(
            "https://realtime.oxylabs.io/v1/queries",
            auth=(USERNAME, PASSWORD),
            json=payload,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"DEBUG: Error fetching URL: {url} - {e}")
        return ""
    
    json_resp = response.json()
    if "results" not in json_resp or not json_resp["results"]:
        print(f"DEBUG: No results found in the response for URL: {url}")
        return ""
    
    html_content = json_resp["results"][0].get("content", "")
    print(f"DEBUG: Retrieved HTML length: {len(html_content)}")
    return html_content

def get_citation_count_from_article(article):
    citation_link = article.find("a", text=lambda x: x and "Cited by" in x)
    if citation_link:
        citation_text = citation_link.get_text()
        try:
            citation_count = citation_text.split("Cited by ")[1].strip()
        except IndexError:
            citation_count = "N/A"
            print(f"DEBUG: Warning: Could not extract citation count from text: {citation_text}")
    else:
        citation_count = "0"
        print("DEBUG: Warning: Could not find citation link in article.")
    print(f"DEBUG: Citation count extracted: {citation_count}")
    return citation_count

def parse_data_from_article(article):
    # Debug print for the raw article snippet (limited)
    print("DEBUG: Parsing an article snippet...")
    title_elem = article.find("h3", {"class": "gs_rt"})
    if not title_elem:
        print("DEBUG: Warning: Title element not found.")
        return {}
    
    try:
        title = title_elem.get_text()
    except Exception as e:
        print(f"DEBUG: Error extracting title: {e}")
        title = "N/A"
    
    # Sometimes the <a> element may not be present
    title_anchor_elems = title_elem.find_all("a")
    if title_anchor_elems:
        title_anchor_elem = title_anchor_elems[0]
        url = title_anchor_elem.get("href", "")
        # The article id might not always be available; we use .get() with a default
        article_id = title_anchor_elem.get("id", "N/A")
    else:
        print("DEBUG: Warning: No anchor element found in title element.")
        url = ""
        article_id = "N/A"
    
    authors_elem = article.find("div", {"class": "gs_a"})
    if authors_elem:
        authors = authors_elem.get_text()
    else:
        authors = "N/A"
        print("DEBUG: Warning: Authors element not found.")
    
    citation_count = get_citation_count_from_article(article)
    
    parsed_data = {
        "title": title,
        "authors": authors,
        "url": url,
        "citations": citation_count,
    }
    print(f"DEBUG: Parsed data: {parsed_data}")
    return parsed_data

def get_url_for_page(url, page_index):
    full_url = url + f"&start={page_index}"
    print(f"DEBUG: Constructed page URL: {full_url}")
    return full_url

def get_data_from_page(url):
    html = get_html_for_page(url)
    if not html:
        print("DEBUG: No HTML returned for URL: " + url)
        return []
    soup = BeautifulSoup(html, "html.parser")
    articles = soup.find_all("div", {"class": "gs_ri"})
    print(f"DEBUG: Found {len(articles)} article(s) on the page.")
    parsed_articles = []
    for article in articles:
        parsed = parse_data_from_article(article)
        if parsed:
            parsed_articles.append(parsed)
    return parsed_articles

# Now get citation data for each paper; here only the first two rows are processed for testing.
citation_counts = []
paper_rows = papers[:2].iterrows()

for index, paper in tqdm(paper_rows, desc="Processing papers"):
    doi = paper['doi']
    search_url = f"https://scholar.google.com/scholar?q={doi}"
    print(f"DEBUG: Searching for DOI: {doi}")
    
    NUM_OF_PAGES = 1
    page_index = 0
    paper_entries = []
    for _ in range(NUM_OF_PAGES):
        page_url = get_url_for_page(search_url, page_index)
        entries = get_data_from_page(page_url)
        if entries:
            paper_entries.extend(entries)
        page_index += 10

    # If no entries were found, print a warning message.
    if not paper_entries:
        print(f"WARNING: No data extracted for paper with DOI: {doi}")
    else:
        citation_counts.extend(paper_entries)

# Optionally, convert the results into a DataFrame and display
citations_df = pd.DataFrame(citation_counts)
print("DEBUG: Final DataFrame:")
print(citations_df)
