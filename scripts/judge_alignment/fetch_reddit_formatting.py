#!/usr/bin/env python3
"""
Fetch original Reddit formatting for questions in balanced_30.csv.

This script searches for the original r/askscience posts and retrieves
the properly formatted question titles and top answers (with newlines,
capitalization, etc.).

Uses Selenium with Google search to find Reddit posts, then Reddit API to fetch content.

Usage:
    python fetch_reddit_formatting.py
"""

import os
import re
import time
import pandas as pd
import praw
from pathlib import Path
from tqdm import tqdm
from urllib.parse import quote_plus
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Paths
SCRIPT_DIR = Path(__file__).parent
INPUT_PATH = SCRIPT_DIR / "balanced_dataset_v2_human" / "balanced_30.csv"
OUTPUT_PATH = SCRIPT_DIR / "balanced_dataset_v2_human" / "balanced_30_formatted.csv"

# Reddit API credentials
REDDIT_CLIENT_ID = "XxtFsMTYONOz3opDovuo6A"
REDDIT_CLIENT_SECRET = "UQsuqpFzrDUN2odyEraeIkCriepecA"
REDDIT_USER_AGENT = "my user agent"


def normalize_text(text):
    """Normalize text for comparison (lowercase, remove extra whitespace)."""
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Markdown links
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def text_similarity(text1, text2):
    """Calculate simple word overlap similarity between two texts."""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def find_matching_comment(submission, normalized_answer, min_similarity=0.3):
    """Find the comment that best matches the normalized answer."""
    submission.comments.replace_more(limit=0)  # Don't expand "more comments"
    
    best_match = None
    best_similarity = min_similarity
    
    for comment in submission.comments.list():
        if hasattr(comment, 'body'):
            similarity = text_similarity(comment.body, normalized_answer)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = comment
    
    return best_match, best_similarity


def setup_selenium_driver():
    """Setup headless Chrome driver for Google searches."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver


def duckduckgo_search_reddit(driver, question, subreddit="askscience"):
    """Use DuckDuckGo with Selenium to find Reddit post URLs for a question."""
    # Try multiple search strategies
    search_queries = [
        f'site:reddit.com/r/{subreddit} {question[:100]}',  # Full question
        f'site:reddit.com {question[:80]}',  # Any subreddit
    ]
    
    all_urls = []
    
    for query in search_queries:
        encoded_query = quote_plus(query)
        url = f"https://duckduckgo.com/?q={encoded_query}"
        
        try:
            driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            # Find all links on the page
            links = driver.find_elements(By.TAG_NAME, "a")
            
            for link in links:
                try:
                    href = link.get_attribute("href")
                    if href and 'reddit.com' in href and '/comments/' in href:
                        # Clean URL - DuckDuckGo sometimes wraps URLs
                        if 'uddg=' in href:
                            # Extract actual URL from DuckDuckGo redirect
                            import urllib.parse
                            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                            if 'uddg' in parsed:
                                href = parsed['uddg'][0]
                        if '&' in href:
                            href = href.split('&')[0]
                        if href not in all_urls:
                            all_urls.append(href)
                except:
                    continue
            
            if all_urls:
                break  # Found results, no need to try other queries
                
        except Exception as e:
            print(f"    DuckDuckGo search error: {e}")
            continue
    
    return all_urls[:10]  # Return up to 10 unique URLs


def extract_post_id(url):
    """Extract Reddit post ID from URL."""
    # URL format: https://www.reddit.com/r/askscience/comments/POST_ID/...
    match = re.search(r'/comments/([a-zA-Z0-9]+)', url)
    return match.group(1) if match else None


def search_reddit_post(reddit, driver, question, answer, subreddit="askscience"):
    """Search for the original Reddit post matching the question and answer."""
    
    # First try DuckDuckGo search (Google blocks headless browsers)
    urls = duckduckgo_search_reddit(driver, question, subreddit)
    
    # Just use the first result - it's usually correct
    if urls:
        url = urls[0]
        try:
            post_id = extract_post_id(url)
            if post_id:
                submission = reddit.submission(id=post_id)
                comment, answer_sim = find_matching_comment(submission, answer)
                
                if comment:
                    return {
                        'reddit_url': f"https://reddit.com{submission.permalink}",
                        'formatted_question': submission.title,
                        'formatted_answer': comment.body,
                        'match_score': answer_sim
                    }
        except Exception as e:
            print(f"    Error fetching post: {e}")
    
    # Fall back to Reddit search
    try:
        search_results = list(reddit.subreddit(subreddit).search(question[:100], sort="relevance", limit=5))
        if search_results:
            submission = search_results[0]
            comment, answer_sim = find_matching_comment(submission, answer)
            if comment:
                return {
                    'reddit_url': f"https://reddit.com{submission.permalink}",
                    'formatted_question': submission.title,
                    'formatted_answer': comment.body,
                    'match_score': answer_sim
                }
    except Exception as e:
        print(f"    Reddit search error: {e}")
    
    return None


def main():
    # Initialize Reddit API
    print("Connecting to Reddit API...")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    
    # Initialize Selenium driver
    print("Setting up Selenium driver for Google search...")
    driver = setup_selenium_driver()
    
    try:
        # Load existing formatted data (to only retry unmatched ones)
        print(f"Loading existing data from {OUTPUT_PATH}")
        existing_df = pd.read_csv(OUTPUT_PATH)
        
        # Find unmatched rows (match_score == 0)
        unmatched = existing_df[existing_df['match_score'] == 0].copy()
        print(f"Found {len(unmatched)} unmatched questions to retry with Google")
        
        if len(unmatched) == 0:
            print("All questions already matched!")
            return
        
        # Process only unmatched questions
        updated_count = 0
        
        for idx, row in tqdm(unmatched.iterrows(), total=len(unmatched), desc="Retrying with Google"):
            question = row['original_question']
            answer = row['original_answer']
            
            result = search_reddit_post(reddit, driver, question, answer)
            
            if result:
                # Update the existing dataframe
                existing_df.loc[idx, 'formatted_question'] = result['formatted_question']
                existing_df.loc[idx, 'formatted_answer'] = result['formatted_answer']
                existing_df.loc[idx, 'reddit_url'] = result['reddit_url']
                existing_df.loc[idx, 'match_score'] = result['match_score']
                updated_count += 1
                print(f"  ✓ Found match for: {question[:50]}...")
            else:
                print(f"  ✗ Still no match for: {question[:50]}...")
            
            # Rate limiting for Google
            time.sleep(3)
        
        # Save updated results
        existing_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved updated data to {OUTPUT_PATH}")
        
        # Summary
        total_matched = (existing_df['match_score'] > 0).sum()
        print(f"\nSummary: {total_matched}/{len(existing_df)} questions matched ({updated_count} new matches)")
    
    finally:
        driver.quit()


if __name__ == "__main__":
    main()

