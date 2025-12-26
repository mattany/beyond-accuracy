#!/usr/bin/env python3
"""
Fetch original Reddit formatting for questions in a CSV file.

This script searches for the original r/askscience posts and retrieves
the properly formatted question titles and top answers (with newlines,
capitalization, etc.).

Uses Selenium with DuckDuckGo search to find Reddit posts, then Reddit API to fetch content.

Usage:
    # Basic usage - process a CSV file
    python fetch_reddit_formatting.py input.csv
    
    # Specify output file
    python fetch_reddit_formatting.py input.csv -o output.csv
    
    # Auto-add _formatted suffix to output
    python fetch_reddit_formatting.py input.csv --suffix _formatted
    
    # Only retry unmatched rows (match_score == 0) from existing output
    python fetch_reddit_formatting.py input.csv --retry-unmatched
    
    # Specify column names
    python fetch_reddit_formatting.py input.csv --question-col "Question" --answer-col "Human Answer"
    
    # Specify subreddit
    python fetch_reddit_formatting.py input.csv --subreddit askscience
"""

import os
import re
import sys
import time
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from urllib.parse import quote_plus

try:
    import praw
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install praw selenium")
    sys.exit(1)

# Reddit API credentials
REDDIT_CLIENT_ID = "XxtFsMTYONOz3opDovuo6A"
REDDIT_CLIENT_SECRET = "UQsuqpFzrDUN2odyEraeIkCriepecA"
REDDIT_USER_AGENT = "reddit_formatting_fetcher/1.0"


def normalize_text(text):
    """Normalize text for comparison (lowercase, remove extra whitespace)."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Markdown links
    text = re.sub(r'[^a-z0-9\s]', '', text)
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
    submission.comments.replace_more(limit=0)
    
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
    """Setup headless Chrome driver for searches."""
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
    search_queries = [
        f'site:reddit.com/r/{subreddit} {question[:100]}',
        f'site:reddit.com {question[:80]}',
    ]
    
    all_urls = []
    
    for query in search_queries:
        encoded_query = quote_plus(query)
        url = f"https://duckduckgo.com/?q={encoded_query}"
        
        try:
            driver.get(url)
            time.sleep(3)
            
            links = driver.find_elements(By.TAG_NAME, "a")
            
            for link in links:
                try:
                    href = link.get_attribute("href")
                    if href and 'reddit.com' in href and '/comments/' in href:
                        if 'uddg=' in href:
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
                break
                
        except Exception as e:
            print(f"    DuckDuckGo search error: {e}")
            continue
    
    return all_urls[:10]


def extract_post_id(url):
    """Extract Reddit post ID from URL."""
    match = re.search(r'/comments/([a-zA-Z0-9]+)', url)
    return match.group(1) if match else None


def search_reddit_post(reddit, driver, question, answer, subreddit="askscience"):
    """Search for the original Reddit post matching the question and answer."""
    
    urls = duckduckgo_search_reddit(driver, question, subreddit)
    
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


def detect_column_names(df):
    """Auto-detect question and answer column names."""
    question_candidates = ['question', 'Question', 'original_question', 'q']
    answer_candidates = ['answer', 'Answer', 'original_answer', 'Human Answer', 'a']
    
    question_col = None
    answer_col = None
    
    for col in question_candidates:
        if col in df.columns:
            question_col = col
            break
    
    for col in answer_candidates:
        if col in df.columns:
            answer_col = col
            break
    
    return question_col, answer_col


def main():
    parser = argparse.ArgumentParser(
        description="Fetch original Reddit formatting for questions in a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file path")
    parser.add_argument("--suffix", default="_formatted", 
                        help="Suffix to add to input filename for output (default: _formatted)")
    parser.add_argument("--question-col", help="Column name for questions (auto-detected if not specified)")
    parser.add_argument("--answer-col", help="Column name for answers (auto-detected if not specified)")
    parser.add_argument("--subreddit", default="askscience", help="Subreddit to search (default: askscience)")
    parser.add_argument("--retry-unmatched", action="store_true",
                        help="Only retry rows with match_score == 0 from existing output")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between requests in seconds (default: 2.0)")
    parser.add_argument("--min-similarity", type=float, default=0.3,
                        help="Minimum similarity score for answer matching (default: 0.3)")
    
    args = parser.parse_args()
    
    # Determine output path
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}{args.suffix}{input_path.suffix}"
    
    # Initialize Reddit API
    print("Connecting to Reddit API...")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    
    # Initialize Selenium driver
    print("Setting up Selenium driver...")
    driver = setup_selenium_driver()
    
    try:
        # Load data
        if args.retry_unmatched and output_path.exists():
            print(f"Loading existing output from {output_path}")
            df = pd.read_csv(output_path)
            
            # Find unmatched rows
            if 'match_score' not in df.columns:
                print("Error: --retry-unmatched requires 'match_score' column in output file")
                return
            
            unmatched_mask = df['match_score'] == 0
            unmatched_count = unmatched_mask.sum()
            print(f"Found {unmatched_count} unmatched questions to retry")
            
            if unmatched_count == 0:
                print("All questions already matched!")
                return
            
            indices_to_process = df[unmatched_mask].index.tolist()
        else:
            print(f"Loading data from {input_path}")
            df = pd.read_csv(input_path)
            indices_to_process = df.index.tolist()
        
        # Detect or use specified column names
        question_col = args.question_col
        answer_col = args.answer_col
        
        if not question_col or not answer_col:
            detected_q, detected_a = detect_column_names(df)
            question_col = question_col or detected_q
            answer_col = answer_col or detected_a
        
        if not question_col or not answer_col:
            print(f"Error: Could not detect column names. Available columns: {list(df.columns)}")
            print("Please specify --question-col and --answer-col")
            return
        
        print(f"Using columns: question='{question_col}', answer='{answer_col}'")
        
        # Add output columns if they don't exist
        if 'original_question' not in df.columns:
            df['original_question'] = df[question_col]
        if 'original_answer' not in df.columns:
            df['original_answer'] = df[answer_col]
        if 'formatted_question' not in df.columns:
            df['formatted_question'] = ''
        if 'formatted_answer' not in df.columns:
            df['formatted_answer'] = ''
        if 'reddit_url' not in df.columns:
            df['reddit_url'] = ''
        if 'match_score' not in df.columns:
            df['match_score'] = 0.0
        
        # Process questions
        print(f"Processing {len(indices_to_process)} questions...")
        matched_count = 0
        
        for idx in tqdm(indices_to_process, desc="Fetching Reddit data"):
            row = df.loc[idx]
            question = row[question_col]
            answer = row[answer_col]
            
            result = search_reddit_post(reddit, driver, question, answer, args.subreddit)
            
            if result:
                df.loc[idx, 'formatted_question'] = result['formatted_question']
                df.loc[idx, 'formatted_answer'] = result['formatted_answer']
                df.loc[idx, 'reddit_url'] = result['reddit_url']
                df.loc[idx, 'match_score'] = result['match_score']
                matched_count += 1
                tqdm.write(f"  ✓ Found: {question[:40]}...")
            else:
                tqdm.write(f"  ✗ No match: {question[:40]}...")
            
            time.sleep(args.delay)
        
        # Save results
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
        
        # Summary
        total_matched = (df['match_score'] > 0).sum()
        print(f"Summary: {total_matched}/{len(df)} questions matched ({matched_count} new matches)")
    
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
