import requests
from pprint import pprint
import os
import shutil
import pandas as pd


category_member_response_limit = 500
category_filtered_keywords = ["film", "Film", "fiction", "Fiction", "about", "Astrology", "astrology", "Astronomy data and publications", "Astronomy education", "History of astronomy", "Astronomical myths"]

def concat_records(dataframe, new_records):
    column_names = ["id", "title", "category", "category_depth"]
    new_dataframe = pd.DataFrame.from_records(data=new_records, columns=column_names)
    if dataframe is None:
        return new_dataframe
    return pd.concat([dataframe, new_dataframe], ignore_index=True)


def clear_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def build_query_params(category_title, category_member_type):
    """
    :param category_title: e.g. Category:Astronomy
    :param category_member_type: page, subcat
    :return:
    """
    return {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_title,
        "prop": "categories",
        "cmlimit": category_member_response_limit,
        "cmtype": category_member_type,
        "format": "json"
    }


def get_direct_pages_for_category(category, titles_df, depth):
    """

    :param category: e.g. Category:Astronomy
    :param titles_df: the dataframe where the page_id and title are stored
    :return: a dataframe
    """

    query_params = build_query_params(category, "page")
    res = requests.get("https://en.wikipedia.org/w/api.php", params=query_params)
    res_json = res.json()
    while res_json.get("continue", False):
        records = [(elem['pageid'], elem["title"], category, depth) for elem in res_json["query"]["categorymembers"]]
        print(f"{len(records)} records in response.", end=" ")
        titles_df = concat_records(titles_df, records)
        query_params.update(res_json["continue"])
        print("continue params: ", end="")
        pprint(res_json["continue"])
        res = requests.get("https://en.wikipedia.org/w/api.php", params=query_params)
        res_json = res.json()
    records = [(elem['pageid'], elem["title"], category, depth) for elem in res_json["query"]["categorymembers"]]
    titles_df = concat_records(titles_df, records)
    return titles_df



def get_child_categories(category):
    query_params = build_query_params(category, "subcat")
    res = requests.get("https://en.wikipedia.org/w/api.php", params=query_params)
    res_json = res.json()
    sub_categories = list()
    while res_json.get("continue", False):
        sub_categories += [elem["title"] for elem in res_json["query"]["categorymembers"]]
        query_params.update(res_json["continue"])
        print("continue params: ", end="")
        pprint(res_json["continue"])
        res = requests.get("https://en.wikipedia.org/w/api.php", params=query_params)
        res_json = res.json()
    sub_categories += [elem["title"] for elem in res_json["query"]["categorymembers"]]
    return sub_categories


def get_output_csv_path(write_path, batch_id):
    return f'{write_path}/datapage_id_title_batch_{batch_id}.csv'


def bfs_over_category(category_title, max_depth=10):
    batch_id = 0
    batch_size = 200
    titles_df = None
    category_output_path_suffix = "".join(category_title.split(" "))
    write_path = f'/Users/mattan/thesis/RAG/{category_output_path_suffix}_pages'
    clear_path(write_path)
    visited = set()

    stack = [(f"Category:{category_title}", 0)]
    while stack:
        category, depth = stack.pop(0)
        visited.add(category)
        titles_df = get_direct_pages_for_category(category, titles_df, depth)
        while len(titles_df) > batch_size:
            print(f"total records in memory: {len(titles_df)}. flushing {batch_size} records to batch {batch_id}")
            titles_df[:batch_size].to_csv(get_output_csv_path(write_path, batch_id), index=False)
            titles_df = titles_df[batch_size:]
            batch_id += 1
        child_categories = get_child_categories(category)
        if depth + 1 <= max_depth:
            new_categories = [(cat, depth + 1) for cat in child_categories if cat not in visited and all([keyword not in cat for keyword in category_filtered_keywords])]
            print(f"source category: {category}. new categories found: {','.join([c[0] for c in new_categories])}")
            stack += new_categories
    print(f"total records in memory: {len(titles_df)}. flushing {len(titles_df)} records to batch {batch_id}")
    titles_df.to_csv(get_output_csv_path(write_path, batch_id))



if __name__ == "__main__":
    bfs_over_category("Solar System", max_depth=5)
