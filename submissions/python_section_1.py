from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        temp_list = []
        for j in range(i, min(i + n, len(lst))):
            temp_list.insert(0, lst[j])
        result.extend(temp_list)
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    
    # Sort the dictionary by keys (lengths) in ascending order
    sorted_result = dict(sorted(result.items()))
    
    return sorted_result

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def recurse(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(recurse(v, new_key).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.extend(recurse({f'{k}[{i}]': item}, parent_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    return recurse(nested_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    
    nums.sort()
    result = []
    backtrack(0)
    return result


import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    return re.findall(date_pattern, text)

# +
import polyline
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    """
    R = 6371000  # Earth radius in meters
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.
        
    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Initialize the distance column
    df['distance'] = 0.0
    
    # Calculate the distance between consecutive points
    for i in range(1, len(df)):
        df.at[i, 'distance'] = haversine(df.at[i-1, 'latitude'], df.at[i-1, 'longitude'], df.at[i, 'latitude'], df.at[i, 'longitude'])
    
    return df


# -

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Rotate the matrix by 90 degrees clockwise
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Create a new matrix to store the results
    transformed = [[0] * n for _ in range(n)]
    
    # Calculate the sum of elements for each row and column
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i]) - rotated[i][j]
            col_sum = sum(rotated[k][j] for k in range(n)) - rotated[i][j]
            transformed[i][j] = row_sum + col_sum
    
    return transformed   


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    def check_complete_week(group):
        days = set(group['timestamp'].dt.day_name())
        if len(days) < 7:
            return False
        daily_coverage = group.groupby(group['timestamp'].dt.date)['timestamp'].apply(lambda x: (x.max() - x.min()).total_seconds() >= 24 * 3600)
        return all(daily_coverage)
    
    completeness = df.groupby(['id', 'id_2']).apply(check_complete_week)
    return 
