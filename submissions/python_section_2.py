import pandas as pd
import numpy as np 
from scipy.spatial import distance


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    dist_matrix = distance.cdist(df.values, df.values, metric='euclidean')
    return pd.DataFrame(dist_matrix, index=df.index, columns=df.index)


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Create an empty DataFrame to store the unrolled data
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])
    
    # Unroll the distance matrix
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            unrolled_df = unrolled_df.append({
                'id_start': df.index[i],
                'id_end': df.columns[j],
                'distance': df.iat[i, j]
            }, ignore_index=True)
    
    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
     # Calculate average distance for each ID
    avg_distances = df.mean(axis=1)
    
    # Get the average distance of the reference ID
    ref_avg_distance = avg_distances[reference_id]
    
    # Calculate the threshold
    lower_threshold = ref_avg_distance * 0.9
    upper_threshold = ref_avg_distance * 1.1
    
    # Find IDs within the threshold
    ids_within_threshold = avg_distances[(avg_distances >= lower_threshold) & (avg_distances <= upper_threshold)].index
    
    # Return as DataFrame
    result_df = pd.DataFrame(ids_within_threshold, columns=['ID'])
    return result_df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6,
    }
    
    # Initialize new columns for each vehicle type
    for vehicle in rate_coefficients:
        df[vehicle] = df['distance'] * rate_coefficients[vehicle]
    
    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define discount factors
    weekday_morning = 0.8
    weekday_midday = 1.2
    weekday_evening = 0.8
    weekend = 0.7
    
    # Generate the time range DataFrame
    time_ranges = pd.date_range("00:00", "23:59", freq="H").time

    # Create a new DataFrame to store the results
    results = pd.DataFrame()

    # Create the 'start_day' and 'end_day' columns
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for day in days_of_week:
        for start_time in time_ranges:
            for _, row in df.iterrows():
                id_start = row['id_start']
                id_end = row['id_end']
                distance = row['distance']
                vehicle_type = row['vehicle_type']

                # Calculate the end time by adding an hour to start time
                end_time = (datetime.datetime.combine(datetime.date.today(), start_time) + datetime.timedelta(hours=1)).time()

                # Determine if it's a weekday or weekend
                if day in ["Saturday", "Sunday"]:
                    discount_factor = weekend
                else:
                    if start_time < datetime.time(10, 0):
                        discount_factor = weekday_morning
                    elif start_time < datetime.time(18, 0):
                        discount_factor = weekday_midday
                    else:
                        discount_factor = weekday_evening

                # Apply the discount factor to the distance to get toll rate
                toll_rate = distance * discount_factor

                # Add the new row to the results DataFrame
                results = results.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'toll_rate': toll_rate
                }, ignore_index=True)

    return results
