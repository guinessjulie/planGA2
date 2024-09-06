import hashlib
import numpy as np
from datetime import datetime
import csv
import random



def save_results_to_csv(results, constraints=['None', 'None', 'None'], filename="floorplan_fitness_results.csv"):
    """
    List of fitness objects will be saved to a CSV file.
    파일이 존재하면 이어서 저장, 존재하지 않으면 새로 생성.
    """
    # filename = create_filename('./testing_results', prefix=constraint, ext='csv')

    # 'a' 모드로 파일을 열어 이어쓰기
    file_exists = False
    try:
        with open(filename, 'r'):
            file_exists = True  # 파일이 이미 존재하는지 체크
    except FileNotFoundError:
        pass

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # 파일이 존재하지 않으면 헤더를 추가
        if not file_exists:
            writer.writerow(['Size_Opt','Adj_Opt', 'Ori_Opt',  'Adjacency', 'Orientation', 'Size', 'Simplicity', 'Rectangularity', 'Regularity', 'PA Ratio', 'Weighted_Fitness'])

        col1, col2, col3 = None, None, None
        col1 = 1 if 'Size' in constraints else 0
        col2 = 1 if 'adjacency' in constraints else 0
        col3 = 1 if 'orientation' in constraints else 0

        # 각 fitness 객체의 결과를 CSV에 기록
        for fitness in results:
            writer.writerow([
                col1,
                col2,
                col3, # 세 가지 constraint 정보를 기록
                fitness.adj_satisfaction,
                fitness.orientation_satisfaction,
                fitness.size_satisfaction,
                fitness.simplicity,
                fitness.rectangularity,
                fitness.regularity,
                fitness.pa_ratio,
                fitness.fitness
            ])

def save_results_to_csv_batch(results, constraint = 'None', filename="floorplan_fitness_results.csv"):
    """
    List of tuples (floorplan, fitness) will be saved to a CSV file.
    """
    filename = create_filename('./testing_results', prefix=constraint, ext='csv' )
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Method','Adjacency','Orientation', 'Size', 'Simplicity' ,'Rectnagularity', 'Regularity','PA Ratio', 'Weighted_Fitness'])
        for fitness in results:
            writer.writerow([constraint,
                             fitness.adj_satisfaction,
                             fitness.orientation_satisfaction,
                             fitness.size_satisfaction,
                             fitness.simplicity,
                             fitness.rectangularity,
                             fitness.regularity,
                             fitness.pa_ratio,
                             fitness.fitness
            ])



def get_month_day_4_digit():
    now = datetime.now()
    return  now.strftime('%m%d')

def generate_unique_id(seed_cell_list):
    seed_str = str(seed_cell_list)
    unique_id = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
    return unique_id


def get_hour_min_4_digit():
    now = datetime.now()
    return now.strftime('%H%M')


def create_folder_by_datetime():
    month_day = get_month_day_4_digit()
    min_sec = get_hour_min_4_digit()
    current_path = os.getcwd()
    full_path = os.path.join(os.getcwd(), month_day,min_sec )
    try:
        os.makedirs(full_path, exist_ok=True)
        # print(f'made folder {full_path}')
        return full_path
    except FileExistsError:
        # print(f'path already exists')
        return current_path
    except PermissionError:
        # print(f'Permission Error')
        return current_path
    except OSError as error:
        # print(f'An error occured: {error}')
        return current_path


def prefix_datetime_string(prefix = None, serial_no = None, path=None):
    formatted_datetime = f'{prefix}_{datetime.now().strftime("%m%d_%H%M%S")}_{serial_no}' \
        if serial_no else f'{prefix}_{datetime.now().strftime("%m%d_%H%M%S")}'
    if path :
        full_path = os.path.join((os.getcwd(), path))
        os.makedirs(full_path, exist_ok=True)
    else:
        full_path = os.path.join(os.getcwd(), formatted_datetime)
        os.makedirs(full_path, exist_ok=True)
    # return formatted_datetime
    return full_path

def create_filename_with_datetime(ext='txt', prefix=None, path=None):
    if path :
        full_path = os.path.join(os.getcwd(), path)
        os.makedirs(full_path, exist_ok=True)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%m%d_%H%M%S")
    filename = f'{prefix}{formatted_datetime}.{ext}'
    if path: filename = os.path.join(full_path, filename)
    return filename


import os


def create_filename(path, prefix=None, postfix=None, filename=None, ext='txt'):
    """
    Constructs a complete file path including prefix and postfix added to the filename.

    Parameters:
    path (str): Relative directory path from the current folder.
    prefix (str): Prefix to prepend to the filename.
    postfix (str): Postfix to append to the filename.
    filename (str): The base filename.

    Returns:
    str: The complete path with the constructed filename.
    """
    # Construct the full path from the current directory
    full_path = os.path.join(os.getcwd(), path)

    # Ensure the directory exists, if not, create it
    os.makedirs(full_path, exist_ok=True)

    # Assemble the filename with prefix and postfix
    full_filename = f"{prefix}{filename}{postfix}.{ext}"

    # Combine the path and filename
    complete_path = os.path.join(full_path, full_filename)
    # print(f'complete_path = {complete_path} returns')
    return complete_path


# Example usage:
# Assuming path = 'data', prefix = 'test_', postfix = '.txt', filename = 'example'
# create_filename('data', 'test_', '.txt', 'example')

def create_filename_in_order(ext='txt', prefix=None, postfix_number=0):
    path = create_folder_by_datetime() # todo to test
    postfix_number += 1
    postfix = str(postfix_number)
    full_path = create_filename(path, prefix, postfix, '', 'png') #todo to test
    # print(f'full_path = {full_path}')
    return full_path, postfix_number

def create_file_name_in_path(path=None, prefix=None, postfix_number = 0):
    if path is None:
        path = create_folder_by_datetime()
    if prefix is None:
        prefix = ''
    postfix = str(postfix_number)
    filename = f'{prefix}_{get_hour_min_4_digit()}_{postfix}'
    full_path = os.path.join(path, filename )
    return full_path

def unique_elements_2d_array(arr_2d):
    unique_elements = set()

    for row in arr_2d:
        for element in row:
            unique_elements.add(element)

    return list(unique_elements)




