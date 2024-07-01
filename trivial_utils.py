from datetime import datetime

def get_month_day_4_digit():
    now = datetime.now()
    return  now.strftime('%m%d')


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
        print(f'made folder {full_path}')
        return full_path
    except FileExistsError:
        print(f'path already exists')
        return current_path
    except PermissionError:
        print(f'Permission Error')
        return current_path
    except OSError as error:
        print(f'An error occured: {error}')
        return current_path


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
    print(f'complete_path = {complete_path} returns')
    return complete_path


# Example usage:
# Assuming path = 'data', prefix = 'test_', postfix = '.txt', filename = 'example'
# create_filename('data', 'test_', '.txt', 'example')

def create_filename_in_order(ext='txt', prefix=None, postfix_number=0):
    path = create_folder_by_datetime() # todo to test
    postfix_number += 1
    postfix = str(postfix_number)
    full_path = create_filename(path,'Step', postfix, '', 'png') #todo to test
    print(f'full_path = {full_path}')
    return full_path, postfix_number


def unique_elements_2d_array(arr_2d):
    unique_elements = set()

    for row in arr_2d:
        for element in row:
            unique_elements.add(element)

    return list(unique_elements)
