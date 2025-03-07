import os.path


def save_data2file(data, path):
    # first check if the directory exists
    # create a directories if it does not exist
    dir_path = os.path.split(path)[0]
    file = os.path.split(path)[1]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # second, determine whether the file exists
    # if it does not exist, write the data directly
    # if it exists, append the data
    if not os.path.exists(path):
        data.to_csv(path, index=False)
    else:
        data.to_csv(path, mode='a', header=False, index=False)

