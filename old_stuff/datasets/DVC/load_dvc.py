import os 
import numpy as np
import pandas as pd


# def load_all_data_in_folder(folder_path):

#     dataset = {}
#     k = 0
#     cage_name_old = ""
#     cage_names = []

#     #go through all files in the folder
#     for file in sorted(os.listdir(folder_path)):
#         dataframe = pd.read_csv(folder_path + '/' + file, sep = '\t', header=None)
#         cage_name = file.split("___")[0]
#         # add k to cage name
#         if cage_name_old == cage_name:
#             k += 1
#         else:  
#             k = 0
#         cage_name_k = cage_name + '-' + str(k)
#         dataset[cage_name_k] = dataframe
#         cage_name_old = cage_name
#         if cage_name not in cage_names:
#             cage_names.append(cage_name)

#     return dataset, cage_names

# def load_all_data_in_folder(folder_path, cages):
#     dataset = {}
#     cage_names = []

#     for cage_folder in sorted(os.listdir(folder_path)):
#         cage_path = os.path.join(folder_path, cage_folder)
#         if os.path.isdir(cage_path):
#             cage_name = cage_folder
#             if cage_name in cages:
#             # Read only the CSV file with "activity" in its name
#                 for file in os.listdir(cage_path):
#                     if "activation" in file and file.endswith(".csv"):
#                         dataframe = pd.read_csv(os.path.join(cage_path, file), header=None)
#                         dataset[cage_name] = dataframe
#                         cage_names.append(cage_name)
#                         break  

#     return dataset, cage_names

# def filter_data(data, selected_cages, selected_days):
#     filtered_data = {}
#     for cage in selected_cages:
#         if cage in data:
#             dataframe = data[cage]
#             # Filter the dataframe based on selected days
#             date = dataframe[0].str.split(' ').str[0]
#             filtered_dataframe = dataframe[date.isin(selected_days)]
#             # filtered_dataframe = dataframe[dataframe[0].str.startswith("2021-12-03")]
#             filtered_data[cage] = filtered_dataframe
#     return filtered_data

def load_data(folder_path, cage_list):
    """
    INPUT: folder_path: path to the folder containing the data
           cage_list: cages and days to concatenate
    OUTPUT: data_concat: dictionary containing the concatenated dataframes
    I updated this to match our new data load method, use the commented out code if you want to use the old method.
    """
    data_concat = {}
    for cage in cage_list.keys():
        days = cage_list[cage]
        file_name = f"{cage}_activation.csv"
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            dataframe = pd.read_csv(file_path, header=0)  # Skip the header row
        else:
            raise FileNotFoundError(f"File {file_name} not found in {file_path}")
        
        # Assuming the date is in the 14th column (index 13)
        date = dataframe.iloc[:, 13].str.split('T').str[0]
        
        filtered_dataframe = dataframe[date.isin(days)]
        del dataframe
        data_concat[cage] = filtered_dataframe

    return data_concat
