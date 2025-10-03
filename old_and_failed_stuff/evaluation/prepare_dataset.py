import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split


def get_cage_days_from_folder(datapath):
    df = {"cage_id": [], "date": []}
    for filename in os.listdir(datapath):
        if filename.endswith(".tsv"):
            cage_id = filename.split("cage_")[1].split("__")[0]
            date = "-".join(filename.split("_")[-3:]).split(".")[0]
            df["cage_id"].append(cage_id.replace("_", "-"))
            df["date"].append(date)
    return pd.DataFrame(df)

def get_cage_days_from_file(filepath):
    df = {"cage_id": [], "date": []}
    filenames = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().split("\t")[0]
            cage_id = line.split("cage_")[1].split("__")[0].replace("_", "-")
            date = "-".join(line.split("_")[-3:]).split(".")[0]
            df["cage_id"].append(cage_id)
            df["date"].append(date)
            filenames.append(line)
    return pd.DataFrame(df), filenames

def get_strain(filename, metadata):
    cage_id = filename.split("cage_")[1].split("__")[0].replace("_", "-")
    date = "-".join(filename.split("_")[-3:]).split(".")[0]
    strain = metadata[(metadata["from"].startswith(date)) & (metadata["cage_human_id"] == cage_id)]["mice_strain"] == 
    return strain

def get_label_array(presencepath, metadatapath, vocabulary = ["presence", "time_of_day", "strain"]):
    frame_number_map = {}
    clip_length = {}
    
    # Get seq counts and check files
    seq_counts = 0
    file_list = []
    metadata = pd.read_csv(metadatapath, sep="\t")

    for filename in os.listdir(presencepath):
        try:
            arr = None
            seq_counts += arr.shape[0]
            file_list.append(filename)
        except:
            print(f"Error with {filename}")

    # Create label array
    label_array = np.zeros((len(vocabulary), seq_counts))
    seq_counts = 0
    for filename in file_list:
        arr = None
        strain = get_strain(filename, metadata)
        clip_length[filename] = arr.shape[0]
        frame_number_map[filename] = (seq_counts, seq_counts+arr.shape[0])
        time_arr = np.zeros_like(arr)
        time_arr[420:1140] = 1
        for i, label in enumerate(vocabulary):
            if label == "presence":
                label_array[i, seq_counts:seq_counts+arr.shape[0]] = arr[:]
            elif label == "time_of_day":
                label_array[i, seq_counts:seq_counts+arr.shape[0]] = time_arr
            elif label == "strain":
                label_array[i, seq_counts:seq_counts+arr.shape[0]] = [strain]*arr.shape[0]
        seq_counts += arr.shape[0]
    return label_array, frame_number_map, clip_length

def convert_data_mabe(presencepath, metadatapath, testfile, output_dir, seeds: list = [41, 42, 43],
):
    df_data, filenames = get_cage_days_from_file(testfile)
    train_files, test_files = train_test_split(filenames, test_size=0.4, random_state=42)
    #TODO find mapping with between train test and presence paths
    seq = {}
    vocabulary = ["presence", "time_of_day", "strain"]
    task_types = ["Discrete"]*len(vocabulary)
    label_array, frame_number_map, clip_length = get_label_array(presencepath, metadatapath,  vocabulary)
    os.makedirs(output_dir, exist_ok=True)
    
    frame_number_map = {}
    data_dict = {"frame_number_map" : frame_number_map,
                "label_array" : label_array,
                "vocabulary" : vocabulary,
                "task_type": task_types}
    
    # Create label dictionary
    np.save(
        os.path.join(
            output_dir, f"label_dictionary_test.npy"
        ),
        data_dict,
        allow_pickle=True,
    )

    # Create frame number map
    np.save(
        os.path.join(output_dir, f"frame_number_map_DVC.npy"),
        frame_number_map,
        allow_pickle=True,
    )
    # Create Split info file
    split_info_dict = {
        "SubmissionTrain": train_files,
        "publicTest": test_files,
        "privateTest": test_files,
    }
    
    with open(
        os.path.join(output_dir, f"split_info_DVC.json"),
        "w",

    ) as f:
        json.dump(split_info_dict, f)

        # Create Tasks info file
        task_info_dict = {
            "seeds": seeds,
            "task_id_list": vocabulary,
            "sequence_level_tasks": ["There_are_no_sequence_task_here"],
        }
        with open(
            os.path.join(
                output_dir,
                f"task_info_DVC.json",
            ),
            "w",
        ) as f:
            json.dump(task_info_dict, f)

        # Create clip length file
        with open(os.path.join(output_dir, f"clip_length_DVC.json"), "w") as f:
            json.dump(clip_length, f)

if __name__ == "__main__":
    data_dir = "/media18/data/andy/EPFL/DVC_data/"
    data_path = os.path.join(data_dir , "DVC_data")
    metadatapath = os.path.join(data_path, "event_periods_meta_parsed.tsv")
    presencepath = os.path.join(data_dir, "DVC_presence")
    testfile = os.path.join(data_dir, "test_files.txt")
    
    output_dir = os.path.join(data_dir, "evaluation_labels")
    
    convert_data_mabe(presencepath, metadatapath, testfile, output_dir=output_dir)