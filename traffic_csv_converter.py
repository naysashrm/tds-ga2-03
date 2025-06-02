#!/usr/bin/env python
"""
Read traffic_csv and convert to .npy format with optional sampling
"""


import os
import argparse
import csv
import glob
import re
import numpy as np
from sessions_plotter import * # Make sure sessions_plotter.py is in the same directory or accessible via PYTHONPATH


# --- Configuration Parameters (for Version B) ---
# Time per session / Delta T between session splits / Minimum session duration
TPS = 60
DELTA_T = 60
MIN_TPS = 50


# Define the root directory where your raw CSVs are located, organized by classes and VPN types.
# This path should be relative to where you run the script (e.g., your FlowPic root directory).
# Example: If your structure is 'FlowPic/raw_csvs/classes/Browse/reg/CICNTTor_Browse.raw.csv',
# then '../raw_csvs/classes/**/**/' is correct if you run from 'FlowPic/TrafficParser/'.
CLASSES_DIR = "../classes_csvs/**/**/"


# --- Helper Functions ---


def export_dataset(dataset, output_path):
   """Exports a single dataset array to a .npy file."""
   print(f"\n[+] Exporting dataset to {output_path}")
   os.makedirs(os.path.dirname(output_path), exist_ok=True)
   np.save(output_path, dataset)
   print(f"[+] Dataset saved. Shape: {dataset.shape}")




def export_class_dataset(dataset, class_dir):
   """
   Exports a combined dataset for a specific class/VPN type to a .npy file
   within its class directory.
   """
   print(f"\n[+] Exporting combined class dataset for: {class_dir}")
   # Derives filename like 'Browse_reg.npy' from path like '../raw_csvs/classes/Browse/reg/'
   output_filename = "_".join(re.findall(r"[\w']+", class_dir)[-2:]) + ".npy"
   output_path = os.path.join(class_dir, output_filename)
   os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
   np.save(output_path, dataset)
   print(f"[+] Combined class dataset saved to {output_path}. Shape: {dataset.shape}")




def traffic_csv_converter(file_path):
   """
   Processes a single traffic CSV file, extracts session data,
   and converts segments into 2D histograms (FlowPics).
   """
   print(f"\n[+] Processing file: {file_path}")
   dataset = []
   counter = 0
   skipped_segments = 0


   try:
       with open(file_path, 'r') as csv_file:
           reader = csv.reader(csv_file)
           # Skip rows if csv has estimatedRowsAboveHeader (e.g. from user file metadata)
           # This is a general CSV reader; specific header handling depends on user's exact CSV format.
           # Assuming first data row is always valid after any potential "header" rows.
           for i, row in enumerate(reader):
               if len(row) < 9: # Minimum expected columns (e.g., up to 'length' + 1 for ts start)
                   # print(f"[!] Skipped row {i} in {file_path}: Not enough columns ({len(row)}).")
                   continue


               try:
                   # session_tuple_key = tuple(row[:8]) # For debugging/identification
                   length = int(row[7]) # Length of timestamps array
                   # Ensure row has enough elements for ts and sizes based on 'length'
                   if (8 + length) > len(row) or (9 + length) > len(row):
                        # print(f"[!] Skipped row {i} in {file_path}: 'length' value ({length}) exceeds row bounds.")
                        continue


                   ts = np.array(row[8:8 + length], dtype=float)
                   sizes = np.array(row[9 + length:], dtype=int)


                   # Only process if there's sufficient data
                   if length > 10: # Minimum number of packets in a session for processing
                       for t in range(int(ts[-1] / DELTA_T - TPS / DELTA_T) + 1):
                           mask = (ts >= t * DELTA_T) & (ts <= (t * DELTA_T + TPS))
                           ts_mask = ts[mask]
                           sizes_mask = sizes[mask]


                           # Check if the segmented session is valid
                           if len(ts_mask) > 10 and (ts_mask[-1] - ts_mask[0]) > MIN_TPS:
                               h = session_2d_histogram(ts_mask, sizes_mask)
                               dataset.append([h])
                               counter += 1
                               if counter % 100 == 0:
                                   print(f"[+] Processed {counter} segments from {os.path.basename(file_path)}")
                           else:
                               skipped_segments += 1
                   else:
                       skipped_segments += 1 # Session too short at original length
               except ValueError as ve:
                   # print(f"[!] Data conversion error in row {i} of {file_path}: {ve}. Skipping segment.")
                   skipped_segments += 1
               except IndexError as ie:
                   # print(f"[!] Index error in row {i} of {file_path}: {ie}. Row format issue? Skipping segment.")
                   skipped_segments += 1
               except Exception as ex:
                   print(f"[!] Unexpected error in row {i} of {file_path}: {ex}. Skipping segment.")
                   skipped_segments += 1


   except FileNotFoundError:
       print(f"[!] CSV file not found: {file_path}. Skipping.")
       return np.array([])
   except Exception as e:
       print(f"[!] Error opening or reading {file_path}: {e}")
       return np.array([])


   print(f"[+] Finished processing {os.path.basename(file_path)}. Extracted {counter} FlowPics, skipped {skipped_segments} segments.")
   return np.asarray(dataset)




def traffic_class_converter(dir_path):
   """
   Combines FlowPics from all CSV files within a given class directory
   into a single NumPy array.
   """
   print(f"[+] Aggregating data for class directory: {dir_path}")
   dataset_list = []
   # Find all .csv files within the directory
   csv_files_in_dir = [os.path.join(dir_path, fn) for fn in os.listdir(dir_path) if fn.endswith(".csv")]


   if not csv_files_in_dir:
       print(f"[!] No .csv files found in {dir_path}. Skipping this directory.")
       return np.array([])


   for file_path in csv_files_in_dir:
       single_csv_dataset = traffic_csv_converter(file_path)
       if single_csv_dataset.size > 0: # Only add if the converter returned data
           dataset_list.append(single_csv_dataset)


   if not dataset_list:
       print(f"[!] No valid FlowPics generated from any CSV in {dir_path}. Returning empty array.")
       return np.array([])


   return np.concatenate(dataset_list, axis=0)




def iterate_all_classes():
   """
   Iterates through all class/VPN type directories defined by CLASSES_DIR,
   processes their CSVs, and exports the combined FlowPics for each.
   """
   print(f"\n--- Starting iteration through all classes in: {CLASSES_DIR} ---")
   found_class_dirs = False
   # Use glob.glob to find directories matching the pattern
   # The /**/ will match any subdirectories (e.g., classes/Browse/reg/)
   # It finds directories like "../raw_csvs/classes/Browse/reg/"
   for class_dir in sorted(glob.glob(CLASSES_DIR)):
       # Ensure it's actually a directory and not a file that matches the pattern
       if os.path.isdir(class_dir):
           # You might want to adjust this filter if you have an 'other' class or similar exclusions
           if "other" not in class_dir.lower():
               found_class_dirs = True
               print(f"\n[+] Working on class directory: {class_dir}")
               dataset = traffic_class_converter(class_dir)
               if dataset.size > 0: # Only export if dataset is not empty
                   export_class_dataset(dataset, class_dir)
               else:
                   print(f"[!] No valid FlowPics generated for {class_dir}. Skipping export for this directory.")


   if not found_class_dirs:
       print(f"[!] No class directories found matching '{CLASSES_DIR}'. Please check the path and your directory structure.")
   print("\n--- Finished iterating through all classes ---")




def random_sampling_dataset(input_array_path, sample_size=2000):
   """
   Loads a .npy dataset, performs random sampling, and saves the sampled dataset.
   This function is not part of the main workflow for 'iterate_all_classes'.
   """
   print(f"\n[+] Sampling from dataset: {input_array_path}")
   try:
       dataset = np.load(input_array_path)
   except FileNotFoundError:
       print(f"[!] Error: Dataset not found at {input_array_path}. Cannot sample.")
       return
   print(f"[+] Original shape: {dataset.shape}")


   if sample_size >= len(dataset):
       print("[!] Sample size >= dataset size. Skipping sampling.")
       return


   p = sample_size / len(dataset)
   print(f"[+] Sampling approximately {p*100:.2f}% of data.")


   # Using np.random.choice to create a mask for sampling
   mask = np.random.choice([True, False], len(dataset), p=[p, 1 - p], replace=False)
   sampled_dataset = dataset[mask]


   sampled_path = os.path.splitext(input_array_path)[0] + "_samp.npy"
   np.save(sampled_path, sampled_dataset)
   print(f"[+] Sampled dataset saved to: {sampled_path}")
   print(f"[+] Sampled shape: {sampled_dataset.shape}")




# --- Main Execution Block ---
if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Convert traffic CSV to .npy format and optionally sample.")
   # Added required=False for --input, as it's not always used when iterate_all_classes is active
   parser.add_argument('--input', type=str, required=False, help='Path to a single traffic CSV file (used if iterate_all_classes is commented out)')
   parser.add_argument('--sample_size', type=int, default=2000, help='Number of samples to extract (default=2000)')
   args = parser.parse_args()


   # This is the primary action for generating all class-specific FlowPic .npy files.
   # UNCOMMENT this line to run the full conversion process.
   iterate_all_classes()


   # The following block is for processing a single input file or sampling,
   # and should be COMMENTED OUT when using iterate_all_classes().
   # if args.input:
   #     if not os.path.isfile(args.input):
   #         raise FileNotFoundError(f"CSV file not found: {args.input}")
   #     dataset = traffic_csv_converter(args.input)
   #     output_path = os.path.splitext(args.input)[0] + ".npy"
   #     export_dataset(dataset, output_path)
   #     random_sampling_dataset(output_path, sample_size=args.sample_size)
   #
   # # Example of how you might use random_sampling_dataset separately after files are generated
   # # input_array_for_sampling = "../raw_csvs/classes/Browse/reg/Browse_reg.npy"
   # # random_sampling_dataset(input_array_for_sampling)
