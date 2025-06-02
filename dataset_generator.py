import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split


# --- Configuration for generating the combined test files ---


# Directory where the individual class/VPN type .npy files (output of traffic_csv_converter.py) are located.
# This assumes traffic_csv_converter.py saved them in a nested structure.
RAW_DATA_DIR = "/content/FlowPic/classes_csvs/"


# Directory where the final combined test datasets will be saved (vpn_x_test.npy, tor_x_test.npy etc.)
DATASET_DIR = "/content/FlowPic/datasets/"


# Proportion of the combined data to be used for the test set
TEST_SIZE = 0.2 # You can adjust this percentage (e.g., 0.1 for 10%)
RANDOM_STATE = 42 # For reproducibility of the split


# List of all class names that you want to include in your combined test dataset.
CLASS_NAMES = [
   "browsing",
   "chat",
   "file_transfer",
   "video",
   "voip"
]


# Mapping of class names to numerical labels. This is essential for creating y_test labels.
CLASS_TO_LABEL = {name: i for i, name in enumerate(CLASS_NAMES)}


# VPN types for which we want to generate combined test sets
VPN_TYPES_TO_GENERATE = [
   "vpn",
   "tor",
   "reg" # 'reg' is for non-VPN regular traffic
]


# --- Helper Function to Load Data ---
def import_array(input_array_path):
   """Imports a numpy array from a given file path."""
   print(f"Importing dataset: {input_array_path}")
   try:
       dataset = np.load(input_array_path)
       print(f"Dataset shape: {dataset.shape}")
       return dataset
   except FileNotFoundError:
       print(f"Error: File not found at {input_array_path}. Skipping.")
       return None
   except Exception as e:
       print(f"Error loading {input_array_path}: {e}. Skipping.")
       return None


# --- Main Generation Logic ---
def generate_combined_vpn_type_test_sets():
   """
   Generates combined x_test and y_test .npy files for each specified VPN type
   (e.g., vpn_x_test.npy, tor_x_test.npy) by combining data from all classes.
   """
   print("\n--- Starting generation of combined VPN/Tor/Reg test datasets ---")


   # Ensure the output directory exists
   os.makedirs(DATASET_DIR, exist_ok=True)


   for vpn_type in VPN_TYPES_TO_GENERATE:
       print(f"\nProcessing data for VPN Type: {vpn_type.upper()}")
       all_x_for_vpn_type = [] # To store all FlowPic images for the current VPN type
       all_y_for_vpn_type = [] # To store corresponding numerical labels for the current VPN type


       for class_name in CLASS_NAMES:
           # Construct the file path pattern assuming the structure created by traffic_csv_converter.py
           # Example: /content/FlowPic/classes_csvs/Browse/vpn/Browse_vpn.npy
           class_vpn_file_pattern = f"{RAW_DATA_DIR}{class_name}/{vpn_type}/{class_name}_{vpn_type}.npy"
           class_npy_files = glob.glob(class_vpn_file_pattern)


           if not class_npy_files:
               print(f"No .npy file found for class '{class_name}' and VPN type '{vpn_type}' at '{class_vpn_file_pattern}'. Skipping this class/VPN type combination.")
               continue # Skip to next class if no file found


           # Assuming there's only one relevant .npy file per class/vpn_type combination
           npy_file_path = class_npy_files[0]
           data_array = import_array(npy_file_path)


           if data_array is not None and data_array.size > 0:
               all_x_for_vpn_type.append(data_array)
               # Create corresponding labels based on the CLASS_NAMES mapping
               label_for_this_class = CLASS_TO_LABEL[class_name]
               labels_for_this_data = np.full(data_array.shape[0], label_for_this_class, dtype=np.int32)
               all_y_for_vpn_type.append(labels_for_this_data)
           else:
               print(f"Skipping {npy_file_path} due to no valid data.")


       if not all_x_for_vpn_type:
           print(f"[!] No data collected for VPN type '{vpn_type}'. Skipping dataset generation for this type.")
           continue # Move to the next VPN type if no data was found


       # Concatenate all collected data and labels for the current VPN type
       combined_x_vpn_type = np.concatenate(all_x_for_vpn_type, axis=0)
       combined_y_vpn_type = np.concatenate(all_y_for_vpn_type, axis=0)


       print(f"Combined data for {vpn_type}: X shape {combined_x_vpn_type.shape}, Y shape {combined_y_vpn_type.shape}")


       # Perform train/test split. We are only interested in the test portion here.
       # stratify ensures that the proportion of classes is maintained in the test set.
       x_train, x_test, y_train, y_test = train_test_split(
           combined_x_vpn_type, combined_y_vpn_type, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=combined_y_vpn_type
       )


       # Save the test portions to the DATASET_DIR with the expected names
       output_x_path = os.path.join(DATASET_DIR, f"{vpn_type}_x_test.npy")
       output_y_path = os.path.join(DATASET_DIR, f"{vpn_type}_y_test.npy")


       np.save(output_x_path, x_test)
       np.save(output_y_path, y_test)


       print(f"Generated {vpn_type}_x_test.npy with shape: {x_test.shape}")
       print(f"Generated {vpn_type}_y_test.npy with shape: {y_test.shape}")


   print("\n--- Combined VPN/Tor/Reg test datasets generation complete ---")


# --- Main Execution Block ---
if __name__ == '__main__':
   generate_combined_vpn_type_test_sets()
