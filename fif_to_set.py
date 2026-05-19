import mne
import os

def convert_fif_to_set(fif_file, set_file):
    # Load the FIF file
    raw = mne.io.read_raw_fif(fif_file, preload=True)

    # Save the data in .set format
    mne.export.export_raw(set_file, raw, fmt='eeglab')

if __name__ == "__main__":    # Example usage
    dir = r"E:\EEGs\TEST_EEG"
    for file in os.listdir(dir):
        if file.endswith(".fif"):
            fif_file = os.path.join(dir, file)
            set_file = os.path.join(dir, file.replace(".fif", ".set"))
            convert_fif_to_set(fif_file, set_file)
            print(f"Converted {fif_file} to {set_file}")