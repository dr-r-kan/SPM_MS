"""

Raw bdf files are 1024 Hz, 136 lead (plus stim lead) which are unreferenced. The stim lead has the event markers as numerical values:
1,2,3,4,5,6,7 and 101,102,103,104,105,106,107

ref_res.sets are at 256 Hz, 0-128 Hz filtered. 131 leads (3 cardinal, 128 lead EEG??) but with annotations (rather than stim lead) for events, with the numerical values as the annot descriptions.

Pipeline:
1. Prioritize suj_N_ref_res.set if it exists
2. Otherwise, use suj_N.bdf if it exists, ensuring alignment with ref_res processing
3. Ensure annotations are properly structured
4. Save equalised FIF files to the configured processed/equalised folder
5. Create columns for original unprocessed EEG and partially processed .set files
6. Mark missing data as None for later exclusion

"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd

DEFAULT_INPUT_DIRECTORY = Path("/Volumes/MINIBLACK/Adrian_behavioural_eeg")
DEFAULT_OUTPUT_DIRECTORY = Path(
    "/Users/rohan/EEG/Microstates_and_Interoception/database/"
)
DEFAULT_PROCESSED_DIRECTORY = Path(
    "/Users/rohan/EEG/Microstates_and_Interoception/equalised/"
)


def find_ref_res_set(eeg_folder: str) -> Optional[str]:
    """
    Looks for a suj_N_ref_res.set file in the EEG folder.
    This is the base reference + resampled file without further processing.

    :param eeg_folder: Path to the EEG folder
    :return: Path to ref_res.set file or None
    """
    if not os.path.isdir(eeg_folder):
        return None

    for file in os.listdir(eeg_folder):
        # Match exactly suj_N_ref_res.set (no additional suffixes)
        # Or match suj_N_ref_res256.set (no additional suffixes)
        if re.match(r"^suj_\d+_ref_res\.set$", file):
            path = os.path.join(eeg_folder, file)
            return path
        elif re.match(r"^suj_\d+_ref_res256\.set$", file):
            path = os.path.join(eeg_folder, file)
            return path

    return None


def find_ref_res_fil_set(eeg_folder: str) -> Optional[str]:
    """
    Looks for a suj_N_ref_res_fil0530.set file in the EEG folder.
    This is the reference + resampled + filtered (0-30 Hz) file.

    :param eeg_folder: Path to the EEG folder
    :return: Path to ref_res_fil0530.set file or None
    """
    if not os.path.isdir(eeg_folder):
        return None

    for file in os.listdir(eeg_folder):
        # Match exactly suj_N_ref_res_fil0530.set (no additional suffixes)
        # Or match suj_N_ref_res256_fil0530.set (no additional suffixes)
        if re.match(r"^suj_\d+_ref_res_fil0530\.set$", file):
            path = os.path.join(eeg_folder, file)
            return path
        elif re.match(r"^suj_\d+_ref_res256_fil0530\.set$", file):
            path = os.path.join(eeg_folder, file)
            return path

    return None


def find_ref_res_fil_chl_set(eeg_folder: str) -> Optional[str]:
    """
    Looks for a suj_N_ref_res_fil0530_chl.set file in the EEG folder.
    This is the reference + resampled + filtered (0-30 Hz) + channel rejected file.

    :param eeg_folder: Path to the EEG folder
    :return: Path to ref_res_fil0530_chl.set file or None
    """
    if not os.path.isdir(eeg_folder):
        return None

    for file in os.listdir(eeg_folder):
        # Match exactly suj_N_ref_res_fil0530_chl.set (no additional suffixes)
        # Or match suj_N_ref_res256_fil0530_chl.set (no additional suffixes)
        if re.match(r"^suj_\d+_ref_res_fil0530_chl\.set$", file):
            path = os.path.join(eeg_folder, file)
            return path
        elif re.match(r"^suj_\d+_ref_res_fil0530_chl_intMARCAS\.set$", file):
            path = os.path.join(eeg_folder, file)
            return path
        elif re.match(r"^suj_\d+_ref_res256_fil0530_chl\.set$", file):
            path = os.path.join(eeg_folder, file)
            return path
        elif re.match(r"^suj_\d+_ref_res256_fil0530_chl_intMARCAS\.set$", file):
            path = os.path.join(eeg_folder, file)
            return path
        elif re.match(r"^suj_\d+_ref_res256_fil0530_CH\.set$", file):
            path = os.path.join(eeg_folder, file)
            return path

    return None


def find_raw_bdf(subject_path: str) -> Optional[str]:
    """
    Searches for a plain suj_N.bdf file in EEG-related folders only.
    This is the raw, unprocessed EEG data.
    Only accepts plain suj_N.bdf files (not intero, extero, mw variants).
    Excludes REST folders as per pipeline requirements.

    :param subject_path: Path to the subject folder
    :return: Path to raw .bdf file or None
    """
    # Check EEG and EEG1, EEG2, etc. folders
    for item in os.listdir(subject_path):
        item_path = os.path.join(subject_path, item)
        if not os.path.isdir(item_path):
            continue

        # Only process EEG-related folders (EEG, EEG1, EEG2, etc.)
        if not (item == "EEG" or item.startswith("EEG")):
            continue

        for file in os.listdir(item_path):
            # Must be plain suj_N.bdf (where N is a number)
            # Pattern: suj_DIGITS.bdf (exactly, no extra suffixes)
            if re.match(r"^suj_\d+\.bdf$", file):
                return os.path.join(item_path, file)

    return None


def load_and_validate_ref_res(ref_res_path: str) -> Optional[mne.io.BaseRaw]:
    """
    Loads a ref_res.set file and validates it contains proper annotations.

    :param ref_res_path: Path to the ref_res.set file
    :return: MNE raw object if valid, None otherwise
    """
    try:
        raw = mne.io.read_raw_eeglab(ref_res_path, preload=False)
        # Check for annotations
        if len(raw.annotations) == 0:
            print(f"Warning: {ref_res_path} has no annotations")
            return raw
        return raw
    except Exception as e:
        print(f"Error loading {ref_res_path}: {e}")
        return None


def load_and_validate_bdf(bdf_path: str) -> Optional[mne.io.BaseRaw]:
    """
    Loads a raw .bdf file and validates it.

    :param bdf_path: Path to the .bdf file
    :return: MNE raw object if valid, None otherwise
    """
    try:
        raw = mne.io.read_raw_bdf(bdf_path, preload=False)
        # Check for stim channel
        stim_channels = mne.pick_types(raw.info, stim=True)
        if len(stim_channels) == 0:
            print(f"Warning: {bdf_path} has no stim channel")
        return raw
    except Exception as e:
        print(f"Error loading {bdf_path}: {e}")
        return None


def get_subject_eeg_files(
    subject_path: str,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], bool]:
    """
    Collects all available EEG files for a subject.

    Returns paths for:
    1. Original_EEG: suj_N.bdf (raw unprocessed data)
    2. Ref_Res_Set: suj_N_ref_res.set (reference + resampled, no filtering)
    3. Ref_Res_Fil_Set: suj_N_ref_res_fil0530.set (reference + resampled + 0-30 Hz filtered)
    4. Ref_Res_Fil_Chl_Set: suj_N_ref_res_fil0530_chl.set (reference + resampled + filtered + channel rejected)

    For inclusion in the pipeline, a subject must have:
    - Original_EEG (suj_N.bdf) AND
    - At least one of the ref_res formats

    :param subject_path: Path to the subject folder
    :return: Tuple of (original_bdf, ref_res_set, ref_res_fil_set, ref_res_fil_chl_set, include_flag)
    """
    eeg_folder = os.path.join(subject_path, "EEG")

    # Find all available files
    original_bdf = find_raw_bdf(subject_path)
    ref_res = find_ref_res_set(eeg_folder)
    ref_res_fil = find_ref_res_fil_set(eeg_folder)
    ref_res_fil_chl = find_ref_res_fil_chl_set(eeg_folder)

    # Subject is included if they have the original BDF or at least one ref_res format
    include = (original_bdf is not None) or (
        (ref_res is not None)
        or (ref_res_fil is not None)
        or (ref_res_fil_chl is not None)
    )

    return (original_bdf, ref_res, ref_res_fil, ref_res_fil_chl, include)


def build_source_register(
    input_directory: str | Path,
    output_directory: str | Path = DEFAULT_PROCESSED_DIRECTORY,
) -> pd.DataFrame:
    """
    Builds a comprehensive source register dataframe with all available EEG file formats.

    Creates columns:
    - Subject_ID: Subject identifier
    - Condition: HTN, NHTN, ANX, or NANX
    - Original_EEG: Raw unprocessed BDF file (suj_N.bdf) or None
    - Ref_Res_Set: Reference + resampled .set file (suj_N_ref_res.set) or None
    - Ref_Res_Fil_Set: Reference + resampled + filtered .set file (suj_N_ref_res_fil0530.set) or None
    - Ref_Res_Fil_Chl_Set: Reference + resampled + filtered + channel rejected .set file (suj_N_ref_res_fil0530_chl.set) or None
    - Include: Whether subject should be included (has BDF and at least one ref_res format)

    :param input_directory: Root directory of EEG data
    :param output_directory: Directory to save processed files
    :return: DataFrame with source register
    """
    Condition_directories = {
        "HTN": "Pacientes",
        "NHTN": "Controles",
        "NANX": "Anxiety_cotrols",
        "ANX": "Anxiety_patients",
    }

    rows = []
    input_directory = Path(input_directory).expanduser()
    output_directory = Path(output_directory).expanduser()

    for condition, subfolder in Condition_directories.items():
        condition_path = input_directory / subfolder
        if not condition_path.is_dir():
            print(f"Warning: condition directory does not exist: {condition_path}")
            continue

        for subject_folder in sorted(os.listdir(condition_path)):
            subject_path = condition_path / subject_folder
            if not subject_path.is_dir():
                continue

            print(f"Processing: {subject_folder} ({condition})")

            # Get all available EEG files
            original_eeg, ref_res_set, ref_res_fil_set, ref_res_fil_chl_set, include = (
                get_subject_eeg_files(str(subject_path))
            )

            normalized_subject_id = normalize_subject_id(subject_folder)
            output_name = f"{normalized_subject_id}.fif"
            rows.append(
                {
                    "Subject_ID": subject_folder,
                    "Normalized_Subject_ID": normalized_subject_id,
                    "Condition": condition,
                    "Original_EEG": original_eeg,
                    "Ref_Res_Set": ref_res_set,
                    "Ref_Res_Fil_Set": ref_res_fil_set,
                    "Ref_Res_Fil_Chl_Set": ref_res_fil_chl_set,
                    "Include": include,
                    "Output_name": str(output_directory / output_name),
                }
            )

    df = pd.DataFrame(rows)
    return df


def is_missing_path(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def existing_path_or_none(value: object) -> Optional[str]:
    if is_missing_path(value):
        return None
    path = str(value)
    return path if os.path.exists(path) else None


def normalize_subject_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"suj[_-]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"suj_{match.group(1)}"
    digit_match = re.search(r"(\d+)", text)
    if digit_match:
        return f"suj_{digit_match.group(1)}"
    safe_text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return safe_text or "unknown_subject"


def describe_file(path: Path) -> str:
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{path} ({size_mb:.2f} MB)"


def verify_written_file(path: str | Path, label: str) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        parent = resolved_path.parent
        visible_entries = (
            sorted(item.name for item in parent.glob("*")) if parent.exists() else []
        )
        preview = ", ".join(visible_entries[:20])
        if len(visible_entries) > 20:
            preview += ", ..."
        raise FileNotFoundError(
            f"{label} was expected at {resolved_path}, but it does not exist. "
            f"Parent exists: {parent.exists()}. Parent contents: [{preview}]"
        )
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{label} exists but is not a file: {resolved_path}")
    return resolved_path


def save_raw_to_fif(raw: mne.io.BaseRaw, save_path: str) -> str:
    destination = Path(save_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving FIF file to {destination}")
    raw.save(destination, overwrite=True)
    written = verify_written_file(destination, "Saved FIF file")
    print(f"Verified saved FIF file: {describe_file(written)}")
    return str(written)


def output_fif_path(row: pd.Series) -> str:
    output_value = row.get("Output_name")
    if not is_missing_path(output_value):
        return str(Path(str(output_value)).expanduser().with_suffix(".fif"))

    subject_id = normalize_subject_id(
        row.get("Normalized_Subject_ID", row.get("Subject_ID", "unknown_subject"))
    )
    return str(DEFAULT_PROCESSED_DIRECTORY / f"{subject_id}.fif")


def filtered_ref_res_path(ref_res_path: str) -> str:
    path = Path(ref_res_path)
    stem = path.stem
    if stem.endswith("_ref_res"):
        new_stem = stem.removesuffix("_ref_res") + "_ref_res_fil0530_rk"
    elif stem.endswith("_ref_res256"):
        new_stem = stem.removesuffix("_ref_res256") + "_ref_res256_fil0530_rk"
    else:
        new_stem = stem + "_fil0530_rk"
    return str(path.with_name(new_stem + ".set"))


def find_stim_channel(raw: mne.io.BaseRaw) -> Optional[str]:
    if "Status" in raw.ch_names:
        return "Status"
    stim_channels = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, stim=True)]
    if stim_channels:
        return stim_channels[0]
    for channel in raw.ch_names:
        if channel.lower() in {"status", "sti 014", "sti014"}:
            return channel
    return None


def align_EEGs(row):
    """
    Code which takes input EEG and gets it all to the same processing stage.
    1. rereferenced to average
    2. resampled to 256 Hz
    3. filtered 0.5-30 Hz
    4. annotations properly structured (as annotations, not as stim lead)

    :param row: dataframe row with EEG file paths in columns: Original_EEG, Ref_Res_Set, Ref_Res_Fil_Set, Ref_Res_Fil_Chl_Set
    """
    ref_res_fil_chl = existing_path_or_none(row.get("Ref_Res_Fil_Chl_Set"))
    ref_res_fil = existing_path_or_none(row.get("Ref_Res_Fil_Set"))
    ref_res = existing_path_or_none(row.get("Ref_Res_Set"))
    original_eeg = existing_path_or_none(row.get("Original_EEG"))

    destination_fif_path = output_fif_path(row)

    if ref_res_fil_chl is not None:
        # Already at final stage; convert it to FIF in the pipeline equalised directory.
        raw = mne.io.read_raw_eeglab(ref_res_fil_chl, preload=True)
        return save_raw_to_fif(raw, destination_fif_path)
    elif ref_res_fil is not None:
        # Already referenced/resampled/filtered; convert it to FIF in the pipeline equalised directory.
        raw = mne.io.read_raw_eeglab(ref_res_fil, preload=True)
        return save_raw_to_fif(raw, destination_fif_path)
    elif ref_res is not None:
        # Load ref_res file, apply filtering, and save as FIF in the pipeline equalised directory.
        raw = mne.io.read_raw_eeglab(ref_res, preload=True)
        raw.filter(0.5, 30, fir_design="firwin", n_jobs=12)
        return save_raw_to_fif(raw, destination_fif_path)
    elif original_eeg is not None:
        # Load raw BDF, apply referencing, resampling, filtering, channel rejection
        raw = mne.io.read_raw_bdf(original_eeg, preload=True)
        raw.set_eeg_reference("average", projection=False)
        raw.resample(256)
        raw.filter(0.5, 30, fir_design="firwin", n_jobs=12)

        # take stim channel and convert to annotations
        # load events from stim lead with values 1,2,3,4,5,6,7 and 101,102,103,104,105,106,107
        stim_channel = find_stim_channel(raw)
        if stim_channel is None:
            raise RuntimeError(
                f"No stim channel found in {original_eeg}. Channels: {raw.ch_names}"
            )

        try:
            events = mne.find_events(
                raw,
                stim_channel=stim_channel,
                shortest_event=1,
                consecutive="increasing",
                initial_event=False,
                verbose=False,
            )
        except Exception as e:
            print(f"Error finding events in {original_eeg}: {e}")
            print("Channels")
            print(raw.ch_names)
            raise e

        # exclude ones with the value 65536 (which can be ignored)
        events = events[events[:, 2] != 65536]
        events = events[events[:, 2] != 65542]

        # create annotations
        onsets = events[:, 0] / raw.info["sfreq"]
        durations = np.zeros(len(events))
        descriptions = [str(int(code)) for code in events[:, 2]]
        annotations = mne.Annotations(
            onset=onsets, duration=durations, description=descriptions
        )
        raw.set_annotations(annotations)
        # Save new processed file as FIF
        saved_path = save_raw_to_fif(raw, destination_fif_path)
        print(f"Saved processed EEG to {saved_path}")
        return saved_path
    else:
        return None


def verify_aligned_outputs_exist(dataframe: pd.DataFrame) -> None:
    missing_rows = []
    for _, row in dataframe.iterrows():
        eeg_path = row.get("Raw_EEG_File_Path")
        if is_missing_path(eeg_path):
            continue
        path = Path(str(eeg_path)).expanduser().resolve()
        if not path.exists():
            missing_rows.append((row.get("Subject_ID"), str(path)))

    if missing_rows:
        preview = "\n".join(
            f"  - {subject_id}: {path}" for subject_id, path in missing_rows[:20]
        )
        if len(missing_rows) > 20:
            preview += f"\n  ... and {len(missing_rows) - 20} more"
        raise FileNotFoundError(
            f"{len(missing_rows)} aligned EEG outputs listed in the register do not exist:\n{preview}"
        )

    verified_count = (
        dataframe["Raw_EEG_File_Path"].apply(lambda x: not is_missing_path(x)).sum()
    )
    print(f"Verified {verified_count} aligned EEG output paths exist.")


def print_output_directory_summary(processed_directory: Path) -> None:
    fif_files = sorted(processed_directory.glob("*.fif"))
    total_bytes = sum(path.stat().st_size for path in fif_files if path.is_file())
    print(
        f"Output directory summary: {processed_directory} contains "
        f"{len(fif_files)} .fif files ({total_bytes / (1024**3):.2f} GiB total)."
    )


def check_annotations_work(dataframe):
    """
    check that annotations are all with descriptions in the set {1,2,3,4,5,6,7,101,102,103,104,105,106,107}

    print any files which do not conform
    """
    valid_descriptions = {
        str(i) for i in [1, 2, 3, 4, 5, 6, 7, 101, 102, 103, 104, 105, 106, 107]
    }
    for idx, row in dataframe.iterrows():
        eeg_path = existing_path_or_none(row.get("Raw_EEG_File_Path"))
        if eeg_path is None:
            print(f"Subject {row['Subject_ID']} has no aligned EEG")
            continue
        raw = mne.io.read_raw_fif(eeg_path, preload=False)
        descriptions = set(raw.annotations.description)
        if not descriptions.issubset(valid_descriptions):
            print(
                f"Subject {row['Subject_ID']} has invalid annotations: {descriptions}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and align Adrian behavioural EEG source data into a source register."
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help=f"Root directory containing condition folders. Default: {DEFAULT_INPUT_DIRECTORY}",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Directory where source_register.csv will be written. Default: {DEFAULT_OUTPUT_DIRECTORY}",
    )
    parser.add_argument(
        "--processed-directory",
        type=Path,
        default=DEFAULT_PROCESSED_DIRECTORY,
        help=f"Directory used for Output_name entries. Default: {DEFAULT_PROCESSED_DIRECTORY}",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Only compile discovered paths; do not load/filter/export EEG files.",
    )
    parser.add_argument(
        "--skip-annotation-check",
        action="store_true",
        help="Skip post-alignment annotation validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = args.output_directory.expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    processed_directory = args.processed_directory.expanduser().resolve()
    processed_directory.mkdir(parents=True, exist_ok=True)
    print(f"Register CSV directory: {output_directory.resolve()}")
    print(f"EEG .fif output directory: {processed_directory}")

    source_register_df = build_source_register(
        args.input_directory, processed_directory
    )

    if args.skip_alignment:
        source_register_df["Raw_EEG_File_Path"] = None
    else:
        source_register_df["Raw_EEG_File_Path"] = source_register_df.apply(
            align_EEGs, axis=1
        )

    verify_aligned_outputs_exist(source_register_df)
    print_output_directory_summary(processed_directory)

    if not args.skip_annotation_check:
        check_annotations_work(source_register_df)

    # Save register
    csv_path = output_directory / "source_register.csv"
    source_register_df.to_csv(csv_path, index=False)
    print("\n" + "=" * 80)
    print(f"Source register saved to {csv_path}")

    # Summarise how many files could be aligned/copied into pp.
    total_subjects = len(source_register_df)
    aligned_mask = source_register_df["Raw_EEG_File_Path"].apply(
        lambda x: not is_missing_path(x)
    )
    copied_ready_mask = aligned_mask & (
        source_register_df["Ref_Res_Fil_Chl_Set"].apply(
            lambda x: not is_missing_path(x)
        )
        | source_register_df["Ref_Res_Fil_Set"].apply(lambda x: not is_missing_path(x))
    )
    ready_subjects = copied_ready_mask.sum()
    processed_subjects = aligned_mask.sum() - ready_subjects
    missing_subjects = total_subjects - aligned_mask.sum()
    print(f"Total subjects: {total_subjects}")
    print(f"Subjects ready to go (no extra processing needed): {ready_subjects}")
    print(f"Subjects processed to align EEGs: {processed_subjects}")
    print(f"Subjects missing EEG data: {missing_subjects}")


# Main execution
if __name__ == "__main__":
    main()
