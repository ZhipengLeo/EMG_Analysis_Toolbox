import re


def parse_force_from_filename(filename: str) -> int:
    """
    Parse force level from EMG h5 filename.

    Example:
        S01_G01_F10_T01_dev0.h5 -> 10

    Returns:
        force level as int (e.g. 10, 30, 50)
    """
    match = re.search(r"_F(\d+)_", filename)
    if match is None:
        raise ValueError(f"Cannot parse force from filename: {filename}")
    return int(match.group(1))

def parse_gesture_from_filename(filename: str) -> int:
    """
    Parse gesture label from EMG h5 filename.

    Example:
        S01_G01_F10_T01_dev0.h5 -> 1
    """
    match = re.search(r"_G(\d+)_", filename)
    if match is None:
        raise ValueError(f"Cannot parse gesture from filename: {filename}")
    return int(match.group(1))

