
# Add the path to the shared utilities directory
import os.path as osp

# Define the shared folder path using join for better compatibility
shared_folder = osp.abspath(osp.join(
    osp.dirname(__file__), os.pardir, os.pardir, os.pardir, 'share'
))

# Add the shared folder to sys.path if it's not already included
import sys
if shared_folder not in sys.path:
    sys.path.insert(1, shared_folder)

# Attempt to import the Storage object
try:
    from notebook_utils import (
        NotebookUtilities, nan, isnan, listdir, makedirs, osp, remove, sep, walk,
        CategoricalDtype, DataFrame, Index, NaT, Series, concat, isna, notnull, read_csv,
        read_excel, read_pickle, to_datetime, math, np, re, subprocess, sys, warnings,
        pickle, display
    )
except ImportError as e:
    print(f"Error importing NotebookUtilities: {e}")

# Initialize with data and saves folder paths
nu = NotebookUtilities(
    data_folder_path=osp.abspath(osp.join(
        osp.dirname(__file__), os.pardir, os.pardir, 'data'
    )),
    saves_folder_path=osp.abspath(osp.join(
        osp.dirname(__file__), os.pardir, os.pardir, 'saves'
    ))
)

from .frvrs_utils import (
    FRVRSUtilities, to_numeric, csv, sm
)
fu = FRVRSUtilities(
    data_folder_path=osp.abspath('../data'),
    saves_folder_path=osp.abspath('../saves')
)

# print(r'\b(' + '|'.join(dir()) + r')\b')