
#!/usr/bin/env python
# Utility Functions to run Jupyter notebooks.
# Dave Babbitt <dave.babbitt@gmail.com>
# Author: Dave Babbitt, Data Scientist
# coding: utf-8

# Soli Deo gloria

from bs4 import BeautifulSoup as bs
from datetime import timedelta
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve
import csv
import humanize
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas as pd
import random
import re
import subprocess
import sys
import urllib
try: import dill as pickle
except:
    try: import pickle5 as pickle
    except: import pickle

import warnings
warnings.filterwarnings("ignore")

class NotebookUtilities(object):
    """
    This class implements the core of the utility
    functions needed to install and run GPTs and 
    also what is common to running Jupyter notebooks.
    
    Examples
    --------
    
    import sys
    import os.path as osp
    sys.path.insert(1, osp.abspath('../py'))
    from notebook_utils import NotebookUtilities
    
    nu = NotebookUtilities(
        data_folder_path=osp.abspath('../data'),
        saves_folder_path=osp.abspath('../saves')
    )
    """
    
    def __init__(self, data_folder_path=None, saves_folder_path=None, verbose=False):
        self.verbose = verbose
        self.pip_command_str = f'{sys.executable} -m pip'
        self.update_modules_list(verbose=verbose)
        
        # Create the data folder if it doesn't exist
        if data_folder_path is None:
            self.data_folder = '../data'
        else:
            self.data_folder = data_folder_path
        os.makedirs(self.data_folder, exist_ok=True)
        if verbose: print('data_folder: {}'.format(osp.abspath(self.data_folder)), flush=True)
        
        # Create the saves folder if it doesn't exist
        if saves_folder_path is None:
            self.saves_folder = '../saves'
        else:
            self.saves_folder = saves_folder_path
        os.makedirs(self.saves_folder, exist_ok=True)
        if verbose: print('saves_folder: {}'.format(osp.abspath(self.saves_folder)), flush=True)
        
        # Create the assumed directories
        self.bin_folder = osp.join(self.data_folder, 'bin'); os.makedirs(self.bin_folder, exist_ok=True)
        self.cache_folder = osp.join(self.data_folder, 'cache'); os.makedirs(self.cache_folder, exist_ok=True)
        self.data_csv_folder = osp.join(self.data_folder, 'csv'); os.makedirs(name=self.data_csv_folder, exist_ok=True)
        self.data_models_folder = osp.join(self.data_folder, 'models'); os.makedirs(name=self.data_models_folder, exist_ok=True)
        self.db_folder = osp.join(self.data_folder, 'db'); os.makedirs(self.db_folder, exist_ok=True)
        self.graphs_folder = osp.join(self.saves_folder, 'graphs'); os.makedirs(self.graphs_folder, exist_ok=True)
        self.indices_folder = osp.join(self.saves_folder, 'indices'); os.makedirs(self.indices_folder, exist_ok=True)
        self.saves_csv_folder = osp.join(self.saves_folder, 'csv'); os.makedirs(name=self.saves_csv_folder, exist_ok=True)
        self.saves_mp3_folder = osp.join(self.saves_folder, 'mp3'); os.makedirs(name=self.saves_mp3_folder, exist_ok=True)
        self.saves_pickle_folder = osp.join(self.saves_folder, 'pkl'); os.makedirs(name=self.saves_pickle_folder, exist_ok=True)
        self.saves_text_folder = osp.join(self.saves_folder, 'txt'); os.makedirs(name=self.saves_text_folder, exist_ok=True)
        self.saves_wav_folder = osp.join(self.saves_folder, 'wav'); os.makedirs(name=self.saves_wav_folder, exist_ok=True)
        self.txt_folder = osp.join(self.data_folder, 'txt'); os.makedirs(self.txt_folder, exist_ok=True)
        
        # Get various model paths
        self.lora_path = osp.abspath(osp.join(self.bin_folder, 'gpt4all-lora-quantized.bin'))
        self.gpt4all_model_path = osp.abspath(osp.join(self.bin_folder, 'gpt4all-lora-q-converted.bin'))
        self.ggjt_model_path = osp.abspath(osp.join(
            self.cache_folder, 'models--LLukas22--gpt4all-lora-quantized-ggjt', 'snapshots', '2e7367a8557085b8267e1f3b27c209e272b8fe6c',
            'ggjt-model.bin'
        ))
        
        # Ensure the Scripts folder is in PATH
        self.anaconda_folder = osp.dirname(sys.executable)
        self.scripts_folder = osp.join(self.anaconda_folder, 'Scripts')
        if self.scripts_folder not in sys.path:
            sys.path.insert(1, self.scripts_folder)

        # Handy list of the different types of encodings
        self.encoding_type = ['latin1', 'iso8859-1', 'utf-8'][2]
        
        # Determine URL from file path
        self.url_regex = re.compile(r'\b(https?|file)://[-A-Z0-9+&@#/%?=~_|$!:,.;]*[A-Z0-9+&@#/%=~_|$]', re.IGNORECASE)
        self.filepath_regex = re.compile(
            r'\b[c-d]:\\(?:[^\\/:*?"<>|\x00-\x1F]{0,254}[^.\\/:*?"<>|\x00-\x1F]\\)*(?:[^\\/:*?"<>|\x00-\x1F]{0,254}[^.\\/:*?"<>|\x00-\x1F])', re.IGNORECASE
        )
        
        # Various aspect ratios
        self.facebook_aspect_ratio = 1.91
        self.twitter_aspect_ratio = 16/9

    ### String Functions ###
    
    def compute_similarity(self, a: str, b: str) -> float:
        """
        Compute the similarity between two strings.

        Parameters
        ----------
        a : str
            The first string.
        b : str
            The second string.

        Returns
        -------
        float
            The similarity between the two strings, as a float between 0 and 1.
        """
        from difflib import SequenceMatcher

        return SequenceMatcher(None, str(a), str(b)).ratio()
    
    def format_timedelta(self, timedelta):
        """
        Formats a timedelta object to a string in the format '0', '30', '1 min', '1:30', '2 min', etc.
        
        Args:
          timedelta: A timedelta object.
        
        Returns:
          A string in the format '0', '30', '1 min', '1:30', '2 min', etc.
        """
        seconds = timedelta.total_seconds()
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        
        if minutes == 0: return str(seconds)
        elif seconds > 0: return f'{minutes}:{seconds:02}'
        else: return f'{minutes} min'

    ### List Functions ###
    
    def conjunctify_nouns(self, noun_list, and_or='and', verbose=False):
        """
        Concatenates a list of nouns into a grammatically correct string with specified conjunctions.
        
        Parameters:
            noun_list (list or str): A list of nouns to be concatenated.
            and_or (str, optional): The conjunction used to join the nouns. Default is 'and'.
            verbose (bool, optional): If True, prints verbose output. Default is False.
        
        Returns:
            str: A string containing the concatenated nouns with appropriate conjunctions.
        
        Example:
            noun_list = ['apples', 'oranges', 'bananas']
            conjunction = 'and'
            result = conjunctify_nouns(noun_list, and_or=conjunction)
            print(result)
            Output: 'apples, oranges, and bananas'
        """
        
        # Handle special cases where noun_list is None or not a list
        if (noun_list is None): return ''
        if (type(noun_list) != list): noun_list = list(noun_list)
        
        # If there are more than two nouns in the list, join the last two nouns with `and_or`
        # Otherwise, join all of the nouns with `and_or`
        if (len(noun_list) > 2):
            last_noun_str = noun_list[-1]
            but_last_nouns_str = ', '.join(noun_list[:-1])
            list_str = f', {and_or} '.join([but_last_nouns_str, last_noun_str])
        elif (len(noun_list) == 2): list_str = f' {and_or} '.join(noun_list)
        elif (len(noun_list) == 1): list_str = noun_list[0]
        else: list_str = ''
        
        # Print verbose output if requested
        if verbose: print(f'Conjunctified noun list: {list_str}')
        
        # Return the conjuncted noun list
        return list_str


    def check_4_doubles(self, item_list, verbose=False):
        if verbose: t0 = time.time()
        rows_list = []
        n = len(item_list)
        for i in range(n-1):
            first_item = item_list[i]
            max_similarity = 0.0
            max_item = first_item
            for j in range(i+1, n):
                second_item = item_list[j]

                # Assume the first item is never identical to the second item
                this_similarity = self.compute_similarity(str(first_item), str(second_item))

                if this_similarity > max_similarity:
                    max_similarity = this_similarity
                    max_item = second_item

            # Get input row in dictionary format; key = col_name
            row_dict = {}
            row_dict['first_item'] = first_item
            row_dict['second_item'] = max_item
            row_dict['first_bytes'] = '-'.join(str(x) for x in bytearray(str(first_item),
                                                                         encoding=self.encoding, errors="replace"))
            row_dict['second_bytes'] = '-'.join(str(x) for x in bytearray(str(max_item),
                                                                          encoding=self.encoding, errors="replace"))
            row_dict['max_similarity'] = max_similarity

            rows_list.append(row_dict)

        column_list = ['first_item', 'second_item', 'first_bytes', 'second_bytes', 'max_similarity']
        item_similarities_df = pd.DataFrame(rows_list, columns=column_list)
        if verbose:
            t1 = time.time()
            print(t1-t0, time.ctime(t1))

        return item_similarities_df

    ### File Functions ###
    
    def get_file_path(self, func):
        """
        Returns the relative or absolute file path where the function is stored.

        Args:
            func: A Python function.

        Returns:
            A string representing the relative or absolute file path where the function is stored.

        Example:
            def my_function(): pass
            file_path = nu.get_file_path(my_function)
            print(os.path.abspath(file_path))
        """
        import inspect
        file_path = inspect.getfile(func)

        # If the function is defined in a Jupyter notebook, return the absolute file path
        if file_path.startswith('<stdin>'): return os.path.abspath(file_path)

        # Otherwise, return the relative file path
        else: return os.path.relpath(file_path)
    
    def show_duplicated_util_fns_search_string(self, util_path=None, github_folder=None):
        """
        Search for duplicate utility function definitions in Jupyter notebooks within a specified GitHub repository folder.
        The function identifies rogue utility function definitions in Jupyter notebooks and prints a regular expression
        pattern to search for instances of these definitions. The intention is to replace these calls with the
        corresponding `nu.` equivalent and remove the duplicates.

        Parameters:
            util_path (str, optional): The path to the utilities file to check for existing utility function definitions.
                                       Defaults to `../py/notebook_utils.py`.
            github_folder (str, optional): The path of the root folder of the GitHub repository containing the notebooks.
                                           Defaults to the parent directory of the current working directory.

        Returns:
            None: The function prints the regular expression pattern to identify rogue utility function definitions.
        """

        # Get a list of rogue functions already in utilities file
        if util_path is None: util_path = '../py/notebook_utils.py'
        utils_regex = re.compile(r'def ([a-z0-9_]+)\(')
        with open(util_path, 'r', encoding='utf-8') as f:
            lines_list = f.readlines()
            utils_set = set()
            for line in lines_list:
                match_obj = utils_regex.search(line)
                if match_obj:
                    scraping_util = match_obj.group(1)
                    utils_set.add(scraping_util)

        # Make a set of rogue util functions
        fn_regex = re.compile(r'\s+"def ([a-z0-9_]+)\(')
        black_list = ['.ipynb_checkpoints', '$Recycle.Bin']
        rows_list = []
        if github_folder is None: github_folder = osp.dirname(osp.abspath(osp.curdir))
        rogue_fns_set = set()
        for sub_directory, directories_list, files_list in os.walk(github_folder):
            if all(map(lambda x: x not in sub_directory, black_list)):
                for file_name in files_list:
                    if file_name.endswith('.ipynb'):
                        file_path = osp.join(sub_directory, file_name)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines_list = f.readlines()
                            for line in lines_list:
                                match_obj = fn_regex.search(line)
                                if match_obj:
                                    fn = match_obj.group(1)
                                    if fn in utils_set: rogue_fns_set.add(fn)
        
        if rogue_fns_set:
            print(f'Search for *.ipynb; file masks in the {github_folder} folder for this pattern:')
            print('\\s+"def (' + '|'.join(rogue_fns_set) + ')\(')
            print('Replace each of the calls to these definitions with calls the the nu. equivalent (and delete the definitions).')

    ### Storage Functions ###

    def csv_exists(self, csv_name, folder_path=None, verbose=False):
        if folder_path is None:
            folder_path = self.saves_csv_folder
        if csv_name.endswith('.csv'): csv_path = osp.join(folder_path, csv_name)
        else: csv_path = osp.join(folder_path, f'{csv_name}.csv')
        if verbose: print(osp.abspath(csv_path), flush=True)

        return osp.isfile(csv_path)

    def load_csv(self, csv_name=None, folder_path=None):
        if folder_path is None:
            csv_folder = self.data_csv_folder
        else:
            csv_folder = osp.join(folder_path, 'csv')
        if csv_name is None:
            csv_path = max([osp.join(csv_folder, f) for f in os.listdir(csv_folder)],
                           key=osp.getmtime)
        else:
            if csv_name.endswith('.csv'):
                csv_path = osp.join(csv_folder, csv_name)
            else:
                csv_path = osp.join(csv_folder, f'{csv_name}.csv')
        data_frame = pd.read_csv(osp.abspath(csv_path), encoding=self.encoding_type)

        return(data_frame)

    def pickle_exists(self, pickle_name: str) -> bool:
        """
        Checks if a pickle file exists.

        Parameters
        ----------
        pickle_name : str
            The name of the pickle file.

        Returns
        -------
        bool
            True if the pickle file exists, False otherwise.
        """
        pickle_path = osp.join(self.saves_pickle_folder, '{}.pkl'.format(pickle_name))

        return osp.isfile(pickle_path)

    def load_data_frames(self, **kwargs):
        frame_dict = {}
        for frame_name in kwargs:
            pickle_path = osp.join(self.saves_pickle_folder, '{}.pkl'.format(frame_name))
            print('Attempting to load {}.'.format(osp.abspath(pickle_path)), flush=True)
            if not osp.isfile(pickle_path):
                csv_name = '{}.csv'.format(frame_name)
                csv_path = osp.join(self.saves_csv_folder, csv_name)
                print('No pickle exists - attempting to load {}.'.format(osp.abspath(csv_path)), flush=True)
                if not osp.isfile(csv_path):
                    csv_path = osp.join(self.data_csv_folder, csv_name)
                    print('No csv exists - trying {}.'.format(osp.abspath(csv_path)), flush=True)
                    if not osp.isfile(csv_path):
                        print('No csv exists - just forget it.', flush=True)
                        frame_dict[frame_name] = None
                    else:
                        frame_dict[frame_name] = self.load_csv(csv_name=frame_name)
                else:
                    frame_dict[frame_name] = self.load_csv(csv_name=frame_name, folder_path=self.saves_folder)
            else:
                frame_dict[frame_name] = self.load_object(frame_name)

        return frame_dict

    def load_object(self, obj_name: str, pickle_path: str = None, download_url: str = None, verbose: bool = False) -> object:
        """
        Load an object from a pickle file.

        Parameters
        ----------
        obj_name : str
            The name of the object to load.
        pickle_path : str, optional
            The path to the pickle file. Defaults to None.
        download_url : str, optional
            The URL to download the pickle file from. Defaults to None.
        verbose : bool, optional
            Whether to print status messages. Defaults to False.

        Returns
        -------
        object
            The loaded object.
        """
        if pickle_path is None:
            pickle_path = osp.join(self.saves_pickle_folder, '{}.pkl'.format(obj_name))
        if not osp.isfile(pickle_path):
            if verbose: print('No pickle exists at {} - attempting to load as csv.'.format(osp.abspath(pickle_path)), flush=True)
            csv_path = osp.join(self.saves_csv_folder, '{}.csv'.format(obj_name))
            if not osp.isfile(csv_path):
                if verbose: print('No csv exists at {} - attempting to download from URL.'.format(osp.abspath(csv_path)), flush=True)
                object = pd.read_csv(download_url, low_memory=False,
                                     encoding=self.encoding_type)
            else:
                object = pd.read_csv(csv_path, low_memory=False,
                                     encoding=self.encoding_type)
            if isinstance(object, pd.DataFrame):
                self.attempt_to_pickle(object, pickle_path, raise_exception=False)
            else:
                with open(pickle_path, 'wb') as handle:

                    # Protocol 4 is not handled in python 2
                    if sys.version_info.major == 2:
                        pickle.dump(object, handle, 2)
                    elif sys.version_info.major == 3:
                        pickle.dump(object, handle, pickle.HIGHEST_PROTOCOL)

        else:
            try:
                object = pd.read_pickle(pickle_path)
            except:
                with open(pickle_path, 'rb') as handle:
                    object = pickle.load(handle)

        if verbose: print('Loaded object {} from {}'.format(obj_name, pickle_path), flush=True)

        return(object)
    
    def save_data_frames(self, include_index=False, verbose=True, **kwargs):
        """
        Saves data frames to CSV files.

        Args:
            include_index: Whether to include the index in the CSV files.
            verbose: Whether to print information about the saved files.
            **kwargs: A dictionary of data frames to save. The keys of the dictionary
                      are the names of the CSV files to save the data frames to.

        Returns:
            None
        """

        # Iterate over the data frames in the kwargs dictionary and save them to CSV files
        for frame_name in kwargs:
            if isinstance(kwargs[frame_name], pd.DataFrame):
                
                # Generate the path to the CSV file
                csv_path = osp.join(self.saves_csv_folder, '{}.csv'.format(frame_name))

                # Print a message about the saved file if verbose is True
                if verbose: print('Saving to {}'.format(osp.abspath(csv_path)), flush=True)

                # Save the data frame to a CSV file
                kwargs[frame_name].to_csv(csv_path, sep=',', encoding=self.encoding_type,
                                          index=include_index)
    
    def store_objects(self, verbose: bool = True, **kwargs: dict) -> None:
        """
        Store objects to pickle files.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print status messages. Defaults to True.
        **kwargs : dict
            The objects to store. The keys of the dictionary are the names of the objects, and the values are the objects themselves.

        Returns
        -------
        None

        """
        for obj_name in kwargs:
            # if hasattr(kwargs[obj_name], '__call__'):
            #     raise RuntimeError('Functions cannot be pickled.')
            pickle_path = osp.join(self.saves_pickle_folder, '{}.pkl'.format(obj_name))
            if isinstance(kwargs[obj_name], pd.DataFrame):
                self.attempt_to_pickle(kwargs[obj_name], pickle_path, raise_exception=False, verbose=verbose)
            else:
                if verbose: print('Pickling to {}'.format(osp.abspath(pickle_path)), flush=True)
                with open(pickle_path, 'wb') as handle:

                    # Protocol 4 is not handled in python 2
                    if sys.version_info.major == 2:
                        pickle.dump(kwargs[obj_name], handle, 2)

                    # Pickle protocol must be <= 4
                    elif sys.version_info.major == 3:
                        pickle.dump(kwargs[obj_name], handle, min(4, pickle.HIGHEST_PROTOCOL))

    def attempt_to_pickle(self, df: pd.DataFrame, pickle_path: str, raise_exception: bool = False, verbose: bool = True) -> None:
        """
        Attempts to pickle a DataFrame to a file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to pickle.
        pickle_path : str
            The path to the pickle file.
        raise_exception : bool, optional
            Whether to raise an exception if the pickle fails. Defaults to False.
        verbose : bool, optional
            Whether to print status messages. Defaults to True.

        Returns
        -------
        None

        """
        try:
            if verbose: print('Pickling to {}'.format(osp.abspath(pickle_path)), flush=True)

            # Protocol 4 is not handled in python 2
            if sys.version_info.major == 2: df.to_pickle(pickle_path, protocol=2)

            # Pickle protocol must be <= 4
            elif sys.version_info.major == 3: df.to_pickle(pickle_path, protocol=min(4, pickle.HIGHEST_PROTOCOL))

        except Exception as e:
            os.remove(pickle_path)
            if verbose: print(e, ": Couldn't save {:,} cells as a pickle.".format(df.shape[0]*df.shape[1]), flush=True)
            if raise_exception: raise
    
    ### Module Functions ###
    
    def get_dir_tree(self, module_name, contains_str=None, not_contains_str=None, verbose=False):
        """
        Gets a list of all attributes in a given module.
        
        Parameters:
        -----------
        module_name : str
            The name of the module to get the directory list for.
        contains_str : str, optional
            If provided, only print attributes containing this substring (case-insensitive).
        not_contains_str : str, optional
            If provided, exclude printing attributes containing this substring (case-insensitive).
        verbose : bool, optional
            If True, print additional information during processing.
        
        Returns:
        --------
        list[str]
            A list of attributes in the module that match the filtering criteria.
        """
    
        # Initialize sets for processed attributes and their suffixes
        dirred_set = set([module_name])
        suffix_set = set([module_name])
    
        # Initialize an unprocessed set of all attributes in the module_name module that don't start with an underscore
        import importlib
        module_obj = importlib.import_module(module_name)
        undirred_set = set([f'module_obj.{fn}' for fn in dir(module_obj) if not fn.startswith('_')])
    
        # Continue processing until the unprocessed set is empty
        while undirred_set:
    
            # Pop the next function or submodule
            fn = undirred_set.pop()
    
            # Extract the suffix of the function or submodule
            fn_suffix = fn.split('.')[-1]
    
            # Check if the suffix has not been processed yet
            if fn_suffix not in suffix_set:
                
                # Add it to processed and suffix sets
                dirred_set.add(fn)
                suffix_set.add(fn_suffix)
    
                try:
                    
                    # Evaluate the 'dir()' function for the attribute and update the unprocessed set with its function or submodule
                    dir_list = eval(f'dir({fn})')
    
                    # Add all of the submodules of the function or submodule to undirred_set if they haven't been processed yet
                    undirred_set.update([f'{fn}.{fn1}' for fn1 in dir_list if not fn1.startswith('_')])
    
                # If there is an error getting the dir() of the function or submodule, just continue to the next iteration
                except: continue
                
        # Apply filtering criteria if provided
        if (not bool(contains_str)) and bool(not_contains_str):
            dirred_set = [fn for fn in dirred_set if (not_contains_str not in fn.lower())]
        elif bool(contains_str) and (not bool(not_contains_str)):
            dirred_set = [fn for fn in dirred_set if (contains_str in fn.lower())]
        elif bool(contains_str) and bool(not_contains_str):
            dirred_set = [fn for fn in dirred_set if (contains_str in fn.lower()) and (not_contains_str not in fn.lower())]
        
        # Remove the importlib object variable name
        dirred_set = set([fn.replace('module_obj', module_name) for fn in dirred_set])
        
        return sorted(dirred_set)
    
    def update_modules_list(self, modules_list: Optional[List[str]] = None, verbose: bool = False) -> None:
        """
        Updates the list of modules that are installed.

        Parameters
        ----------
        modules_list : Optional[List[str]], optional
            The list of modules to update. If None, the list of installed modules will be used. Defaults to None.
        verbose : bool, optional
            Whether to print status messages. Defaults to False.

        Returns
        -------
        None
        """

        if modules_list is None: self.modules_list = [
            o.decode().split(' ')[0] for o in subprocess.check_output(f'{self.pip_command_str} list'.split(' ')).splitlines()[2:]
            ]
        else: self.modules_list = modules_list

        if verbose: print('Updated modules list to {}'.format(self.modules_list), flush=True)
    
    def ensure_module_installed(self, module_name: str, upgrade: bool = False, verbose: bool = True) -> None:
        """
        Checks if a module is installed and installs it if it is not.

        Parameters
        ----------
        module_name : str
            The name of the module to check for.
        upgrade : bool, optional
            Whether to upgrade the module if it is already installed. Defaults to False.
        verbose : bool, optional
            Whether to print status messages. Defaults to True.

        Returns
        -------
        None
        """

        if module_name not in self.modules_list:
            command_str = f'{self.pip_command_str} install {module_name}'
            if upgrade: command_str += ' --upgrade'
            if verbose: print(command_str, flush=True)
            else: command_str += ' --quiet'
            output_str = subprocess.check_output(command_str.split(' '))
            if verbose:
                for line_str in output_str.splitlines(): print(line_str.decode(), flush=True)
            self.update_modules_list(verbose=verbose)
    
    ### URL and Soup Functions ###
    
    def get_filename_from_url(self, url, verbose=False):
        """
        Extracts the filename from a given URL.

        Parameters:
        -----------
        url : str
            The URL from which to extract the filename.
        verbose : bool, optional
            If True, print additional information (default is False).

        Returns:
        --------
        str
            The extracted filename from the URL.
        """

        # Import the urllib module for URL parsing
        import urllib

        # Parse the URL and extract the filename from the path
        file_name = urllib.parse.urlparse(url).path.split('/')[-1]
        
        # Print verbose information if verbose flag is True
        if verbose: print(f"Extracted filename from '{url}': '{file_name}'")

        return file_name
    
    def download_file(self, url, download_dir=None, exist_ok=False, verbose=False):
        """
        Downloads a file from the internet.

        Args:
            url: The URL of the file to download.
            download_dir: The directory to download the file to. If None, the file
                          will be downloaded to the `downloads` subdirectory of the data folder.
            exist_ok: If True, the function will not raise an error if the file
                      already exists.
            verbose: If True, the function will print progress information to the
                     console.

        Returns:
            The path to the downloaded file.
        """

        # Get the file name from the URL
        file_name = self.get_filename_from_url(url, verbose=verbose)

        # If the download directory is not specified, use the downloads subdirectory
        if download_dir is None: download_dir = osp.join(self.data_folder, 'downloads')

        # Create the download directory if it does not exist
        os.makedirs(download_dir, exist_ok=True)

        # Compute the path to the downloaded file
        file_path = osp.join(download_dir, file_name)

        # If the file does not exist or if exist_ok is True, download the file
        if exist_ok or (not osp.isfile(file_path)):
            import urllib
            urllib.request.urlretrieve(url, file_path)
    
        return file_path

    def get_page_soup(self, page_url_or_filepath, verbose=True):
        """
        Gets the BeautifulSoup soup object for a given page URL or filepath.

        Args:
            page_url_or_filepath (str): The URL or filepath of the page to get the soup object for.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            BeautifulSoup: The BeautifulSoup soup object for the given page.
        """

        # Check if the page URL or filepath is a URL
        match_obj = self.url_regex.search(page_url_or_filepath)
        if match_obj:

            # If the page URL or filepath is a URL, open it using urllib.request.urlopen()
            with urllib.request.urlopen(page_url_or_filepath) as response: page_html = response.read()

        else:

            # If the page URL or filepath is not a URL, open it using open()
            with open(page_url_or_filepath, 'r', encoding='utf-8') as f: page_html = f.read()

        # Parse the page HTML using BeautifulSoup
        page_soup = bs(page_html, 'html.parser')

        # If verbose output is enabled, print the page URL or filepath
        if verbose: print(f'Getting soup object for: {page_url_or_filepath}')

        # Return the page soup object
        return page_soup
    
    def get_wiki_tables(self, tables_url_or_filepath, verbose=True):
        """
        Gets a list of DataFrames from Wikipedia tables.

        Args:
            tables_url_or_filepath: The URL or filepath to the Wikipedia page containing the tables.
            verbose: Whether to print verbose output.

        Returns:
            A list of DataFrames containing the data from the Wikipedia tables.

        Raises:
            Exception: If there is an error getting the Wikipedia page or the tables from the page.
        """
        table_dfs_list = []
        try:

            # Get the BeautifulSoup object for the Wikipedia page
            page_soup = self.get_page_soup(tables_url_or_filepath, verbose=verbose)

            # Find all the tables on the Wikipedia page
            table_soups_list = page_soup.find_all('table', attrs={'class': 'wikitable'})

            # Recursively get the DataFrames for all the tables on the Wikipedia page
            table_dfs_list = []
            for table_soup in table_soups_list: table_dfs_list += self.get_page_tables(str(table_soup), verbose=False)

            # If verbose is True, print a sorted list of the tables by their number of rows and columns
            if verbose: print(sorted([(i, df.shape) for (i, df) in enumerate(table_dfs_list)], key=lambda x: x[1][0]*x[1][1], reverse=True))

        except Exception as e:

            # If there is an error, print the error message
            if verbose: print(str(e).strip())

            # Recursively get the DataFrames for the tables on the Wikipedia page again, but with verbose=False
            table_dfs_list = self.get_page_tables(tables_url_or_filepath, verbose=False)

        # Return the list of DataFrames
        return table_dfs_list

    def get_page_tables(self, tables_url_or_filepath, verbose=True):
        '''
        import sys
        sys.path.insert(1, '../py')
        from notebook_utils import NotebookUtilities
        import os
        nu = NotebookUtilities(data_folder_path=os.path.abspath('../data'))
        tables_url = 'https://en.wikipedia.org/wiki/Provinces_of_Afghanistan'
        page_tables_list = nu.get_page_tables(tables_url)
        '''
        if self.url_regex.fullmatch(tables_url_or_filepath) or self.filepath_regex.fullmatch(tables_url_or_filepath):
            tables_df_list = pd.read_html(tables_url_or_filepath)
        else:
            f = io.StringIO(tables_url_or_filepath)
            tables_df_list = pd.read_html(f)
        if verbose:
            print(sorted([(i, df.shape) for (i, df) in enumerate(tables_df_list)],
                         key=lambda x: x[1][0]*x[1][1], reverse=True))

        return tables_df_list
    
    ### Pandas Functions ###
    
    def get_row_dictionary(self, value_obj, row_dict={}, key_prefix=''):
        '''
        This function takes a value_obj (either a dictionary, list or scalar value) and creates a flattened dictionary from it, where
        keys are made up of the keys/indices of nested dictionaries and lists. The keys are constructed with a key_prefix
        (which is updated as the function traverses the value_obj) to ensure uniqueness. The flattened dictionary is stored in the
        row_dict argument, which is updated at each step of the function.

        Parameters
        ----------
        value_obj : dict, list, scalar value
            The object to be flattened into a dictionary.
        row_dict : dict, optional
            The dictionary to store the flattened object.
        key_prefix : str, optional
            The prefix for constructing the keys in the row_dict.

        Returns
        ----------
        row_dict : dict
            The flattened dictionary representation of the value_obj.
        '''
        
        # Check if the value is a dictionary
        if type(value_obj) == dict:
            
            # Iterate through the dictionary 
            for k, v, in value_obj.items():
                
                # Recursively call get_row_dictionary() with the dictionary key as part of the prefix
                row_dict = get_row_dictionary(
                    v, row_dict=row_dict, key_prefix=f'{key_prefix}_{k}'
                )
                
        # Check if the value is a list
        elif type(value_obj) == list:
            
            # Get the minimum number of digits in the list length
            list_length = len(value_obj)
            digits_count = min(len(str(list_length)), 2)
            
            # Iterate through the list
            for i, v in enumerate(value_obj):
                
                # Add leading zeros to the index
                if (i == 0) and (list_length == 1):
                    i = ''
                else:
                    i = str(i).zfill(digits_count)
                
                # Recursively call get_row_dictionary() with the list index as part of the prefix
                row_dict = get_row_dictionary(
                    v, row_dict=row_dict, key_prefix=f'{key_prefix}{i}'
                )
                
        # If value is neither a dictionary nor a list
        else:
            
            # Add the value to the row dictionary
            if key_prefix.startswith('_') and (key_prefix[1:] not in row_dict):
                key_prefix = key_prefix[1:]
            row_dict[key_prefix] = value_obj
        
        return row_dict
    
    def get_column_descriptions(self, df, column_list=None, verbose=False):
        
        if column_list is None:
            column_list = df.columns
        g = df.columns.to_series().groupby(df.dtypes).groups
        rows_list = []
        for dtype, dtype_column_list in g.items():
            for column_name in dtype_column_list:
                if column_name in column_list:
                    mask_series = df[column_name].isnull()
                    
                    # Get input row in dictionary format; key = col_name
                    row_dict = {}
                    row_dict['column_name'] = column_name
                    row_dict['dtype'] = str(dtype)
                    row_dict['count_blanks'] = df[column_name].isnull().sum()
                    
                    # Count how many unique numbers there are
                    try:
                        row_dict['count_uniques'] = len(df[column_name].unique())
                    except Exception:
                        row_dict['count_uniques'] = math.nan
                    
                    # Count how many zeroes the column has
                    try:
                        row_dict['count_zeroes'] = int((df[column_name] == 0).sum())
                    except Exception:
                        row_dict['count_zeroes'] = math.nan
                    
                    # Check to see if the column has any dates
                    date_series = pd.to_datetime(df[column_name], errors='coerce')
                    null_series = date_series[~date_series.notnull()]
                    row_dict['has_dates'] = (null_series.shape[0] < date_series.shape[0])
                    
                    # Show the minimum value in the column
                    try:
                        row_dict['min_value'] = df[~mask_series][column_name].min()
                    except Exception:
                        row_dict['min_value'] = math.nan
                    
                    # Show the maximum value in the column
                    try:
                        row_dict['max_value'] = df[~mask_series][column_name].max()
                    except Exception:
                        row_dict['max_value'] = math.nan
                    
                    # Show whether the column contains only integers
                    try:
                        row_dict['only_integers'] = (df[column_name].apply(lambda x: float(x).is_integer())).all()
                    except Exception:
                        row_dict['only_integers'] = float('nan')
    
                    rows_list.append(row_dict)
    
        columns_list = ['column_name', 'dtype', 'count_blanks', 'count_uniques', 'count_zeroes', 'has_dates',
                        'min_value', 'max_value', 'only_integers']
        blank_ranking_df = pd.DataFrame(rows_list, columns=columns_list)
        
        return(blank_ranking_df)
    
    def get_session_groupby(self, frvrs_logs_df=None, mask_series=None, extra_column=None):
        if frvrs_logs_df is None: frvrs_logs_df = self.load_object('frvrs_logs_df')
        if (mask_series is None) and (extra_column is None):
            gb = frvrs_logs_df.sort_values(['elapsed_time']).groupby(['session_uuid'])
        elif (mask_series is None) and (extra_column is not None):
            gb = frvrs_logs_df.sort_values(['elapsed_time']).groupby(['session_uuid', extra_column])
        elif (mask_series is not None) and (extra_column is None):
            gb = frvrs_logs_df[mask_series].sort_values(['elapsed_time']).groupby(['session_uuid'])
        elif (mask_series is not None) and (extra_column is not None):
            gb = frvrs_logs_df[mask_series].sort_values(['elapsed_time']).groupby(['session_uuid', extra_column])
    
        return gb
    
    def get_inf_nan_mask(self, x_list, y_list):
        """
        Returns a mask indicating which elements of x_list and y_list are not inf or nan.
        
        Args:
        x_list: A list of numbers.
        y_list: A list of numbers.
        
        Returns:
        A numpy array of booleans, where True indicates that the corresponding element
        of x_list and y_list is not inf or nan.
        """
        
        import numpy as np
        
        # Check if the input lists are empty.
        if not x_list or not y_list: return np.array([], dtype=bool)
        
        # Create masks indicating which elements of x_list and y_list are not inf or nan.
        x_mask = np.logical_and(np.logical_not(np.isinf(x_list)), np.logical_not(np.isnan(x_list)))
        y_mask = np.logical_and(np.logical_not(np.isinf(y_list)), np.logical_not(np.isnan(y_list)))
        
        # Return a mask indicating which elements of both x_list and y_list are not inf or nan.
        return np.logical_and(x_mask, y_mask)

    def get_statistics(self, describable_df, columns_list):
        df = describable_df.describe().rename(index={'std': 'SD'})
        
        if ('mode' not in df.index):
            
            # Create the mode row dictionary
            row_dict = {cn: describable_df[cn].mode().tolist()[0] for cn in columns_list}
            
            # Convert the row dictionary to a data frame to match the df structure
            row_df = pd.DataFrame([row_dict], index=['mode'])
            
            # Append the row data frame to the df data frame
            df = pd.concat([df, row_df], axis='index', ignore_index=False)
        
        if ('median' not in df.index):
            
            # Create the median row dictionary
            row_dict = {cn: describable_df[cn].median() for cn in columns_list}
            
            # Convert the row dictionary to a data frame to match the df structure
            row_df = pd.DataFrame([row_dict], index=['median'])
            
            # Append the row data frame to the df data frame
            df = pd.concat([df, row_df], axis='index', ignore_index=False)
        
        index_list = ['mean', 'mode', 'median', 'SD', 'min', '25%', '50%', '75%', 'max']
        mask_series = df.index.isin(index_list)
        
        return df[mask_series].reindex(index_list)
    
    def show_time_statistics(self, describable_df, columns_list):
        df = self.get_statistics(describable_df, columns_list).applymap(lambda x: self.format_timedelta(timedelta(milliseconds=int(x))), na_action='ignore').T
        df.SD = df.SD.map(lambda x: '±' + str(x))
        display(df)
    
    def modalize_columns(self, frvrs_logs_df, columns_list, new_column_):
        columns_list= [
            'injury_record_id', 'injury_treated_id'
        ]
        mask_series = (frvrs_logs_df[columns_list].apply(pd.Series.nunique, axis='columns') == 1)
        frvrs_logs_df.loc[~mask_series, 'injury_id'] = np.nan
        def f(srs):
            cn = srs.first_valid_index()
            
            return srs[cn]
        frvrs_logs_df.loc[mask_series, 'injury_id'] = frvrs_logs_df[mask_series][columns_list].apply(f, axis='columns')
    
        return frvrs_logs_df
    
    ### LLM Functions ###
    
    def download_lora_model(self, verbose=False):
        if not osp.exists(self.lora_path):
            download_url = f'https://the-eye.eu/public/AI/models/downloadnomic-ai/gpt4all/{osp.basename(self.lora_path)}'
            
            # Download the file from the URL using requests
            import requests
            response = requests.get(download_url)
            
            # Create the necessary directories if they don't exist
            os.makedirs(osp.dirname(self.lora_path), exist_ok=True)

            # Save the downloaded file to disk
            with open(self.lora_path, 'wb') as f:
                f.write(response.content)
    
    def convert_lora_model_to_gpt4all(self, verbose=False):
        if not osp.exists(self.gpt4all_model_path):
            converter_path = osp.abspath(osp.join(self.scripts_folder, 'pyllamacpp-convert-gpt4all.exe'))
            llama_file = osp.abspath(osp.join(llama_folder, 'tokenizer.model'))
            command_str = f'{converter_path} {self.lora_path} {llama_file} {self.gpt4all_model_path}'
            if verbose: print(command_str, flush=True)
            output_str = subprocess.check_output(command_str.split(' '))
            if verbose:
                for line_str in output_str.splitlines(): print(line_str.decode(), flush=True)
    
    ### 3D Point Functions ###
    
    def get_coordinates(self, second_point, first_point=None):
        """
        Get the coordinates of two 3D points.
    
        Parameters
        ----------
        second_point : str
            The coordinates of the second point as a string.
        first_point : str, optional
            The coordinates of the first point as a string. If not provided, the default values (0, 0, 0) will be used.
    
        Returns
        -------
        tuple of float
            The coordinates of the two points.
    
        """
        if first_point is None:
            x1 = 0.0  # The x-coordinate of the first point
            y1 = 0.0  # The y-coordinate of the first point
            z1 = 0.0  # The z-coordinate of the first point
        else:
            location_tuple = eval(first_point)
            x1 = location_tuple[0]  # The x-coordinate of the first point
            y1 = location_tuple[1]  # The y-coordinate of the first point
            z1 = location_tuple[2]  # The z-coordinate of the first point
        location_tuple = eval(second_point)
        x2 = location_tuple[0]  # The x-coordinate of the second point
        y2 = location_tuple[1]  # The y-coordinate of the second point
        z2 = location_tuple[2]  # The z-coordinate of the second point
    
        return x1, x2, y1, y2, z1, z2
    
    def get_euclidean_distance(self, second_point, first_point=None):
        """
        Calculates the Euclidean distance between two 3D points.
    
        Returns:
            float: The Euclidean distance between the two points.
        """
        x1, x2, y1, y2, z1, z2 = self.get_coordinates(second_point, first_point=first_point)
    
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    
    def get_absolute_position(self, second_point, first_point=None):
        """
        Calculates the absolute position of a point relative to another point.
    
        Parameters
        ----------
        second_point : tuple
            The coordinates of the second point.
        first_point : tuple, optional
            The coordinates of the first point. If not specified,
            the origin is retrieved from get_coordinates.
    
        Returns
        -------
        tuple
            The absolute coordinates of the second point.
        """
        x1, x2, y1, y2, z1, z2 = self.get_coordinates(second_point, first_point=first_point)
    
        return (round(x1 + x2, 1), round(y1 + y2, 1), round(z1 + z2, 1))
    
    ### Sub-sampling Functions ###
    
    def get_minority_combinations(self, sample_df, groupby_columns):
        """
        Get the minority combinations of a DataFrame.
        
        Args:
            sample_df: A Pandas DataFrame.
            groupby_columns: A list of column names to group by.
        
        Returns:
            A Pandas DataFrame containing a single sample row of each of the four smallest groups.
        """
        df = pd.DataFrame([], columns=sample_df.columns)
        for bool_tuple in sample_df.groupby(groupby_columns).size().sort_values().index.tolist()[:4]:
            
            # Filter the name in the column to the corresponding value of the tuple
            mask_series = True
            for cn, cv in zip(groupby_columns, bool_tuple): mask_series &= (sample_df[cn] == cv)
            
            # Append a random single record from the filtered data frame
            df = pd.concat([df, sample_df[mask_series].sample(1)], axis='index')
        
        return df
    
    def get_random_subdictionary(self, super_dict, n=5):
        keys = list(super_dict.keys())
        import random
        random_keys = random.sample(keys, n)
        sub_dict = {}
        for key in random_keys: sub_dict[key] = super_dict[key]
            
        return sub_dict
    
    ### Plotting Functions ###
    
    def visualize_player_movement(self, session_mask, title=None, save_only=False, frvrs_logs_df=None, verbose=False):
        """
        Visualizes the player movement for the given session mask in a 2D plot.
    
        Args:
            session_mask (pandas.Series): A boolean mask indicating which rows of the frvrs_logs_df DataFrame belong to the current session.
            title (str, optional): The title of the plot, if saving.
            save_only (bool, optional): Whether to only save the plot to a PNG file and not display it.
            frvrs_logs_df (pandas.DataFrame, optional): A DataFrame containing the FRVRS logs.
                If `None`, the DataFrame will be loaded from the disk.
            verbose (bool, optional): Whether to print verbose output.
    
        Returns:
            None: The function either displays the plot or saves it to a file.
    
        Note:
        - This function visualizes player movement based on data in the DataFrame `frvrs_logs_df`.
        - It can display player positions, locations, and teleportations.
        - Use `session_mask` to filter the data for a specific session.
        - Set `save_only` to True to save the plot as a PNG file with the specified `title`.
        - Set `verbose` to True to enable verbose printing.
        """
    
        # Load the FRVRS logs if not provided.
        if frvrs_logs_df is None: frvrs_logs_df = self.load_object('frvrs_logs_df')
    
        # Check if saving to a file is requested
        if save_only:
            assert title is not None, "To save, you need a title"
            file_path = os.path.join(self.saves_folder, 'png', re.sub(r'\W+', '_', str(title)).strip('_').lower() + '.png')
            filter = not os.path.exists(file_path)
        else: filter = True
    
        # If the filter is True, then visualize the player movement
        if filter:
            import textwrap
    
            # Turn off interactive plotting if saving to a file
            if save_only: plt.ioff()
    
            # Create a figure and add a subplot
            fig, ax = plt.subplots(figsize=(18, 9))
            
            # Show the positions of patients recorded and engaged at our time group and session UUID
            color_cycler = self.get_color_cycler(frvrs_logs_df[session_mask].groupby('patient_id').size().shape[0])
            location_cns_list = [
                'patient_demoted_position', 'patient_engaged_position', 'patient_record_position', 'player_gaze_location'
            ]
            for (patient_id, df1), face_color_dict in zip(frvrs_logs_df[session_mask].sort_values(['elapsed_time']).groupby([
                'patient_id'
            ]), color_cycler()):
                x_dim = []; y_dim = []; z_dim = []
                for location_cn in location_cns_list:
                    mask_series = df1[location_cn].isnull()
                    srs = df1[~mask_series][location_cn].map(lambda x: eval(x))
                    x_dim.extend(srs.map(lambda x: x[0]).values)
                    y_dim.extend(srs.map(lambda x: x[1]).values)
                    z_dim.extend(srs.map(lambda x: x[2]).values)
    
                face_color = face_color_dict['color']
                
                # Pick from among the sort columns whichever value is not null and use that in the label
                columns_list = ['patient_demoted_sort', 'patient_engaged_sort', 'patient_record_sort']
                srs = df1[columns_list].apply(pd.Series.notnull, axis=1).sum()
                mask_series = (srs > 0)
                cn = srs[mask_series].index.tolist()[0]
                if (type(patient_id) == tuple): patient_id = patient_id[0]
    
                # Generate a wrapped label
                label = patient_id.replace(' Root', ' (') + df1[cn].dropna().tolist()[-1] + ')'
                label = '\n'.join(textwrap.wrap(label, width=20))
    
                # Plot the ball and chain
                ax.plot(x_dim, z_dim, color=face_color, alpha=1.0, label=label)
                ax.scatter(x_dim, z_dim, color=face_color, alpha=1.0)
                
                # Get the first of the movement coordinates and label the patient there
                coords_set = set()
                for x, z in zip(x_dim, z_dim):
                    coords_tuple = (x, z)
                    coords_set.add(coords_tuple)
    
                for coords_tuple in coords_set:
                    x, y = coords_tuple
                    plt.annotate(label, (x, y), textcoords='offset points', xytext=(0, -8), ha='center', va='center')
                    break
            
            # Visualize locations
            x_dim = []; z_dim = []; label = ''
            mask_series = (frvrs_logs_df.action_type == 'PLAYER_LOCATION') & session_mask
            locations_df = frvrs_logs_df[mask_series]
            if locations_df.shape[0]:
                label = 'locations'
                locations_df = locations_df.sort_values(['elapsed_time'])
                for player_location_location in locations_df.player_location_location:
                    player_location_location = eval(player_location_location)
                    x_dim.append(player_location_location[0])
                    z_dim.append(player_location_location[2])
                if verbose: print(x_dim, z_dim)
            
            # Chain or, maybe, ball
            if (len(x_dim) < 2): ax.scatter(x_dim, z_dim, alpha=1.0, label=label)
            else: ax.plot(x_dim, z_dim, alpha=1.0, label=label)
            
            # Visualize teleportations
            x_dim = []; z_dim = []; label = ''
            mask_series = (frvrs_logs_df.action_type == 'TELEPORT') & session_mask
            teleports_df = frvrs_logs_df[mask_series]
            if teleports_df.shape[0]:
                label = 'teleportations'
                teleports_df = teleports_df.sort_values(['elapsed_time'])
                for teleport_location in teleports_df.teleport_location:
                    teleport_location = eval(teleport_location)
                    x_dim.append(teleport_location[0])
                    z_dim.append(teleport_location[2])
                if verbose: print(x_dim, z_dim)
            
            # Chain or, maybe, ball
            if (len(x_dim) < 2): ax.scatter(x_dim, z_dim, alpha=1.0, label=label)
            else: ax.plot(x_dim, z_dim, alpha=1.0, label=label)
            
            # Add labels
            ax.set_xlabel('X')
            ax.set_ylabel('Z')

            # Move the left and right borders to make room for the legend
            left_lim, right_lim = ax.get_xlim()
            xlim_tuple = ax.set_xlim(left_lim-1.5, right_lim+1.5)
            ax.legend(loc='best')
            
            # Add title, if any
            if title is not None: ax.set_title(title)
            
            # Save the figure to PNG
            if save_only:
                plt.savefig(file_path, bbox_inches='tight')
                plt.ion()
    
    def visualize_extreme_player_movement(
        self, df, sorting_column, mask_series=None, is_ascending=True, humanize_type='precisedelta',
        title_str='slowest action to control time', frvrs_logs_df=None, verbose=False
    ):
        '''
        Get time group with some edge case and visualize the player movement there
        '''
        
        if mask_series is None: mask_series = [True] * df.shape[0]
        df1 = df[mask_series].sort_values(
            [sorting_column], ascending=[is_ascending]
        ).head(1)
        if df1.shape[0]:
            session_uuid = df1.session_uuid.squeeze()
            time_group = df1.time_group.squeeze()
            if frvrs_logs_df is None: frvrs_logs_df = self.load_object('frvrs_logs_df')
            base_mask_series = (frvrs_logs_df.session_uuid == session_uuid) & (frvrs_logs_df.time_group == time_group)
            
            title = f'Location Map for UUID {session_uuid} ({humanize.ordinal(time_group+1)} Scene)'
            title += f' showing trainee with the {title_str} ('
            if is_ascending:
                column_value = df1[sorting_column].min()
            else:
                column_value = df1[sorting_column].max()
            if verbose: display(column_value)
            if (humanize_type == 'precisedelta'):
                title += humanize.precisedelta(timedelta(milliseconds=column_value)) + ')'
            elif (humanize_type == 'percentage'):
                title += str(100 * column_value) + '%)'
            elif (humanize_type == 'intword'):
                title += humanize.intword(column_value) + ')'
            self.visualize_player_movement(base_mask_series, title=title, frvrs_logs_df=frvrs_logs_df)
    
    def get_color_cycler(self, n):
        """
        color_cycler = self.get_color_cycler(len(possible_cause_list))
        for possible_cause, face_color_dict in zip(possible_cause_list, color_cycler()):
            face_color = face_color_dict['color']
        """
        color_cycler = None
        from cycler import cycler
        import numpy as np
        if n < 9:
            color_cycler = cycler('color', plt.cm.Accent(np.linspace(0, 1, n)))
        elif n < 11:
            color_cycler = cycler('color', plt.cm.tab10(np.linspace(0, 1, n)))
        elif n < 13:
            color_cycler = cycler('color', plt.cm.Paired(np.linspace(0, 1, n)))
        else:
            color_cycler = cycler('color', plt.cm.tab20(np.linspace(0, 1, n)))
        
        return color_cycler
    
    def first_order_linear_scatterplot(self, df, xname, yname,
                                       xlabel_str='Overall Capitalism (explanatory variable)',
                                       ylabel_str='World Bank Gini % (response variable)',
                                       x_adj='capitalist', y_adj='unequal',
                                       title='"Wealth inequality is huge in the capitalist societies"',
                                       idx_reference='United States', annot_reference='most evil',
                                       aspect_ratio=None,
                                       least_x_xytext=(40, -10), most_x_xytext=(-150, 55),
                                       least_y_xytext=(-200, -10), most_y_xytext=(45, 0),
                                       reference_xytext=(-75, 25), color_list=None):
        """
        Create a first-order (linear) scatter plot assuming the data frame
        has an index labeled with strings.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The data frame to be plotted.
        xname : str
            The name of the x-axis variable.
        yname : str
            The name of the y-axis variable.
        xlabel_str : str, optional
            The label for the x-axis. Defaults to 'Overall Capitalism (explanatory variable)'.
        ylabel_str : str, optional
            The label for the y-axis. Defaults to 'World Bank Gini % (response variable)'.
        x_adj : str, optional
            The adjective to use for the x-axis variable in the annotations. Default is 'capitalist'.
        y_adj : str, optional
            The adjective to use for the y-axis variable in the annotations. Default is 'unequal'.
        title : str, optional
            The title of the plot. Defaults to '"Wealth inequality is huge in the capitalist societies"'.
        idx_reference : str, optional
            The index of the data point to be used as the reference point for the annotations. Default is 'United States'.
        annot_reference : str, optional
            The reference text to be used for the annotation of the reference point. Default is 'most evil'.
        aspect_ratio : float, optional
            The aspect ratio of the plot. Default is the Facebook aspect ratio (1.91).
        least_x_xytext : tuple[float, float], optional
            The xytext position for the annotation of the least x-value data point. Default is (40, -10).
        most_x_xytext : tuple[float, float], optional
            The xytext position for the annotation of the most x-value data point. Default is (-150, 55).
        least_y_xytext : tuple[float, float], optional
            The xytext position for the annotation of the least y-value data point. Default is (-200, -10).
        most_y_xytext : tuple[float, float], optional
            The xytext position for the annotation of the most y-value data point. Default is (45, 0).
        reference_xytext : tuple[float, float], optional
            The xytext position for the annotation of the reference point. Default is (-75, 25).
        color_list : list[str], optional
            The list of colors to be used for the scatter plot. Default is None, which will use a default color scheme.
    
        Returns
        -------
        figure: matplotlib.figure.Figure
            The figure object for the generated scatter plot.
        """
    
        if aspect_ratio is None: aspect_ratio = self.facebook_aspect_ratio
        fig_width = 18
        fig_height = fig_width / aspect_ratio
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111, autoscale_on=True)
        line_kws = dict(color='k', zorder=1, alpha=.25)
    
        if color_list is None: scatter_kws = dict(s=30, lw=.5, edgecolors='k', zorder=2)
        else: scatter_kws = dict(s=30, lw=.5, edgecolors='k', zorder=2, color=color_list)
    
        import seaborn as sns
        merge_axes_subplot = sns.regplot(x=xname, y=yname, scatter=True, data=df, ax=ax,
                                         scatter_kws=scatter_kws, line_kws=line_kws)
    
        if not xlabel_str.endswith(' (explanatory variable)'): xlabel_str = f'{xlabel_str} (explanatory variable)'
        xlabel_text = plt.xlabel(xlabel_str)
    
        if not ylabel_str.endswith(' (response variable)'): ylabel_str = f'{ylabel_str} (response variable)'
        ylabel_text = plt.ylabel(ylabel_str)
    
        kwargs = dict(textcoords='offset points', ha='left', va='bottom',
                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        xdata = df[xname].values
        least_x = xdata.min()
        most_x = xdata.max()
        
        ydata = df[yname].values
        most_y = ydata.max()
        least_y = ydata.min()
        
        least_x_tried = most_x_tried = least_y_tried = most_y_tried = False
    
        for label, x, y in zip(df.index, xdata, ydata):
            if (x == least_x) and not least_x_tried:
                annotation = plt.annotate('{} (least {})'.format(label, x_adj),
                                          xy=(x, y), xytext=least_x_xytext, **kwargs)
                least_x_tried = True
            elif (x == most_x) and not most_x_tried:
                annotation = plt.annotate('{} (most {})'.format(label, x_adj),
                                          xy=(x, y), xytext=most_x_xytext, **kwargs)
                most_x_tried = True
            elif (y == least_y) and not least_y_tried:
                annotation = plt.annotate('{} (least {})'.format(label, y_adj),
                                          xy=(x, y), xytext=least_y_xytext, **kwargs)
                least_y_tried = True
            elif (y == most_y) and not most_y_tried:
                annotation = plt.annotate('{} (most {})'.format(label, y_adj),
                                          xy=(x, y), xytext=most_y_xytext, **kwargs)
                most_y_tried = True
            elif (label == idx_reference):
                annotation = plt.annotate('{} ({})'.format(label, annot_reference),
                                          xy=(x, y), xytext=reference_xytext, **kwargs)
    
        title_obj = fig.suptitle(t=title, x=0.5, y=0.91)
        
        # Get r squared value
        inf_nan_mask = self.get_inf_nan_mask(xdata.tolist(), ydata.tolist())
        from scipy.stats import pearsonr
        pearsonr_tuple = pearsonr(xdata[inf_nan_mask], ydata[inf_nan_mask])
        pearson_r = pearsonr_tuple[0]
        pearsonr_statement = str('%.2f' % pearson_r)
        coefficient_of_determination_statement = str('%.2f' % pearson_r**2)
        p_value = pearsonr_tuple[1]
    
        if p_value < 0.0001: pvalue_statement = '<0.0001'
        else: pvalue_statement = '=' + str('%.4f' % p_value)
    
        s_str = r'$r^2=' + coefficient_of_determination_statement + ',\ p' + pvalue_statement + '$'
        text_tuple = ax.text(0.75, 0.9, s_str, alpha=0.5, transform=ax.transAxes, fontsize='x-large')
        
        return fig
    
    def plot_line_with_error_bars(self, df, xname, xlabel, xtick_text_fn, yname, ylabel, ytick_text_fn, title):
        
        # Drop rows with NaN values, group by patient ranking, and calculate mean and standard deviation
        groupby_list = [xname]
        columns_list = [xname, yname]
        aggs_list = ['mean', 'std']
        df = df.dropna(subset=columns_list).groupby(groupby_list)[yname].agg(aggs_list).reset_index()
        
        # Create the figure and subplot
        fig, ax = plt.subplots(figsize=(18, 9))
        
        # Plot the line with error bars
        ax.errorbar(
            x=df[xname],
            y=df['mean'],
            yerr=df['std'],
            label=ylabel,
            fmt='-o',  # Line style with markers
        )
        
        # Set plot title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Humanize x tick labels
        xticklabels_list = []
        for text_obj in ax.get_xticklabels():
            text_obj.set_text(xtick_text_fn(text_obj))
            xticklabels_list.append(text_obj)
        ax.set_xticklabels(xticklabels_list)
        
        # Humanize y tick labels
        yticklabels_list = []
        for text_obj in ax.get_yticklabels():
            text_obj.set_text(ytick_text_fn(text_obj))
            yticklabels_list.append(text_obj)
        ax.set_yticklabels(yticklabels_list);
    
    def plot_histogram(self, df, xname, xlabel, xtick_text_fn, title, ylabel=None, xticks_are_temporal=False, ax=None, color=None, bins=100):
        """
        Plots a histogram of a DataFrame column.
        
        Args:
            df: A Pandas DataFrame.
            xname: The name of the column to plot the histogram of.
            xlabel: The label for the x-axis.
            xtick_text_fn: A function that takes a text object as input and returns a new
            text object to be used as the tick label.
            title: The title of the plot.
            ylabel: The label for the y-axis.
            ax: A matplotlib axis object. If None, a new figure and axis will be created.
        
        Returns:
            A matplotlib axis object.
        """
        
        # Create the figure and subplot
        if ax is None: fig, ax = plt.subplots(figsize=(18, 9))
        
        # Plot the histogram with centered bars
        df[xname].hist(ax=ax, bins=bins, align='mid', edgecolor='black', color=color)
        
        # Set the grid, title and labels
        plt.grid(False)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ylabel is None: ylabel = 'Count of Instances in Bin'
        ax.set_ylabel(ylabel)

        if xticks_are_temporal:
        
            # Set the minor x-axis tick labels to every 30 seconds
            thirty_seconds = 1_000 * 30
            minor_ticks = np.arange(0, df[xname].max() + thirty_seconds, thirty_seconds)
            ax.set_xticks(minor_ticks, minor=True)
            
            # Set the major x-axis tick labels to every 5 minutes
            if (len(minor_ticks) > 84):
                five_minutes = 1_000 * 60 * 5
                major_ticks = np.arange(0, df[xname].max() + five_minutes, five_minutes)
                ax.set_xticks(major_ticks)
            
            # Set the major x-axis tick labels to every 60 seconds
            else:
                sixty_seconds = 1_000 * 60
                major_ticks = np.arange(0, df[xname].max() + sixty_seconds, sixty_seconds)
                ax.set_xticks(major_ticks)
        
        # Humanize x tick labels
        xticklabels_list = []
        for text_obj in ax.get_xticklabels():
            
            # Call the xtick text function to convert numerical values into minutes and seconds format
            text_obj.set_text(xtick_text_fn(text_obj))
            
            xticklabels_list.append(text_obj)
        # print(len(xticklabels_list))
        if (len(xticklabels_list) > 17): ax.set_xticklabels(xticklabels_list, rotation=90)
        else: ax.set_xticklabels(xticklabels_list)
        
        # Humanize y tick labels
        yticklabels_list = []
        for text_obj in ax.get_yticklabels():
            text_obj.set_text(humanize.intword(int(text_obj.get_position()[1])))
            yticklabels_list.append(text_obj)
        ax.set_yticklabels(yticklabels_list)
        
        return ax
    
    def plot_grouped_box_and_whiskers(self, transformable_df, x_column_name, y_column_name, x_label, y_label, transformer_name='min', is_y_temporal=True):    
        import seaborn as sns
        
        # Get the transformed data frame
        if transformer_name is None: transformed_df = transformable_df
        else:
            groupby_columns = ['session_uuid', 'scene_index']
            transformed_df = transformable_df.groupby(groupby_columns).filter(
                lambda df: not df[y_column_name].isnull().any()
            ).groupby(groupby_columns).transform(transformer_name).reset_index(drop=False).sort_values(y_column_name)
        
        # Create a figure and subplots
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        
        # Create a box plot of the y column grouped by the x column
        sns.boxplot(
            x=x_column_name,
            y=y_column_name,
            showmeans=True,
            data=transformed_df,
            ax=ax
        )
        
        # Rotate the x-axis labels to prevent overlapping
        plt.xticks(rotation=45)
        
        # Label the x- and y-axis
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Humanize y tick labels
        if is_y_temporal:
            yticklabels_list = []
            for text_obj in ax.get_yticklabels():
                text_obj.set_text(
                    humanize.precisedelta(timedelta(milliseconds=text_obj.get_position()[1])).replace(', ', ',\n').replace(' and ', ' and\n')
                )
                yticklabels_list.append(text_obj)
            ax.set_yticklabels(yticklabels_list);
        
        plt.show()