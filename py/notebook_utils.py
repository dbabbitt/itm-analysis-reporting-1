
#!/usr/bin/env python
# Utility Functions to run Jupyter notebooks.
# Dave Babbitt <dave.babbitt@gmail.com>
# Author: Dave Babbitt, Data Scientist
# coding: utf-8

# Soli Deo gloria

from difflib import SequenceMatcher
from typing import List, Optional
try: import dill as pickle
except:
    try: import pickle5 as pickle
    except: import pickle
import pandas as pd
import os
import os.path as osp
import sys
import subprocess
import csv

import warnings
warnings.filterwarnings("ignore")

class NotebookUtilities(object):
    """
    This class implements the core of the utility
    functions needed to install and run GPTs to 
    common to running Jupyter notebooks.
    
    Examples
    --------
    
    import sys
    import os
    sys.path.insert(1, osp.abspath('../py'))
    from notebook_utils import NotebookUtilities
    
    tu = NotebookUtilities(
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
            self.cache_folder, 'models--LLukas22--gpt4all-lora-quantized-ggjt', 'snapshots', '2e7367a8557085b8267e1f3b27c209e272b8fe6c', 'ggjt-model.bin'
        ))
        
        # Ensure the Scripts folder is in PATH
        self.anaconda_folder = osp.dirname(sys.executable)
        self.scripts_folder = osp.join(self.anaconda_folder, 'Scripts')
        if self.scripts_folder not in sys.path:
            sys.path.insert(1, self.scripts_folder)

        # Handy list of the different types of encodings
        self.encoding_type = ['latin1', 'iso8859-1', 'utf-8'][2]
    
    def similar(self, a: str, b: str) -> float:
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

        return SequenceMatcher(None, str(a), str(b)).ratio()
    
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

    def load_dataframes(self, **kwargs):
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
    
    def save_dataframes(self, include_index=False, verbose=True, **kwargs):
        for frame_name in kwargs:
            if isinstance(kwargs[frame_name], pd.DataFrame):
                csv_path = osp.join(self.saves_csv_folder, '{}.csv'.format(frame_name))
                if verbose: print('Saving to {}'.format(osp.abspath(csv_path)), flush=True)
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

        if modules_list is None:
            self.modules_list = [o.decode().split(' ')[0] for o in subprocess.check_output(f'{self.pip_command_str} list'.split(' ')).splitlines()[2:]]
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
    
    def get_filename_from_url(self, url, verbose=False):
        import urllib
        file_name = urllib.parse.urlparse(url).path.split('/')[-1]
        
        return file_name
    
    def download_file(self, url, download_dir=None, exist_ok=False, verbose=False):
        '''Download a file from the internet'''
        file_name = self.get_filename_from_url(url, verbose=verbose)
        if download_dir is None:
            download_dir = osp.join(self.data_folder, 'downloads')
        os.makedirs(download_dir, exist_ok=True)
        file_path = osp.join(download_dir, file_name)
        if exist_ok or (not osp.isfile(file_path)):
            import urllib
            urllib.request.urlretrieve(url, file_path)
    
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
    
    def conjunctify_nouns(self, noun_list, and_or='and', verbose=False):
        if noun_list is None:
            
            return ''
        if type(noun_list) != list:
            noun_list = list(noun_list)
        if len(noun_list) > 2:
            last_noun_str = noun_list[-1]
            but_last_nouns_str = ', '.join(noun_list[:-1])
            list_str = f', {and_or} '.join([but_last_nouns_str, last_noun_str])
        elif len(noun_list) == 2:
            list_str = f' {and_or} '.join(noun_list)
        elif len(noun_list) == 1:
            list_str = noun_list[0]
        else:
            list_str = ''
        
        return list_str
    
    def get_color_cycler(self, n):
        """
        color_cycler = nu.get_color_cycler(len(possible_cause_list))
        for possible_cause, face_color_dict in zip(possible_cause_list, color_cycler()):
            face_color = face_color_dict['color']
        """
        color_cycler = None
        from cycler import cycler
        import matplotlib.pyplot as plt
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