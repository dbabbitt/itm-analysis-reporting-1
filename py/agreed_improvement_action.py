#!/usr/bin/env python
# Utility Functions to meet the agreed improvement actions.
# Dave Babbitt <dave.babbitt@bigbear.ai>
# Author: Dave Babbitt, Machine Learning Engineer
# coding: utf-8

# To run this in windows, open the containing folder into cmd, and type something like:
# C:\Users\DaveBabbitt\anaconda3\python.exe agreed_improvement_action.py
# To run this in Ubuntu, type something like:
# clear; cd /mnt/c/Users/DaveBabbitt/Downloads; /home/dbabbitt/anaconda3/envs/itm_analysis_reporting/bin/python agreed_improvement_action.py

# agreed_improvement_action.py in its attached .zip file creates everything (including the ANOVA aggregation) in the saves folder
# using everything in the data folder using only one .py file.

IS_DEBUG = True
if IS_DEBUG: print("\nLoad agreed_improvement_action libraries")
from numpy import nan, isnan
from os import listdir as listdir, makedirs as makedirs, path as osp, remove as remove, sep as sep, walk as walk
from pandas import CategoricalDtype, DataFrame, Index, NaT, Series, concat, get_dummies, isna, notnull, read_csv, read_excel, to_datetime, to_numeric
from re import MULTILINE, search, split, sub
from scipy.stats import f_oneway, ttest_ind, kruskal, norm
import csv
import inspect
import itertools
import json
import math
import numpy as np
import re
import statsmodels.api as sm
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')

# Check for presence of 'get_ipython' function (exists in Jupyter)
try:
    get_ipython()
    from IPython.display import display
except NameError:
    display = lambda message: print(message)

# Check if pandas is installed and import relevant functions
try:
    from pandas.core.arrays.numeric import is_integer_dtype, is_float_dtype
    is_integer = lambda srs: is_integer_dtype(srs)
    is_float = lambda srs: is_float_dtype(srs)
except:
    
    # Use numpy functions if this version of pandas is not available
    is_integer = lambda srs: any(map(lambda value: np.issubdtype(type(value), np.integer), srs.tolist()))
    is_float = lambda srs: any(map(lambda value: np.issubdtype(type(value), np.floating), srs.tolist()))

class NotebookUtilities(object):
    """
    This class implements the core of the utility
    functions needed to install and run GPTs and 
    also what is common to running Jupyter notebooks.
    
    Example:
        import sys
        import os.path as osp
        sys.path.insert(1, osp.abspath('../py'))
        from FRVRS import nu
    """
    
    def __init__(self, data_folder_path=None, saves_folder_path=None, verbose=False):
        
        # Create the data folder if it doesn't exist
        if data_folder_path is None: self.data_folder = 'data'
        else: self.data_folder = data_folder_path
        makedirs(self.data_folder, exist_ok=True)
        if verbose: print('data_folder: {}'.format(osp.abspath(self.data_folder)), flush=True)
        
        # Create the saves folder if it doesn't exist
        if saves_folder_path is None: self.saves_folder = 'saves'
        else: self.saves_folder = saves_folder_path
        makedirs(self.saves_folder, exist_ok=True)
        if verbose: print('saves_folder: {}'.format(osp.abspath(self.saves_folder)), flush=True)
        
        # Create the only directories we use
        self.saves_csv_folder = osp.join(self.saves_folder, 'csv'); makedirs(name=self.saves_csv_folder, exist_ok=True)
        self.saves_text_folder = osp.join(self.saves_folder, 'txt'); makedirs(name=self.saves_text_folder, exist_ok=True)

        # Handy list of the different types of encodings
        self.encoding_types_list = ['utf-8', 'latin1', 'iso8859-1']
        self.encoding_type = self.encoding_types_list[0]
    
    ### String Functions ###
    
    
    
    
    ### List Functions ###
    
    
    @staticmethod
    def conjunctify_nouns(noun_list=None, and_or='and', verbose=False):
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
        if not isinstance(noun_list, list): noun_list = list(noun_list)
        
        # If there are more than two nouns in the list, join the last two nouns with `and_or`
        # Otherwise, join all of the nouns with `and_or`
        if (len(noun_list) > 2):
            last_noun_str = noun_list[-1]
            but_last_nouns_str = ', '.join(noun_list[:-1])
            list_str = f', {and_or} '.join([but_last_nouns_str, last_noun_str])
        elif (len(noun_list) == 2): list_str = f' {and_or} '.join(noun_list)
        elif (len(noun_list) == 1): list_str = noun_list[0]
        else: list_str = ''
        
        # Print debug output if requested
        if verbose: print(f'noun_list="{noun_list}", and_or="{and_or}", list_str="{list_str}"')
        
        # Return the conjuncted noun list
        return list_str
    
    
    ### File Functions ###
    
    
    ### Storage Functions ###

    
    def load_csv(self, csv_name=None, folder_path=None):
        """
        Loads a CSV file from the specified folder or the default CSV folder,
        returning the data as a pandas DataFrame.
        
        Parameters:
            csv_name (str, optional): The name of the CSV file (with or without the '.csv' extension).
                If None, loads the most recently modified CSV file in the specified or default folder.
            folder_path (str, optional): The path to the folder containing the CSV file.
                If None, uses the default data_csv_folder specified in the class.
        
        Returns:
            pandas.DataFrame: The data from the CSV file as a pandas DataFrame.
        """
        
        # Set folder path if not provided
        if folder_path is None: csv_folder = self.data_csv_folder
        else: csv_folder = osp.join(folder_path, 'csv')
        
        # Determine the CSV file path based on the provided name or the most recently modified file in the folder
        if csv_name is None:
            
            # If no specific CSV file is named, load the most recently modified CSV file
            csv_path = max([osp.join(csv_folder, f) for f in listdir(csv_folder)], key=osp.getmtime)
        else:
            
            # If a specific CSV file is named, construct the full path to the CSV file
            if csv_name.endswith('.csv'): csv_path = osp.join(csv_folder, csv_name)
            else: csv_path = osp.join(csv_folder, f'{csv_name}.csv')
        
        # Load the CSV file as a pandas DataFrame using the class-specific encoding
        data_frame = read_csv(osp.abspath(csv_path), encoding=self.encoding_type)
        
        return data_frame
    
    
    def save_data_frames(self, include_index=False, verbose=True, **kwargs):
        """
        Saves data frames to CSV files.

        Parameters:
            include_index: Whether to include the index in the CSV files.
            verbose: Whether to print information about the saved files.
            **kwargs: A dictionary of data frames to save. The keys of the dictionary
                      are the names of the CSV files to save the data frames to.

        Returns:
            None
        """

        # Iterate over the data frames in the kwargs dictionary and save them to CSV files
        for frame_name in kwargs:
            if isinstance(kwargs[frame_name], DataFrame):
                
                # Generate the path to the CSV file
                csv_path = osp.join(self.saves_csv_folder, '{}.csv'.format(frame_name))

                # Print a message about the saved file if verbose is True
                if verbose: print('Saving to {}'.format(osp.abspath(csv_path)), flush=True)

                # Save the data frame to a CSV file
                kwargs[frame_name].to_csv(csv_path, sep=',', encoding=self.encoding_type,
                                          index=include_index)
    
    
    ### Module Functions ###
    
    
    ### URL and Soup Functions ###
    
    
    ### Pandas Functions ###
    
    
    @staticmethod
    def get_statistics(describable_df, columns_list, verbose=False):
        """
        Calculates and returns descriptive statistics for a subset of columns in a Pandas DataFrame.
        
        Parameters:
            describable_df (pandas.DataFrame): The DataFrame to calculate descriptive statistics for.
            columns_list (list of str): A list of specific columns to calculate statistics for.
            verbose (bool): If True, display debug information.
        
        Returns:
            pandas.DataFrame: A DataFrame containing the descriptive statistics for the analyzed columns.
                The returned DataFrame includes the mean, mode, median, standard deviation (SD),
                minimum, 25th percentile, 50th percentile (median), 75th percentile, and maximum.
        """
        
        # Calculate basic descriptive statistics for the specified columns
        df = describable_df[columns_list].describe().rename(index={'std': 'SD'})
        
        # If the mode is not already included in the statistics, calculate it
        if ('mode' not in df.index):
            
            # Create the mode row dictionary
            row_dict = {cn: describable_df[cn].mode().iloc[0] for cn in columns_list}
            
            # Convert the row dictionary to a data frame to match the df structure
            row_df = DataFrame([row_dict], index=['mode'])
            
            # Append the row data frame to the df data frame
            df = concat([df, row_df], axis='index', ignore_index=False)
        
        # If the median is not already included in the statistics, calculate it
        if ('median' not in df.index):
            
            # Create the median row dictionary
            row_dict = {cn: describable_df[cn].median() for cn in columns_list}
            
            # Convert the row dictionary to a data frame to match the df structure
            row_df = DataFrame([row_dict], index=['median'])
            
            # Append the row data frame to the df data frame
            df = concat([df, row_df], axis='index', ignore_index=False)
        
        # Define the desired index order for the resulting DataFrame
        index_list = ['mean', 'mode', 'median', 'SD', 'min', '25%', '50%', '75%', 'max']
        
        # Create a boolean mask to select rows with desired index values
        mask_series = df.index.isin(index_list)
        df = df[mask_series].reindex(index_list)
        
        # If verbose is True, print additional information
        if verbose:
            print(f'columns_list: {columns_list}')
            display(describable_df)
            display(df)
        
        # Return the filtered DataFrame containing the selected statistics
        return df
    
    
    @staticmethod
    def modalize_columns(df, columns_list, new_column_name):
        """
        Create a new column in a DataFrame representing the modal value of specified columns.
        
        Parameters:
            df (pandas.DataFrame): The input DataFrame.
            columns_list (list): The list of column names from which to calculate the modal value.
            new_column_name (str): The name of the new column to create.
        
        Returns:
            pandas.DataFrame: The modified DataFrame with the new column representing the modal value.
        """
        
        # Ensure that all columns are in the data frame
        columns_list = list(set(df.columns).intersection(set(columns_list)))
        
        # Create a mask series indicating rows with one unique value across the specified columns
        mask_series = (df[columns_list].apply(Series.nunique, axis='columns') == 1)
        
        # Replace non-unique or missing values with NaN
        df.loc[~mask_series, new_column_name] = nan
        
        # Define a function to extract the first valid value in each row
        f = lambda srs: srs[srs.first_valid_index()]
        
        # For rows with identical values in specified columns, set the new column to the modal value
        df.loc[mask_series, new_column_name] = df[mask_series][columns_list].apply(f, axis='columns')
    
        return df
    
    
    def get_flattened_dictionary(self, value_obj, row_dict={}, key_prefix=''):
        """
        This function takes a value_obj (either a dictionary, list or scalar value) and creates a flattened
        dictionary from it, where keys are made up of the keys/indices of nested dictionaries and lists. The
        keys are constructed with a key_prefix (which is updated as the function traverses the value_obj) to
        ensure uniqueness. The flattened dictionary is stored in the row_dict argument, which is updated at
        each step of the function.
        
        Parameters:
            value_obj (dict, list, scalar value): The object to be flattened into a dictionary.
            row_dict (dict, optional): The dictionary to store the flattened object.
            key_prefix (str, optional): The prefix for constructing the keys in the row_dict.
        
        Returns:
            row_dict (dict): The flattened dictionary representation of the value_obj.
        """
        
        # Check if the value is a dictionary
        if isinstance(value_obj, dict):
            
            # Iterate through the dictionary 
            for k, v, in value_obj.items():
                
                # Recursively call get row dictionary with the dictionary key as part of the prefix
                row_dict = self.get_flattened_dictionary(
                    v, row_dict=row_dict, key_prefix=f'{key_prefix}_{k}'
                )
                
        # Check if the value is a list
        elif isinstance(value_obj, list):
            
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
                
                # Recursively call get row dictionary with the list index as part of the prefix
                row_dict = self.get_flattened_dictionary(
                    v, row_dict=row_dict, key_prefix=f'{key_prefix}{i}'
                )
        else:
            # If value is neither a dictionary nor a list
            # Add the value to the row dictionary
            if key_prefix.startswith('_') and (key_prefix[1:] not in row_dict):
                key_prefix = key_prefix[1:]
            row_dict[key_prefix] = value_obj
        
        return row_dict
    
    
    @staticmethod
    def get_numeric_columns(df, is_na_dropped=True):
        """
        Identify numeric columns in a DataFrame.
        
        Parameters:
            df (pandas.DataFrame): The DataFrame to search for numeric columns.
            is_na_dropped (bool, optional): Whether to drop columns with all NaN values. Default is True.
        
        Returns:
            list: A list of column names containing numeric values.
        
        Notes:
            This function identifies numeric columns by checking if the data in each column
            can be interpreted as numeric. It checks for integer, floating-point, and numeric-like
            objects.
        
        Examples:
            import pandas as pd
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3], 'C': ['a', 'b', 'c']})
            nu.get_numeric_columns(df)
            ['A', 'B']
        """

        # Initialize an empty list to store numeric column names
        numeric_columns = []

        # Iterate over DataFrame columns to identify numeric columns
        for cn in df.columns:
            if is_integer(df[cn]) or is_float(df[cn]):
                numeric_columns.append(cn)

        # Optionally drop columns with all NaN values
        if is_na_dropped: numeric_columns = df[numeric_columns].dropna(axis='columns', how='all').columns

        # Sort and return the list of numeric column names
        return sorted(numeric_columns)
    
    
    ### LLM Functions ###
    
    
    ### 3D Point Functions ###
    
    
    ### Sub-sampling Functions ###
    
    
    ### Plotting Functions ###

nu = NotebookUtilities(
    data_folder_path=osp.abspath('data'),
    saves_folder_path=osp.abspath('saves')
)


class FRVRSUtilities(object):
    """
    This class implements the core of the utility
    functions needed to manipulate FRVRS logger
    data.
    
    Examples
    --------
    
    import sys
    import os.path as osp
    sys.path.insert(1, osp.abspath('../py'))
    from FRVRS import fu
    """
    
    
    def __init__(self, data_folder_path=None, saves_folder_path=None, verbose=False):
        self.verbose = verbose
        
        if IS_DEBUG: print("Create the data folder if it doesn't exist")
        if data_folder_path is None: self.data_folder = 'data'
        else: self.data_folder = data_folder_path
        makedirs(self.data_folder, exist_ok=True)
        if verbose: print('data_folder: {}'.format(osp.abspath(self.data_folder)), flush=True)
        
        if IS_DEBUG: print("Create the saves folder if it doesn't exist")
        if saves_folder_path is None: self.saves_folder = 'saves'
        else: self.saves_folder = saves_folder_path
        makedirs(self.saves_folder, exist_ok=True)
        if verbose: print('saves_folder: {}'.format(osp.abspath(self.saves_folder)), flush=True)
        
        if IS_DEBUG: print("FRVRS log constants")
        self.data_logs_folder = osp.join(self.data_folder, 'logs'); makedirs(name=self.data_logs_folder, exist_ok=True)
        self.scene_groupby_columns = ['session_uuid', 'scene_id']
        
        if IS_DEBUG: print("List of action types to consider as user actions")
        self.known_mcivr_metrics_types = [
            'BAG_ACCESS', 'BAG_CLOSED', 'INJURY_RECORD', 'INJURY_TREATED', 'PATIENT_DEMOTED', 'PATIENT_ENGAGED', 'BREATHING_CHECKED', 'PATIENT_RECORD', 'PULSE_TAKEN', 'SP_O2_TAKEN',
            'S_A_L_T_WALKED', 'TRIAGE_LEVEL_WALKED', 'S_A_L_T_WALK_IF_CAN', 'TRIAGE_LEVEL_WALK_IF_CAN', 'S_A_L_T_WAVED', 'TRIAGE_LEVEL_WAVED', 'S_A_L_T_WAVE_IF_CAN',
            'TRIAGE_LEVEL_WAVE_IF_CAN', 'TAG_APPLIED', 'TAG_DISCARDED', 'TAG_SELECTED', 'TELEPORT', 'TOOL_APPLIED', 'TOOL_DISCARDED', 'TOOL_HOVER', 'TOOL_SELECTED', 'VOICE_CAPTURE',
            'VOICE_COMMAND', 'BUTTON_CLICKED', 'PLAYER_LOCATION', 'PLAYER_GAZE', 'SESSION_START', 'SESSION_END'
        ]
        self.action_types_list = [
            'TELEPORT', 'S_A_L_T_WALK_IF_CAN', 'TRIAGE_LEVEL_WALK_IF_CAN', 'S_A_L_T_WAVE_IF_CAN', 'TRIAGE_LEVEL_WAVE_IF_CAN', 'PATIENT_ENGAGED',
            'PULSE_TAKEN', 'BAG_ACCESS', 'TOOL_HOVER', 'TOOL_SELECTED', 'INJURY_TREATED', 'TOOL_APPLIED', 'TAG_SELECTED', 'TAG_APPLIED',
            'BAG_CLOSED', 'TAG_DISCARDED', 'TOOL_DISCARDED'
        ]
        
        # According to the PatientEngagementWatcher class in the engagement detection code, this Euclidean distance, if the patient has been looked at, triggers enagement
        # The engagement detection code spells out the responder etiquette:
        # 1) if you don't want to trigger a patient walking by, don't look at them, 
        # 2) if you teleport to someone, you must look at them to trigger engagement
        patient_lookup_distance = 2.5
        
        if IS_DEBUG: print("List of command messages to consider as user actions; added Open World commands 20240429")
        self.command_columns_list = ['voice_command_message', 'button_command_message']
        self.command_messages_list = [
            'walk to the safe area', 'wave if you can', 'are you hurt', 'reveal injury', 'lay down', 'where are you',
            'can you hear', 'anywhere else', 'what is your name', 'hold still', 'sit up/down', 'stand up'
        ] + ['can you breathe', 'show me', 'stand', 'walk', 'wave']
        
        if IS_DEBUG: print("List of action types that assume 1-to-1 interaction")
        self.responder_negotiations_list = ['PULSE_TAKEN', 'INJURY_TREATED', 'TAG_APPLIED', 'TOOL_APPLIED']#, 'PATIENT_ENGAGED'
        
        if IS_DEBUG: print("List of columns that contain only boolean values")
        self.boolean_columns_list = [
            'injury_record_injury_treated_with_wrong_treatment', 'injury_record_injury_treated',
            'injury_treated_injury_treated_with_wrong_treatment', 'injury_treated_injury_treated'
        ]
        
        if IS_DEBUG: print("List of columns that contain patientIDs")
        self.patient_id_columns_list = [
            'patient_demoted_patient_id', 'patient_record_patient_id', 'injury_record_patient_id', 's_a_l_t_walk_if_can_patient_id',
            's_a_l_t_walked_patient_id', 's_a_l_t_wave_if_can_patient_id', 's_a_l_t_waved_patient_id', 'patient_engaged_patient_id',
            'pulse_taken_patient_id', 'injury_treated_patient_id', 'tool_applied_patient_id', 'tag_applied_patient_id',
            'player_gaze_patient_id', 'breathing_checked_patient_id', 'sp_o2_taken_patient_id', 'triage_level_walked_patient_id'
        ]
        
        if IS_DEBUG: print("List of columns that contain locationIDs")
        self.location_id_columns_list = [
            'teleport_location', 'patient_demoted_position', 'patient_record_position', 'injury_record_injury_injury_locator',
            's_a_l_t_walk_if_can_sort_location', 's_a_l_t_walked_sort_location', 's_a_l_t_wave_if_can_sort_location',
            's_a_l_t_waved_sort_location', 'patient_engaged_position', 'bag_access_location', 'injury_treated_injury_injury_locator',
            'bag_closed_location', 'tag_discarded_location', 'tool_discarded_location', 'player_location_location',
            'player_gaze_location', 'triage_level_walked_location'
        ]
        
        if IS_DEBUG: print("List of columns with injuryIDs")
        self.injury_id_columns_list = ['injury_record_id', 'injury_treated_id']
        
        if IS_DEBUG: print("Patient SORT designations")
        self.patient_sort_columns_list = ['patient_demoted_sort', 'patient_record_sort', 'patient_engaged_sort']
        self.patient_sort_order = ['still', 'waver', 'walker']
        self.sort_category_order = CategoricalDtype(categories=self.patient_sort_order, ordered=True)
        
        if IS_DEBUG: print("Patient SALT designations")
        self.patient_salt_columns_list = ['patient_demoted_salt', 'patient_record_salt', 'patient_engaged_salt']
        self.patient_salt_order = ['DEAD', 'EXPECTANT', 'IMMEDIATE', 'DELAYED', 'MINIMAL']
        self.salt_category_order = CategoricalDtype(categories=self.patient_salt_order, ordered=True)
        
        if IS_DEBUG: print("Patient pulse designations")
        self.pulse_columns_list = ['patient_demoted_pulse', 'patient_record_pulse', 'patient_engaged_pulse']
        self.patient_pulse_order = ['none', 'faint', 'fast', 'normal']
        self.pulse_category_order = CategoricalDtype(categories=self.patient_pulse_order, ordered=True)
        
        if IS_DEBUG: print("Patient breath designations")
        self.breath_columns_list = ['patient_demoted_breath', 'patient_record_breath', 'patient_engaged_breath', 'patient_checked_breath']
        self.patient_breath_order = ['none', 'collapsedLeft', 'collapsedRight', 'restricted', 'fast', 'normal']
        self.breath_category_order = CategoricalDtype(categories=self.patient_breath_order, ordered=True)
        
        if IS_DEBUG: print("Patient hearing designations")
        self.hearing_columns_list = ['patient_record_hearing', 'patient_engaged_hearing']
        self.patient_hearing_order = ['none', 'limited', 'normal']
        self.hearing_category_order = CategoricalDtype(categories=self.patient_hearing_order, ordered=True)
        
        if IS_DEBUG: print("Patient mood designations")
        self.mood_columns_list = ['patient_demoted_mood', 'patient_record_mood', 'patient_engaged_mood']
        self.patient_mood_order = ['dead', 'unresponsive', 'agony', 'upset', 'calm', 'low', 'normal', 'none']
        self.mood_category_order = CategoricalDtype(categories=self.patient_mood_order, ordered=True)
        
        if IS_DEBUG: print("Patient pose designations")
        self.pose_columns_list = ['patient_demoted_pose', 'patient_record_pose', 'patient_engaged_pose']
        self.patient_pose_order = ['dead', 'supine', 'fetal', 'agony', 'sittingGround', 'kneeling', 'upset', 'standing', 'recovery', 'calm']
        self.pose_category_order = CategoricalDtype(categories=self.patient_pose_order, ordered=True)
        
        # Hemorrhage control procedures list
        self.hemorrhage_control_procedures_list = ['tourniquet', 'woundpack']
        
        if IS_DEBUG: print("Injury required procedure designations")
        self.injury_required_procedure_columns_list = ['injury_record_required_procedure', 'injury_treated_required_procedure']
        self.injury_required_procedure_order = [
            'tourniquet', 'gauzePressure', 'decompress', 'chestSeal', 'woundpack', 'ivBlood', 'airway', 'epiPen', 'burnDressing', 'splint', 'ivSaline', 'painMeds', 'blanket', 'none'
        ]
        self.required_procedure_category_order = CategoricalDtype(categories=self.injury_required_procedure_order, ordered=True)
        self.required_procedure_to_tool_type_dict = {
            'airway': 'Nasal Airway',
            'blanket': 'Blanket',
            'burnDressing': 'Burn Dressing',
            'chestSeal': 'ChestSeal',
            'decompress': 'Needle',
            'epiPen': 'EpiPen',
            'gauzePressure': 'Gauze_Dressing',
            'ivBlood': 'IV Blood',
            'ivSaline': 'IV Saline',
            'none': '',
            'painMeds': 'Pain Meds',
            'splint': 'SAM Splint',
            'tourniquet': 'Tourniquet',
            'woundpack': 'Gauze_Pack',
        }
        
        if IS_DEBUG: print("Injury severity designations")
        self.injury_severity_columns_list = ['injury_record_severity', 'injury_treated_severity']
        self.injury_severity_order = ['high', 'medium', 'low']
        self.severity_category_order = CategoricalDtype(categories=self.injury_severity_order, ordered=True)
        
        if IS_DEBUG: print("Injury body region designations")
        self.body_region_columns_list = ['injury_record_body_region', 'injury_treated_body_region']
        self.injury_body_region_order = ['head', 'neck', 'chest', 'abdomen', 'leftLeg', 'rightLeg', 'rightHand', 'rightArm', 'leftHand', 'leftArm']
        self.body_region_category_order = CategoricalDtype(categories=self.injury_body_region_order, ordered=True)
        
        if IS_DEBUG: print("Pulse name designations")
        self.pulse_name_order = ['pulse_none', 'pulse_faint', 'pulse_fast', 'pulse_normal']
        self.pulse_name_category_order = CategoricalDtype(categories=self.pulse_name_order, ordered=True)
        
        if IS_DEBUG: print("Tool type designations")
        self.tool_type_columns_list = ['tool_hover_type', 'tool_selected_type', 'tool_applied_type', 'tool_discarded_type']
        self.tool_type_order = [
            'Tourniquet', 'Gauze_Pack', 'Hemostatic Gauze', 'ChestSeal', 'Occlusive', 'IV Blood', 'IV_Blood', 'Needle', 'Naso', 'Nasal Airway', 'Burn Dressing', 'Burn_Dressing',
            'Gauze_Dressing', 'Gauze', 'EpiPen', 'IV Saline', 'IV_Saline', 'Pain Meds', 'Pain_Meds', 'Pulse Oximeter', 'Pulse_Oximeter', 'SAM Splint', 'SAM_Splint', 'Shears',
            'SurgicalTape', 'Blanket'
        ]
        self.tool_type_category_order = CategoricalDtype(categories=self.tool_type_order, ordered=True)
        self.tool_type_to_required_procedure_dict = {
            'Blanket': 'blanket',
            'Burn Dressing': 'burnDressing',
            'Burn_Dressing': 'burnDressing',
            'ChestSeal': 'chestSeal',
            'EpiPen': 'epiPen',
            'Gauze': 'gauzePressure',
            'Gauze_Dressing': 'gauzePressure',
            'Gauze_Pack': 'woundpack',
            'Hemostatic Gauze': 'woundpack',
            'IV Blood': 'ivBlood',
            'IV Saline': 'ivSaline',
            'IV_Blood': 'ivBlood',
            'IV_Saline': 'ivSaline',
            'Nasal Airway': 'airway',
            'Naso': 'airway',
            'Needle': 'decompress',
            'Occlusive': 'chestSeal',
            'Pain Meds': 'painMeds',
            'Pain_Meds': 'painMeds',
            'Pulse Oximeter': 'none',
            'Pulse_Oximeter': 'none',
            'SAM Splint': 'splint',
            'SAM_Splint': 'splint',
            'Shears': 'none',
            'SurgicalTape': 'none',
            'Tourniquet': 'tourniquet',
        }
        
        if IS_DEBUG: print("Tool data designations")
        self.tool_applied_data_order = ['right_chest', 'left_chest', 'right_underarm', 'left_underarm']
        self.tool_applied_data_category_order = CategoricalDtype(categories=self.tool_applied_data_order, ordered=True)
        
        # MCI-VR metrics types dictionary
        self.action_type_to_columns = {
            'BAG_ACCESS': {
                'bag_access_location': 4,
            },
            'BAG_CLOSED': {
                'bag_closed_location': 4,
            },
            'INJURY_RECORD': {
                'injury_record_id': 4,
                'injury_record_patient_id': 5,
                'injury_record_required_procedure': 6,
                'injury_record_severity': 7,
                'injury_record_body_region': 8,
                'injury_record_injury_treated': 9,
                'injury_record_injury_treated_with_wrong_treatment': 10,
                'injury_record_injury_injury_locator': 11,
            },
            'INJURY_TREATED': {
                'injury_treated_id': 4,
                'injury_treated_patient_id': 5,
                'injury_treated_required_procedure': 6,
                'injury_treated_severity': 7,
                'injury_treated_body_region': 8,
                'injury_treated_injury_treated': 9,
                'injury_treated_injury_treated_with_wrong_treatment': 10,
                'injury_treated_injury_injury_locator': 11,
            },
            'PATIENT_DEMOTED': {
                'patient_demoted_health_level': 4,
                'patient_demoted_health_time_remaining': 5,
                'patient_demoted_patient_id': 6,
                'patient_demoted_position': 7,
                'patient_demoted_rotation': 8,
                'patient_demoted_salt': 9,
                'patient_demoted_sort': 10,
                'patient_demoted_pulse': 11,
                'patient_demoted_breath': 12,
                'patient_demoted_hearing': 13,
                'patient_demoted_mood': 14,
                'patient_demoted_pose': 15,
            },
            'PATIENT_ENGAGED': {
                'patient_engaged_health_level': 4,
                'patient_engaged_health_time_remaining': 5,
                'patient_engaged_patient_id': 6,
                'patient_engaged_position': 7,
                'patient_engaged_rotation': 8,
                'patient_engaged_salt': 9,
                'patient_engaged_sort': 10,
                'patient_engaged_pulse': 11,
                'patient_engaged_breath': 12,
                'patient_engaged_hearing': 13,
                'patient_engaged_mood': 14,
                'patient_engaged_pose': 15,
            },
            'BREATHING_CHECKED': {
                'patient_checked_breath': 4,
                'breathing_checked_patient_id': 5,
            },
            'PATIENT_RECORD': {
                'patient_record_health_level': 4,
                'patient_record_health_time_remaining': 5,
                'patient_record_patient_id': 6,
                'patient_record_position': 7,
                'patient_record_rotation': 8,
                'patient_record_salt': 9,
                'patient_record_sort': 10,
                'patient_record_pulse': 11,
                'patient_record_breath': 12,
                'patient_record_hearing': 13,
                'patient_record_mood': 14,
                'patient_record_pose': 15,
            },
            'PULSE_TAKEN': {
                'pulse_taken_pulse_name': 4,
                'pulse_taken_patient_id': 5,
            },
            'SP_O2_TAKEN': {
                'sp_o2_taken_level': 4,
                'sp_o2_taken_patient_id': 5,
            },
            'S_A_L_T_WALKED': {
                's_a_l_t_walked_sort_location': 4,
                's_a_l_t_walked_sort_command_text': 5,
                's_a_l_t_walked_patient_id': 6,
            },
            'TRIAGE_LEVEL_WALKED': {
                'triage_level_walked_location': 4,
                'triage_level_walked_command_text': 5,
                'triage_level_walked_patient_id': 6,
            },
            'S_A_L_T_WALK_IF_CAN': {
                's_a_l_t_walk_if_can_sort_location': 4,
                's_a_l_t_walk_if_can_sort_command_text': 5,
                's_a_l_t_walk_if_can_patient_id': 6,
            },
            'TRIAGE_LEVEL_WALK_IF_CAN': {
                'triage_level_walk_if_can_location': 4,
                'triage_level_walk_if_can_command_text': 5,
                'triage_level_walk_if_can_patient_id': 6,
            },
            'S_A_L_T_WAVED': {
                's_a_l_t_waved_sort_location': 4,
                's_a_l_t_waved_sort_command_text': 5,
                's_a_l_t_waved_patient_id': 6,
            },
            'TRIAGE_LEVEL_WAVED': {
                'triage_level_waved_location': 4,
                'triage_level_waved_command_text': 5,
                'triage_level_waved_patient_id': 6,
            },
            'S_A_L_T_WAVE_IF_CAN': {
                's_a_l_t_wave_if_can_sort_location': 4,
                's_a_l_t_wave_if_can_sort_command_text': 5,
                's_a_l_t_wave_if_can_patient_id': 6,
            },
            'TRIAGE_LEVEL_WAVE_IF_CAN': {
                'triage_level_wave_if_can_location': 4,
                'triage_level_wave_if_can_command_text': 5,
                'triage_level_wave_if_can_patient_id': 6,
            },
            'TAG_APPLIED': {
                'tag_applied_patient_id': 4,
                'tag_applied_type': 5,
            },
            'TAG_DISCARDED': {
                'tag_discarded_type': 4,
                'tag_discarded_location': 5,
            },
            'TAG_SELECTED': {
                'tag_selected_type': 4,
            },
            'TELEPORT': {
                'teleport_location': 4,
            },
            'TOOL_APPLIED': {
                'tool_applied_patient_id': 4,
                'tool_applied_type': 5,
                'tool_applied_attachment_point': 6,
                'tool_applied_tool_location': 7,
                'tool_applied_sender': 8,
                'tool_applied_attach_message': 9,
            },
            'TOOL_DISCARDED': {
                'tool_discarded_type': 4,
                'tool_discarded_count': 5,
                'tool_discarded_location': 6,
            },
            'TOOL_HOVER': {
                'tool_hover_type': 4,
                'tool_hover_count': 5,
            },
            'TOOL_SELECTED': {
                'tool_selected_type': 4,
                'tool_selected_count': 5,
            },
            'VOICE_CAPTURE': {
                'voice_capture_message': 4,
                'voice_capture_command_description': 5,
            },
            'VOICE_COMMAND': {
                'voice_command_message': 4,
                'voice_command_command_description': 5,
            },
            'BUTTON_CLICKED': {
                'button_command_message': 4,
            },
            'PLAYER_LOCATION': {
                'player_location_location': 4,
                'player_location_left_hand_location': 5,
                'player_location_right_hand_location': 6,
            },
            'PLAYER_GAZE': {
                'player_gaze_location': 4,
                'player_gaze_patient_id': 5,
                'player_gaze_distance_to_patient': 6,
                'player_gaze_direction_of_gaze': 7,
            },
        }
        
        # The patients lists from the March 25th ITM BBAI Exploratory analysis email
        self.desert_patients_list = [
            'Open World Marine 1 Female', 'Open World Marine 2 Male', 'Open World Civilian 1 Male', 'Open World Civilian 2 Female'
        ]
        self.desert_patients_list += [c + ' Root' for c in self.desert_patients_list]
        self.jungle_patients_list = [
            'Open World Marine 1 Male', 'Open World Marine 2 Female', 'Open World Marine 3 Male', 'Open World Marine 4 Male'
        ]
        self.jungle_patients_list += [c + ' Root' for c in self.jungle_patients_list]
        self.submarine_patients_list = ['Navy Soldier 1 Male', 'Navy Soldier 2 Male', 'Navy Soldier 3 Male', 'Navy Soldier 4 Female']
        self.submarine_patients_list += [c + ' Root' for c in self.submarine_patients_list]
        self.urban_patients_list = ['Marine 1 Male', 'Marine 2 Male', 'Marine 3 Male', 'Marine 4 Male', 'Civilian 1 Female']
        self.urban_patients_list += [c + ' Root' for c in self.urban_patients_list]

    ### String Functions ###
    
    
    ### List Functions ###
    
    
    ### File Functions ###
    
    
    ### Session Functions ###
    
    
    @staticmethod
    def get_logger_version(session_df_or_file_path, verbose=False):
        """
        Retrieve the unique logger version associated with the given session DataFrame.

        Parameters
        ----------
        session_df_or_file_path : pandas.DataFrame or str
            DataFrame containing session data or file path.
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Returns
        -------
        str
            The unique logger version associated with the session DataFrame.
        """
        
        # Assume there are only three versions
        if isinstance(session_df_or_file_path, str):
            with open(session_df_or_file_path, 'r', encoding=nu.encoding_type) as f:
                file_content = f.read()
                if ',1.3,' in file_content: logger_version = 1.3
                elif ',1.4,' in file_content: logger_version = 1.4
                else: logger_version = 1.0
        else:
            
            # Extract the unique logger version from the session DataFrame
            logger_version = session_df_or_file_path.logger_version.unique().item()
        
        # Print verbose output
        if verbose: print('Logger version: {}'.format(logger_version))
        
        # Return the unique logger version
        return logger_version
    
    
    @staticmethod
    def get_last_still_engagement(actual_engagement_order, verbose=False):
        """
        A utility method to retrieve the timestamp of the last engagement of still patients.

        Parameters:
            actual_engagement_order (array_like): A 2D array containing engagement information for patients.
            verbose (bool, optional): If True, prints debug output. Default is False.

        Returns:
            last_still_engagement (float): The timestamp of the last engagement of still patients.

        Notes:
            This method assumes that the input array contains columns in the following order:
            1. 'patient_id'
            2. 'engagement_start'
            3. 'location_tuple'
            4. 'patient_sort'
            5. 'predicted_priority'
            6. 'injury_severity'
        """
        
        # Get the chronological order of engagement starts for each patient in the scene
        columns_list = ['patient_id', 'engagement_start', 'location_tuple', 'patient_sort', 'predicted_priority', 'injury_severity']
        df = DataFrame(actual_engagement_order, columns=columns_list)
        
        # Filter out only the still patients
        mask_series = (df.patient_sort == 'still')
        
        # Get the maximum engagement start from that subset
        last_still_engagement = df[mask_series].engagement_start.max()
        
        return last_still_engagement
    
    
    @staticmethod
    def get_actual_engagement_distance(actual_engagement_order, verbose):
        """
        Calculate the total distance covered during actual engagements.
        
        Parameters:
            actual_engagement_order (list of tuples):
                A chronologically-ordered list containing tuples of engagements, 
                where each tuple contains information about the engagement.
                The third element of the tuple must be a location tuple.
            verbose (bool, optional): If True, prints debug information. Default is False.
        
        Returns:
            float: The total Euclidean distance covered during actual engagements.
        """
        
        # Filter out all the non-locations of the non-engaged
        actual_engagement_order = [engagement_tuple for engagement_tuple in actual_engagement_order if engagement_tuple[2] is not None]
        
        # Add the Euclidean distances between the successive engagment locations of a chronologically-ordered list
        actual_engagement_distance = sum([
            math.sqrt(
                (first_tuple[2][0] - last_tuple[2][0])**2 + (first_tuple[2][1] - last_tuple[2][1])**2
            ) for first_tuple, last_tuple in zip(actual_engagement_order[:-1], actual_engagement_order[1:])
        ])
        
        return actual_engagement_distance
    
    
    def get_distance_deltas_data_frame(self, logs_df, verbose=False):
        """
        Compute various metrics related to engagement distances and ordering for scenes in logs dataframe.
        
        Parameters:
            logs_df (pandas DataFrame): Dataframe containing logs of engagement scenes.
            verbose (bool, optional): Verbosity flag for debugging output, by default False.
        
        Returns:
            pandas.DataFrame: DataFrame containing computed metrics for each scene.
        
        Notes:
            This function computes metrics such as patient count, engagement order, last still engagement, actual engagement distance,
            measure of right ordering, and adherence to SALT protocol for each scene in the logs DataFrame.
        """
        rows_list = []
        for (session_uuid, scene_id), scene_df in logs_df.groupby(self.scene_groupby_columns):
            row_dict = {}
            for cn in self.scene_groupby_columns: row_dict[cn] = eval(cn)
            
            # Get patient_count
            patient_count = self.get_patient_count(scene_df)
            row_dict['patient_count'] = patient_count
            
            # Get the chronological order of engagement starts for each patient in the scene
            actual_engagement_order = self.get_actual_engagement_order(scene_df, include_noninteracteds=True, verbose=False)
            assert len(actual_engagement_order) == patient_count, f"There are {patient_count} patients in this scene and only {len(actual_engagement_order)} engagement tuples:\n{scene_df[~scene_df.patient_id.isnull()].patient_id.unique().tolist()}\n{actual_engagement_order}"
            unengaged_patient_count = 0; engaged_patient_count = 0
            for engagement_tuple in actual_engagement_order:
                if engagement_tuple[1] < 0:
                    column_name = f'unengaged_patient{unengaged_patient_count:0>2}_metadata'
                    unengaged_patient_count += 1
                else:
                    column_name = f'engaged_patient{engaged_patient_count:0>2}_metadata'
                    engaged_patient_count += 1
                column_value = '|'.join([str(x) for x in list(engagement_tuple)])
                if not isna(column_value): row_dict[column_name] = column_value
            
            # Get last still engagement and subtract the scene start
            last_still_engagement = self.get_last_still_engagement(actual_engagement_order, verbose=verbose)
            row_dict['last_still_engagement'] = last_still_engagement - self.get_scene_start(scene_df)
            
            # Actual
            row_dict['actual_engagement_distance'] = self.get_actual_engagement_distance(actual_engagement_order, verbose=verbose)
            
            # Calculate the measure of right ordering
            row_dict['measure_of_right_ordering'] = self.get_measure_of_right_ordering(scene_df, verbose=verbose)
            
            rows_list.append(row_dict)
        distance_delta_df = DataFrame(rows_list)
        
        # Add the adherence to SALT protocol column
        mask_series = (distance_delta_df.measure_of_right_ordering == 1.0)
        distance_delta_df['adherence_to_salt'] = mask_series
        
        return distance_delta_df
    
    
    ### Scene Functions ###
    
    
    @staticmethod
    def get_scene_start(scene_df, verbose=False):
        """
        Get the start time of the scene DataFrame run.
        
        Parameters
        ----------
        scene_df : pandas.DataFrame
            DataFrame containing data for a specific scene.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        
        Returns
        -------
        float
            The start time of the scene in milliseconds.
        """
        
        # Find the minimum elapsed time to get the scene start time
        run_start = scene_df.action_tick.min()
        
        if verbose: print('Scene start: {}'.format(run_start))
        
        # Return the start time of the scene
        return run_start
    
    
    @staticmethod
    def get_last_engagement(scene_df, verbose=False):
        """
        Get the last time a patient was engaged in the given scene DataFrame.
        
        Parameters
        ----------
        scene_df : pandas.DataFrame
            DataFrame containing scene-specific data.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        
        Returns
        -------
        int
            Action tick of the last patient engagement in the scene DataFrame, in milliseconds.
        """
        
        # Get the mask for the PATIENT_ENGAGED actions
        action_mask_series = (scene_df.action_type == 'PATIENT_ENGAGED')
        
        # Find the maximum elapsed time among rows satisfying the action mask
        last_engagement = scene_df[action_mask_series].action_tick.max()
        
        # Print verbose output if enabled
        if verbose: print('Last engagement time: {}'.format(last_engagement))
        
        # Return the last engagement time
        return last_engagement
    
    
    @staticmethod
    def get_scene_type(scene_df, verbose=False):
        """
        Gets the type of a scene.
        
        Parameters
        ----------
        scene_df : pandas.DataFrame
            DataFrame containing data for a specific scene.
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Returns
        -------
        str
            The type of the scene, e.g., 'Triage', 'Orientation', etc.
        """
        
        # Check if the scene_type column exists in the scene data frame, and get the unique value if it is
        if 'scene_type' in scene_df.columns:
            scene_types = scene_df.scene_type.unique()
            if verbose: print(f'scene_types={scene_types}')
            if scene_types.shape[0] > 1: raise Exception(f'scene_types={scene_types} has more than one entry')
            else: scene_type = scene_types.item()
        
        else:
            
            # Default scene type is 'Triage'
            scene_type = 'Triage'
            
            # Check if all of the patient IDs in the scene DataFrame contain the string 'mike', and, if so, set the scene type to 'Orientation'
            if all(scene_df.patient_id.dropna().transform(lambda srs: srs.str.lower().str.contains('mike'))): scene_type = 'Orientation'
        
        # Return the scene type
        return scene_type
    
    
    @staticmethod
    def get_scene_end(scene_df, verbose=False):
        """
        Calculate the end time of a scene based on the maximum elapsed time in the input DataFrame.
        
        Parameters
        ----------
        scene_df : pandas.DataFrame
            A Pandas DataFrame containing the scene data.
        verbose : bool, optional
            Whether to print additional information. Default is False.
        
        Returns
        -------
        int
            End time of the scene calculated as the maximum elapsed time, in milliseconds.
        """
        
        # Get the maximum elapsed time in the scene
        run_end = scene_df.action_tick.max()
        
        # If verbose is True, print the end time
        if verbose: print(f'Scene end time calculated: {run_end}')
        
        # Return the end time
        return run_end
    
    
    @staticmethod
    def get_patient_count(scene_df, verbose=False):
        """
        Calculates the number of unique patient IDs in a given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data, including 'patient_id' column.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: Number of unique patients in the scene DataFrame.
        """
        
        # Count the number of unique patient IDs
        patient_count = scene_df.patient_id.nunique()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Number of unique patients: {patient_count}')
            display(scene_df)
        
        # Return the calculated patient count
        return patient_count
    
    
    @staticmethod
    def get_injury_treatments_count(scene_df, verbose=False):
        """
        Calculates the number of records where injury treatment attempts were logged for a given scene.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of records where injury treatment attempts were logged.
        """
        
        # Filter for treatments with injury_treated set to True
        mask_series = (scene_df.injury_treated_injury_treated == True)
        
        # Count the number of treatments
        injury_treatments_count = scene_df[mask_series].shape[0]
        if verbose:
            print(f'Number of records where injury treatment attempts were logged: {injury_treatments_count}')
            display(scene_df)
        
        # Return the count of records where injuries were treated
        return injury_treatments_count
    
    
    @staticmethod
    def get_injury_not_treated_count(scene_df, verbose=False):
        """
        Get the count of records in a scene DataFrame where injuries were not treated.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of patients who have not received injury treatment.
        """
        
        # Loop through each injury and make a determination if it's treated or not
        injury_not_treated_count = 0
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            for injury_id, injury_df in patient_df.groupby('injury_id'):
                mask_series = (injury_df.action_type == 'INJURY_TREATED')
                if not mask_series.any():
                    injury_not_treated_count += 1
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Number of records where injuries were not treated: {injury_not_treated_count}')
            display(scene_df)
        
        # Return the count of records where injuries were not treated
        return injury_not_treated_count
    
    
    def get_injury_correctly_treated_count(self, scene_df, verbose=False):
        """
        Get the count of records in a scene DataFrame where injuries were correctly treated.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: Count of records where injuries were correctly treated in the scene DataFrame.
        
        Note:
            The FRVRS logger has trouble logging multiple tool applications,
            so injury_treated_injury_treated_with_wrong_treatment == True
            remains and another INJURY_TREATED is never logged, even though
            the right tool was applied after that.
        """
        
        # Loop through each injury ID and make a determination if it's treated or not
        injury_correctly_treated_count = 0
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            for injury_id, injury_df in patient_df.groupby('injury_id'):
                is_correctly_treated = self.get_is_injury_correctly_treated(injury_df, patient_df, verbose=verbose)
                if is_correctly_treated: injury_correctly_treated_count += 1
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Number of records where injuries were correctly treated: {injury_correctly_treated_count}')
            display(scene_df)
        
        # Return the count of records where injuries were correctly treated
        return injury_correctly_treated_count
    
    
    def get_injury_wrongly_treated_count(self, scene_df, verbose=False):
        """
        Calculates the number of patients whose injuries have been incorrectly treated in a given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): The DataFrame containing scene data.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of patients whose injuries have been incorrectly treated.
        
        Note:
            The FRVRS logger has trouble logging multiple tool applications,
            so injury_treated_injury_treated_with_wrong_treatment == True
            remains and another INJURY_TREATED is never logged, even though
            the right tool was applied after that.
        """
        
        # Get the count of all the patient injuries
        all_patient_injuries_count = 0
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            all_patient_injuries_count += patient_df.injury_id.nunique()
        
        # Get the count of all correctly treated injuries
        correctly_treated_count = self.get_injury_correctly_treated_count(scene_df)
        
        # Get the count of all untreated injuries
        not_treated_count = self.get_injury_not_treated_count(scene_df)
        
        # Count the number of patients whose injuries have been incorrectly treated
        injury_wrongly_treated_count = all_patient_injuries_count - correctly_treated_count - not_treated_count
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Injury wrongly treated count: {injury_wrongly_treated_count}')
            display(scene_df)
        
        return injury_wrongly_treated_count
    
    
    @staticmethod
    def get_pulse_taken_count(scene_df, verbose=False):
        """
        Count the number of 'PULSE_TAKEN' actions in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of PULSE_TAKEN actions in the scene DataFrame.
        """
        
        # Create a boolean mask to filter 'PULSE_TAKEN' actions
        mask_series = scene_df.action_type.isin(['PULSE_TAKEN'])
        
        # Use the mask to filter the DataFrame and count the number of 'PULSE_TAKEN' actions
        pulse_taken_count = scene_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Number of PULSE_TAKEN actions: {pulse_taken_count}')
            display(scene_df)
        
        # Return the count of 'PULSE_TAKEN' actions
        return pulse_taken_count
    
    
    @staticmethod
    def get_teleport_count(scene_df, verbose=False):
        """
        Count the number of 'TELEPORT' actions in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of TELEPORT actions in the scene DataFrame.
        """
    
        # Create a boolean mask to filter TELEPORT action types
        mask_series = scene_df.action_type.isin(['TELEPORT'])
        
        # Count the number of actions
        teleport_count = scene_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Number of TELEPORT actions: {teleport_count}')
            display(scene_df)
        
        return teleport_count
    
    
    @staticmethod
    def get_voice_capture_count(scene_df, verbose=False):
        """
        Calculates the number of VOICE_CAPTURE actions in a given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of VOICE_CAPTURE actions in the scene DataFrame.
        """
        
        # Filter for actions with the type "VOICE_CAPTURE"
        mask_series = scene_df.action_type.isin(['VOICE_CAPTURE'])
        
        # Count the number of "VOICE_CAPTURE" actions
        voice_capture_count = scene_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Number of VOICE_CAPTURE actions: {voice_capture_count}')
            display(scene_df)
        
        # Return the count of 'VOICE_CAPTURE' actions
        return voice_capture_count
    
    
    @staticmethod
    def get_walk_command_count(scene_df, verbose=False):
        """
        Count the number of 'walk to the safe area' voice command events in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of walk to the safe area voice commands in the scene DataFrame.
        """
        
        # Filter for voice commands with the message "walk to the safe area"
        mask_series = scene_df.voice_command_message.isin(['walk to the safe area'])
        
        # Count the number of "walk to the safe area" voice commands
        walk_command_count = scene_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f"Number of 'walk to the safe area' voice command events: {walk_command_count}")
            display(scene_df)
        
        # Return the count of 'walk to the safe area' voice command events
        return walk_command_count
    
    
    @staticmethod
    def get_wave_command_count(scene_df, verbose=False):
        """
        Calculates the number of wave if you can voice commands in a given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of wave if you can voice commands in the scene DataFrame.
        """
        
        # Filter for voice commands with the message "wave if you can"
        mask_series = scene_df.voice_command_message.isin(['wave if you can'])
        
        # Count the number of "wave if you can" voice commands
        wave_command_count = scene_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f"Number of 'wave if you can' voice command events: {wave_command_count}")
            display(scene_df)
        
        # Return the count of 'wave if you can' voice command events
        return wave_command_count
    
    
    @staticmethod
    def get_first_engagement(scene_df, verbose=False):
        """
        Get the action tick of the first 'PATIENT_ENGAGED' action in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: Action tick of the first 'PATIENT_ENGAGED' action in the scene DataFrame.
        """
        
        # Filter for actions with the type "PATIENT_ENGAGED"
        action_mask_series = (scene_df.action_type == 'PATIENT_ENGAGED')
        
        # Get the action tick of the first 'PATIENT_ENGAGED' action
        first_engagement = scene_df[action_mask_series].action_tick.min()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Action tick of the first PATIENT_ENGAGED action: {first_engagement}')
            display(scene_df)
        
        # Return the action tick of the first 'PATIENT_ENGAGED' action
        return first_engagement
    
    
    @staticmethod
    def get_first_treatment(scene_df, verbose=False):
        """
        Get the action tick of the first 'INJURY_TREATED' action in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: Action tick of the first 'INJURY_TREATED' action in the scene DataFrame.
        """
        
        # Filter for actions with the type "INJURY_TREATED"
        action_mask_series = (scene_df.action_type == 'INJURY_TREATED')
        
        # Get the action tick of the first 'INJURY_TREATED' action
        first_treatment = scene_df[action_mask_series].action_tick.min()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Action tick of the first INJURY_TREATED action: {first_treatment}')
            display(scene_df)
        
        # Return the action tick of the first 'INJURY_TREATED' action
        return first_treatment
    
    
    def get_is_scene_aborted(self, scene_df, verbose=False):
        """
        Gets whether or not a scene is aborted.
        
        Parameters
        ----------
        scene_df : pandas.DataFrame
            DataFrame containing data for a specific scene.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        
        Returns
        -------
        bool
            True if the scene is aborted, False otherwise.
        """
        
        # Check if the is_scene_aborted column exists in the scene data frame, and get the unique value if it is
        if ('is_scene_aborted' in scene_df.columns) and (scene_df.is_scene_aborted.unique().shape[0] > 0):
            is_scene_aborted = scene_df.is_scene_aborted.unique().item()
        else: is_scene_aborted = False
        if not is_scene_aborted:
            
            # Calculate the time duration between scene start and last engagement
            scene_start = self.get_scene_start(scene_df, verbose=verbose)
        
            # Get the last engagement time
            last_engagement = self.get_last_engagement(scene_df, verbose=verbose)
        
            # Calculate the time difference between the scene start and the last engagement
            start_to_last_engagement = last_engagement - scene_start
            
            # Define the threshold for scene abortion (sixteen minutes)
            sixteen_minutes = 1_000 * 60 * 16
            
            # Determine if the scene is aborted based on the time duration
            is_scene_aborted = bool(start_to_last_engagement > sixteen_minutes)
        
        # Return True if the scene is aborted, False otherwise
        return is_scene_aborted
    
    
    def get_triage_time(self, scene_df, verbose=False):
        """
        Calculates the triage time for a scene.
        
        Parameters
        ----------
        scene_df : pandas.DataFrame
            DataFrame containing scene data.
        verbose : bool, optional
            Whether to print verbose output.
        
        Returns
        -------
        int
            Triage time in milliseconds.
        """
        
        # Get the scene start and end times
        time_start = self.get_scene_start(scene_df, verbose=verbose)
        time_stop = self.get_scene_end(scene_df, verbose=verbose)
        
        # Calculate the triage time
        triage_time = time_stop - time_start
        
        return triage_time
    
    
    def get_total_actions_count(self, scene_df, verbose=False):
        """
        Calculates the total number of user actions within a given scene DataFrame,
        including voice commands with specific messages.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: Total number of user actions in the scene DataFrame.
        """
        
        # Create a boolean mask to filter action types that are user-initiated (TELEPORT, S_A_L_T_WALK_IF_CAN, TRIAGE_LEVEL_WALK_IF_CAN, S_A_L_T_WAVE_IF_CAN, TRIAGE_LEVEL_WAVE_IF_CAN, PATIENT_ENGAGED, PULSE_TAKEN, BAG_ACCESS, TOOL_HOVER, TOOL_SELECTED, INJURY_TREATED, TOOL_APPLIED, TAG_SELECTED, TAG_APPLIED, BAG_CLOSED, TAG_DISCARDED, and TOOL_DISCARDED)
        mask_series = scene_df.action_type.isin(self.action_types_list)
        
        # Include VOICE_COMMAND actions with specific user-initiated messages in the mask (walk to the safe area, wave if you can, are you hurt, reveal injury, lay down, where are you, can you hear, anywhere else, what is your name, hold still, sit up/down, stand up, can you breathe, show me, stand, walk, and wave)
        mask_series |= ((scene_df.action_type == 'VOICE_COMMAND') & (scene_df.voice_command_message.isin(self.command_messages_list)))
        
        # Count the number of user actions for the current group
        total_actions_count = scene_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Total number of user actions: {total_actions_count}')
            display(scene_df)
        
        # Return the total number of user actions
        return total_actions_count
    
    
    def get_actual_and_ideal_patient_sort_sequences(self, scene_df, verbose=False):
        """
        Extracts the actual and ideal sequences of first interactions from a scene dataframe.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing patient interactions with columns, including 'patient_sort' and 'patient_id'.
            verbose (bool, optional): Whether to print intermediate results for debugging. Defaults to False.
        
        Returns:
            tuple: A tuple of three elements:
                actual_sequence (pandas.Series): The actual sequence of first interactions, sorted.
                ideal_sequence (pandas.Series): Series of ideal patient interactions based on SORT categories.
                sort_dict (dict): Dictionary containing lists of first interactions for each SORT category.
        
        Note:
            Only SORT categories included in `self.patient_sort_order` are considered.
            None values in the resulting lists indicate missing interactions.
            When the walkers walk too close to the responder they can trigger the PATIENT_ENGAGED action
            which incorrectly assigns them as a patient that has been seen. 
        """
        
        # Group patients by their SORT category and get lists of their elapsed times
        sort_dict = {}
        for sort, patient_sort_df in scene_df.groupby('patient_sort'):
            
            # Only consider SORT categories included in the patient_sort_order
            if sort in self.patient_sort_order:
                
                # Loop through the SORT patients to add their first interactions to the action list
                action_list = []
                for patient_id in patient_sort_df.patient_id.unique():
                    mask_series = (scene_df.patient_id == patient_id)
                    if mask_series.any(): action_list.append(self.get_first_patient_interaction(scene_df[mask_series]))
                
                # Sort the list of first interactions
                if verbose: print(sort, action_list)
                sort_dict[sort] = sorted([action for action in action_list if not isna(action)])
        
        # Get the whole ideal and actual sequences
        ideal_sequence = []
        for sort in self.patient_sort_order: ideal_sequence.extend(sort_dict.get(sort, []))
        ideal_sequence = Series(data=ideal_sequence)
        actual_sequence = ideal_sequence.sort_values(ascending=True)
        
        return actual_sequence, ideal_sequence, sort_dict
    
    
    def get_measure_of_ordering(self, actual_sequence, ideal_sequence, verbose=False):
        """
        Calculate the measure of ordering between actual and ideal sequences using the adjusted R-squared value.
        
        Parameters:
            actual_sequence (array-like): The observed sequence of actions or events.
            ideal_sequence (array-like): The ideal or expected sequence of actions or events.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            float: The R-squared adjusted value as a measure of ordering.
                Returns NaN if model fitting fails.
        """
        
        # Initialize the measure of ordering to NaN
        measure_of_ordering = nan
        
        # Prepare data for regression model
        X, y = ideal_sequence.values.reshape(-1, 1), actual_sequence.values.reshape(-1, 1)
        
        # Fit regression model and calculate R-squared adjusted if data is present
        if X.shape[0]:
            X1 = sm.add_constant(X)  # Add constant for intercept
            
            # Handle model fitting exceptions
            try: measure_of_ordering = sm.OLS(y, X1).fit().rsquared_adj
            except: measure_of_ordering = nan
        
        # Print additional information if verbose is True
        if verbose: print(f'The measure of ordering for patients: {measure_of_ordering}')
        
        return measure_of_ordering
    
    
    def get_measure_of_right_ordering(self, scene_df, verbose=False):
        """
        Calculates a measure of right ordering for patients based on their SORT category and elapsed times.
        The measure of right ordering is an R-squared adjusted value, where a higher value indicates better right ordering.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            float: The measure of right ordering for patients.
        """
        measure_of_right_ordering = nan
        
        # Extract the actual and ideal sequences of first interactions from the scene in terms of still/waver/walker
        actual_sequence, ideal_sequence, _ = self.get_actual_and_ideal_patient_sort_sequences(scene_df, verbose=verbose)
        
        # Calculate the R-squared adjusted value as a measure of right ordering
        measure_of_right_ordering = self.get_measure_of_ordering(actual_sequence, ideal_sequence, verbose=verbose)
        
        # If verbose is True, print additional information
        if verbose:
            print(f'The measure of right ordering for patients: {measure_of_right_ordering}')
            display(scene_df)
        
        return measure_of_right_ordering
    
    
    def get_percent_hemorrhage_controlled(self, scene_df, verbose=False):
        """
        Calculates the percentage of hemorrhage-related injuries that have been controlled in a given scene DataFrame.
        
        The percentage is based on the count of 'hemorrhage_control_procedures_list' procedures
        recorded in the 'injury_record_required_procedure' column compared to the count of those
        procedures being treated and marked as controlled in the 'injury_treated_required_procedure' column.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            float: Percentage of hemorrhage cases successfully controlled.
        
        Note:
            The logs have instances of a TOOL_APPLIED but no INJURY_TREATED preceding it. But, we already know the injury
            that the patient has and the correct tool for every patient because we assigned those ahead of time.
        """
        
        # Loop through each injury, examining its required procedures and wrong treatments
        hemorrhage_count = 0; controlled_count = 0
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            is_patient_dead = self.get_is_patient_dead(patient_df, verbose=verbose)
            if not is_patient_dead:
                for injury_id, injury_df in patient_df.groupby('injury_id'):
                    
                    # Check if an injury record or treatment exists for a hemorrhage-related procedure
                    is_injury_hemorrhage = self.get_is_injury_hemorrhage(injury_df, verbose=verbose)
                    if is_injury_hemorrhage:
                        
                        # Count any injuries requiring hemorrhage control procedures
                        hemorrhage_count += 1
                        
                        # Check if the injury was treated correctly
                        is_correctly_treated = self.get_is_injury_correctly_treated(injury_df, patient_df, verbose=verbose)
                        
                        # See if there are any tools applied that are associated with the hemorrhage injuries
                        is_tool_applied_correctly = self.get_is_hemorrhage_tool_applied(injury_df, patient_df, verbose=verbose)
                        
                        # Count any hemorrhage-related injuries that have been treated, and not wrong, and not counted twice
                        if is_correctly_treated or is_tool_applied_correctly: controlled_count += 1
        
        if verbose: print(f'Injuries requiring hemorrhage control procedures: {hemorrhage_count}')
        if verbose: print(f"Hemorrhage-related injuries that have been treated: {controlled_count}")
        
        # Calculate the percentage of controlled hemorrhage-related injuries
        try: percent_controlled = 100 * controlled_count / hemorrhage_count
        except ZeroDivisionError: percent_controlled = nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Percentage of hemorrhage controlled: {percent_controlled:.2f}%')
            columns_list = ['injury_required_procedure', 'injury_record_required_procedure', 'injury_treated_required_procedure', 'injury_treated_injury_treated_with_wrong_treatment']
            display(scene_df[columns_list].drop_duplicates())
        
        # Return the percentage of hemorrhage cases controlled
        return percent_controlled
    
    
    def get_time_to_last_hemorrhage_controlled(self, scene_df, verbose=False):
        """
        Calculate the time to the last hemorrhage controlled event for patients in the given scene DataFrame.
        
        The time is determined based on the 'hemorrhage_control_procedures_list' procedures being treated
        and marked as controlled. The function iterates through patients in the scene to find the maximum
        time to hemorrhage control.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The time to the last hemorrhage control action, or 0 if no hemorrhage control actions exist.
        """
        
        # Get the start time of the scene
        scene_start = self.get_scene_start(scene_df)
        
        # Initialize the last controlled time to 0
        last_controlled_time = 0
        
        # Iterate through patients in the scene
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            
            # Check if the patient is hemorrhaging and not dead
            if self.get_is_patient_hemorrhaging(patient_df, verbose=verbose) and not self.get_is_patient_dead(patient_df, verbose=verbose):
                
                # Get the time to hemorrhage control for the patient
                controlled_time = self.get_time_to_hemorrhage_control(patient_df, scene_start=scene_start, use_dead_alternative=False, verbose=verbose)
                
                # Update the last controlled time if the current controlled time is greater
                last_controlled_time = max(controlled_time, last_controlled_time)
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Time to last hemorrhage controlled: {last_controlled_time} milliseconds')
            display(scene_df)
        
        # Return the time to the last hemorrhage controlled event
        return last_controlled_time
    
    
    def get_time_to_hemorrhage_control_per_patient(self, scene_df, verbose=False):
        """
        According to the paper we define time to hemorrhage control per patient like this:
        Duration of time from when the patient was first approached by the participant until
        the time hemorrhage treatment was applied (with a tourniquet or wound packing)
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The time it takes to control hemorrhage for the scene, per patient, in action ticks.
        """
        
        # Iterate through patients in the scene
        times_list = []
        patient_count = 0
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            
            # Check if the patient is hemorrhaging and not dead
            if self.get_is_patient_hemorrhaging(patient_df, verbose=verbose) and not self.get_is_patient_dead(patient_df, verbose=verbose):
                
                action_tick = self.get_time_to_hemorrhage_control(patient_df, scene_start=self.get_first_patient_interaction(patient_df), use_dead_alternative=False)
                times_list.append(action_tick)
                patient_count += 1
        
        # Calculate the hemorrhage control per patient
        try: time_to_hemorrhage_control_per_patient = sum(times_list) / patient_count
        except ZeroDivisionError: time_to_hemorrhage_control_per_patient = nan
        
        return time_to_hemorrhage_control_per_patient
    
    
    def get_actual_engagement_order(self, scene_df, include_noninteracteds=False, verbose=False):
        """
        Get the chronological order of engagement starts for each patient in a scene.
        
        Parameters:
            - scene_df (pandas.DataFrame): DataFrame containing scene data, including patient IDs, action types,
              action ticks, location IDs, patient sorts, and DTR triage priority model predictions.
            - verbose (bool, optional): If True, prints debug information. Default is False.
        
        Returns:
            - engagement_order (list): List of tuples containing engagement information ordered chronologically:
                - patient_id (int): The ID of the patient.
                - engagement_start (int): The action tick at which the engagement started.
                - location_tuple ((int, int)): A tuple representing the (x, z) coordinates of the engagement location.
                - patient_sort (str or None): The patient's SORT designation, if available.
                - dtr_triage_priority_model_prediction (float): The patient's predicted priority
                - injury_severity (str): The patient's severity
        """
        engagement_starts_list = []
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            
            # Get the cluster ID, if available
            mask_series = ~patient_df.patient_sort.isnull()
            patient_sort = (
                patient_df[mask_series].sort_values('action_tick').iloc[-1].patient_sort
                if mask_series.any()
                else None
            )
            
            # Get the predicted priority
            if 'dtr_triage_priority_model_prediction' in patient_df.columns:
                mask_series = ~patient_df.dtr_triage_priority_model_prediction.isnull()
                predicted_priority = (
                    patient_df[mask_series].dtr_triage_priority_model_prediction.mean()
                    if mask_series.any()
                    else None
                )
            else: predicted_priority = None
            
            # Get the maximum injury severity
            injury_severity = self.get_maximum_injury_severity(patient_df)
            
            # Check if the responder even interacted with this patient
            mask_series = patient_df.action_type.isin(self.responder_negotiations_list)
            if mask_series.any():
                df = patient_df[mask_series].sort_values('action_tick')
                
                # Get the first engagement start that has a location
                mask_series = ~df.location_id.isnull()
                if mask_series.any():
                    engagement_start = df[mask_series].iloc[0].action_tick
                    engagement_location = eval(df[mask_series].iloc[0].location_id) # Evaluate string to get tuple
                    location_tuple = (engagement_location[0], engagement_location[2])
                else:
                    engagement_start = df.iloc[0].action_tick
                    location_tuple = (0.0, 0.0)
                
                # Add engagement information to the list
                engagement_tuple = (patient_id, engagement_start, location_tuple, patient_sort, predicted_priority, injury_severity)
                engagement_starts_list.append(engagement_tuple)
            
            # Add -99999 for engagement_start if you're including non-interacted-with patients
            elif include_noninteracteds:
                engagement_tuple = (patient_id, -99999, None, patient_sort, predicted_priority, injury_severity)
                engagement_starts_list.append(engagement_tuple)
        
        # Sort the starts list chronologically
        engagement_order = sorted(engagement_starts_list, key=lambda x: x[1], reverse=False)
        
        return engagement_order
    
    
    ### Patient Functions ###
    
    
    @staticmethod
    def get_is_patient_dead(patient_df, verbose=False):
        """
        Check if the patient is considered dead based on information in the given patient DataFrame.
        
        The function examines the 'patient_record_salt' and 'patient_engaged_salt' columns to
        determine if the patient is marked as 'DEAD' or 'EXPECTANT'. If both columns are empty,
        the result is considered unknown (NaN).
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool or numpy.nan: True if the patient is considered dead, False if not, and numpy.nan if unknown.
        """
        
        # Handle missing values in both patient_record_salt and patient_engaged_salt
        if patient_df.patient_record_salt.isnull().all() and patient_df.patient_engaged_salt.isnull().all(): is_patient_dead = nan
        else:
            
            # Check the patient_record_salt field
            mask_series = ~patient_df.patient_record_salt.isnull()
            
            # Check the patient_engaged_salt field if patient_record_salt is not available
            if mask_series.any():
                patient_record_salt = patient_df[mask_series].patient_record_salt.iloc[0]
                is_patient_dead = patient_record_salt in ['DEAD', 'EXPECTANT']
            else:
                
                # Check 'patient_engaged_salt' for patient status if 'patient_record_salt' is empty
                mask_series = ~patient_df.patient_engaged_salt.isnull()
                
                # If both columns are empty, the result is unknown
                if mask_series.any():
                    patient_engaged_salt = patient_df[mask_series].patient_engaged_salt.iloc[0]
                    is_patient_dead = patient_engaged_salt in ['DEAD', 'EXPECTANT']
                else: is_patient_dead = nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Is patient considered dead: {is_patient_dead}')
            display(patient_df)
        
        # Return True if the patient is considered dead, False if not, and numpy.nan if unknown
        return is_patient_dead
    
    
    def get_first_patient_interaction(self, patient_df, verbose=False):
        """
        Get the action tick of the first patient interaction of a specific type.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The action tick of the first responder negotiation action, or None if no such action exists.
        """
        
        # Filter for actions involving responder negotiations
        mask_series = patient_df.action_type.isin(self.responder_negotiations_list)
        
        # If there are responder negotiation actions, find the first action tick
        if mask_series.any(): engagement_start = patient_df[mask_series]['action_tick'].min()
        else: engagement_start = nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'First patient interaction: {engagement_start}')
            display(patient_df[mask_series].dropna(axis='columns', how='all').T)
        
        # Return the action tick of the first patient interaction or numpy.nan if no data is available
        return engagement_start
    
    
    def get_is_patient_hemorrhaging(self, patient_df, verbose=False):
        """
        Check if a patient is hemorrhaging based on injury record and required procedures.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool: True if the patient has an injury record indicating hemorrhage, False otherwise.
        """
        
        # Create a mask to check if injury record requires hemorrhage control procedures
        mask_series = patient_df.injury_required_procedure.isin(self.hemorrhage_control_procedures_list)
        is_hemorrhaging = mask_series.any()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Is the patient hemorrhaging: {is_hemorrhaging}')
            display(patient_df)
        
        return is_hemorrhaging
    
    
    def get_time_to_hemorrhage_control(self, patient_df, scene_start, use_dead_alternative=False, verbose=False):
        """
        Calculate the time it takes to control hemorrhage for the patient by getting the injury treatments
        where the responder is not confused from the bad feedback.
        
        According to the paper we define time to hemorrhage control like this:
        Duration of time from the scene start time until the time that the last patient who requires life
        threatening bleeding has been treated with hemorrhage control procedures (when the last
        tourniquet or wound packing was applied).
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            scene_start (int): The action tick of the first interaction with the patient.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The time it takes to control hemorrhage for the patient, in action ticks.
        """
        controlled_time = 0
        is_patient_dead = self.get_is_patient_dead(patient_df, verbose=verbose)
        
        # Use time to correct triage tag (black of gray) an alternative measure of proper treatment
        if is_patient_dead:
            if use_dead_alternative:
                
                # Get the last SALT value for the patient
                try:
                    mask_series = ~patient_df.patient_record_salt.isnull()
                    salt_srs = patient_df[mask_series].sort_values('action_tick').iloc[-1]
                    action_tick = salt_srs.action_tick
                    last_salt = salt_srs.patient_record_salt
                except Exception: last_salt = None
                if verbose: print(f'last_salt = {last_salt}')
                
                # Get the predicted tag based on the maximum salt value
                try: predicted_tag = self.salt_to_tag_dict.get(last_salt, None)
                except Exception: predicted_tag = None
                if verbose: print(f'predicted_tag = {predicted_tag}')
                
                # Update the maximum hemorrhage control time with the time to correct triage tag
                if predicted_tag in ['black', 'gray']:
                    controlled_time = max(controlled_time, action_tick - scene_start)
                    if verbose: print(f'controlled_time = {controlled_time}')
        
        else:
            
            # Define columns for merging
            on_columns_list = ['injury_id']
            merge_columns_list = ['action_tick'] + on_columns_list
            
            # Loop through hemorrhage control procedures
            for control_procedure in self.hemorrhage_control_procedures_list:
                
                # Identify hemorrhage events for the current procedure
                mask_series = (patient_df.injury_record_required_procedure == control_procedure)
                if mask_series.any():
                    hemorrhage_df = patient_df[mask_series][merge_columns_list]
                    if verbose: display(hemorrhage_df)
                    
                    # Identify controlled hemorrhage events for the current procedure
                    mask_series = (patient_df.injury_treated_required_procedure == control_procedure)
                    if not mask_series.any():
                        tool_type = self.required_procedure_to_tool_type_dict[control_procedure]
                        mask_series = patient_df.tool_applied_type.isnull() | patient_df.tool_applied_type.map(lambda x: x == tool_type)
                        controlled_df = patient_df[mask_series]
                        controlled_df.injury_id.fillna(method='ffill', inplace=True)
                        controlled_df.injury_id.fillna(method='bfill', inplace=True)
                        controlled_df = controlled_df[merge_columns_list]
                    else: controlled_df = patient_df[mask_series][merge_columns_list]
                    if verbose: display(controlled_df)
                    
                    # Merge hemorrhage and controlled hemorrhage events
                    df = hemorrhage_df.merge(controlled_df, on=on_columns_list, suffixes=('_hemorrhage', '_controlled'))
                    if verbose: display(df)
                    
                    # Update the maximum hemorrhage control time
                    controlled_time = max(controlled_time, df.action_tick_controlled.max() - scene_start)
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Action ticks to control hemorrhage for the patient: {controlled_time}')
            display(patient_df)
        
        return controlled_time
    
    
    @staticmethod
    def get_maximum_injury_severity(patient_df, verbose=False):
        """
        Compute the maximum injury severity from a DataFrame of patient data.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient data, including a column 'injury_severity'
                which indicates the severity of the injury.
            verbose (bool, optional): If True, print debug information. Default is False.
        
        Returns:
            maximum_injury_severity (float or None):
                The maximum severity of injury found in the DataFrame, or None if
                no valid severity values are present.
        
        Notes:
            This function assumes that the DataFrame contains a column named 'injury_severity'
            which represents the severity of injuries, with higher values indicating more severe
            injuries. If 'injury_severity' is missing or contains non-numeric values, this
            function will return None.
        """
        
        # Filter out rows where injury severity is missing
        mask_series = ~patient_df.injury_severity.isnull()
        
        # Find the minimum severity among valid values
        # since higher values indicate more severe injuries
        maximum_injury_severity = patient_df[mask_series].injury_severity.min()
        
        return maximum_injury_severity
    
    
    ### Injury Functions ###
    
    
    def get_is_injury_correctly_treated(self, injury_df, patient_df=None, verbose=False):
        """
        Determine whether the given injury was correctly treated.
        
        Parameters:
            injury_df (pandas.DataFrame): DataFrame containing injury-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool: True if the injury was correctly treated, False otherwise.
        
        Note:
            The FRVRS logger has trouble logging multiple tool applications,
            so injury_treated_injury_treated_with_wrong_treatment == True
            remains and another INJURY_TREATED is never logged, even though
            the right tool was applied after that.
        """
        mask_series = ~injury_df.injury_record_required_procedure.isnull()
        required_procedure = injury_df[mask_series].injury_record_required_procedure.mode().squeeze()
        if isinstance(required_procedure, Series):
            mask_series = ~injury_df.injury_treated_required_procedure.isnull()
            required_procedure = injury_df[mask_series].injury_treated_required_procedure.mode().squeeze()
        assert not isinstance(required_procedure, Series), "You have no required procedures"
        is_correctly_treated = (required_procedure == 'none')
        if (not is_correctly_treated) and (patient_df is None):
            
            # Create a mask to identify treated injuries
            mask_series = (injury_df.injury_treated_injury_treated == True)
            
            # Add to that mask to identify correctly treated injuries
            mask_series &= (injury_df.injury_treated_injury_treated_with_wrong_treatment == False)
            
            # Return True if there are correctly treated attempts, False otherwise
            is_correctly_treated = mask_series.any()
            
        elif patient_df is not None:
            millisecond_threshold = 3
            mask_series = (injury_df.action_type == 'INJURY_TREATED')
            action_ticks_list = sorted(injury_df[mask_series].action_tick.unique())
            for action_tick in action_ticks_list:
                mask_series = patient_df.action_tick.map(
                    lambda ts: abs(ts - action_tick) < millisecond_threshold
                ) & patient_df.action_type.isin(['TOOL_APPLIED'])
                if mask_series.any():
                    is_correctly_treated = any(
                        [(required_procedure == self.tool_type_to_required_procedure_dict.get(tool_type)) for tool_type in patient_df[mask_series].tool_applied_type]
                    )
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Was the injury correctly treated: {is_correctly_treated}')
            print('\n\n')
            display(injury_df.dropna(axis='columns', how='all').T)
        
        return is_correctly_treated
    
    
    def get_is_injury_hemorrhage(self, injury_df, verbose=False):
        """
        Determine whether the given injury is a hemorrhage based on the injury record or treatment record.
        
        Parameters:
            injury_df (pandas.DataFrame): DataFrame containing injury-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool: True if the injury is a hemorrhage, False otherwise.
        """
        
        # Check if either the injury record or treatment record indicates hemorrhage
        mask_series = injury_df.injury_required_procedure.isin(self.hemorrhage_control_procedures_list)
        is_hemorrhage = mask_series.any()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Is the injury a hemorrhage: {is_hemorrhage}')
            print('\n\n')
            display(injury_df.dropna(axis='columns', how='all').T)
        
        return is_hemorrhage
    
    
    def get_is_hemorrhage_tool_applied(self, injury_df, logs_df, verbose=False):
        """
        Checks if a hemorrhage control tool was applied for a given injury.
        
        Parameters:
            injury_df (pandas.DataFrame): DataFrame containing injury data. Must include columns for
                - patient_id (int): Unique identifier for the patient.
                - injury_id (int): Unique identifier for the injury. (Optional)
                - injury_record_required_procedure (str): Required procedure for the injury record.
            
            logs_df (pandas.DataFrame): DataFrame containing log records. Must include columns for
                - patient_id (int): Unique identifier for the patient.
                - tool_applied_type (str): Type of tool applied during the procedure.
            
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool: True if a hemorrhage control tool was applied for the injury, False otherwise.
        """
        
        # Get the entire patient record
        mask_series = ~injury_df.patient_id.isnull()
        patient_id = injury_df[mask_series].patient_id.tolist()[0]
        mask_series = (logs_df.patient_id == patient_id)
        patient_df = logs_df[mask_series]
        
        # See if there are any tools applied that are associated with the hemorrhage injuries
        is_tool_applied_correctly = False
        record_mask_series = injury_df.injury_record_required_procedure.isin(self.hemorrhage_control_procedures_list)
        for required_procedure in injury_df[record_mask_series].injury_record_required_procedure.unique():
            tool_type = self.required_procedure_to_tool_type_dict[required_procedure]
            is_tool_applied_correctly = is_tool_applied_correctly or patient_df.tool_applied_type.map(lambda x: x == tool_type).any()
        
        if verbose:
            
            # Get the injury ID
            mask_series = ~injury_df.injury_id.isnull()
            injury_id = injury_df[mask_series].injury_id.tolist()[0]
            
            print(f'A hemorrhage-related TOOL_APPLIED event can be associated with the injury ({injury_id}): {is_tool_applied_correctly}')
        
        return is_tool_applied_correctly
    
    
    ### Rasch Analysis Scene Functions ###
    
    
    def get_stills_value(self, scene_df, verbose=False):
        """
        0=All Stills not visited first, 1=All Stills visited first
        """
        
        # Extract the actual and ideal sequences of first interactions from the scene in terms of still/waver/walker
        actual_sequence, ideal_sequence, sort_dict = self.get_actual_and_ideal_patient_sort_sequences(scene_df, verbose=verbose)
        
        # Truncate both sequences to the head at the stills length and compare them; they both should have all stills
        still_len = len(sort_dict.get('still', []))
        ideal_sequence = ideal_sequence.tolist()[:still_len]
        actual_sequence = actual_sequence.tolist()[:still_len]
        
        # If they are, output a 1 (All Stills visited first), if not, output a 0 (All Stills not visited first)
        is_stills_visited_first = int(actual_sequence == ideal_sequence)
        
        return is_stills_visited_first
    
    
    def get_walkers_value(self, scene_df, verbose=False):
        """
        0=All Walkers not visited last, 1=All Walkers visited last
        """
        
        # Extract the actual and ideal sequences of first interactions from the scene in terms of still/waver/walker
        actual_sequence, ideal_sequence, sort_dict = self.get_actual_and_ideal_patient_sort_sequences(scene_df, verbose=verbose)
        
        # Truncate both sequences to the tail at the walkers length and compare them; they both should have all walkers
        walker_len = len(sort_dict.get('walker', []))
        ideal_sequence = ideal_sequence.tolist()[-walker_len:]
        actual_sequence = actual_sequence.tolist()[-walker_len:]
        
        # If they are, output a 1 (All Walkers visited last), if not, output a 0 (All Walkers not visited last)
        is_walkers_visited_last = int(actual_sequence == ideal_sequence)
        
        return is_walkers_visited_last
    
    
    @staticmethod
    def get_wave_value(scene_df, verbose=False):
        """
        0=No Wave Command issued, 1=Wave Command issued
        """
        
        # Check in the scene if there are any WAVE_IF_CAN actions
        mask_series = scene_df.action_type.isin(['S_A_L_T_WAVE_IF_CAN', 'TRIAGE_LEVEL_WAVE_IF_CAN'])
        
        # If there are, output a 1 (Wave Command issued), if not, output a 0 (No Wave Command issued)
        is_wave_command_issued = int(scene_df[mask_series].shape[0] > 0)
        
        return is_wave_command_issued
    
    
    @staticmethod
    def get_walk_value(scene_df, verbose=False):
        """
        0=No Walk Command issued, 1=Walk Command issued
        """
        
        # Check in the scene if there are any WALK_IF_CAN actions
        mask_series = scene_df.action_type.isin(['S_A_L_T_WALK_IF_CAN', 'TRIAGE_LEVEL_WALK_IF_CAN'])
        
        # If there are, output a 1 (Walk Command issued), if not, output a 0 (No Walk Command issued)
        is_walk_command_issued = int(scene_df[mask_series].shape[0] > 0)
        
        return is_walk_command_issued
    
    
    ### Rasch Analysis Patient Functions ###
    
    
    ### Pandas Functions ###
    
    
    @staticmethod
    def set_scene_indices(file_df):
        """
        Section off player actions by session start and end. We are finding log entries above the first SESSION_START and below the last SESSION_END.
        
        Parameters:
            file_df: A Pandas DataFrame containing the player action data with its index reset.
        
        Returns:
            A Pandas DataFrame with the `scene_id` and `is_scene_aborted` columns added.
        """
    
        # Set the whole file to zero first
        file_df = file_df.sort_values('action_tick')
        scene_id = 0
        file_df['scene_id'] = scene_id
        file_df['is_scene_aborted'] = False
        
        # Delineate runs by the session end below them
        mask_series = (file_df.action_type == 'SESSION_END')
        lesser_idx = file_df[mask_series].index.min()
        mask_series &= (file_df.index > lesser_idx)
        while mask_series.any():
            
            # Find this session end as the bottom
            greater_idx = file_df[mask_series].index.min()
            
            # Add everything above that to this run
            mask_series = (file_df.index > lesser_idx) & (file_df.index <= greater_idx)
            scene_id += 1
            file_df.loc[mask_series, 'scene_id'] = scene_id
            
            # Delineate runs by the session end below them
            lesser_idx = greater_idx
            mask_series = (file_df.action_type == 'SESSION_END') & (file_df.index > lesser_idx)
        
        # Find the last session end
        mask_series = (file_df.action_type == 'SESSION_END')
        lesser_idx = file_df[mask_series].index.max()
        
        # Add everything below that to the next run
        mask_series = (file_df.index > lesser_idx)
        file_df.loc[mask_series, 'scene_id'] = scene_id + 1
        
        # The session end command signifies the end of the session and anything after that is junk
        file_df.loc[mask_series, 'is_scene_aborted'] = True
        
        # Convert the scene index column to int64
        file_df.scene_id = file_df.scene_id.astype('int64')
        
        return file_df
    
    
    def set_mcivr_metrics_types(self, action_type, df, row_index, row_series, verbose=False):
        """
        Ingest all the auxiliary data based on action type out of numbered columns into
        named columns by setting the MCI-VR metrics types for a given action type and
        row series.
    
        Parameters:
            action_type: The action type.
            df: The DataFrame containing the MCI-VR metrics.
            row_index: The index of the row in the DataFrame to set the metrics for.
            row_series: The row series containing the MCI-VR metrics.
    
        Returns:
            The DataFrame containing the MCI-VR metrics with new columns.
        """
        
        # List of indexes to delete
        drop_index_list = []
        
        # Set the metrics types for each action type
        if (action_type == 'BAG_ACCESS'): # BagAccess
            df.loc[row_index, 'bag_access_location'] = row_series[4] # Location
        elif (action_type == 'BAG_CLOSED'): # BagClosed
            df.loc[row_index, 'bag_closed_location'] = row_series[4] # Location
        elif (action_type == 'INJURY_RECORD'): # InjuryRecord
            df.loc[row_index, 'injury_record_id'] = row_series[4] # Id
            df.loc[row_index, 'injury_record_patient_id'] = row_series[5] # patientId
            df.loc[row_index, 'injury_record_required_procedure'] = row_series[6] # requiredProcedure
            df.loc[row_index, 'injury_record_severity'] = row_series[7] # severity
            df.loc[row_index, 'injury_record_body_region'] = row_series[8] # bodyRegion
            df.loc[row_index, 'injury_record_injury_treated'] = row_series[9] # injuryTreated
            df.loc[row_index, 'injury_record_injury_treated_with_wrong_treatment'] = row_series[10] # injuryTreatedWithWrongTreatment
            df.loc[row_index, 'injury_record_injury_injury_locator'] = row_series[11] # injuryLocator
        elif (action_type == 'INJURY_TREATED'): # InjuryTreated
            df.loc[row_index, 'injury_treated_id'] = row_series[4] # Id
            df.loc[row_index, 'injury_treated_patient_id'] = row_series[5] # patientId
            df.loc[row_index, 'injury_treated_required_procedure'] = row_series[6] # requiredProcedure
            df.loc[row_index, 'injury_treated_severity'] = row_series[7] # severity
            df.loc[row_index, 'injury_treated_body_region'] = row_series[8] # bodyRegion
            df.loc[row_index, 'injury_treated_injury_treated'] = row_series[9] # injuryTreated
            df.loc[row_index, 'injury_treated_injury_treated_with_wrong_treatment'] = row_series[10] # injuryTreatedWithWrongTreatment
            df.loc[row_index, 'injury_treated_injury_injury_locator'] = row_series[11] # injuryLocator
        elif (action_type == 'PATIENT_DEMOTED'): # PatientDemoted
            df.loc[row_index, 'patient_demoted_health_level'] = row_series[4] # healthLevel
            df.loc[row_index, 'patient_demoted_health_time_remaining'] = row_series[5] # healthTimeRemaining
            df.loc[row_index, 'patient_demoted_id'] = row_series[6] # id
            df.loc[row_index, 'patient_demoted_position'] = row_series[7] # position
            df.loc[row_index, 'patient_demoted_rotation'] = row_series[8] # rotation
            df.loc[row_index, 'patient_demoted_salt'] = row_series[9] # salt
            df.loc[row_index, 'patient_demoted_sort'] = row_series[10] # sort
            df.loc[row_index, 'patient_demoted_pulse'] = row_series[11] # pulse
            df.loc[row_index, 'patient_demoted_breath'] = row_series[12] # breath
            df.loc[row_index, 'patient_demoted_hearing'] = row_series[13] # hearing
            df.loc[row_index, 'patient_demoted_mood'] = row_series[14] # mood
            df.loc[row_index, 'patient_demoted_pose'] = row_series[15] # pose
        elif (action_type == 'PATIENT_ENGAGED'): # PatientEngaged
            df.loc[row_index, 'patient_engaged_health_level'] = row_series[4] # healthLevel
            df.loc[row_index, 'patient_engaged_health_time_remaining'] = row_series[5] # healthTimeRemaining
            df.loc[row_index, 'patient_engaged_id'] = row_series[6] # id
            df.loc[row_index, 'patient_engaged_position'] = row_series[7] # position
            df.loc[row_index, 'patient_engaged_rotation'] = row_series[8] # rotation
            df.loc[row_index, 'patient_engaged_salt'] = row_series[9] # salt
            df.loc[row_index, 'patient_engaged_sort'] = row_series[10] # sort
            df.loc[row_index, 'patient_engaged_pulse'] = row_series[11] # pulse
            df.loc[row_index, 'patient_engaged_breath'] = row_series[12] # breath
            df.loc[row_index, 'patient_engaged_hearing'] = row_series[13] # hearing
            df.loc[row_index, 'patient_engaged_mood'] = row_series[14] # mood
            df.loc[row_index, 'patient_engaged_pose'] = row_series[15] # pose
        elif (action_type == 'BREATHING_CHECKED'):
            df.loc[row_index, 'patient_checked_breath'] = row_series[4]
            df.loc[row_index, 'patient_checked_id'] = row_series[5]
        elif (action_type == 'PATIENT_RECORD'): # PatientRecord
            df.loc[row_index, 'patient_record_health_level'] = row_series[4] # healthLevel
            df.loc[row_index, 'patient_record_health_time_remaining'] = row_series[5] # healthTimeRemaining
            df.loc[row_index, 'patient_record_id'] = row_series[6] # id
            df.loc[row_index, 'patient_record_position'] = row_series[7] # position
            df.loc[row_index, 'patient_record_rotation'] = row_series[8] # rotation
            df.loc[row_index, 'patient_record_salt'] = row_series[9] # salt
            df.loc[row_index, 'patient_record_sort'] = row_series[10] # sort
            df.loc[row_index, 'patient_record_pulse'] = row_series[11] # pulse
            df.loc[row_index, 'patient_record_breath'] = row_series[12] # breath
            df.loc[row_index, 'patient_record_hearing'] = row_series[13] # hearing
            df.loc[row_index, 'patient_record_mood'] = row_series[14] # mood
            df.loc[row_index, 'patient_record_pose'] = row_series[15] # pose
        elif (action_type == 'PULSE_TAKEN'): # PulseTaken
            df.loc[row_index, 'pulse_taken_pulse_name'] = row_series[4] # pulseName
            df.loc[row_index, 'pulse_taken_patient_id'] = row_series[5] # patientId
        elif (action_type == 'SP_O2_TAKEN'):
            df.loc[row_index, 'sp_o2_taken_level'] = row_series[4]
            df.loc[row_index, 'sp_o2_taken_patient_id'] = row_series[5]
        elif (action_type == 'S_A_L_T_WALKED'): # SALTWalked
            df.loc[row_index, 's_a_l_t_walked_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_walked_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_walked_patient_id'] = row_series[6]
        elif (action_type == 'TRIAGE_LEVEL_WALKED'):
            df.loc[row_index, 'triage_level_walked_location'] = row_series[4]
            df.loc[row_index, 'triage_level_walked_command_text'] = row_series[5]
            df.loc[row_index, 'triage_level_walked_patient_id'] = row_series[6]
        elif (action_type == 'S_A_L_T_WALK_IF_CAN'): # SALTWalkIfCan
            df.loc[row_index, 's_a_l_t_walk_if_can_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_walk_if_can_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_walk_if_can_patient_id'] = row_series[6] # patientId
        elif (action_type == 'TRIAGE_LEVEL_WALK_IF_CAN'):
            df.loc[row_index, 'triage_level_walk_if_can_location'] = row_series[4]
            df.loc[row_index, 'triage_level_walk_if_can_command_text'] = row_series[5]
            df.loc[row_index, 'triage_level_walk_if_can_patient_id'] = row_series[6]
        elif (action_type == 'S_A_L_T_WAVED'): # SALTWave
            df.loc[row_index, 's_a_l_t_waved_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_waved_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_waved_patient_id'] = row_series[6] # patientId
        elif (action_type == 'TRIAGE_LEVEL_WAVED'):
            df.loc[row_index, 'triage_level_waved_location'] = row_series[4]
            df.loc[row_index, 'triage_level_waved_command_text'] = row_series[5]
            df.loc[row_index, 'triage_level_waved_patient_id'] = row_series[6]
        elif (action_type == 'S_A_L_T_WAVE_IF_CAN'): # SALTWaveIfCan
            df.loc[row_index, 's_a_l_t_wave_if_can_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_wave_if_can_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_wave_if_can_patient_id'] = row_series[6] # patientId
        elif (action_type == 'TRIAGE_LEVEL_WAVE_IF_CAN'):
            df.loc[row_index, 'triage_level_wave_if_can_location'] = row_series[4]
            df.loc[row_index, 'triage_level_wave_if_can_command_text'] = row_series[5]
            df.loc[row_index, 'triage_level_wave_if_can_patient_id'] = row_series[6]
        elif (action_type == 'TAG_APPLIED'): # TagApplied
            df.loc[row_index, 'tag_applied_patient_id'] = row_series[4] # patientId
            df.loc[row_index, 'tag_applied_type'] = row_series[5] # type
        elif (action_type == 'TAG_DISCARDED'): # TagDiscarded
            df.loc[row_index, 'tag_discarded_type'] = row_series[4] # Type
            df.loc[row_index, 'tag_discarded_location'] = row_series[5] # Location
        elif (action_type == 'TAG_SELECTED'): # TagSelected
            df.loc[row_index, 'tag_selected_type'] = row_series[4] # Type
        elif (action_type == 'TELEPORT'): # Teleport
            df.loc[row_index, 'teleport_location'] = row_series[4] # Location
        elif (action_type == 'TOOL_APPLIED'): # ToolApplied
            if verbose: df.loc[row_index, 'tool_applied_row_shape'] = row_series.shape
            tool_applied_patient_id = row_series[4]
            if ' Root' in tool_applied_patient_id:
                df.loc[row_index, 'tool_applied_patient_id'] = tool_applied_patient_id # patientId
            df.loc[row_index, 'tool_applied_type'] = row_series[5] # type
            df.loc[row_index, 'tool_applied_attachment_point'] = row_series[6] # attachmentPoint
            df.loc[row_index, 'tool_applied_tool_location'] = row_series[7] # toolLocation
            
            # Find the attachMessage and infer if a data column exists from there
            for attach_message_idx in range(10, 8, -1):
                if str(row_series[attach_message_idx]).startswith('Applied'): break
            sender_idx = attach_message_idx - 1
            if (sender_idx == 9): df.loc[row_index, 'tool_applied_data'] = row_series[8] # data
            
            df.loc[row_index, 'tool_applied_sender'] = row_series[sender_idx] # sender
            df.loc[row_index, 'tool_applied_attach_message'] = row_series[attach_message_idx] # attachMessage
        elif (action_type == 'TOOL_DISCARDED'): # ToolDiscarded
            df.loc[row_index, 'tool_discarded_type'] = row_series[4] # Type
            df.loc[row_index, 'tool_discarded_count'] = row_series[5] # Count
            df.loc[row_index, 'tool_discarded_location'] = row_series[6] # Location
        elif (action_type == 'TOOL_HOVER'): # ToolHover
            df.loc[row_index, 'tool_hover_type'] = row_series[4] # Type
            df.loc[row_index, 'tool_hover_count'] = row_series[5] # Count
        elif (action_type == 'TOOL_SELECTED'): # ToolSelected
            df.loc[row_index, 'tool_selected_type'] = row_series[4] # Type
            df.loc[row_index, 'tool_selected_count'] = row_series[5] # Count
        elif (action_type == 'VOICE_CAPTURE'): # VoiceCapture
            df.loc[row_index, 'voice_capture_message'] = row_series[4] # Message
            df.loc[row_index, 'voice_capture_command_description'] = row_series[5] # commandDescription
        elif (action_type == 'VOICE_COMMAND'): # VoiceCommand
            df.loc[row_index, 'voice_command_message'] = row_series[4] # Message
            df.loc[row_index, 'voice_command_command_description'] = row_series[5] # commandDescription
        elif (action_type == 'BUTTON_CLICKED'):
            df.loc[row_index, 'button_command_message'] = row_series[4]
        elif (action_type == 'PLAYER_LOCATION'): # PlayerLocation
            df.loc[row_index, 'player_location_location'] = row_series[4] # Location (x,y,z)
            df.loc[row_index, 'player_location_left_hand_location'] = row_series[5] # Left Hand Location (x,y,z); deactivated in v1.3
            df.loc[row_index, 'player_location_right_hand_location'] = row_series[6] # Right Hand Location (x,y,z); deactivated in v1.3
        elif (action_type == 'PLAYER_GAZE'): # PlayerGaze
            if ' Root' in row_series[4]:
                df.loc[row_index, 'player_gaze_patient_id'] = row_series[4] # PatientID
                df.loc[row_index, 'player_gaze_location'] = row_series[5] # Location (x,y,z)
            elif ' Root' in row_series[5]:
                df.loc[row_index, 'player_gaze_location'] = row_series[4] # Location (x,y,z)
                df.loc[row_index, 'player_gaze_patient_id'] = row_series[5] # PatientID
            else:
                print(row_series); raise
            df.loc[row_index, 'player_gaze_distance_to_patient'] = row_series[6] # Distance to Patient
            df.loc[row_index, 'player_gaze_direction_of_gaze'] = row_series[7] # Direction of Gaze (vector3)
        elif (action_type == 'Participant ID'):
            drop_index_list.append(row_index)
        elif action_type not in self.known_mcivr_metrics_types:
            raise Exception(f"\n\n{action_type} not in in self.known_mcivr_metrics_types:\n{row_series}")
        
        # Delete rows at specified indexes
        df = df.drop(index=drop_index_list)
        
        return df
    
    
    def process_files(self, sub_directory_df, sub_directory, file_name, verbose=False):
        """
        Process files and update the subdirectory dataframe.
        
        This function takes a Pandas DataFrame representing the subdirectory and the name of the file to process.
        It then reads the file, parses it into a DataFrame, and appends the DataFrame to the subdirectory DataFrame.
        
        Parameters:
            sub_directory_df (pandas.DataFrame): The DataFrame for the current subdirectory.
            sub_directory (str): The path of the current subdirectory.
            file_name (str): The name of the file to process.
        
        Returns:
            pandas.DataFrame: The updated subdirectory DataFrame.
        """
        
        # Construct the full path to the file and get the logger version
        file_path = osp.join(sub_directory, file_name)
        version_number = self.get_logger_version(file_path, verbose=verbose)
        
        if IS_DEBUG: print("Read CSV file using a CSV reader")
        rows_list = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for values_list in reader:
                if (values_list[-1] == ''): values_list.pop(-1)
                rows_list.append({i: v for i, v in enumerate(values_list)})
        file_df = DataFrame(rows_list)
        
        # Ignore small files and return the subdirectory data frame unharmed
        if (file_df.shape[1] < 16): return sub_directory_df
        
        # Remove column 4 and rename all the numbered colums above that
        if (version_number > 1.0):
            file_df.drop(4, axis='columns', inplace=True)
            file_df.columns = list(range(file_df.shape[1]))
        
        # Add file name and logger version to the data frame
        file_dir_suffix = osp.abspath(sub_directory).replace(osp.abspath(self.data_logs_folder) + sep, '')
        file_df['csv_file_subpath'] = '/'.join(file_dir_suffix.split(sep)) + '/' + file_name
        file_df['logger_version'] = float(version_number)
        
        # Name the global columns
        columns_list = ['action_type', 'action_tick', 'event_time', 'session_uuid']
        file_df.columns = columns_list + file_df.columns.tolist()[len(columns_list):]
        
        # Parse the third column as a date column
        if ('event_time' in file_df.columns):
            
            # Attempt to infer the format automatically
            if sub_directory.endswith('v.1.0'): file_df['event_time'] = to_datetime(file_df['event_time'], format='%m/%d/%Y %H:%M')
            else: file_df['event_time'] = to_datetime(file_df['event_time'], infer_datetime_format=True)
        
        if IS_DEBUG: print("Set the MCIVR metrics types")
        for row_index, row_series in file_df.iterrows(): file_df = self.set_mcivr_metrics_types(row_series.action_type, file_df, row_index, row_series, verbose=verbose)
        
        if IS_DEBUG: print("Section off player actions by session start and end")
        file_df = self.set_scene_indices(file_df.reset_index(drop=True))
        
        # Append the data frame for the current file to the data frame for the current subdirectory
        sub_directory_df = concat([sub_directory_df, file_df], axis='index')
        
        return sub_directory_df
    
    
    def concatonate_logs(self, logs_folder=None, verbose=False):
        """
        Concatenates all the CSV files in the given logs folder into a single data frame.
        
        Parameters
        ----------
        logs_folder : str, optional
            The path to the folder containing the CSV files. The default value is the data logs folder.
        
        Returns
        -------
        DataFrame
            A DataFrame containing all the data from the CSV files in the given logs folder.
        """
        logs_df = DataFrame([])
        
        # Iterate over the subdirectories, directories, and files in the logs folder
        if logs_folder is None: logs_folder = self.data_logs_folder
        for sub_directory, directories_list, files_list in walk(logs_folder):
            
            # Create a data frame to store the data for the current subdirectory
            sub_directory_df = DataFrame([])
            
            # Iterate over the files in the current subdirectory
            for file_name in files_list:
                
                # If the file is a CSV file, merge it into the subdirectory data frame
                if file_name.endswith('.csv'): sub_directory_df = self.process_files(sub_directory_df, sub_directory, file_name, verbose=verbose)
            
            # Append the data frame for the current subdirectory to the main data frame
            logs_df = concat([logs_df, sub_directory_df], axis='index')
        
        # Convert event time to a datetime
        if ('event_time' in logs_df.columns): logs_df['event_time'] = to_datetime(logs_df['event_time'], format='mixed')
        
        # Convert elapsed time to an integer
        if ('action_tick' in logs_df.columns):
            logs_df.action_tick = to_numeric(logs_df.action_tick, errors='coerce')
            mask_series = ~logs_df.action_tick.isnull()
            logs_df = logs_df[mask_series]
            logs_df.action_tick = logs_df.action_tick.astype('int64')
        
        logs_df = logs_df.reset_index(drop=True)
        
        return logs_df
    
    
    def convert_column_to_categorical(self, column_name, df, verbose=False):
        if (column_name in df.columns):
            name_parts_list = column_name.split('_')
            
            # Find the order attribute
            attribute_name = 'XXXX'
            for i in range(3):
                if not hasattr(self, attribute_name):
                    attribute_name = f"{'_'.join(name_parts_list[i:])}_order"
                else:
                    break
        
            # Check if the order attribute exists
            if hasattr(self, attribute_name):
                
                # Check for missing elements
                mask_series = ~df[column_name].isnull()
                feature_set = set(df[mask_series][column_name].unique())
                order_set = set(eval(f"self.{attribute_name}"))
                assert feature_set.issubset(order_set), f"You're missing {feature_set.difference(order_set)} from self.{attribute_name}"
                
                # Find the category attribute
                attribute_name = 'XXXX'
                for i in range(3):
                    if not hasattr(self, attribute_name):
                        attribute_name = f"{'_'.join(name_parts_list[i:])}_category_order"
                    else:
                        break
                
                # Check if the category attribute exists
                if hasattr(self, attribute_name):
                    if verbose: print(f"\nConvert {column_name} column to categorical")
                    df[column_name] = df[column_name].astype(eval(f"self.{attribute_name}"))
                else:
                    if verbose: print(f"AttributeError: 'FRVRSUtilities' object has no attribute '{attribute_name}'")
                
            if verbose: print(df[column_name].nunique())
            if verbose: display(df.groupby(column_name).size().to_frame().rename(columns={0: 'record_count'}).sort_values('record_count', ascending=False).head(20))
        
        return df
    
    
    def add_modal_column(self, new_column_name, df, verbose=False):
        if (new_column_name not in df.columns):
            name_parts_list = new_column_name.split('_')
            if verbose: print(f"\nModalize into one {' '.join(name_parts_list)} column if possible")
            
            # Find the columns list attribute
            attribute_name = 'XXXX'
            for i in range(3):
                if not hasattr(self, attribute_name):
                    attribute_name = f"{'_'.join(name_parts_list[i:])}_columns_list"
                else:
                    break

            df = nu.modalize_columns(df, eval(f"self.{attribute_name}"), new_column_name)
            df = self.convert_column_to_categorical(df, new_column_name, verbose=verbose)
        
        return df
    
    
    ### Plotting Functions ###
    
    
    ### Open World Functions ###
    
    
    def add_encounter_layout_column(self, csv_stats_df, json_stats_df, verbose=False):
        """
        Add a new column to the JSON statistics DataFrame indicating the environment of each encounter.
        
        Parameters:
            csv_stats_df (pandas.DataFrame):
                DataFrame containing statistics from CSV files.
            json_stats_df (pandas.DataFrame):
                DataFrame containing statistics from JSON files.
            verbose (bool, optional):
                If True, print verbose output, by default False.
        
        Returns:
            None
        """
        if verbose: 
            
            # Print the shape of the CSV stats DataFrame
            print(csv_stats_df.shape)
        
        new_column_name = 'encounter_layout'
        
        # Loop through each session and scene in the CSV stats dataset
        for (session_uuid, scene_id), scene_df in csv_stats_df.groupby(self.scene_groupby_columns):
            if verbose:
                
                # Print the unique patient IDs for each scene
                print(scene_df.patient_id.unique().tolist())
            
            # Loop through each environment and get the patients list for that environment
            for env_str in ['desert', 'jungle', 'submarine', 'urban']:
                patients_list = eval(f'{env_str}_patients_list')
                
                # Check if all patients are in that scene
                if all(map(lambda patient_id: patient_id in scene_df.patient_id.unique().tolist(), patients_list)):
                    
                    # If so, find the corresponding session in the JSON stats dataset and add that environment to it as a new column
                    mask_series = (json_stats_df.session_uuid == session_uuid)
                    json_stats_df.loc[mask_series, new_column_name] = env_str.title()
        
        if verbose:
            
            # Print the shape of the JSON stats DataFrame
            print(json_stats_df.shape) # (43, 3541)
            
            # Display the count of records for each environment
            display(json_stats_df.groupby(new_column_name, dropna=False).size().to_frame().rename(columns={0: 'record_count'}))
    
    
    def add_medical_role_column(self, json_stats_df, anova_df, verbose=False):
        """
        Add a medical role column to the dataframe by merging with JSON stats.
        
        Parameters:
            json_stats_df (pandas.DataFrame):
                Dataframe containing JSON statistics.
            anova_df (pandas.DataFrame):
                Dataframe to merge with JSON statistics dataframe.
            verbose (bool, optional):
                If True, print verbose output, by default False.
        
        Returns:
            None
        """
        
        # Use the "MedRole" key from the JSON stats to determine the integer value
        new_column = 'MedRole'
        column_name = 'medical_role'
        if new_column in json_stats_df.columns:
            
            # Determine the intersection of JSON stats columns and the columns in your dataframe to merge on (usually participant_id and session_uuid)
            on_columns = sorted(set(anova_df.columns).intersection(set(json_stats_df.columns)))
            
            # Filter only those "merge on" columns and the "MedRole" column on the right side of the merge in order for your dataframe to do a left outer join with the JSON stats dataframe
            columns_list = on_columns + [new_column]
            anova_df = anova_df.merge(
                json_stats_df[columns_list], on=on_columns, how='left'
            ).rename(columns={new_column: column_name})
            
            # Decode the integer value by means of the Labels column in the Metrics_Evaluation_Dataset_organization_for_BBAI spreadsheet provided by CACI and map that to the new column
            anova_df[column_name] = anova_df[column_name].map(
                lambda cv: get_value_description('MedRole', cv)
            ).replace('', nan)
            
            if verbose: print(anova_df.shape)
            if verbose: print(anova_df.columns.tolist())
            if verbose: display(anova_df.groupby(column_name).size().to_frame().rename(columns={0: 'record_count'}).sort_values(
                'record_count', ascending=False
            ).head(5))
    
    
    def get_configData_scenarioData_difficulty(self, json_stats_df, anova_df, verbose=False):
        """
        Calculate the average difficulty level from the scenarioData dictionary within the configData dictionary 
        of the JSON data for a session and participant.
        
        Parameters:
            json_stats_df (pandas.DataFrame):
                DataFrame containing JSON data for the session and participant.
            anova_df (pandas.DataFrame):
                DataFrame containing data for ANOVA analysis.
            verbose (bool, optional):
                If True, print debug output. Default is False.
        
        Returns:
            float: The average difficulty level.
        """
        
        # Find the scenarioData dictionary within the configData dictionary of the JSON data for that session and participant
        
        # Inside, you will find three keys: description, difficulty, and name. The difficulty key is semi-continously numeric, and you can average it for whatever grouping you need
        pass
    
    
    def add_prioritize_severity_column(self, merge_df, new_column_name='prioritize_high_injury_severity_patients', verbose=False):
        """
        Adds a new column to the DataFrame indicating if a row (patient) has the highest injury severity among engaged patients (engaged_patientXX_injury_severity).
        
        Parameters:
            merge_df (pandas.DataFrame):
                The DataFrame to add the new column to.
            new_column_name (str, optional):
                The name of the new column to be added. Defaults to 'prioritize_high_injury_severity_patients'.
            verbose (bool, optional):
                If True, print debug messages. Default is False.
        
        Returns:
            tuple:
                A tuple containing the modified DataFrame and a list of the newly added columns:
                - pandas.DataFrame: The modified DataFrame with the new column added.
                - list: A list containing the name of the newly added column ([new_column_name]).
        
        Raises:
            ValueError:
                If the new column name already exists in the DataFrame.
        
        Notes:
            - If the specified new_column_name already exists in merge_df, the function does nothing.
            - The new column will contain 1 for patients with the highest injury severity among the provided injury severity columns, and 0 otherwise.
        """
        if new_column_name in merge_df.columns:
            raise ValueError(f"Column '{new_column_name}' already exists in the DataFrame.")
        if 'engaged_patient00_injury_severity' not in merge_df.columns:
            rows_list = []
            for (session_uuid, scene_id), scene_df in merge_df.groupby(self.scene_groupby_columns):
                row_dict = {}
                for cn in self.scene_groupby_columns: row_dict[cn] = eval(cn)
                
                # Get the chronological order of engagement starts for each patient in the scene
                engagement_starts_list = []
                for patient_id, patient_df in scene_df.groupby('patient_id'):
                    
                    # Get the cluster ID, if available
                    mask_series = ~patient_df.patient_sort.isnull()
                    patient_sort = (
                        patient_df[mask_series].sort_values('action_tick').iloc[-1].patient_sort
                        if mask_series.any()
                        else None
                    )
                    
                    # Get the predicted priority
                    if 'dtr_triage_priority_model_prediction' in patient_df.columns:
                        mask_series = ~patient_df.dtr_triage_priority_model_prediction.isnull()
                        predicted_priority = (
                            patient_df[mask_series].dtr_triage_priority_model_prediction.mean()
                            if mask_series.any()
                            else None
                        )
                    else: predicted_priority = None
                    
                    # Get the maximum injury severity
                    injury_severity = self.get_maximum_injury_severity(patient_df)
                    
                    # Check if the responder even interacted with this patient
                    mask_series = patient_df.action_type.isin(self.responder_negotiations_list)
                    if mask_series.any():
                        df = patient_df[mask_series].sort_values('action_tick')
                        
                        # Get the first engagement start that has a location
                        mask_series = ~df.location_id.isnull()
                        if mask_series.any():
                            engagement_start = df[mask_series].iloc[0].action_tick
                            engagement_location = eval(df[mask_series].iloc[0].location_id) # Evaluate string to get tuple
                            location_tuple = (engagement_location[0], engagement_location[2])
                        else:
                            engagement_start = df.iloc[0].action_tick
                            location_tuple = (0.0, 0.0)
                        
                        # Add engagement information to the list
                        engagement_tuple = (patient_id, engagement_start, location_tuple, patient_sort, predicted_priority, injury_severity)
                        engagement_starts_list.append(engagement_tuple)
                    
                    # Add -99999 for engagement_start if you're including non-interacted-with patients
                    else:
                        engagement_tuple = (patient_id, -99999, None, patient_sort, predicted_priority, injury_severity)
                        engagement_starts_list.append(engagement_tuple)
                
                # Sort the starts list chronologically
                actual_engagement_order = sorted(engagement_starts_list, key=lambda x: x[1], reverse=False)
                
                assert len(actual_engagement_order) == self.get_patient_count(scene_df), f"There are {patient_count} patients in this scene and only {len(actual_engagement_order)} engagement tuples:\n{scene_df[~scene_df.patient_id.isnull()].patient_id.unique().tolist()}\n{actual_engagement_order}"
                
                unengaged_patient_count = 0; engaged_patient_count = 0
                for engagement_tuple in actual_engagement_order:
                    if engagement_tuple[1] < 0:
                        column_name = f'unengaged_patient{unengaged_patient_count:0>2}_metadata'
                        unengaged_patient_count += 1
                    else:
                        column_name = f'engaged_patient{engaged_patient_count:0>2}_metadata'
                        engaged_patient_count += 1
                    column_value = '|'.join([str(x) for x in list(engagement_tuple)])
                    if not isna(column_value): row_dict[column_name] = column_value
                
                rows_list.append(row_dict)
            distance_delta_df = DataFrame(rows_list)
            
            # Merge the distance delta dataset with the original dataset
            metadata_columns = sorted([cn for cn in distance_delta_df.columns if cn.endswith('_metadata')])
            on_columns = sorted(set(merge_df.columns).intersection(set(distance_delta_df.columns)))
            columns_list = on_columns + metadata_columns
            assert set(columns_list).issubset(set(distance_delta_df.columns)), "You've lost access to the metadata columns"
            merge_df = merge_df.merge(distance_delta_df[columns_list], on=on_columns, how='left').drop_duplicates()
            
            # Break up the metadata columns into their own columns
            for cn in metadata_columns:
                str_prefix = split('_metadata', cn, 0)[0]
                
                # Split the pipe-delimited values into a DataFrame
                split_df = merge_df[cn].str.split('|', expand=True)
                
                # Change the column names to reflect the content
                split_df.columns = [
                    f'{str_prefix}_patient_id', f'{str_prefix}_engagement_start', f'{str_prefix}_location_tuple', f'{str_prefix}_patient_sort',
                    f'{str_prefix}_predicted_priority', f'{str_prefix}_injury_severity'
                ]
                
                # Make engagement_start an integer
                split_df[f'{str_prefix}_engagement_start'] = to_numeric(split_df[f'{str_prefix}_engagement_start'], errors='coerce')
                
                # Add the split columns to the original DataFrame
                merge_df = concat([merge_df, split_df], axis='columns')
                
                # Drop the original column and the empty predicted priority column
                merge_df = merge_df.drop(columns=[cn, f'{str_prefix}_predicted_priority'])
            
        # Add the prioritize patients column to the original dataset
        prioritize_columns = [new_column_name]
        merge_df[new_column_name] = 0
        def f(srs):
            is_maxed = nan  # Using numpy nan for missing values
            injury_severity_list = []
            
            # Loop through columns looking for injury severity columns
            for column_name, column_value in srs.iteritems():
                if column_name.endswith('_injury_severity') and not isna(column_value):
                    injury_severity_list.append(column_value)
            
            # Check if any injury severity was found
            if injury_severity_list:
                
                # Check if engaged patient has the maximum severity
                is_maxed = int(srs.engaged_patient00_injury_severity == max(injury_severity_list))
            
            return is_maxed
        
        # Apply the function to each row to get the new column added
        merge_df[new_column_name] = merge_df.apply(f, axis='columns')
        
        if verbose:
            
            # Display summary table if verbose mode is enabled
            display(merge_df.groupby(new_column_name, dropna=False).size().to_frame().rename(columns={0: 'record_count'}))
        
        return merge_df, prioritize_columns

fu = FRVRSUtilities(
    data_folder_path=osp.abspath('data'),
    saves_folder_path=osp.abspath('saves')
)

# In the zip there are 51 folders, (51 JSON, 51 CSV).
# All the files are named appropriated in the folder/csv/json UUID_ParticipantID.
# Some of the internal Participants IDs might be off because the moderator forgot to enter a Participant ID or didn't enter
# the Participant ID correctly so we needed to figure out which participant it was.
# So only utilize the UUID and Participant ID that is on the file name to identify and ignore the internal Participant IDs.
if IS_DEBUG: print("\nGet all the Open World logs into one data frame")
csv_stats_df = DataFrame([])
logs_path = osp.join(nu.data_folder, 'logs', 'Human_Sim_Metrics_Data_4-12-2024')
directories_list = listdir(logs_path)
for dir_name in directories_list:
    
    # Add the CSVs to the data frame
    folder_path = osp.join(logs_path, dir_name)
    df = fu.concatonate_logs(logs_folder=folder_path)
    
    session_uuid, participant_id = dir_name.split('_')
    df['session_uuid'] = session_uuid
    df['participant_id'] = int(participant_id)
    
    # Remove numerically-named columns
    columns_list = [x for x in df.columns if not search(r'\d+', str(x))]
    df = df[columns_list]
    
    # Convert 'TRUE' and 'FALSE' to boolean values
    for cn in fu.boolean_columns_list:
        df[cn] = df[cn].map({'TRUE': True, 'FALSE': False, 'True': True, 'False': False})
    
    # Convert the nulls into NaNs
    for cn in df.columns: df[cn] = df[cn].replace(['null', 'nan'], nan)
    
    # Append the data frame for the current subdirectory to the main data frame and break the participant ID loop
    csv_stats_df = concat([csv_stats_df, df], axis='index')

csv_stats_df = csv_stats_df.reset_index(drop=True).drop_duplicates()
csv_stats_df['csv_file_name'] = csv_stats_df.csv_file_subpath.map(lambda x: str(x).split('/')[-1])

if IS_DEBUG: print("\nCheck for proper ingestion (duplicate file ingestion, et al)")
assert len(csv_stats_df.columns) > 4, "Nothing ingested"
assert csv_stats_df.participant_id.nunique() == 26, f"Participant count should be 26, it's {csv_stats_df.participant_id.nunique()} instead"

if IS_DEBUG: print(csv_stats_df.groupby('logger_version').size().to_frame().rename(columns={0: 'record_count'})) # 276926

if IS_DEBUG: print("\nFilter all the rows that have more than one unique value in the file_name column for each value in the session_uuid column")
mask_series = (csv_stats_df.groupby('session_uuid').csv_file_subpath.transform(Series.nunique) > 1)
assert not mask_series.any(), "You have duplicate files"

if IS_DEBUG: print("\nCheck that all your junk scenes are the last scenes")
if IS_DEBUG: print(csv_stats_df.groupby('is_scene_aborted').size().to_frame().rename(columns={0: 'record_count'}))
mask_series = csv_stats_df.is_scene_aborted
for (session_uuid, scene_id), scene_df in csv_stats_df[mask_series].groupby(fu.scene_groupby_columns):
    mask_series = (csv_stats_df.session_uuid == session_uuid)
    max_scene_id = csv_stats_df[mask_series].scene_id.max()
    assert max_scene_id == scene_id, "You've got junk scenes in strange places"

if IS_DEBUG: print("\nModalize separate columns into one")
csv_stats_df = fu.add_modal_column('patient_id', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('injury_id', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('location_id', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('patient_sort', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('patient_pulse', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('patient_salt', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('patient_hearing', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('patient_breath', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('patient_mood', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('patient_pose', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('injury_severity', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('injury_required_procedure', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('injury_body_region', csv_stats_df, verbose=IS_DEBUG)
csv_stats_df = fu.add_modal_column('tool_type', csv_stats_df, verbose=IS_DEBUG)

csv_stats_df = fu.convert_column_to_categorical(csv_stats_df, 'pulse_taken_pulse_name', verbose=IS_DEBUG)
csv_stats_df = fu.convert_column_to_categorical(csv_stats_df, 'tool_applied_data', verbose=IS_DEBUG)

# Remove the Unity suffix from all patient_id columns
# The one without "Root" is the ID that CACI sets for it. Unity
# then takes the ID and adds "Root" to the end when it
# creates the hierarchy, so there's less room for human
# error. They're going to match perfectly.
for cn in fu.patient_id_columns_list + ['patient_id']:
    if cn in csv_stats_df.columns:
        mask_series = ~csv_stats_df[cn].isnull()
        csv_stats_df.loc[mask_series, cn] = csv_stats_df[mask_series][cn].map(lambda x: str(x).replace(' Root', ''))

# Remove the patients not in our lists
mask_series = csv_stats_df.patient_id.isin(fu.ow_patients_list)
if IS_DEBUG: pre_count = csv_stats_df.shape[0]
csv_stats_df = csv_stats_df[mask_series]
if IS_DEBUG: print(f"\nFiltered out {pre_count - csv_stats_df.shape[0]} patients not in the Desert, Jungle, Submarine, or Urban patients lists")

columns_list = ['voice_command_command_description', 'voice_capture_message']
if not csv_stats_df[columns_list].applymap(lambda x: '[PERSON]' in str(x), na_action='ignore').sum().sum():
    if IS_DEBUG: print("\nMask voice capture PII")
    try:
        import spacy
        try: nlp = spacy.load('en_core_web_sm')
        except OSError as e:
            print(str(e).strip())
            command_str = f'{sys.executable} -m spacy download en_core_web_sm --quiet'
            import subprocess
            subprocess.run(command_str.split())
            nlp = spacy.load('en_core_web_sm')
        import en_core_web_sm
        nlp = en_core_web_sm.load()
        
        mask_series = csv_stats_df.voice_command_command_description.isnull() & csv_stats_df.voice_capture_message.isnull()
        df = csv_stats_df[~mask_series]
        def mask_pii(srs):
            for idx in columns_list:
                new_text = srs[idx]
                if notnull(new_text):
                    doc = nlp(new_text)
                    for entity in doc.ents:
                        if entity.label_ == 'PERSON': new_text = sub('\\b' + entity.text + '\\b', '[PERSON]', new_text)
                    srs[idx] = new_text
        
            return srs
        
        for row_index, row_series in df.apply(mask_pii, axis='columns')[columns_list].iterrows():
            for column_name, column_value in row_series.items():
                if notnull(column_value): csv_stats_df.loc[row_index, column_name] = column_value
    except Exception as e: print(f'{e.__class__.__name__} error in PII masking: {str(e).strip()}')


if IS_DEBUG: print("\nGet column and value descriptions dataset")
file_path = osp.join(nu.data_folder, 'xlsx', 'Metrics_Evaluation_Dataset_organization_for_BBAI.xlsx')
dataset_organization_df = read_excel(file_path)

# Fix the doubled up descriptions
mask_series = dataset_organization_df.Labels.map(lambda x: ';' in str(x))
for row_index, label in dataset_organization_df[mask_series].Labels.items():
    labels_list = split(' *; *', str(label), 0)
    dataset_organization_df.loc[row_index, 'Labels'] = labels_list[0]
    
    # Append the new row to the DataFrame
    new_row = dataset_organization_df.loc[row_index].copy()
    new_row['Labels'] = labels_list[1]
    dataset_organization_df = concat([dataset_organization_df, new_row], ignore_index=True)

# Append the AD_Del_Omni_Text row to the DataFrame
mask_series = (dataset_organization_df.Variable == 'AD_Del_Omni')
new_row = dataset_organization_df.loc[mask_series].copy()
new_row['Variable'] = 'AD_Del_Omni_Text'
dataset_organization_df = concat([dataset_organization_df, new_row], ignore_index=True)

if IS_DEBUG: print("\nGet the column name description dictionary")
mask_series = ~dataset_organization_df.Description.isnull()
df = dataset_organization_df[mask_series]
COLUMN_NAME_DESCRIPTION_DICT = df.set_index('Variable').Description.to_dict()
new_description_dict = COLUMN_NAME_DESCRIPTION_DICT.copy()
for k, v in COLUMN_NAME_DESCRIPTION_DICT.items():
    new_description_dict[k] = v
    if (not k.endswith('_Text')):
        new_key_name = f'{k}_Text'
        new_description_dict[new_key_name] = new_description_dict.get(new_key_name, v)
COLUMN_NAME_DESCRIPTION_DICT = new_description_dict.copy()

if IS_DEBUG: print("\nCreate the value description function")
numeric_categories_mask_series = dataset_organization_df.Labels.map(lambda x: '=' in str(x))
value_descriptions_columns = dataset_organization_df[numeric_categories_mask_series].Variable.unique().tolist()
def get_value_description(column_name, column_value):
    """
    Get the description of a given value for a specific column.
    
    Parameters:
        column_name (str):
            The name of the column.
        column_value (Any):
            The value of the column.
    
    Returns:
        str: The description of the value.
    """
    value_description = ''
    
    # Check if the column value is not NaN
    if not isna(column_value):
        
        # Create a boolean mask to filter the dataset_organization_df
        mask_series = (dataset_organization_df.Variable == column_name) & ~dataset_organization_df.Labels.isnull()
        
        # Check if there are any matching rows
        if mask_series.any():
            
            # Filter the DataFrame using the mask
            df = dataset_organization_df[mask_series]
            
            # Create a new mask to find rows with labels matching the column value
            mask_series = df.Labels.map(lambda label: split(' *= *', str(label), 0)[0] == str(int(float(column_value))))
            
            # Check if there are any matching rows
            if mask_series.any():
                
                # Get the label for the matching row
                label = df[mask_series].Labels.squeeze()
                
                # Extract the description from the label
                value_description = split(' *= *', str(label), 0)[1]
    
    return value_description
if IS_DEBUG: print("\nCreate the distance delta dataframe")
distance_delta_df = fu.get_distance_deltas_data_frame(csv_stats_df)

if IS_DEBUG: print("\nAdd the agony column")
if 'has_patient_in_agony' not in distance_delta_df.columns:
    distance_delta_df['has_patient_in_agony'] = False
    for (session_uuid, scene_id), idx_df in distance_delta_df.groupby(fu.scene_groupby_columns):
        
        # Get the whole scene history
        mask_series = True
        for cn in fu.scene_groupby_columns: mask_series &= (csv_stats_df[cn] == eval(cn))
        scene_df = csv_stats_df[mask_series]
        
        # Get whether any patient in the scene is in agony
        mask_series = False
        for cn in fu.mood_columns_list: mask_series |= (scene_df[cn] == 'agony')
        
        # Mark the scene in distance delta as agonistic
        if mask_series.any(): distance_delta_df.loc[idx_df.index, 'has_patient_in_agony'] = True

if 'cluster_label' not in distance_delta_df.columns:
    if IS_DEBUG: print("Set appropriate parameters for DBSCAN based on what gives 4 clusters")
    from sklearn.cluster import DBSCAN
    columns_list = ['actual_engagement_distance']
    X = distance_delta_df[columns_list].values
    dbscan = DBSCAN(eps=5, min_samples=1)
    dbscan.fit(X)
    distance_delta_df['cluster_label'] = dbscan.labels_

if IS_DEBUG: print("\nScene Stats Created for Metrics Evaluation Open World")

rows_list = []
engagment_columns_list = ['patient_id', 'engagement_start', 'location_tuple', 'patient_sort', 'predicted_priority', 'injury_severity']
for (session_uuid, scene_id), idx_df in distance_delta_df.groupby(fu.scene_groupby_columns):
    row_dict = list(idx_df.T.to_dict().values())[0]
    
    # Get the whole scene history
    mask_series = True
    for cn in fu.scene_groupby_columns: mask_series &= (csv_stats_df[cn] == eval(cn))
    scene_df = csv_stats_df[mask_series]
    
    if scene_df.shape[0]:
        row_dict['participant_id'] = scene_df.participant_id.iloc[0]
        
        # Get the count of all the patient injuries
        all_patient_injuries_count = 0
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            all_patient_injuries_count += patient_df.injury_id.nunique()
        row_dict['all_patient_injuries_count'] = all_patient_injuries_count
        
        # Get all the FRVRS utils scalar scene values
        row_dict['first_engagement'] = fu.get_first_engagement(scene_df)
        row_dict['first_treatment'] = fu.get_first_treatment(scene_df)
        row_dict['injury_correctly_treated_count'] = fu.get_injury_correctly_treated_count(scene_df)
        row_dict['injury_not_treated_count'] = fu.get_injury_not_treated_count(scene_df)
        row_dict['injury_wrongly_treated_count'] = fu.get_injury_wrongly_treated_count(scene_df)
        row_dict['is_scene_aborted'] = fu.get_is_scene_aborted(scene_df)
        row_dict['last_engagement'] = fu.get_last_engagement(scene_df)
        row_dict['percent_hemorrhage_controlled'] = fu.get_percent_hemorrhage_controlled(scene_df)
        row_dict['pulse_taken_count'] = fu.get_pulse_taken_count(scene_df)
        row_dict['scene_end'] = fu.get_scene_end(scene_df)
        row_dict['scene_start'] = fu.get_scene_start(scene_df)
        row_dict['scene_type'] = fu.get_scene_type(scene_df)
        row_dict['stills_value'] = fu.get_stills_value(scene_df)
        row_dict['teleport_count'] = fu.get_teleport_count(scene_df)
        row_dict['time_to_hemorrhage_control_per_patient'] = fu.get_time_to_hemorrhage_control_per_patient(scene_df)
        row_dict['time_to_last_hemorrhage_controlled'] = fu.get_time_to_last_hemorrhage_controlled(scene_df)
        row_dict['total_actions_count'] = fu.get_total_actions_count(scene_df)
        row_dict['triage_time'] = fu.get_triage_time(scene_df)
        row_dict['voice_capture_count'] = fu.get_voice_capture_count(scene_df)
        row_dict['walk_command_count'] = fu.get_walk_command_count(scene_df)
        row_dict['walk_value'] = fu.get_walk_value(scene_df)
        row_dict['walkers_value'] = fu.get_walkers_value(scene_df)
        row_dict['wave_command_count'] = fu.get_wave_command_count(scene_df)
        row_dict['wave_value'] = fu.get_wave_value(scene_df)
    
    rows_list.append(row_dict)
scene_stats_df = DataFrame(rows_list).drop_duplicates()

# Check for miscalculated injury columns
assert (
    scene_stats_df.all_patient_injuries_count == scene_stats_df[
        ['injury_correctly_treated_count', 'injury_not_treated_count', 'injury_wrongly_treated_count']
    ].sum(axis='columns')
).all(), "The sum of the various injury counts does not match the total injury count for all rows."

if 'MedExp' not in scene_stats_df.columns:
    file_path = osp.join(nu.data_folder, 'xlsx', 'participant_data_0420.xlsx')
    participant_data_df = read_excel(file_path).rename(columns={'ParticipantID': 'participant_id', 'Date': 'participation_date'})
    participant_data_df.participation_date = to_datetime(participant_data_df.participation_date, format='%m/%d/%Y')
    
    if IS_DEBUG: print("\nColumns to merge the participant dataset with the scene stats on:")
    on_columns = sorted(set(scene_stats_df.columns).intersection(set(participant_data_df.columns)))
    if IS_DEBUG: print(on_columns)
    if on_columns:
        scene_stats_df = scene_stats_df.merge(
            participant_data_df, how='left', on=on_columns
        )
    
    columns_list = [cn for cn in scene_stats_df.columns if cn.lower().startswith('participant') and cn.lower().endswith('id')]
    if len(columns_list) == 2:
        if IS_DEBUG: print("Check if the various partipant id columns are inconsistent")
        df = scene_stats_df[columns_list]
        for cn in columns_list: df[cn] = df[cn].map(lambda x: str(x).strip())
        mask_series = (df[columns_list[0]] != df[columns_list[1]])
        if mask_series.any():
            if IS_DEBUG: print("The various partipant id columns are inconsistent")

if IS_DEBUG: print('\nCheck if all the patient IDs in any run are some variant of Mike and designate those runs as "Orientation"')
new_column_name = 'scene_type'
if (new_column_name in scene_stats_df.columns): scene_stats_df = scene_stats_df.drop(columns=new_column_name)
if (new_column_name not in scene_stats_df.columns): scene_stats_df[new_column_name] = 'Triage'
column_value = 'Orientation'
if (column_value not in scene_stats_df.scene_type):
    for (session_uuid, scene_id), scene_df in csv_stats_df.groupby(fu.scene_groupby_columns):
        patients_list = sorted(scene_df[~scene_df.patient_id.isnull()].patient_id.unique())
        is_mike_series = Series(patients_list).map(lambda x: 'mike' in str(x).lower())
        if is_mike_series.all():
            mask_series = True
            for cn in fu.scene_groupby_columns: mask_series &= (scene_stats_df[cn] == eval(cn))
            scene_stats_df.loc[mask_series, new_column_name] = column_value
    if IS_DEBUG: display(
        scene_stats_df.groupby([new_column_name]).size().to_frame().rename(columns={0: 'record_count'}).sort_values(new_column_name, ascending=False).head(20)
    )

new_column_name = 'is_scene_aborted'
if (new_column_name not in scene_stats_df.columns):
    if IS_DEBUG: print("\nAny runs longer than that 16 minutes are probably an instance of someone taking off the headset and setting it on the ground.")
    # 1 second = 1,000 milliseconds; 1 minute = 60 seconds
    scene_stats_df[new_column_name] = False
    for (session_uuid, scene_id), scene_df in csv_stats_df.groupby(fu.scene_groupby_columns):
        mask_series = True
        for cn in fu.scene_groupby_columns: mask_series &= (scene_stats_df[cn] == eval(cn))
        scene_stats_df.loc[mask_series, new_column_name] = fu.get_is_scene_aborted(scene_df)
if IS_DEBUG: print(scene_stats_df.groupby(new_column_name).size().to_frame().rename(columns={0: 'record_count'}))

new_column_name = 'is_a_one_triage_file'
if (new_column_name not in scene_stats_df.columns):
    if IS_DEBUG: print("\nAdd the is_a_one_triage_file column")
    scene_stats_df[new_column_name] = False
        
    #Assume a 1:1 correspondence between file name and UUID from the logs data frame build
    for session_uuid, session_df in scene_stats_df.groupby('session_uuid'):
        
        # Filter in the triage files in this UUID
        mask_series = (scene_stats_df.session_uuid == session_uuid) & (scene_stats_df.scene_type == 'Triage')
        
        # Get whether the file has only one triage run
        triage_scene_count = len(scene_stats_df[mask_series].groupby('scene_id').groups)
        is_a_one_triage_file = bool(triage_scene_count == 1)
        
        scene_stats_df.loc[session_df.index, new_column_name] = is_a_one_triage_file
if IS_DEBUG: print(scene_stats_df.groupby(new_column_name).size().to_frame().rename(columns={0: 'record_count'}))

if IS_DEBUG: print("\nData Fixes for Metrics Evaluation Open World")

if IS_DEBUG: print("\nFix the encounter_layout column based on the set of patients in the scene")
new_column_name = 'encounter_layout'
encounter_layouts_list = ['Desert', 'Jungle', 'Submarine', 'Urban']
for (session_uuid, scene_id), scene_df in csv_stats_df.groupby(fu.scene_groupby_columns):
    for env_str in encounter_layouts_list:
        patients_list = eval(f'{env_str.lower()}_patients_list')
        if all(map(lambda patient_id: patient_id in scene_df.patient_id.unique().tolist(), patients_list)):
            mask_series = (scene_stats_df.session_uuid == session_uuid) & (scene_stats_df.scene_id == scene_id)
            scene_stats_df.loc[mask_series, new_column_name] = env_str
if IS_DEBUG: print(scene_stats_df.groupby(new_column_name, dropna=False).size().to_frame().rename(columns={0: 'record_count'}))

scene_columns_set = set(scene_stats_df.columns)
logs_columns_set = set(csv_stats_df.columns)
intersection_columns = set(['is_scene_aborted'])

drop_columns = sorted(scene_columns_set.intersection(logs_columns_set).intersection(intersection_columns))
if drop_columns:
    if IS_DEBUG: print("\nDrop the logs columns already recorded in the scene stats data frames")
    if IS_DEBUG: print(drop_columns)
    csv_stats_df = csv_stats_df.drop(columns=drop_columns)

columns_list = ['encounter_layout', 'scene_type', 'teleport_count']
if IS_DEBUG: display(
    scene_stats_df[columns_list].groupby(columns_list, dropna=False).size().to_frame().rename(columns={0: 'record_count'}).sort_values(columns_list, ascending=[False]*len(columns_list))
)

mask_series = scene_stats_df.encounter_layout.isin(encounter_layouts_list)
if IS_DEBUG: pre_count = scene_stats_df.shape[0]
scene_stats_df = scene_stats_df[mask_series]
if IS_DEBUG: print(f"\nFiltered out {pre_count - scene_stats_df.shape[0]} unnamed encounter layouts")

mask_series = scene_stats_df.scene_type.isin(['Orientation'])
if IS_DEBUG: pre_count = scene_stats_df.shape[0]
scene_stats_df = scene_stats_df[~mask_series]
if IS_DEBUG: print(f"\nFiltered out {pre_count - scene_stats_df.shape[0]} orientation scenes")

mask_series = (scene_stats_df.teleport_count < 1)
if IS_DEBUG: pre_count = scene_stats_df.shape[0]
scene_stats_df = scene_stats_df[~mask_series]
if IS_DEBUG: print(f"\nFiltered out {pre_count - scene_stats_df.shape[0]} scenes with no teleports")

nu.save_data_frames(metrics_evaluation_open_world_scene_stats_df=scene_stats_df, verbose=IS_DEBUG)
nu.save_data_frames(metrics_evaluation_open_world_csv_stats_df=csv_stats_df, verbose=IS_DEBUG)


# load data frames to get a reliable representation
if IS_DEBUG: print('\nCreate the final dataframe that we should use for analysis')


if IS_DEBUG: print("\nColumns to merge the scene stats dataset with the CSV stats on:")
on_columns = sorted(set(csv_stats_df.columns).intersection(set(scene_stats_df.columns)))
if IS_DEBUG: print(on_columns)

if IS_DEBUG: print('\nThe scene stats dataset columns we want to have in the merge:')
metadata_columns = sorted([cn for cn in scene_stats_df.columns if cn.endswith('_metadata')])
analysis_columns = sorted(set([
    'actual_engagement_distance', 'first_engagement', 'first_treatment', 'injury_correctly_treated_count', 'injury_not_treated_count',
    'injury_treatments_count', 'injury_wrongly_treated_count', 'last_engagement', 'last_still_engagement', 'measure_of_right_ordering', 'patient_count',
    'percent_hemorrhage_controlled', 'pulse_taken_count', 'stills_value', 'teleport_count', 'time_to_hemorrhage_control_per_patient',
    'time_to_last_hemorrhage_controlled', 'total_actions_count', 'triage_time', 'voice_capture_count', 'walk_command_count', 'walk_value', 'walkers_value',
    'wave_command_count', 'wave_value'
] + metadata_columns).intersection(set(scene_stats_df.columns)))
if IS_DEBUG: print(analysis_columns)

if IS_DEBUG: print("\nMerge the scene stats with the CSV stats")
survey_columns = ['AD_KDMA_Sim', 'AD_KDMA_Text', 'PropTrust', 'ST_KDMA_Sim', 'ST_KDMA_Text', 'YrsMilExp']
columns_list = on_columns + analysis_columns + survey_columns
assert set(columns_list).issubset(set(scene_stats_df.columns)), "You've lost access to the analysis columns"
merge_df = csv_stats_df.merge(scene_stats_df[columns_list], on=on_columns, how='left').drop_duplicates()

if IS_DEBUG: print("\nBreak up the metadata columns into their own columns")
metadata_columns = sorted([cn for cn in merge_df.columns if cn.endswith('_metadata')])
for cn in metadata_columns:
    str_prefix = split('_metadata', cn, 0)[0]
    
    # Split the pipe-delimited values into a DataFrame
    split_df = merge_df[cn].str.split('|', expand=True)
    
    # Change the column names to reflect the content
    split_df.columns = [
        f'{str_prefix}_patient_id', f'{str_prefix}_engagement_start', f'{str_prefix}_location_tuple', f'{str_prefix}_patient_sort',
        f'{str_prefix}_predicted_priority', f'{str_prefix}_injury_severity'
    ]
    
    # Make engagement_start an integer
    # split_df[f'{str_prefix}_engagement_start'] = split_df[f'{str_prefix}_engagement_start'].fillna(0).astype(int)
    split_df[f'{str_prefix}_engagement_start'] = to_numeric(split_df[f'{str_prefix}_engagement_start'], errors='coerce')
    
    # Add the split columns to the original DataFrame
    merge_df = concat([merge_df, split_df], axis='columns')
    
    # Drop the original column
    merge_df = merge_df.drop(columns=[cn, f'{str_prefix}_predicted_priority'])

patient_id_columns = sorted([cn for cn in merge_df.columns if cn.endswith('_patient_id')])
engagement_start_columns = sorted([cn for cn in merge_df.columns if cn.endswith('_engagement_start')])
location_tuple_columns = sorted([cn for cn in merge_df.columns if cn.endswith('_location_tuple')])
patient_sort_columns = sorted([cn for cn in merge_df.columns if cn.endswith('_patient_sort')])
injury_severity_columns = sorted([cn for cn in merge_df.columns if cn.endswith('_injury_severity')])

new_column_name = 'prioritize_high_injury_severity_patients'
if new_column_name not in merge_df.columns:
    if IS_DEBUG: print("\nAdd the prioritize patients columns")
    prioritize_columns = [new_column_name]
    merge_df[new_column_name] = 0
    def f(srs):
        is_maxed = nan
        injury_severity_list = []
        for column_name, column_value in srs.iteritems():
            if column_name.endswith('_injury_severity') and not isna(column_value):
                injury_severity_list.append(column_value)
        if injury_severity_list:
            is_maxed = int(srs.engaged_patient00_injury_severity == max(injury_severity_list))
        return is_maxed
    merge_df[new_column_name] = merge_df.apply(f, axis='columns')
    if IS_DEBUG: print(merge_df.groupby(new_column_name, dropna=False).size().to_frame().rename(columns={0: 'record_count'}))

if IS_DEBUG: print('\nThe merge dataset columns we want to have in the groupby:')
columns_list = sorted(set(
    on_columns + survey_columns + analysis_columns + patient_id_columns + engagement_start_columns
    + location_tuple_columns + patient_sort_columns + injury_severity_columns + prioritize_columns
).intersection(set(merge_df.columns)))
if IS_DEBUG: print(columns_list)

assert set(columns_list).issubset(set(merge_df.columns)), "You've lost access to the survey columns"
if IS_DEBUG: print("\nThe numeric columns we want to take the mean of:")
df = merge_df[columns_list]
numeric_columns = sorted(set(nu.get_numeric_columns(df)).difference(set(
    on_columns + patient_id_columns + location_tuple_columns + patient_sort_columns + injury_severity_columns
)))
if IS_DEBUG: print(numeric_columns)

if IS_DEBUG: print("\nThe other columns we do not want to take the mean of:")
other_columns = sorted(set(df.columns).difference(set(
    numeric_columns + [
        'injury_record_patient_id', 'injury_treated_patient_id', 'pulse_taken_patient_id', 'tag_applied_patient_id',
        'tool_applied_patient_id', 'player_gaze_patient_id', 'triage_level_walk_if_can_patient_id', 'triage_level_walked_patient_id',
        'triage_level_wave_if_can_patient_id', 'triage_level_waved_patient_id'
    ]
)))
if IS_DEBUG: print(other_columns)

left_df = merge_df[numeric_columns+on_columns].groupby(on_columns).mean().reset_index(drop=False).rename(
    columns={cn: 'mean_'+cn for cn in numeric_columns}
).dropna(axis='columns', how='all')
right_df = merge_df[other_columns].drop_duplicates().dropna(axis='columns', how='all')
if IS_DEBUG: print("\nColumns to merge the unmeaned half of the merge with the meaned half of the merge on:")
on_columns = sorted(set(left_df.columns).intersection(set(right_df.columns)))
if IS_DEBUG: print(on_columns)

if IS_DEBUG: print("\nAggregate the data from the merged datasets and group by participant, session, and scene to get the means of the numeric columns")
anova_df = left_df.merge(right_df, on=on_columns, how='outer').drop_duplicates()
assert set(['mean_'+cn for cn in survey_columns]).issubset(set(anova_df.columns)), "You've lost acces to the survey columns (YrsMilExp, et al)"
for cn in survey_columns: anova_df['mean_'+cn] = anova_df['mean_'+cn].fillna(0)
assert len(anova_df.groupby(['participant_id', 'scene_id', 'session_uuid']).groups.keys()) == anova_df.shape[0], "You have duplicate rows in anova_df"

if IS_DEBUG: print("\nAdd medical role back in")
new_column = 'MedRole'
column_name = 'medical_role'
if new_column in scene_stats_df.columns:
    on_columns = sorted(set(anova_df.columns).intersection(set(scene_stats_df.columns)))
    columns_list = on_columns + [new_column]
    anova_df = anova_df.merge(
        scene_stats_df[columns_list], on=on_columns, how='left'
    ).rename(columns={new_column: column_name})
    anova_df[column_name] = anova_df[column_name].map(
        lambda cv: get_value_description('MedRole', cv)
    ).replace('', nan)
if IS_DEBUG: print(anova_df.groupby(column_name).size().to_frame().rename(columns={0: 'record_count'}).sort_values(
    'record_count', ascending=False
).head(5))

if IS_DEBUG: print("\nAdd the sim environment back in")
new_column = 'encounter_layout'
if new_column in scene_stats_df.columns:
    on_columns = sorted(set(anova_df.columns).intersection(set(scene_stats_df.columns)))
    columns_list = on_columns + [new_column]
    anova_df = anova_df.merge(
        scene_stats_df[columns_list], on=on_columns, how='left'
    )
if IS_DEBUG: print(anova_df.groupby(new_column).size().to_frame().rename(columns={0: 'record_count'}).sort_values(
    'record_count', ascending=False
).head(5))

if IS_DEBUG: print("\nStore the results and show the new data frame columns")
mean_survey_columns = ['mean_' + cn for cn in survey_columns]
columns_list = anova_df.columns.tolist()
nu.save_data_frames(metrics_evaluation_open_world_anova_df=anova_df[columns_list], verbose=IS_DEBUG)


def entitle_column_name(column_name):
    """
    Entitles a column name based on specified rules.
    
    Parameters:
        column_name (str):
            The name of the column to be entitled.
    
    Returns:
        str
            The entitled column name.
    
    Notes:
        - If the column name starts with 'mean_' and the substring after 'mean_' exists in the column name description dictionary,
          the entitled name is obtained from the corresponding value in the dictionary, prepended with 'Average ' if necessary.
        - Otherwise, the column name is split into parts, and each part is processed to ensure proper capitalization and word replacements.
    """
    
    # Check if the column name starts with 'mean_' and is present in the column name description dictionary
    if column_name.startswith('mean_') and (column_name[5:] in COLUMN_NAME_DESCRIPTION_DICT):
        entitled_name = COLUMN_NAME_DESCRIPTION_DICT[column_name[5:]]
        
        # Prepend 'Average ' if the entitled name does not already start with it
        if not entitled_name.startswith('Average '):
            entitled_name = 'Average ' + entitled_name
    else:
        new_parts_list = []
        old_parts_list = [op for op in split('_', column_name, 0) if op]  # Split the column name by underscores
        for name_part in old_parts_list:
            
            # Check if the part contains a capital letter followed by lowercase letters (camelCase)
            if search('[A-Z][a-z]+', name_part):
                humps_list = [hp for hp in split('([A-Z][a-z]+)', name_part, 0) if hp]  # Split camelCase parts
                for i, hump_part in enumerate(humps_list):
                    
                    # Capitalize each part if it is all lowercase
                    if hump_part == hump_part.lower():
                        humps_list[i] = hump_part.title()
                    
                    # Replace specific abbreviations with full words
                    elif hump_part == 'Sim':
                        humps_list[i] = 'Simulation'
                    elif hump_part == 'Yrs':
                        humps_list[i] = 'Years of'
                    elif hump_part == 'Mil':
                        humps_list[i] = 'Military'
                    elif hump_part == 'Exp':
                        humps_list[i] = 'Experience'
                new_parts_list.extend(humps_list)
            else:
                
                # Process parts that are not in camelCase
                if name_part == name_part.lower():
                    
                    # Capitalize if the part is all lowercase and meets certain conditions
                    if (len(name_part) > 2) and (name_part != 'uuid'):
                        name_part = name_part.title()
                    
                    # Convert certain parts to uppercase
                    elif name_part not in ['to', 'of', 'per']:
                        name_part = name_part.upper()
                new_parts_list.append(name_part)
        
        # Replace 'Mean' with 'Average' if it is the first part
        if new_parts_list[0] == 'Mean':
            new_parts_list[0] = 'Average'
        entitled_name = ' '.join(new_parts_list)
    
    return entitled_name

if IS_DEBUG: print('\nWrite up the steps to do the ANOVA stats columns calculations')
comment_regex = re.compile('^ *# ([^\r\n]+)', MULTILINE)
function_call_dict = {
    'encounter_layout': 'fu.add_encounter_layout_column', 'medical_role': 'fu.add_medical_role_column',
    'mean_prioritize_high_injury_severity_patients': 'fu.add_prioritize_severity_column'
}
engaged_patient_columns_list = []
file_path = osp.join(nu.saves_text_folder, 'how_to_do_calculations.txt')
with open(file_path, mode='w', encoding=nu.encoding_type) as f: print('', file=f)
with open(file_path, mode='a', encoding=nu.encoding_type) as f:
    for cn in anova_df.columns:
        if 'engaged_patient' in cn:
            engaged_patient_columns_list.append(cn)
            continue
        print('', file=f)
        print(f'{cn} ({entitle_column_name(cn)})', file=f)
        print('Steps Needed to do Calculations:', file=f)
        comments_list = []
        if cn in ['participant_id', 'scene_id', 'session_uuid']:
            if cn == 'scene_id':
                comments_list.append('The scene_id is derived from the CSV SESSION_START and SESSION_END entries')
            else:
                comments_list.append(
                    f'The {cn} is found in both the CSV and the JSON data in the'
                    + ' "Human_Sim_Metrics_Data 4-12-2024.zip" file provided by CACI'
                )
        else:
            comments_list.append('Group your dataset by participant_id, session_uuid, and scene_id')
            if cn in mean_survey_columns:
                comments_list.extend([
                    f'Find the {cn.replace("mean_", "")} column in the participant_data_0420 spreadsheet provided by CACI for that participant',
                    f'The {cn.replace("mean_", "")} value is semi-continously numeric, and you can average it for whatever grouping you need'
                ])
            else:
                if cn in function_call_dict:
                    function_call = function_call_dict[cn]
                else:
                    function_call = cn.replace('mean_', 'fu.get_')
                source_code = inspect.getsource(eval(function_call))
                comments_list.extend([comment_str for comment_str in comment_regex.findall(source_code) if comment_str and ('verbose' not in comment_str)])
        for i, comment_str in enumerate(comments_list):
            if not comment_str.endswith('.'):
                comment_str += '.'
            print(f'{i+1}. {comment_str}', file=f)
    if engaged_patient_columns_list:
        print(
            f'\n\n\nNOTE: All the patients metadata columns ({nu.conjunctify_nouns(engaged_patient_columns_list)}) are byproducts of the production of the prioritize patients columns.',
            file=f
        )
if IS_DEBUG: print(f'Saving to {osp.abspath(file_path)}')


if IS_DEBUG: print("\nWrite up the statistical analysis of the ANOVA dataset")
def compare_columns(anova_df, groupby_column, columns_list, dataset_organization_df, text_io_wrapper=None):
    """
    Compare columns using linear regression across groups defined by a specified column.
    
    Parameters:
        anova_df (pandas.DataFrame):
            The dataframe containing the data to be analyzed.
        groupby_column (str):
            The column used for grouping the data.
        columns_set (set):
            The set of columns to be compared.
        dataset_organization_df (pandas.DataFrame):
            The dataframe containing information about the organization of the dataset.
        text_io_wrapper (_io.TextIOWrapper, optional):
            If a text IO wrapper, print the results to the open file, by default None.
    
    Returns:
        None
    """
    
    # Encode the group-by column into dummy variables
    dummy_groupby = get_dummies(anova_df[groupby_column], prefix=groupby_column)
    
    # Concatenate the dummy variables with the dataframe
    anova_with_dummies_df = concat([anova_df, dummy_groupby], axis=1)
    
    # Add a constant term to the independent variables
    anova_with_dummies_df['const'] = 1
    
    # Group data by groupby columns
    grouped_data = anova_with_dummies_df.groupby(groupby_column)
    
    # Iterate over columns for comparison
    if groupby_column in columns_list: columns_list.pop(groupby_column)
    for cn in columns_list: perform_linear_regression(
        cn, ' '.join([w.title() for w in cn.split('_')]), groupby_column, grouped_data, anova_with_dummies_df, dataset_organization_df,
        text_io_wrapper=text_io_wrapper
    )


def perform_linear_regression(cn, xname, groupby_column, grouped_data, anova_with_dummies_df, dataset_organization_df, text_io_wrapper=None):
    """
    Perform linear regression analysis.
    
    Parameters:
        cn (str):
            The name of the dependent variable column.
        xname (str):
            The name of the independent variable.
        groupby_column (str):
            The name of the column used for grouping.
        grouped_data (pandas.core.groupby.generic.DataFrameGroupBy):
            Grouped data for the specified groupby_column.
        anova_with_dummies_df (pandas.DataFrame):
            DataFrame with dummy variables for ANOVA.
        dataset_organization_df (pandas.DataFrame):
            DataFrame containing organizational information about the dataset.
        text_io_wrapper (_io.TextIOWrapper, optional):
            If a text IO wrapper, print the results to the open file, by default None.
    """
    statements_list = []
    
    # Generate regression columns
    regression_columns = [cn for cn in anova_with_dummies_df.columns if cn.startswith(f'{groupby_column}_')]
    columns_list = [cn, groupby_column, 'const'] + regression_columns
    
    # Filter and drop NaN values from the DataFrame
    df = anova_with_dummies_df[columns_list].dropna()
    cn_data = df[cn]
    statements_list.append(f"\n{cn.replace('mean_', '')} ({entitle_column_name(cn).replace('Average ', '')}) grouped by {groupby_column.replace('mean_', '')} ({entitle_column_name(groupby_column).replace('Average ', '')})")
    
    # Perform a test for normality
    try: statements_list, p_value = perform_shapiro_wilk(cn, cn_data, statements_list)
    except: statements_list, p_value = perform_shapiro_francia(cn, cn_data, statements_list)

    if grouped_data.ngroups > 1: statements_list, theoretical_quantiles, transformable_df = compare_groups(
        statements_list, p_value, grouped_data, cn_data, cn, df, groupby_column
    )
    
    # Perform linear regression
    results = sm.OLS(cn_data, df[['const'] + regression_columns]).fit()
    
    # Append the regression equation
    if not isna(results.params['const']): statements_list = append_regression_equation(
        statements_list, regression_columns, dataset_organization_df, groupby_column, results, cn
    )
    
    # Save the results
    if len(statements_list) > 1:
        print('\n'.join(statements_list), file=text_io_wrapper)


def perform_shapiro_wilk(cn, cn_data, statements_list):
    """
    Perform Shapiro-Wilk test for normality.
    
    Parameters:
        cn (str):
            The name of the dataset being tested.
        cn_data (array_like):
            The data to be tested for normality.
        statements_list (list):
            A list to which the test results will be appended.
    
    Returns:
        list
            A list containing the appended test results.
        float
            The p-value of the Shapiro-Wilk test.
    """
    
    # Import the Shapiro-Wilk test from SciPy
    from scipy.stats import shapiro_wilk
    
    # Perform Shapiro-Wilk test
    stat, p_value = shapiro_wilk(cn_data)
    
    # Append the test results
    if not isna(stat): statements_list.append(f'Shapiro-Wilk Test for normality ({cn}): Statistic: {stat:.4f}, p-value: {p_value:.4f}')
    
    return statements_list, p_value


def perform_shapiro_francia(cn, cn_data, statements_list):
    """
    Perform Shapiro-Francia test for normality.

    Parameters
    ----------
    cn : str
        Column name or identifier.
    cn_data : array_like
        Input data.
    statements_list : list
        List to append the test results.

    Returns
    -------
    tuple
        A tuple containing the updated statements_list and the p-value of the test.
    """

    # Import necessary function from scipy.stats
    from scipy.stats import shapiro
    
    # Perform Shapiro-Francia test
    stat, p_value = shapiro(cn_data)
    
    # Append the test results
    if not isna(stat): statements_list.append(f"Shapiro-Francia Test for normality ({cn.replace('mean_', '')}): Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    
    return statements_list, p_value


def compare_groups(statements_list, p_value, grouped_data, cn_data, cn, anova_with_dummies_df, groupby_column):
    """
    Compare groups using either ANOVA or Kruskal-Wallis test depending on the normality assumption.
    
    Parameters:
        statements_list (list):
            List to which statements about the statistical comparisons are appended.
        p_value (float):
            The p-value obtained from normality test.
        grouped_data (pandas.DataFrame):
            Data grouped according to the variable of interest.
        cn_data (pandas.Series):
            Series containing the counts of each group.
        cn (int):
            Total number of observations.
        anova_with_dummies_df (pandas.DataFrame):
            Data with dummy variables encoded for ANOVA.
        groupby_column (str):
            Column name by which the data is grouped.
        
    Returns:
        statements_list (list):
            Updated list of statements about the statistical comparisons.
        theoretical_quantiles (None or array_like):
            Theoretical quantiles for normality test, if applicable.
        transformable_df (pandas.DataFrame):
            Dataframe containing pairwise comparisons if significant difference exists, otherwise empty.
    """
    theoretical_quantiles = None; transformable_df = DataFrame([])
    
    # Fail to reject the null hypothesis of normality
    if (p_value >= 0.05): statements_list, theoretical_quantiles, p_value = perform_anova(statements_list, grouped_data, cn_data, cn)
    
    # The data does not come from a normal distribution: consider non-parametric tests
    else: statements_list, p_value = perform_kruskal_wallis(statements_list, grouped_data, anova_with_dummies_df, groupby_column, cn, cn_data)
    
    # Test shows a significant difference (p-value < 0.05)
    if p_value < 0.05: transformable_df, statements_list = compare_the_pairs(
        anova_with_dummies_df, groupby_column, grouped_data, statements_list, cn_data, cn
    )
    
    return statements_list, theoretical_quantiles, transformable_df


def perform_anova(statements_list, grouped_data, cn_data, cn):
    """
    Perform one-way ANOVA on grouped data.
    
    Parameters:
        statements_list (list):
            List to store ANOVA results.
        grouped_data (iterable):
            Grouped data to perform ANOVA on.
        cn_data (numpy.ndarray):
            Data corresponding to the column 'cn'.
        cn (str):
            Column name for which ANOVA is performed.
    
    Returns:
        tuple
            A tuple containing:
            - statements_list (list):
                List of statements with ANOVA results.
            - theoretical_quantiles (numpy.ndarray):
                Theoretical quantiles from a standard normal distribution.
            - p_value (float):
                p-value from ANOVA.
    """
    
    # Perform ANOVA on values to get F statistic and p-value
    f_statistic, p_value = f_oneway(*[group[cn] for name, group in grouped_data])
    
    # Append the results
    if not isna(p_value): statements_list.append(f"One-way ANOVA for {cn}: F statistic = {f_statistic:.4f}, p-value = {p_value:.4f}")
    
    # Calculate the sum of squares between groups (SS_between)
    mean_overall = cn_data.mean()  # Mean of all values
    SS_between = sum(len(group) * (group[cn].mean() - mean_overall)**2 for name, group in grouped_data)
    
    # Calculate the total sum of squares (SS_total)
    SS_total = sum((value - mean_overall)**2 for name, group in grouped_data for value in group[cn])
    
    # Compute eta-squared (η²) and append the results
    try: eta_squared = SS_between / SS_total
    except: eta_squared = nan
    if not isna(eta_squared): statements_list.append(f"    η² (effect size) = {eta_squared:.4f}")
    
    # Generate theoretical quantiles from a standard normal distribution
    theoretical_quantiles = norm.ppf(cn_data.rank() / (len(cn_data) + 1))
    
    return statements_list, theoretical_quantiles, p_value


def perform_kruskal_wallis(statements_list, grouped_data, anova_with_dummies_df, groupby_column, cn, cn_data):
    """
    Perform Kruskal-Wallis test on grouped data.
    
    Parameters:
        statements_list (list):
            List to store the result statements.
        grouped_data (DataFrameGroupBy):
            Grouped data to perform the test on.
        anova_with_dummies_df (pandas.DataFrame):
            DataFrame used for ANOVA with dummy variables.
        groupby_column (str):
            Name of the column used for grouping.
        cn (str):
            Column name for the data to perform the test on.
        cn_data (str):
            Data for the given column.
    
    Returns:
        tuple
            A tuple containing the updated statements list and the p-value.
    """
    
    # Perform Kruskal-Wallis test on values
    kruskal_statistic, p_value = kruskal(*[group[cn] for name, group in grouped_data])
    
    # Access additional test details from the result object
    kruskal_results = kruskal(*[group[cn] for name, group in grouped_data])
    
    # Extract chi-squared statistic (equivalent to H statistic)
    chi_squared = kruskal_results.statistic
    
    # Degrees of freedom (number of groups - 1)
    degrees_of_freedom = len(anova_with_dummies_df[groupby_column].unique()) - 1
    
    # Append the complete results
    statements_list.append(f"Kruskal-Wallis test for {cn.replace('mean_', '')}: H statistic = {kruskal_statistic:.4f}, p-value = {p_value:.4f}")
    statements_list.append(f"    χ² = {chi_squared:.4f}, df = {degrees_of_freedom}, p-value = {p_value:.4f}")
    
    # Epsilon squared calculation
    epsilon_squared = calculate_epsilon_squared(anova_with_dummies_df, groupby_column, kruskal_results, cn, grouped_data, cn_data)
    
    # Append the results
    if not isna(epsilon_squared): statements_list.append(f"    ε² (effect size) = {epsilon_squared:.4f}")
    
    return statements_list, p_value


def calculate_epsilon_squared(anova_with_dummies_df, groupby_column, kruskal_results, cn, data_by_env, cn_data):
    """
    Calculate epsilon squared (effect size) for a one-way ANOVA or Kruskal-Wallis test.
    
    Parameters:
        anova_with_dummies_df (pandas.DataFrame):
            Dataframe containing the data for ANOVA with dummy variables.
        groupby_column (str):
            Column name to group by in the dataframe.
        kruskal_results (scipy.stats.KruskalResult):
            Result object from Kruskal-Wallis test.
        cn (str):
            Column name of the response variable.
        data_by_env (pandas.groupby.DataFrameGroupBy):
            Grouped data by environment.
        cn_data (pandas.Series):
            Series containing response variable data.
    
    Returns:
        float
            Epsilon squared (effect size) value.
    
    Notes:
        Epsilon squared (ε²) is a measure of the proportion of variance in the dependent variable
        accounted for by the independent variable in ANOVA or Kruskal-Wallis tests.
        
        This function computes epsilon squared based on the formula:
        ε² = SSbetween / SStotal
        where SSbetween is the sum of squares between groups and SStotal is the total sum of squares.
        If calculation of SSbetween fails, SSwithin is computed alternatively.
    
    References:
        - Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. Journal of the American Statistical Association, 47(260), 583-621.
        - Rosenthal, R., & Rosnow, R. L. (2008). Essentials of behavioral research: Methods and data analysis. McGraw-Hill Education.
    """
    
    # Epsilon squared (effect size)
    n = len(anova_with_dummies_df)  # Total number of observations
    k = len(anova_with_dummies_df[groupby_column].unique())  # Number of groups
    
    # Calculate manually
    try:
        ss_between = kruskal_results.statistic * n / (k * (k - 1))
        ss_total = kruskal_results.ss  # Access total sum of squares from results
        epsilon_squared = ss_between / ss_total
    
    # Alternative way to calculate sum of squares within groups (SSwithin)
    except:
        ss_within = sum([group[cn].var(ddof=0) * len(group) for name, group in data_by_env])
        
        # Total sum of squares (SStotal) - using variance of entire data
        ss_total = cn_data.var(ddof=0) * len(anova_with_dummies_df)
        
        # Epsilon squared calculation
        epsilon_squared = (kruskal_results.statistic * n) / (k * (k - 1) * ss_within / ss_total)
    
    return epsilon_squared


def compare_the_pairs(anova_with_dummies_df, groupby_column, grouped_data, statements_list, cn_data, cn):
    """
    Compare pairs of environments based on a continuous variable.

    Parameters
    ----------
    anova_with_dummies_df : DataFrame
        DataFrame containing the data with dummy variables.
    groupby_column : str
        Column name for grouping the data.
    grouped_data : DataFrameGroupBy
        Grouped DataFrame based on groupby_column.
    statements_list : list
        List to append the comparison statements.
    cn_data : Series
        Series containing continuous variable data.
    cn : str
        Name of the continuous variable.

    Returns
    -------
    DataFrame
        DataFrame containing transformable data.
    list
        Updated list of comparison statements.
    """
    transformable_df = DataFrame([])
    
    # Compare the pairs
    mask_series = ~anova_with_dummies_df[groupby_column].isnull()
    for pair in itertools.combinations(anova_with_dummies_df[mask_series][groupby_column].unique(), 2):
        env1 = pair[0]; env2 = pair[1]
        env1_data = grouped_data.get_group(env1)[cn]
        env2_data = grouped_data.get_group(env2)[cn]
        t_statistic, p_value = ttest_ind(env1_data, env2_data)
        if p_value < 0.05: statements_list.append(f"    t-test between {env1} and {env2}: t = {t_statistic:.4f}, p = {p_value:.4f}")
    
    # Display a box and whiskers plot
    mask_series = ~cn_data.isnull()
    if mask_series.any():
        transformable_df = anova_with_dummies_df[mask_series]
    
    return transformable_df, statements_list


def append_regression_equation(statements_list, dummy_columns_list, dataset_organization_df, groupby_column, results, cn):
    """
    Append the regression equation to a list of statements.
    
    Parameters:
        statements_list (list):
            List to which the regression equation statement will be appended.
        dummy_columns_list (list):
            List of column names used in the regression equation.
        dataset_organization_df (DataFrame):
            DataFrame containing the dataset and its organization.
        groupby_column (str):
            Name of the column used for grouping.
        results (RegressionResults):
            Result object obtained from regression analysis.
        cn (str):
            Name of the dependent variable.
        
    Returns:
        list
            Updated list of statements.
    """
    
    # Append header for the regression equation
    statements_list.append("Regression Equation:")
    
    # Construct format string for the regression equation
    format_str = "{} = {:.4f} + " + ' + '.join([f"{{:.4f}} * {dc.replace('mean_', '').replace('.0', '')}" for dc in dummy_columns_list])
    
    # Create a mask to filter the dataset for the specified groupby_column
    mask_series = (dataset_organization_df.Variable == groupby_column.replace('_Text', ''))
    
    # Check if the mask has any True values
    if mask_series.any():
        equation_dict = {}
        
        # Extract and parse key-value pairs from the Labels column
        for kv in dataset_organization_df[mask_series].Labels:
            kv_split = split(' *= *', kv, 0)
            if len(kv_split) == 2: equation_dict[float(kv_split[0])] = sub('[^A-Za-z0-9]+', '_', kv_split[1]).strip('_')
        
        # Update format string to replace coefficients with corresponding labels
        for k, v in equation_dict.items(): format_str = format_str.replace('_' + str(k), '_' + v)
    
    # Format and append the regression equation statement
    statements_list.append(format_str.format(
        cn.replace('mean_', ''), results.params['const'], *[results.params[dc] for dc in dummy_columns_list]
    ))
    
    return statements_list

if IS_DEBUG: print("\nWrite up the section prompts")
old_COLUMN_NAME_DESCRIPTION_DICT = COLUMN_NAME_DESCRIPTION_DICT.copy()
kdma_columns = [cn for cn in anova_df.columns if 'KDMA' in cn]
file_path = osp.join(nu.saves_text_folder, 'add_a_section_prompt.txt')
with open(file_path, mode='w', encoding=nu.encoding_type) as f: print('', file=f)
with open(file_path, mode='a', encoding=nu.encoding_type) as f:
    for question_number, feature_title, feature_description, analysis_column in zip(
        range(1, 13),
        [
            'Engagement Order', 'Military Medical Experience', 'Medical Role', 'Untreated Injuries', 'Correctly Treated Injuries',
            'Total Distance Travelled between Engagements', 'Wave Command Counts',
            'Walk Command Counts', 'High Injury Severity', 'Pulse Taken Counts',
            'Hemorrhage Control Time'
        ],
        [
            'responders who prioritize engaging still patients', 'years of military medical experience', 'medical roles', 'count of injuries not treated', 'count of injuries treated correctly',
            'total distance traveled between patient engagements', 'the number of wave commands (commands to "wave if you can hear me") issued',
            'the number of walk commands issued (commands to "walk to the safe space if you can")', 'responders who prioritize high-injury-severity patients', 'the number of pulses taken',
            'time to last hemorrhage controlled'
        ],
        [
            'stills_value', 'YrsMilExp', 'medical_role', 'injury_not_treated_count', 'injury_correctly_treated_count',
            'actual_engagement_distance', 'wave_command_count',
            'walk_command_count', 'prioritize_high_injury_severity_patients', 'pulse_taken_count',
            'time_to_last_hemorrhage_controlled'
        ]
    ):
        print(f"""
Based on the results below, add a blurb to the Analysis section given these particular tests to compare {feature_description} with all KDMA measures we investigated: ST_KDMA_Sim (SoarTech simulator probe responses), ST_KDMA_Text (SoarTech text probe responses), AD_KDMA_Sim (ADEPT simulator probe responses), and AD_KDMA_Text (ADEPT text probe responses). ADEPT (AD) probe responses are measuring Moral Desert (MD) – an attribute that assesses to what extent someone prioritizes patients based on the patient’s moral responsibility for the situation. SoarTech (ST) probe responses are measuring Maximization (Max) – an attribute that assesses to what extent someone performs an exhaustive search through choice alternatives prior to deciding (as opposed to satisficing where people search until they find a good enough option).

""", file=f)
        
        # Calculate Pearson correlation coefficient between the analysis column and each KDMA column
        if analysis_column not in anova_df.columns:
            analysis_column_name = f'mean_{analysis_column}'
        else:
            analysis_column_name = analysis_column
        if is_integer(anova_df[analysis_column_name]) or is_float(anova_df[analysis_column_name]):
            for col in kdma_columns:
                df = anova_df[[analysis_column_name, col]].dropna()
                correlation = df[analysis_column_name].corr(df[col])
                kdma_column_name = col.replace('mean_', '')
                print(f'Correlation between {analysis_column} and {kdma_column_name}: {correlation:.4f}', file=f)
        
        # if IS_DEBUG: print(f"The {analysis_column_name} column has {anova_df[analysis_column_name].nunique()} unique values")
        if anova_df[analysis_column_name].nunique() < 6:
            
            if analysis_column == 'stills_value':
                labels_dict = {0: 'All Stills not Visited First', 1: 'All Stills Visited First'}
            elif analysis_column == 'prioritize_high_injury_severity_patients':
                labels_dict = {0: 'Highest severity patient not engaged first', 1: 'Highest severity patient engaged first'}
            else:
                labels_dict = {column_value: f'{column_value} {feature_description}' for column_value, _ in anova_df.groupby(analysis_column_name)}
            COLUMN_NAME_DESCRIPTION_DICT = old_COLUMN_NAME_DESCRIPTION_DICT.copy()
            COLUMN_NAME_DESCRIPTION_DICT.update(labels_dict)
            
            if IS_DEBUG: print("\nThe columns we want to group by:")
            groupby_columns = [analysis_column_name]
            if IS_DEBUG: print(groupby_columns)
            
            if IS_DEBUG: print("\nThe numeric columns we want to analyze:")
            if IS_DEBUG: print(kdma_columns)
            
            if IS_DEBUG: print("\nContributing factors:")
            for groupby_column in groupby_columns:
                compare_columns(anova_df, groupby_column, kdma_columns, dataset_organization_df, text_io_wrapper=f)
            
            print("""\n\nThough the Kruskal-Wallis test is relatively robust to outliers, we have calculated the IQR fence here:""", file=f)
            for column_value, df in anova_df.groupby(analysis_column_name):
                statements_list = ['\nOutlier test for ' + labels_dict.get(column_value, f'{column_value} {feature_description}')]
                for cn in kdma_columns:
                    outlier_dict = nu.get_statistics(df, [cn]).to_dict()[cn]
                    mask_series = (df[cn] < outlier_dict['25%']) | (df[cn] > outlier_dict['75%'])
                    if mask_series.any():
                        statements_list.append(
                            f"For {cn.replace('mean_', '')}, {mask_series.sum():,} out of {df.shape[0]:,} data points are considered potential outliers"
                            + f" (IQR = ({outlier_dict['25%']:.2f}, {outlier_dict['75%']:.2f}))."
                        )
                if len(statements_list) > 1:
                    print('\n'.join(statements_list), file=f)
        
        print(f"""
    
    KDMA Values and {feature_title} (Question {question_number})
    
    
    a) Question:
    Is there a correlation between the {feature_description} and KDMA scores?
    
    
    b) Analysis:
    
    
    c) Conclusions:
    
    
    d) Other Questions (one could ask of the data):
    
    
    e) Possible Confounding Factors:""", file=f)
COLUMN_NAME_DESCRIPTION_DICT = old_COLUMN_NAME_DESCRIPTION_DICT.copy()