
#!/usr/bin/env python
# Utility Functions to manipulate FRVRS logger data.
# Dave Babbitt <dave.babbitt@gmail.com>
# Author: Dave Babbitt, Data Scientist
# coding: utf-8

# Soli Deo gloria

from datetime import timedelta
from pandas import DataFrame, concat, to_datetime
import csv
import humanize
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas as pd
import random
import re

import warnings
warnings.filterwarnings("ignore")

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
    from frvrs_utils import FRVRSUtilities
    
    fu = FRVRSUtilities(
        data_folder_path=osp.abspath('../data'),
        saves_folder_path=osp.abspath('../saves')
    )
    """
    
    
    def __init__(self, data_folder_path=None, saves_folder_path=None, verbose=False):
        self.verbose = verbose
        
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
        
        # FRVRS log constants
        self.data_logs_folder = osp.join(self.data_folder, 'logs'); os.makedirs(name=self.data_logs_folder, exist_ok=True)
        self.scene_groupby_columns = ['session_uuid', 'scene_index']
        self.patient_groupby_columns = self.scene_groupby_columns + ['patient_id']
        self.injury_groupby_columns = self.patient_groupby_columns + ['injury_id']
        self.right_ordering_list = ['still', 'waver', 'walker']
        
        # List of action types to consider as simulation loggings that can't be directly read by the responder
        self.simulation_actions_list = ['INJURY_RECORD', 'PATIENT_RECORD', 'S_A_L_T_WALKED', 'S_A_L_T_WAVED']
        
        # List of action types to consider as user actions
        self.action_types_list = [
            'TELEPORT', 'S_A_L_T_WALK_IF_CAN', 'S_A_L_T_WAVE_IF_CAN', 'PATIENT_ENGAGED', 'PULSE_TAKEN', 'BAG_ACCESS',
            'TOOL_HOVER', 'TOOL_SELECTED', 'INJURY_TREATED', 'TOOL_APPLIED', 'TAG_SELECTED', 'TAG_APPLIED',
            'BAG_CLOSED', 'TAG_DISCARDED', 'TOOL_DISCARDED'
        ]
        
        # List of command messages to consider as user actions
        self.command_messages_list = [
            'walk to the safe area', 'wave if you can', 'are you hurt', 'reveal injury', 'lay down', 'where are you',
            'can you hear', 'anywhere else', 'what is your name', 'hold still', 'sit up/down', 'stand up'
        ]
        
        # List of action types that assume 1-to-1 interaction
        self.responder_negotiations_list = ['PULSE_TAKEN', 'PATIENT_ENGAGED', 'INJURY_TREATED', 'TAG_APPLIED', 'TOOL_APPLIED', 'PLAYER_GAZE']
        
        # Delayed is yellow per Nick
        self.salt_to_tag_dict = {'DEAD': 'black', 'EXPECTANT': 'gray', 'IMMEDIATE': 'red', 'DELAYED': 'yellow', 'MINIMAL': 'green'}
        self.sort_to_color_dict = {'still': 'black', 'waver': 'red', 'walker': 'green'}
        
        # Reordered per Ewart so that the display is from left to right as follows: dead, expectant, immediate, delayed, minimal, not tagged
        self.salt_types = ['DEAD', 'EXPECTANT', 'IMMEDIATE', 'DELAYED', 'MINIMAL']
        self.tag_colors = ['black', 'gray', 'red', 'yellow', 'green', 'Not Tagged']
        self.error_table_df = pd.DataFrame([
            {'DEAD': 'Exact', 'EXPECTANT': 'Critical', 'IMMEDIATE': 'Critical', 'DELAYED': 'Critical', 'MINIMAL': 'Critical'},
            {'DEAD': 'Over',  'EXPECTANT': 'Exact',    'IMMEDIATE': 'Critical', 'DELAYED': 'Critical', 'MINIMAL': 'Critical'},
            {'DEAD': 'Over',  'EXPECTANT': 'Over',     'IMMEDIATE': 'Exact',    'DELAYED': 'Over',     'MINIMAL': 'Over'},
            {'DEAD': 'Over',  'EXPECTANT': 'Over',     'IMMEDIATE': 'Under',    'DELAYED': 'Exact',    'MINIMAL': 'Over'},
            {'DEAD': 'Over',  'EXPECTANT': 'Over',     'IMMEDIATE': 'Under',    'DELAYED': 'Under',    'MINIMAL': 'Exact'}
        ], columns=self.salt_types, index=self.tag_colors[:-1])
        
        # Define the custom categorical orders
        self.colors_category_order = pd.CategoricalDtype(categories=self.tag_colors, ordered=True)
        self.salt_category_order = pd.CategoricalDtype(categories=self.salt_types, ordered=True)
        self.error_values = ['Exact', 'Critical', 'Over', 'Under']
        self.errors_category_order = pd.CategoricalDtype(categories=self.error_values, ordered=True)
        
        # Hemorrhage control procedures list
        self.hemorrhage_control_procedures_list = ['tourniquet', 'woundpack']
        
        # Sort and SALT columns
        self.salt_columns_list = ['patient_demoted_salt', 'patient_record_salt', 'patient_engaged_salt']
        self.sort_columns_list = ['patient_demoted_sort', 'patient_record_sort', 'patient_engaged_sort']

    ### String Functions ###
    
    
    def format_timedelta(self, timedelta, minimum_unit='seconds'):
        """
        Formats a timedelta object to a string.
        
        If the minimum_unit is seconds, the format is
        '0 sec', '30 sec', '1 min', '1:30', '2 min', etc.
        
        If the minimum_unit is minutes, the format is
        '0:00', '0:30', '1:00', '1:30', '2:00', etc.
        
        Parameters:
            timedelta: A timedelta object.
        
        Returns:
            A string in the appropriate format.
        """
        seconds = timedelta.total_seconds()
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        
        if (minimum_unit == 'seconds'):
            if minutes == 0: formatted_string = f'{seconds} sec'
            elif seconds > 0: formatted_string = f'{minutes}:{seconds:02}'
            else: formatted_string = f'{minutes} min'
        elif (minimum_unit == 'minutes'):
            formatted_string = f'{minutes}:{seconds:02}'
        
        return formatted_string
    
    
    ### List Functions ###
    
    
    def replace_consecutive_elements(self, actions_list, element='PATIENT_ENGAGED'):
        """
        Replaces consecutive elements in a list with a count of how many there are in a row.
        
        Parameters:
            actions_list: A list of elements.
            element: The element to replace consecutive occurrences of.
        
        Returns:
            A list with the consecutive elements replaced with a count of how many there are in a row.
        """
        result = []
        count = 0
        for i in range(len(actions_list)):
            if (actions_list[i] == element): count += 1
            else:
                if (count > 0): result.append(f'{element} x{str(count)}')
                result.append(actions_list[i])
                count = 0
        
        # Handle the last element
        if (count > 0): result.append(f'{element} x{str(count)}')
        
        return(result)
    
    
    ### File Functions ###
    
    
    def get_new_file_name(self, old_file_name):
        """
        Generate a new file name based on the timestamp extracted from the old file.
        
        Parameters
        ----------
        old_file_name : str
            The name of the old log file.

        Returns
        -------
        str
            The new file name with the updated timestamp.
        """

        # Construct the full path of the old file
        old_file_path = '../data/logs/' + old_file_name
        
        # Open the old file and read its content using the CSV reader
        with open(old_file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            
            # Extract the date string from the first row of the CSV file
            for values_list in reader:
                date_str = values_list[2]
                break
            
            # Try to parse the date string into a datetime object
            from datetime import datetime
            try: date_obj = datetime.strptime(date_str, '%m/%d/%Y %H:%M')
            except ValueError: date_obj = datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p')
            
            # Format the datetime object into a new file name in the format YY.MM.DD.HHMM.csv
            new_file_name = date_obj.strftime('%y.%m.%d.%H%M.csv')
            
            # Get the new file path by replacing the old file name with the new file name
            new_file_path = old_file_name.replace(old_file_name.split('/')[-1], new_file_name)
            
            # Return the updated file path with the new file name
            return new_file_path
    
    
    ### Session Functions ###
    
    
    def get_session_groupby(self, frvrs_logs_df=None, mask_series=None, extra_column=None):
        """
        Group the FRVRS logs DataFrame by session UUID, with optional additional grouping by an extra column,
        based on the provided mask and extra column parameters.

        Parameters
        ----------
        frvrs_logs_df : pd.DataFrame, optional
            DataFrame containing the FRVRS logs data, by default None. If None, loads the data from file.
        mask_series : pd.Series, optional
            Boolean mask to filter rows of frvrs_logs_df, by default None.
        extra_column : str, optional
            Additional column for further grouping, by default None.

        Returns
        -------
        pd.DataFrameGroupBy
            GroupBy object grouped by session UUID, and, if provided, the extra column.
        """
        
        # Check if frvrs_logs_df is provided, if not, load it from file
        if frvrs_logs_df is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=self.data_folder,
                saves_folder_path=self.saves_folder
            )
            frvrs_logs_df = nu.load_object('frvrs_logs_df')
        
        # Apply grouping based on the provided parameters
        if (mask_series is None) and (extra_column is None):
            gb = frvrs_logs_df.sort_values(['action_tick']).groupby(['session_uuid'])
        elif (mask_series is None) and (extra_column is not None):
            gb = frvrs_logs_df.sort_values(['action_tick']).groupby(['session_uuid', extra_column])
        elif (mask_series is not None) and (extra_column is None):
            gb = frvrs_logs_df[mask_series].sort_values(['action_tick']).groupby(['session_uuid'])
        elif (mask_series is not None) and (extra_column is not None):
            gb = frvrs_logs_df[mask_series].sort_values(['action_tick']).groupby(['session_uuid', extra_column])
        
        # Return the grouped data
        return gb
    
    
    def get_is_a_one_triage_file(self, session_df, file_name=None, verbose=False):
        """
        Check if a session DataFrame has only one triage run.

        Parameters
        ----------
        session_df : pd.DataFrame
            DataFrame containing session data for a specific file.
        file_name : str, optional
            The name of the file to be checked, by default None.
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Returns
        -------
        bool
            True if the file contains only one triage run, False otherwise.
        """
        # Check if 'is_a_one_triage_file' column exists in the session data frame
        if 'is_a_one_triage_file' in session_df.columns: is_a_one_triage_file = session_df.is_a_one_triage_file.unique().item()
        else:
            
            # Filter out the triage files in this file name
            mask_series = (session_df.scene_type == 'Triage') & (session_df.is_scene_aborted == False)
            mask_series &= (session_df.file_name == file_name)
            
            # Add the scene type for each run
            actions_list = []
            for scene_index, scene_df in session_df[mask_series].groupby('scene_index'): actions_list.append(self.get_scene_type(scene_df))
            
            # Get whether the file has only one triage run
            is_a_one_triage_file = bool(actions_list.count('Triage') == 1)
        
        if verbose:
            print('File name: {}'.format(file_name))
            print('Is a one triage file: {}'.format(is_a_one_triage_file))
        
        # Return True if the file has only one triage run, False otherwise
        return is_a_one_triage_file
    
    
    def get_file_name(self, session_df, verbose=False):
        """
        Retrieve the unique file name associated with the given session DataFrame.

        Parameters
        ----------
        session_df : pd.DataFrame
            DataFrame containing session data.
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Returns
        -------
        str
            The unique file name associated with the session DataFrame.
        """
        
        # Extract the unique file name from the session DataFrame
        file_name = session_df.file_name.unique().item()
        
        if verbose: print('File name: {}'.format(file_name))
        
        # Return the unique file name
        return file_name
    
    
    def get_logger_version(self, session_df, verbose=False):
        """
        Retrieve the unique logger version associated with the given session DataFrame.

        Parameters
        ----------
        session_df : pd.DataFrame
            DataFrame containing session data.
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Returns
        -------
        str
            The unique logger version associated with the session DataFrame.
        """
        
        # Extract the unique logger version from the session DataFrame
        logger_version = session_df.logger_version.unique().item()
        
        # Print verbose output
        if verbose: print('Logger version: {}'.format(logger_version))
        
        # Return the unique logger version
        return logger_version
    
    
    def get_is_duplicate_file(self, session_df, verbose=False):
        """
        Check if a session DataFrame is a duplicate file, i.e., if there is more than one unique file name for the session UUID.
        
        Parameters
        ----------
        session_df : pd.DataFrame
            DataFrame containing session data for a specific session UUID.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        
        Returns
        -------
        bool
            True if the file has duplicate names for the same session UUID, False otherwise.
        """
        
        # Filter all the rows that have more than one unique value in the file_name for the session_uuid
        is_duplicate_file = bool(session_df.file_name.unique().shape[0] > 1)
        
        if verbose: print('Is duplicate file: {}'.format(is_duplicate_file))
        
        # Return True if the file has duplicate names, False otherwise
        return is_duplicate_file
    
    
    ### Scene Functions ###
    
    
    def get_scene_start(self, scene_df, verbose=False):
        """
        Get the start time of the scene DataFrame run.
        
        Parameters
        ----------
        scene_df : pd.DataFrame
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
    
    
    def get_last_engagement(self, scene_df, verbose=False):
        """
        Get the last time a patient was engaged in the given scene DataFrame.
        
        Parameters
        ----------
        scene_df : pd.DataFrame
            DataFrame containing scene-specific data.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        
        Returns
        -------
        int
            Timestamp of the last patient engagement in the scene DataFrame, in milliseconds.
        """
        
        # Get the mask for the PATIENT_ENGAGED actions
        action_mask_series = (scene_df.action_type == 'PATIENT_ENGAGED')
        
        # Find the maximum elapsed time among rows satisfying the action mask
        last_engagement = scene_df[action_mask_series].action_tick.max()
        
        # Print verbose output if enabled
        if verbose: print('Last engagement time: {}'.format(last_engagement))
        
        # Return the last engagement time
        return last_engagement
    
    
    def get_is_scene_aborted(self, scene_df, verbose=False):
        """
        Gets whether or not a scene is aborted.
        
        Parameters
        ----------
        scene_df : pd.DataFrame
            DataFrame containing data for a specific scene.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        
        Returns
        -------
        bool
            True if the scene is aborted, False otherwise.
        """
        
        # Check if the is_scene_aborted column exists in the scene data frame, and get the unique value if it is
        if 'is_scene_aborted' in scene_df.columns: is_scene_aborted = scene_df.is_scene_aborted.unique().item()
        
        else:
            
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
    
    
    def get_scene_type(self, scene_df, verbose=False):
        """
        Gets the type of a scene.
        
        Parameters
        ----------
        scene_df : pd.DataFrame
            DataFrame containing data for a specific scene.
        verbose : bool, optional
            Whether to print verbose output, by default False.

        Returns
        -------
        str
            The type of the scene, e.g., 'Triage', 'Orientation', etc.
        """
        
        # Check if the scene_type column exists in the scene data frame, and get the unique value if it is
        if 'scene_type' in scene_df.columns: scene_type = scene_df.scene_type.unique().item()
        
        else:
            
            # Default scene type is 'Triage'
            scene_type = 'Triage'
            
            # Check if all of the patient IDs in the scene DataFrame contain the string 'mike', and, if so, set the scene type to 'Orientation'
            if scene_df.patient_id.transform(lambda srs: all(srs.str.lower().str.contains('mike'))): scene_type = 'Orientation'
        
        # Return the scene type
        return scene_type
    
    
    def get_scene_end(self, scene_df, verbose=False):
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
    
    
    def get_patient_count(self, scene_df, verbose=False):
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
    
    
    def get_dead_patients(self, scene_df, verbose=False):
        """
        Get a list of unique patient IDs corresponding to patients marked as DEAD or EXPECTANT in a scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            list: List of unique patient IDs marked as DEAD or EXPECTANT in the scene DataFrame.
        """
        
        # Filter the treat-as-dead columns and combine them into a mask series
        mask_series = False
        for column_name in self.salt_columns_list: mask_series |= scene_df[column_name].isin(['DEAD', 'EXPECTANT'])
        
        # Extract the list of dead patients from the filtered mask series
        dead_list = scene_df[mask_series].patient_id.unique().tolist()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Dead patients: {dead_list}')
            display(scene_df)
        
        # Return the list of unique patient IDs marked as DEAD or EXPECTANT
        return dead_list
    
    
    def get_still_patients(self, scene_df, verbose=False):
        """
        Get a list of unique patient IDs corresponding to patients marked as 'still' in a scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            list: List of unique patient IDs marked as 'still' in the scene DataFrame.
        """
        
        # Filter the '_sort' columns and combine them into a mask series
        mask_series = False
        for column_name in self.sort_columns_list: mask_series |= (scene_df[column_name] == 'still')
        
        # Extract the list of still patients from the filtered mask series
        still_list = scene_df[mask_series].patient_id.unique().tolist()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'List of patients marked as still: {still_list}')
            display(scene_df)
        
        # Return the list of unique patient IDs marked as 'still'
        return still_list
    
    
    def get_total_actions(self, scene_df, verbose=False):
        """
        Calculates the total number of user actions within a given scene DataFrame,
        including voice commands with specific messages.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: Total number of user actions in the scene DataFrame.
        """
        
        # Create a boolean mask to filter action types
        mask_series = scene_df.action_type.isin(self.action_types_list)
        
        # Include VOICE_COMMAND actions with specific messages in the mask
        mask_series |= ((scene_df.action_type == 'VOICE_COMMAND') & (scene_df.voice_command_message.isin(self.command_messages_list)))
        
        # Count the number of user actions for the current group
        total_actions = scene_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Total number of user actions: {total_actions}')
            display(scene_df)
        
        # Return the total number of user actions
        return total_actions
    
    
    def get_injury_treated_count(self, scene_df, verbose=False):
        """
        Calculates the number of patients who have received injury treatment in a given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of patients who have received injury treatment.
        """
        
        # Filter for patients with injury_treated set to True
        mask_series = (scene_df.injury_treated_injury_treated == True)
        
        # Count the number of patients who have received injury treatment
        injury_treated_count = scene_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Number of records where injuries were treated: {injury_treated_count}')
            display(scene_df)
        
        # Return the count of records where injuries were treated
        return injury_treated_count
    
    
    def get_injury_not_treated_count(self, scene_df, verbose=False):
        """
        Get the count of records in a scene DataFrame where injuries were not treated.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of patients who have not received injury treatment.
        """
        
        # Create a boolean mask to filter records where injuries were not treated
        mask_series = (scene_df.injury_treated_injury_treated == False)
        
        # Use the mask to filter the DataFrame and count the number of untreated injuries
        injury_not_treated_count = scene_df[mask_series].shape[0]
        
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
        
        # Filter for patients with injury_treated set to True
        mask_series = (scene_df.injury_treated_injury_treated == True)
        
        # Exclude cases where the FRVRS logger incorrectly logs injury_treated_injury_treated_with_wrong_treatment as True
        # This addresses issues with the FRVRS logger logging multiple tool applications
        mask_series &= (scene_df.injury_treated_injury_treated_with_wrong_treatment == False)
        
        # Count the number of patients whose injuries have been correctly treated
        injury_correctly_treated_count = scene_df[mask_series].shape[0]
        
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
        
        # Filter for patients with injury_treated set to True
        mask_series = (scene_df.injury_treated_injury_treated == True)
        
        # Include cases where the FRVRS logger incorrectly logs injury_treated_injury_treated_with_wrong_treatment as True
        mask_series &= (scene_df.injury_treated_injury_treated_with_wrong_treatment == True)
        
        # Count the number of patients whose injuries have been incorrectly treated
        injury_wrongly_treated_count = scene_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Injury wrongly treated count: {injury_wrongly_treated_count}')
            display(scene_df)
        
        return injury_wrongly_treated_count
    
    
    def get_pulse_taken_count(self, scene_df, verbose=False):
        """
        Count the number of 'PULSE_TAKEN' actions in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of "PULSE_TAKEN" actions in the scene DataFrame.
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
    
    
    def get_teleport_count(self, scene_df, verbose=False):
        """
        Count the number of 'TELEPORT' actions in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of "TELEPORT" actions in the scene DataFrame.
        """
    
        # Create a boolean mask to filter actions
        mask_series = scene_df.action_type.isin(['TELEPORT'])
        
        # Count the number of actions
        teleport_count = scene_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Number of TELEPORT actions: {teleport_count}')
            display(scene_df)
        
        # Return the count of actions
        return teleport_count
    
    
    def get_voice_capture_count(self, scene_df, verbose=False):
        """
        Calculates the number of "VOICE_CAPTURE" actions in a given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of "VOICE_CAPTURE" actions in the scene DataFrame.
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
    
    
    def get_walk_command_count(self, scene_df, verbose=False):
        """
        Count the number of 'walk to the safe area' voice command events in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of "walk to the safe area" voice commands in the scene DataFrame.
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
    
    
    def get_wave_command_count(self, scene_df, verbose=False):
        """
        Calculates the number of "wave if you can" voice commands in a given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of "wave if you can" voice commands in the scene DataFrame.
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
        
        # Initialize the measure of right ordering to NaN
        measure_of_right_ordering = np.nan
        
        # Group patients by their SORT category and get lists of their elapsed times
        sort_dict = {}
        for sort, patient_sort_df in scene_df.groupby('patient_sort'):
            
            # Only consider SORT categories included in the right_ordering_list
            if sort in self.right_ordering_list:
                
                # Loop through the SORT patients to add their first interactions to the action list
                action_list = []
                for patient_id in patient_sort_df.patient_id.unique():
                    mask_series = (scene_df.patient_id == patient_id)
                    patient_actions_df = scene_df[mask_series]
                    action_list.append(self.get_first_patient_interaction(patient_actions_df))
                
                # Sort the list of first interactions
                sort_dict[sort] = sorted(action_list)
        
        # Calculate the R-squared adjusted value as a measure of right ordering
        ideal_sequence = []
        for sort in self.right_ordering_list: ideal_sequence.extend(sort_dict.get(sort, []))
        
        ideal_sequence = pd.Series(data=ideal_sequence)
        actual_sequence = ideal_sequence.sort_values(ascending=True)
        X, y = ideal_sequence.values.reshape(-1, 1), actual_sequence.values.reshape(-1, 1)
        if X.shape[0]:
            import statsmodels.api as sm
            X1 = sm.add_constant(X)
            try: measure_of_right_ordering = sm.OLS(y, X1).fit().rsquared_adj
            except: measure_of_right_ordering = np.nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'The measure of right ordering for patients: {measure_of_right_ordering}')
            display(scene_df)
        
        return measure_of_right_ordering
    
    
    def get_first_engagement(self, scene_df, verbose=False):
        """
        Get the timestamp of the first 'PATIENT_ENGAGED' action in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: Timestamp of the first 'PATIENT_ENGAGED' action in the scene DataFrame.
        """
        
        # Filter for actions with the type "PATIENT_ENGAGED"
        action_mask_series = (scene_df.action_type == 'PATIENT_ENGAGED')
        
        # Get the timestamp of the first 'PATIENT_ENGAGED' action
        first_engagement = scene_df[action_mask_series].action_tick.min()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Timestamp of the first PATIENT_ENGAGED action: {first_engagement}')
            display(scene_df)
        
        # Return the timestamp of the first 'PATIENT_ENGAGED' action
        return first_engagement
    
    
    def get_first_treatment(self, scene_df, verbose=False):
        """
        Get the timestamp of the first 'INJURY_TREATED' action in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: Timestamp of the first 'INJURY_TREATED' action in the scene DataFrame.
        """
        
        # Filter for actions with the type "INJURY_TREATED"
        action_mask_series = (scene_df.action_type == 'INJURY_TREATED')
        
        # Get the timestamp of the first 'INJURY_TREATED' action
        first_treatment = scene_df[action_mask_series].action_tick.min()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Timestamp of the first INJURY_TREATED action: {first_treatment}')
            display(scene_df)
        
        # Return the timestamp of the first 'INJURY_TREATED' action
        return first_treatment
    
    
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
        """
        
        # Filter for injuries requiring hemorrhage control procedures
        mask_series = scene_df.injury_record_required_procedure.isin(self.hemorrhage_control_procedures_list)
        hemorrhage_count = scene_df[mask_series].shape[0]
        
        # Filter for hemorrhage-related injuries that have been treated
        mask_series = scene_df.injury_treated_required_procedure.isin(self.hemorrhage_control_procedures_list)
        controlled_count = scene_df[mask_series].shape[0]
        
        # Calculate the percentage of controlled hemorrhage-related injuries
        try: percent_controlled = 100 * controlled_count / hemorrhage_count
        except ZeroDivisionError: percent_controlled = np.nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Percentage of hemorrhage controlled: {percent_controlled:.2f}%')
            display(scene_df)
        
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
            
            # Check if the patient is hemorrhaging
            if self.is_patient_hemorrhaging(patient_df):
                
                # Get the time to hemorrhage control for the patient
                controlled_time = self.get_time_to_hemorrhage_control(patient_df, scene_start=scene_start, verbose=verbose)
                
                # Update the last controlled time if the current controlled time is greater
                last_controlled_time = max(controlled_time, last_controlled_time)
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Time to last hemorrhage controlled: {last_controlled_time} milliseconds')
            display(scene_df)
        
        # Return the time to the last hemorrhage controlled event
        return last_controlled_time
    
    
    ### Patient Functions ###
    
    
    def is_correct_bleeding_tool_applied(self, patient_df, verbose=False):
        """
        Determines whether the correct bleeding control tool (tourniquet or packing gauze) has been applied to a patient in a given scene DataFrame.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool: True if the correct bleeding control tool has been applied, False otherwise.
        """
        
        # Filter for actions where a bleeding control tool was applied
        mask_series = patient_df.tool_applied_sender.isin(['AppliedTourniquet', 'AppliedPackingGauze'])
        
        # Check if any correct bleeding control tools are applied to the patient
        correct_tool_applied = bool(patient_df[mask_series].shape[0])
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Correct bleeding control tool applied: {correct_tool_applied}')
            display(patient_df)
        
        # Return True if the correct tool is applied, False otherwise
        return correct_tool_applied
    
    
    def is_patient_dead(self, patient_df, verbose=False):
        """
        """
        if patient_df.patient_record_salt.isnull().all() and patient_df.patient_engaged_salt.isnull().all(): is_patient_dead = np.nan
        else:
            
            # Add EXPECTANT to the treat-as-dead list per Nick
            mask_series = ~patient_df.patient_record_salt.isnull()
            if patient_df[mask_series].shape[0]:
                patient_record_salt = patient_df[mask_series].patient_record_salt.iloc[0]
                is_patient_dead = (patient_record_salt.isin['DEAD', 'EXPECTANT'])
            else:
                mask_series = ~patient_df.patient_engaged_salt.isnull()
                if patient_df[mask_series].shape[0]:
                    patient_engaged_salt = patient_df[mask_series].patient_engaged_salt.iloc[0]
                    is_patient_dead = (patient_engaged_salt.isin['DEAD', 'EXPECTANT'])
                else: is_patient_dead = np.nan
        
        return is_patient_dead
    
    
    def is_patient_still(self, patient_df, verbose=False):
        """
        """
        if patient_df.patient_record_sort.isnull().all() and patient_df.patient_engaged_sort.isnull().all(): is_patient_still = np.nan
        else:
            mask_series = ~patient_df.patient_record_sort.isnull()
            if patient_df[mask_series].shape[0]:
                patient_record_sort = patient_df[mask_series].patient_record_sort.iloc[0]
                is_patient_still = (patient_record_sort == 'still')
            else:
                mask_series = ~patient_df.patient_engaged_sort.isnull()
                if patient_df[mask_series].shape[0]:
                    patient_engaged_sort = patient_df[mask_series].patient_engaged_sort.iloc[0]
                    is_patient_still = (patient_engaged_sort == 'still')
                else: is_patient_still = np.nan
        
        return is_patient_still
    
    
    def get_max_salt(self, patient_df=None, session_uuid=None, scene_index=None, random_patient_id=None, verbose=False):
        """
        """
        if patient_df is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=self.data_folder,
                saves_folder_path=self.saves_folder
            )
            frvrs_logs_df = nu.load_object('frvrs_logs_df')
            scene_mask_series = (frvrs_logs_df.session_uuid == session_uuid) & (frvrs_logs_df.scene_index == scene_index)
            
            # Get a random patient from within the scene
            if random_patient_id is None: random_patient_id = random.choice(frvrs_logs_df[scene_mask_series].patient_id.unique())
        
            # Get the patient data frame
            mask_series = scene_mask_series & (frvrs_logs_df.patient_id == random_patient_id)
            patient_df = frvrs_logs_df[mask_series]
        
        # Get the max salt value
        mask_series = patient_df.patient_record_salt.isnull()
        try: max_salt = patient_df[~mask_series].patient_record_salt.max()
        except Exception: max_salt = np.nan
        
        if session_uuid is not None: return random_patient_id, max_salt
        else: return max_salt
    
    
    def get_last_tag(self, patient_df, verbose=False):
        """
        """
        
        # Get the last tag value
        mask_series = patient_df.tag_applied_type.isnull()
        try: last_tag = patient_df[~mask_series].tag_applied_type.iloc[-1]
        except Exception: last_tag = np.nan
        
        return last_tag
    
    
    def is_tag_correct(self, patient_df, verbose=False):
        """
        """
        
        # Ensure non-null tag applied type and patient record SALT
        mask_series = ~patient_df.tag_applied_type.isnull() & ~patient_df.patient_record_salt.isnull()
        assert patient_df[mask_series].shape[0], "You need to have both a tag_applied_type and a patient_record_salt"
        
        # Get the last tag value
        last_tag = self.get_last_tag(patient_df)
        
        # Get the max salt value
        max_salt = self.get_max_salt(patient_df)
        
        # Get the predicted tag value
        try: predicted_tag = self.salt_to_tag_dict.get(max_salt, np.nan)
        except Exception: predicted_tag = np.nan
        
        # Get if tag is correct
        try: is_tag_correct = bool(last_tag == predicted_tag)
        except Exception: is_tag_correct = np.nan
        
        return is_tag_correct
    
    
    def get_first_patient_interaction(self, patient_df, verbose=False):
        """
        """
        mask_series = patient_df.action_type.isin(self.responder_negotiations_list)
        engagement_start = patient_df[mask_series].action_tick.min()
        
        return engagement_start
    
    
    def get_last_patient_interaction(self, patient_df, verbose=False):
        """
        """
        mask_series = (patient_df.action_type.isin(self.action_types_list))
        mask_series |= ((patient_df.action_type == 'VOICE_COMMAND') & (patient_df.voice_command_message.isin(self.command_messages_list)))
        engagement_end = patient_df[mask_series].action_tick.max()
        
        return engagement_end
    
    
    def is_patient_gazed_at(self, patient_df, verbose=False):
        """
        """
        
        # Did the responder gaze at this patient at least once?
        mask_series = (patient_df.action_type == 'PLAYER_GAZE')
        gazed_at_patient = bool(patient_df[mask_series].shape[0])
        
        return gazed_at_patient
    
    
    def get_wanderings(self, patient_df, verbose=False):
        """
        """
        x_dim = []; z_dim = []
        location_cns_list = [
            'patient_demoted_position', 'patient_engaged_position', 'patient_record_position', 'player_gaze_location'
        ]
        
        # Create the melt data frame
        mask_series = False
        for location_cn in location_cns_list: mask_series |= ~patient_df[location_cn].isnull()
        columns_list = ['action_tick'] + location_cns_list
        melt_df = patient_df[mask_series][columns_list].sort_values('action_tick')
        
        # Split the location columns into two columns using melt and convert the location into a tuple
        df = melt_df.melt(id_vars=['action_tick'], value_vars=location_cns_list).sort_values('action_tick')
        mask_series = ~df['value'].isnull()
        if verbose: display(df[mask_series])
        srs = df[mask_series]['value'].map(lambda x: eval(x))
        
        # Grab the x and z dimensions and return them
        x_dim.extend(srs.map(lambda x: x[0]).values)
        z_dim.extend(srs.map(lambda x: x[2]).values)
        
        return x_dim, z_dim
    
    
    def get_wrapped_label(self, patient_df, verbose=False):
        """
        """
        
        # Pick from among the sort columns whichever value is not null and use that in the label
        columns_list = ['patient_demoted_sort', 'patient_engaged_sort', 'patient_record_sort']
        srs = patient_df[columns_list].apply(pd.Series.notnull, axis='columns').sum()
        mask_series = (srs > 0)
        sort_cns_list = srs[mask_series].index.tolist()[0]
    
        # Generate a wrapped label
        patient_id = patient_df.patient_id.unique().item()
        label = patient_id.replace(' Root', ' (') + patient_df[sort_cns_list].dropna().iloc[-1] + ')'
        import textwrap
        label = '\n'.join(textwrap.wrap(label, width=20))
        
        return label
    
    
    def is_patient_hemorrhaging(self, patient_df, verbose=False):
        """
        """
        mask_series = patient_df.injury_record_required_procedure.isin(self.hemorrhage_control_procedures_list)
        
        return bool(patient_df[mask_series].shape[0])
    
    
    def get_time_to_hemorrhage_control(self, patient_df, scene_start=None, verbose=False):
        """
        """
        
        # Get the injury treatments where the responder is not confused from the bad feedback
        if scene_start is None: scene_start = self.get_first_patient_interaction(patient_df)
        controlled_time = 0
        on_columns_list = ['injury_id']
        merge_columns_list = ['action_tick'] + on_columns_list
        for control_procedure in self.hemorrhage_control_procedures_list:
            
            mask_series = patient_df.injury_record_required_procedure.isin([control_procedure])
            hemorrhage_df = patient_df[mask_series][merge_columns_list]
            if verbose: display(hemorrhage_df)
            
            mask_series = patient_df.injury_treated_required_procedure.isin([control_procedure])
            controlled_df = patient_df[mask_series][merge_columns_list]
            if verbose: display(controlled_df)
            
            df = hemorrhage_df.merge(controlled_df, on=on_columns_list, suffixes=('_hemorrhage', '_controlled'))
            if verbose: display(df)
            
            controlled_time = max(controlled_time, df.action_tick_controlled.max() - scene_start)
        
        return controlled_time
    
    
    def get_patient_engagement_count(self, patient_df, verbose=False):
        """
        """
        mask_series = (patient_df.action_type == 'PATIENT_ENGAGED')
        
        return patient_df[mask_series].shape[0]
    
    
    ### Injury Functions ###
    
    
    def is_injury_correctly_treated(self, injury_df, verbose=False):
        """
        """
        mask_series = (injury_df.injury_treated_injury_treated == True)
        
        # The FRVRS logger has trouble logging multiple tool applications,
        # so injury_treated_injury_treated_with_wrong_treatment == True
        # remains and another INJURY_TREATED is never logged, even though
        # the right tool was applied after that.
        mask_series &= (injury_df.injury_treated_injury_treated_with_wrong_treatment == False)
        
        return bool(injury_df[mask_series].shape[0])
    
    
    def is_hemorrhage_controlled(self, injury_df, verbose=False):
        """
        """
        mask_series = injury_df.injury_record_required_procedure.isin(self.hemorrhage_control_procedures_list)
        mask_series |= injury_df.injury_treated_required_procedure.isin(self.hemorrhage_control_procedures_list)
        
        return bool(injury_df[mask_series].shape[0] == 2)
    
    
    def is_injury_hemorrhage(self, injury_df, verbose=False):
        """
        """
        mask_series = injury_df.injury_record_required_procedure.isin(self.hemorrhage_control_procedures_list)
        mask_series |= injury_df.injury_treated_required_procedure.isin(self.hemorrhage_control_procedures_list)
        
        return bool(1 <= injury_df[mask_series].shape[0] <= 2)
    
    
    def is_bleeding_correctly_treated(self, injury_df, verbose=False):
        """
        """
        mask_series = injury_df.injury_treated_required_procedure.isin(self.hemorrhage_control_procedures_list)
        mask_series &= (injury_df.injury_treated_injury_treated == True)
        
        # The FRVRS logger has trouble logging multiple tool applications,
        # so injury_treated_injury_treated_with_wrong_treatment == True
        # remains and another INJURY_TREATED is never logged, even though
        # the right tool was applied after that.
        mask_series &= (injury_df.injury_treated_injury_treated_with_wrong_treatment == False)
        
        bleeding_treated = bool(injury_df[mask_series].shape[0])
        
        return bleeding_treated
    
    
    def get_injury_correctly_treated_time(self, injury_df, verbose=False):
        """
        """
        mask_series = (injury_df.injury_treated_injury_treated == True)
        
        # The FRVRS logger has trouble logging multiple tool applications,
        # so injury_treated_injury_treated_with_wrong_treatment == True
        # remains and another INJURY_TREATED is never logged, even though
        # the right tool was applied after that.
        mask_series &= (injury_df.injury_treated_injury_treated_with_wrong_treatment == False)
        
        bleeding_treated_time = injury_df[mask_series].action_tick.max()
        
        return bleeding_treated_time
    
    ### Plotting Functions ###
    
    
    def visualize_order_of_engagement(self, scene_df, nu=None, verbose=False):
        """
        """
        
        # Get the engagement order
        engagement_starts_list = []
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            engagement_start = self.get_first_patient_interaction(patient_df)
            mask_series = (patient_df.action_tick == engagement_start) & ~patient_df.location_id.isnull()
            if patient_df[mask_series].shape[0]:
                engagement_location = eval(patient_df[mask_series].location_id.tolist()[0])
                location_tuple = (engagement_location[0], engagement_location[2])
                engagement_tuple = (patient_id, engagement_start, location_tuple)
                engagement_starts_list.append(engagement_tuple)
        engagement_order = sorted(engagement_starts_list, key=lambda x: x[1], reverse=False)
        
        # Create a figure and add a subplot
        fig, ax = plt.subplots(figsize=(18, 9))
        
        # Show the positions of patients recorded and engaged at our scene
        if nu is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=self.data_folder,
                saves_folder_path=self.saves_folder
            )
        color_cycler = nu.get_color_cycler(scene_df.groupby('patient_id').size().shape[0])
        
        for engagement_tuple, face_color_dict in zip(engagement_order, color_cycler()):
    
            # Get the entire patient history
            patient_id = engagement_tuple[0]
            mask_series = (scene_df.patient_id == patient_id)
            patient_df = scene_df[mask_series]
            
            # Get all the wanderings
            x_dim, z_dim = self.get_wanderings(patient_df)
            
            # Generate a wrapped label
            label = self.get_wrapped_label(patient_df)
            
            # Plot the ball and chain
            face_color = face_color_dict['color']
            ax.plot(x_dim, z_dim, color=face_color, alpha=1.0, label=label)
            ax.scatter(x_dim, z_dim, color=face_color, alpha=1.0)
            
            # Get the first of the engagement coordinates and label the patient there
            location_tuple = engagement_tuple[2]
            plt.annotate(label, location_tuple, textcoords='offset points', xytext=(0, -8), ha='center', va='center')
        
        # Create new FancyArrowPatch objects for each arrow and add the arrows to the plot
        import matplotlib.patches as mpatches
        for (first_tuple, second_tuple) in zip(engagement_order[:-1], engagement_order[1:]):
            first_location = first_tuple[2]
            second_location = second_tuple[2]
            if verbose: print(first_location, second_location)
            arrow_obj = mpatches.FancyArrowPatch(
                first_location, second_location,
                mutation_scale=50, color='grey', linewidth=2, alpha=0.25
            )
            ax.add_patch(arrow_obj)
        
        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        
        # Move the left and right borders to make room for the legend
        left_lim, right_lim = ax.get_xlim()
        xlim_tuple = ax.set_xlim(left_lim-1.5, right_lim+1.5)
        ax.legend(loc='best')
        
        # Add title
        title = f'Order-of-Engagement Map for UUID {scene_df.session_uuid.unique().item()} ({humanize.ordinal(scene_df.scene_index.unique().item()+1)} Scene)'
        ax.set_title(title);
    
    
    def visualize_player_movement(self, scene_mask, title=None, save_only=False, nu=None, verbose=False):
        """
        Visualizes the player movement for the given session mask in a 2D plot.
    
        Parameters:
            scene_mask (pandas.Series): A boolean mask indicating which rows of the frvrs_logs_df DataFrame belong to the current scene.
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
        - Use `scene_mask` to filter the data for a specific scene.
        - Set `save_only` to True to save the plot as a PNG file with the specified `title`.
        - Set `verbose` to True to enable verbose printing.
        """
        
        # Load the notebook utilities if not provided
        if nu is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=self.data_folder,
                saves_folder_path=self.saves_folder
            )
        frvrs_logs_df = nu.load_object('frvrs_logs_df')
    
        # Check if saving to a file is requested
        if save_only:
            assert title is not None, "To save, you need a title"
            file_path = osp.join(self.saves_folder, 'png', re.sub(r'\W+', '_', str(title)).strip('_').lower() + '.png')
            filter = not osp.exists(file_path)
        else: filter = True
    
        # If the filter is True, then visualize the player movement
        if filter:
            import textwrap
            
            # Turn off interactive plotting if saving to a file
            if save_only: plt.ioff()
            
            # Create a figure and add a subplot
            fig, ax = plt.subplots(figsize=(18, 9))
            
            # Show the positions of patients recorded and engaged at our time group and session UUID
            color_cycler = nu.get_color_cycler(frvrs_logs_df[scene_mask].groupby('patient_id').size().shape[0])
            for (patient_id, patient_df), face_color_dict in zip(frvrs_logs_df[scene_mask].sort_values('action_tick').groupby('patient_id'), color_cycler()):
                
                # Get all the wanderings
                x_dim, z_dim = self.get_wanderings(patient_df)
                
                # Generate a wrapped label
                label = self.get_wrapped_label(patient_df)
    
                # Plot the ball and chain
                face_color = face_color_dict['color']
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
            mask_series = (frvrs_logs_df.action_type == 'PLAYER_LOCATION') & scene_mask
            locations_df = frvrs_logs_df[mask_series]
            if locations_df.shape[0]:
                label = 'locations'
                locations_df = locations_df.sort_values(['action_tick'])
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
            mask_series = (frvrs_logs_df.action_type == 'TELEPORT') & scene_mask
            teleports_df = frvrs_logs_df[mask_series]
            if teleports_df.shape[0]:
                label = 'teleportations'
                teleports_df = teleports_df.sort_values(['action_tick'])
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
        title_str='slowest action to control time', nu=None, verbose=False
    ):
        """
        Get time group with some edge case and visualize the player movement there
        """
        
        if mask_series is None: mask_series = [True] * df.shape[0]
        df1 = df[mask_series].sort_values(
            [sorting_column], ascending=[is_ascending]
        ).head(1)
        if df1.shape[0]:
            session_uuid = df1.session_uuid.squeeze()
            scene_index = df1.scene_index.squeeze()
            if nu is None:
                from notebook_utils import NotebookUtilities
                nu = NotebookUtilities(
                    data_folder_path=self.data_folder,
                    saves_folder_path=self.saves_folder
                )
            frvrs_logs_df = nu.load_object('frvrs_logs_df')
            base_mask_series = (frvrs_logs_df.session_uuid == session_uuid) & (frvrs_logs_df.scene_index == scene_index)
            
            title = f'Location Map for UUID {session_uuid} ({humanize.ordinal(scene_index+1)} Scene)'
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
            self.visualize_player_movement(base_mask_series, title=title, nu=nu)
    
    
    def show_timelines(self, frvrs_logs_df=None, random_session_uuid=None, random_scene_index=None, color_cycler=None, verbose=False):
        """
        """
        if frvrs_logs_df is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=self.data_folder,
                saves_folder_path=self.saves_folder
            )
            frvrs_logs_df = nu.load_object('frvrs_logs_df')
        
        # Get a random session
        if random_session_uuid is None:
            random_session_uuid = random.choice(self.frvrs_logs_df.session_uuid.unique())
        
        # Get a random scene from within the session
        if random_scene_index is None:
            mask_series = (self.frvrs_logs_df.session_uuid == random_session_uuid)
            random_scene_index = random.choice(self.frvrs_logs_df[mask_series].scene_index.unique())
        
        # Get the scene mask
        scene_mask_series = (self.frvrs_logs_df.session_uuid == random_session_uuid) & (self.frvrs_logs_df.scene_index == random_scene_index)
        
        # Get the event time and elapsed time of each person engaged
        mask_series = scene_mask_series & self.frvrs_logs_df.action_type.isin([
            'PATIENT_ENGAGED', 'INJURY_TREATED', 'PULSE_TAKEN', 'TAG_APPLIED', 'TOOL_APPLIED'
        ])
        columns_list = ['patient_id', 'action_tick']
        patient_engagements_df = self.frvrs_logs_df[mask_series][columns_list].sort_values(['action_tick'])
        if verbose: display(patient_engagements_df)
        if patient_engagements_df.shape[0]:
            
            # For each patient, get a timeline of every reference on or before engagement
            if color_cycler is None:
                from notebook_utils import NotebookUtilities
                nu = NotebookUtilities()
                color_cycler = nu.get_color_cycler(len(patient_engagements_df.patient_id.unique()))
            hlineys_list = []; hlinexmins_list = []; hlinexmaxs_list = []; hlinecolors_list = []; hlinelabels_list = []
            hlineaction_types_list = []; vlinexs_list = []
            left_lim = 999999; right_lim = -999999
            for (random_patient_id, patient_df), (y, face_color_dict) in zip(patient_engagements_df.groupby('patient_id'), enumerate(color_cycler())):
                
                # Get the broad horizontal line parameters
                hlineys_list.append(y)
                face_color = face_color_dict['color']
                hlinecolors_list.append(face_color)
                hlinelabels_list.append(random_patient_id)
                
                # Create the filter for the first scene
                mask_series = scene_mask_series & (self.frvrs_logs_df.patient_id == random_patient_id)
                action_tick = patient_df.action_tick.max()
                mask_series &= (self.frvrs_logs_df.action_tick <= action_tick)
                
                df1 = self.frvrs_logs_df[mask_series].sort_values(['action_tick'])
                if verbose: display(df1)
                if df1.shape[0]:
                    
                    # Get the fine horizontal line parameters and plot dimensions
                    xmin = df1.action_tick.min(); hlinexmins_list.append(xmin);
                    if xmin < left_lim: left_lim = xmin
                    xmax = df1.action_tick.max(); hlinexmaxs_list.append(xmax);
                    if xmax > right_lim: right_lim = xmax
                    
                    # Get the vertical line parameters
                    mask_series = df1.action_type.isin(['SESSION_END', 'SESSION_START'])
                    for x in df1[mask_series].action_tick:
                        vlinexs_list.append(x)
                    
                    # Get the action type annotation parameters
                    mask_series = df1.action_type.isin(['INJURY_TREATED', 'PATIENT_ENGAGED', 'PULSE_TAKEN', 'TAG_APPLIED', 'TOOL_APPLIED'])
                    for label, action_type_df in df1[mask_series].groupby('action_type'):
                        for x in action_type_df.action_tick:
                            annotation_tuple = (label.lower().replace('_', ' '), x, y)
                            hlineaction_types_list.append(annotation_tuple)
        
        ax = plt.figure(figsize=(18, 9)).add_subplot(1, 1, 1)
        
        # Add the timelines to the figure subplot axis
        if verbose: print(hlineys_list, hlinexmins_list, hlinexmaxs_list, hlinecolors_list)
        line_collection_obj = ax.hlines(hlineys_list, hlinexmins_list, hlinexmaxs_list, colors=hlinecolors_list)
        
        # Label each timeline with the appropriate patient name
        for label, x, y in zip(hlinelabels_list, hlinexmins_list, hlineys_list):
            plt.annotate(label.replace(' Root', ''), (x, y), textcoords='offset points', xytext=(0, -8), ha='left')
        
        # Annotate the action types along their timeline
        for annotation_tuple in hlineaction_types_list:
            label, x, y = annotation_tuple
            plt.annotate(label, (x, y), textcoords='offset points', xytext=(0, 0), va='center', rotation=90, fontsize=6)
        
        # Mark any session boundaries with a vertical line
        ymin, ymax = ax.get_ylim()
        line_collection_obj = ax.vlines(vlinexs_list, ymin=ymin, ymax=ymax)
        
        # Remove the ticks and tick labels from the y axis
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Move the top and right border out so that the annotations don't cross it
        plt.subplots_adjust(top=1.5)
        xlim_tuple = ax.set_xlim(left_lim-10_000, right_lim+10_000)
        
        # Set the title and labels
        ax.set_title(f'Multi-Patient Timeline for UUID {random_session_uuid} and Scene {random_scene_index}')
        ax.set_xlabel('Elapsed Time since Scene Start')
        
        # tick_labels = ax.get_xticklabels()
        # print(tick_labels)
        
        from matplotlib.text import Text
        ax.set_xticklabels([
            Text(300000.0, 0, humanize.precisedelta(timedelta(milliseconds=300000.0)).replace(', ', ',\n').replace(' and ', ' and\n')),
            Text(400000.0, 0, humanize.precisedelta(timedelta(milliseconds=400000.0)).replace(', ', ',\n').replace(' and ', ' and\n')),
            Text(500000.0, 0, humanize.precisedelta(timedelta(milliseconds=500000.0)).replace(', ', ',\n').replace(' and ', ' and\n')),
            Text(600000.0, 0, humanize.precisedelta(timedelta(milliseconds=600000.0)).replace(', ', ',\n').replace(' and ', ' and\n')),
            Text(700000.0, 0, humanize.precisedelta(timedelta(milliseconds=700000.0)).replace(', ', ',\n').replace(' and ', ' and\n')),
            Text(800000.0, 0, humanize.precisedelta(timedelta(milliseconds=800000.0)).replace(', ', ',\n').replace(' and ', ' and\n')),
            Text(900000.0, 0, humanize.precisedelta(timedelta(milliseconds=900000.0)).replace(', ', ',\n').replace(' and ', ' and\n')),
            Text(1000000.0, 0, humanize.precisedelta(timedelta(milliseconds=1000000.0)).replace(', ', ',\n').replace(' and ', ' and\n')),
            Text(1100000.0, 0, humanize.precisedelta(timedelta(milliseconds=1100000.0)).replace(', ', ',\n').replace(' and ', ' and\n'))
        ]);
    
        return random_session_uuid, random_scene_index
    
    
    def plot_grouped_box_and_whiskers(self, transformable_df, x_column_name, y_column_name, x_label, y_label, transformer_name='min', is_y_temporal=True):
        """
        """
        import seaborn as sns
        
        # Get the transformed data frame
        if transformer_name is None: transformed_df = transformable_df
        else:
            groupby_columns = self.scene_groupby_columns
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
            from datetime import timedelta
            for text_obj in ax.get_yticklabels():
                text_obj.set_text(
                    humanize.precisedelta(timedelta(milliseconds=text_obj.get_position()[1])).replace(', ', ',\n').replace(' and ', ' and\n')
                )
                yticklabels_list.append(text_obj)
            ax.set_yticklabels(yticklabels_list);
        
        plt.show()
    
    
    def show_gaze_timeline(self, frvrs_logs_df=None, random_session_uuid=None, random_scene_index=None, consecutive_cutoff=600, patient_color_dict=None, verbose=False):
        """
        """
        if frvrs_logs_df is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=self.data_folder,
                saves_folder_path=self.saves_folder
            )
            frvrs_logs_df = nu.load_object('frvrs_logs_df')
        
        # Get a random session
        if random_session_uuid is None:
            mask_series = (frvrs_logs_df.action_type.isin(['PLAYER_GAZE']))
            mask_series &= ~frvrs_logs_df.player_gaze_patient_id.isnull()
            random_session_uuid = random.choice(frvrs_logs_df[mask_series].session_uuid.unique())
        
        # Get a random scene from within the session
        if random_scene_index is None:
            mask_series = (frvrs_logs_df.session_uuid == random_session_uuid)
            mask_series &= (frvrs_logs_df.action_type.isin(['PLAYER_GAZE']))
            mask_series &= ~frvrs_logs_df.player_gaze_patient_id.isnull()
            random_scene_index = random.choice(frvrs_logs_df[mask_series].scene_index.unique())
        
        # Get the scene mask
        scene_mask_series = (frvrs_logs_df.session_uuid == random_session_uuid) & (frvrs_logs_df.scene_index == random_scene_index)
        
        # Get the elapsed time of the player gaze
        mask_series = scene_mask_series & (frvrs_logs_df.action_type.isin(['PLAYER_GAZE', 'SESSION_END', 'SESSION_START']))
        columns_list = ['action_tick', 'patient_id']
        patient_gazes_df = frvrs_logs_df[mask_series][columns_list].sort_values(['action_tick'])
        patient_gazes_df['time_diff'] = patient_gazes_df.action_tick.diff()
        for patient_id in patient_gazes_df.patient_id.unique():
            patient_gazes_df = self.replace_consecutive_rows(
                patient_gazes_df, 'patient_id', patient_id, time_diff_column='time_diff', consecutive_cutoff=consecutive_cutoff
            )
        if verbose: display(patient_gazes_df.tail(20))
        if patient_gazes_df.shape[0]:
            
            # For the patient, get a timeline of every reference of gazement
            hlineys_list = []; hlinexmins_list = []; hlinexmaxs_list = []; hlinelabels_list = []
            hlineaction_types_list = []; vlinexs_list = []
            left_lim = 999999; right_lim = -999999
            
            # Get the broad horizontal line parameters
            y = 0
            hlineys_list.append(y)
            hlinelabels_list.append('Player Gaze')
            
            # Get the fine horizontal line parameters and plot dimensions
            xmin = patient_gazes_df.action_tick.min(); hlinexmins_list.append(xmin);
            if xmin < left_lim: left_lim = xmin
            xmax = patient_gazes_df.action_tick.max(); hlinexmaxs_list.append(xmax);
            if xmax > right_lim: right_lim = xmax
            
            for row_index, row_series in patient_gazes_df.iterrows():
                # for cn in columns_list: exec(f'{cn} = row_series.{cn}')
                action_tick = row_series.action_tick
                patient_id = row_series.patient_id
                time_diff = row_series.time_diff
                
                # Get the player gaze annotation parameters
                if not pd.isna(patient_id):
                    annotation_tuple = (patient_id, action_tick, y)
                    hlineaction_types_list.append(annotation_tuple)
        
        ax = plt.figure(figsize=(18, 9)).add_subplot(1, 1, 1)
        
        # Add the timelines to the figure subplot axis
        if verbose: print(hlineys_list, hlinexmins_list, hlinexmaxs_list)
        line_collection_obj = ax.hlines(hlineys_list, hlinexmins_list, hlinexmaxs_list)
        
        # Label each timeline with the appropriate patient name
        for label, x, y in zip(hlinelabels_list, hlinexmins_list, hlineys_list): plt.annotate(label, (x, y), textcoords='offset points', xytext=(0, -8), ha='left')
        
        # Annotate the patients who have been gazed at along their timeline
        if patient_color_dict is None:
            scene_df = frvrs_logs_df[scene_mask_series]
            patient_color_dict = {}
            mask_series = ~scene_df.patient_record_salt.isnull()
            for patient_id, patient_df in scene_df[mask_series].groupby('patient_id'):
                patient_color_dict[patient_id] = self.salt_to_tag_dict[self.get_max_salt(patient_df=patient_df)]
        consec_regex = re.compile(r' x\d+$')
        for annotation_tuple in hlineaction_types_list:
            patient_id, x, y = annotation_tuple
            color = patient_color_dict[consec_regex.sub('', patient_id)]
            if verbose: print(patient_id, color)
            plt.annotate(patient_id, (x, y), textcoords='offset points', xytext=(0, 0), va='center', rotation=90, fontsize=7, color=color)
        
        # Mark any session boundaries with a vertical line
        ymin, ymax = ax.get_ylim()
        line_collection_obj = ax.vlines(vlinexs_list, ymin=ymin, ymax=ymax)
        
        # Remove the ticks and tick labels from the y axis
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Move the top and right border out so that the annotations don't cross it
        # plt.subplots_adjust(top=1.5)
        # xlim_tuple = ax.set_xlim(left_lim, right_lim+10_000)
        
        # Set the title and labels
        ax.set_title(f'Player Gaze Timeline for UUID {random_session_uuid} and Scene {random_scene_index}')
        ax.set_xlabel('Elapsed Time since Scene Start')
        
        # Humanize x tick labels
        xticklabels_list = []
        for text_obj in ax.get_xticklabels():
            text_obj.set_text(humanize.precisedelta(timedelta(milliseconds=text_obj.get_position()[0])).replace(', ', ',\n').replace(' and ', ' and\n'))
            xticklabels_list.append(text_obj)
        ax.set_xticklabels(xticklabels_list);
    
        return random_session_uuid, random_scene_index
    
    ### Pandas Functions ###

    def get_statistics(self, describable_df, columns_list):
        """
        """
        df = describable_df[columns_list].describe().rename(index={'std': 'SD'})
        
        if ('mode' not in df.index):
            
            # Create the mode row dictionary
            row_dict = {cn: describable_df[cn].mode().iloc[0] for cn in columns_list}
            
            # Convert the row dictionary to a data frame to match the df structure
            row_df = DataFrame([row_dict], index=['mode'])
            
            # Append the row data frame to the df data frame
            df = concat([df, row_df], axis='index', ignore_index=False)
        
        if ('median' not in df.index):
            
            # Create the median row dictionary
            row_dict = {cn: describable_df[cn].median() for cn in columns_list}
            
            # Convert the row dictionary to a data frame to match the df structure
            row_df = DataFrame([row_dict], index=['median'])
            
            # Append the row data frame to the df data frame
            df = concat([df, row_df], axis='index', ignore_index=False)
        
        index_list = ['mean', 'mode', 'median', 'SD', 'min', '25%', '50%', '75%', 'max']
        mask_series = df.index.isin(index_list)
        
        return df[mask_series].reindex(index_list)
    
    
    def show_time_statistics(self, describable_df, columns_list):
        """
        """
        df = self.get_statistics(describable_df, columns_list).applymap(lambda x: self.format_timedelta(timedelta(milliseconds=int(x))), na_action='ignore').T
        df.SD = df.SD.map(lambda x: '±' + str(x))
        display(df)
    
    
    def set_scene_indexes(self, df):
        """
        Section off player actions by session start and end. We are finding log entries above the first SESSION_START and below the last SESSION_END.
        
        Parameters:
            df: A Pandas DataFrame containing the player action data with its index reset.
        
        Returns:
            A Pandas DataFrame with the `scene_index` column added.
        """
    
        # Set the whole file to zero first
        df = df.sort_values('action_tick')
        scene_index = 0
        df['scene_index'] = scene_index
        
        # Delineate runs by the session end below them
        mask_series = (df.action_type == 'SESSION_END')
        lesser_idx = df[mask_series].index.min()
        mask_series &= (df.index > lesser_idx)
        while df[mask_series].shape[0]:
            
            # Find this session end as the bottom
            greater_idx = df[mask_series].index.min()
            
            # Add everything above that to this run
            mask_series = (df.index > lesser_idx) & (df.index <= greater_idx)
            scene_index += 1
            df.loc[mask_series, 'scene_index'] = scene_index
            
            # Delineate runs by the session end below them
            lesser_idx = greater_idx
            mask_series = (df.action_type == 'SESSION_END') & (df.index > lesser_idx)
        
        # Find the last session start
        mask_series = (df.action_type == 'SESSION_START')
        lesser_idx = df[mask_series].index.max()
        
        # Add everything below that to the last run
        mask_series = (df.index >= lesser_idx)
        df.loc[mask_series, 'scene_index'] = scene_index
        
        # Convert the scene index column to int64
        df.scene_index = df.scene_index.astype('int64')
        
        return df
    
    
    def set_mcivr_metrics_types(self, action_type, df, row_index, row_series):
        """
        Set the MCI-VR metrics types for a given action type and row series.
    
        Parameters:
            action_type: The action type.
            df: The DataFrame containing the MCI-VR metrics.
            row_index: The index of the row in the DataFrame to set the metrics for.
            row_series: The row series containing the MCI-VR metrics.
    
        Returns:
            The DataFrame containing the MCI-VR metrics with new columns.
        """
    
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
        elif (action_type == 'S_A_L_T_WALKED'): # SALTWalked
            df.loc[row_index, 's_a_l_t_walked_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_walked_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_walked_patient_id'] = row_series[6] # patientId
        elif (action_type == 'S_A_L_T_WALK_IF_CAN'): # SALTWalkIfCan
            df.loc[row_index, 's_a_l_t_walk_if_can_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_walk_if_can_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_walk_if_can_patient_id'] = row_series[6] # patientId
        elif (action_type == 'S_A_L_T_WAVED'): # SALTWave
            df.loc[row_index, 's_a_l_t_waved_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_waved_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_waved_patient_id'] = row_series[6] # patientId
        elif (action_type == 'S_A_L_T_WAVE_IF_CAN'): # SALTWaveIfCan
            df.loc[row_index, 's_a_l_t_wave_if_can_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_wave_if_can_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_wave_if_can_patient_id'] = row_series[6] # patientId
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
            tool_applied_patient_id = row_series[4]
            if ' Root' in tool_applied_patient_id:
                df.loc[row_index, 'tool_applied_patient_id'] = tool_applied_patient_id # patientId
            df.loc[row_index, 'tool_applied_type'] = row_series[5] # type
            df.loc[row_index, 'tool_applied_attachment_point'] = row_series[6] # attachmentPoint
            df.loc[row_index, 'tool_applied_tool_location'] = row_series[7] # toolLocation
            df.loc[row_index, 'tool_applied_data'] = row_series[8] # data
            df.loc[row_index, 'tool_applied_sender'] = row_series[9] # sender
            df.loc[row_index, 'tool_applied_attach_message'] = row_series[10] # attachMessage
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
    
        return df
    
    
    def process_files(self, sub_directory_df, sub_directory, file_name):
        """
        """
        file_path = osp.join(sub_directory, file_name)
        try:
            version_number = '1.0'
            file_df = pd.read_csv(file_path, header=None, index_col=False)
        except:
            version_number = '1.3'
            rows_list = []
            with open(file_path, 'r') as f:
                import csv
                reader = csv.reader(f, delimiter=',', quotechar='"')
                for values_list in reader:
                    if (values_list[-1] == ''): values_list.pop(-1)
                    rows_list.append({i: v for i, v in enumerate(values_list)})
            file_df = DataFrame(rows_list)
        
        # Ignore the files that are too small; return the subdirectory data frame unharmed
        if (file_df.shape[1] < 16): return sub_directory_df
        
        # Find the columns that look like they have nothing but a version number in them
        VERSION_REGEX = re.compile(r'^\d\.\d$')
        is_version_there = lambda x: re.match(VERSION_REGEX, str(x)) is not None
        srs = file_df.applymap(is_version_there, na_action='ignore').sum()
        columns_list = srs[srs == file_df.shape[0]].index.tolist()
        
        # Remove column 4 and rename all the numbered colums above that
        if 4 in columns_list:
            version_number = file_df[4].unique().item()
            file_df.drop(4, axis='columns', inplace=True)
            file_df.columns = list(range(file_df.shape[1]))
        
        # Add the file name and logger version to the data frame
        file_df['file_name'] = '/'.join(sub_directory.split(os.sep)[1:]) + '/' + file_name
        if is_version_there(version_number): file_df['logger_version'] = float(version_number)
        else: file_df['logger_version'] = 1.0
        
        # Name the global columns
        columns_list = ['action_type', 'action_tick', 'event_time', 'session_uuid']
        file_df.columns = columns_list + file_df.columns.tolist()[len(columns_list):]
        
        # Parse the third column as a date column
        if ('event_time' in file_df.columns):
            if sub_directory.endswith('v.1.0'): file_df['event_time'] = to_datetime(file_df['event_time'], format='%m/%d/%Y %H:%M')
            # elif sub_directory.endswith('v.1.3'): file_df['event_time'] = to_datetime(file_df['event_time'], format='%m/%d/%Y %I:%M:%S %p')
            else: file_df['event_time'] = to_datetime(file_df['event_time'], format='mixed')
        
        # Set the MCIVR metrics types
        for row_index, row_series in file_df.iterrows(): file_df = self.set_mcivr_metrics_types(row_series.action_type, file_df, row_index, row_series)
        
        # Section off player actions by session start and end
        file_df = self.set_scene_indexes(file_df)
        
        # Append the data frame for the current file to the data frame for the current subdirectory
        sub_directory_df = concat([sub_directory_df, file_df], axis='index')
    
        return sub_directory_df
    
    
    def concatonate_logs(self, logs_folder=None):
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
        frvrs_logs_df = DataFrame([])
        
        # Iterate over the subdirectories, directories, and files in the logs folder
        if logs_folder is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=self.data_folder,
                saves_folder_path=self.saves_folder
            )
            logs_folder = nu.data_logs_folder
        for sub_directory, directories_list, files_list in os.walk(logs_folder):
            
            # Create a data frame to store the data for the current subdirectory
            sub_directory_df = DataFrame([])
            
            # Iterate over the files in the current subdirectory
            for file_name in files_list:
                
                # If the file is a CSV file, merge it into the subdirectory data frame
                if file_name.endswith('.csv'): sub_directory_df = self.process_files(sub_directory_df, sub_directory, file_name)
            
            # Append the data frame for the current subdirectory to the main data frame
            frvrs_logs_df = pd.concat([frvrs_logs_df, sub_directory_df], axis='index')
        
        # Convert event time to a datetime
        if ('event_time' in frvrs_logs_df.columns): frvrs_logs_df['event_time'] = pd.to_datetime(frvrs_logs_df['event_time'], format='mixed')
        
        # Convert elapsed time to an integer
        if ('action_tick' in frvrs_logs_df.columns):
            frvrs_logs_df['action_tick'] = pd.to_numeric(frvrs_logs_df['action_tick'], errors='coerce')
            frvrs_logs_df['action_tick'] = frvrs_logs_df['action_tick'].astype('int64')
        
        frvrs_logs_df = frvrs_logs_df.reset_index(drop=True)
        
        return frvrs_logs_df
    
    
    def split_df_by_teleport(self, df, nu=None, verbose=False):
        """
        """
        print(teleport_rows, df.index.tolist()); raise
        split_dfs = []
        current_df = DataFrame()
        if nu is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=self.data_folder,
                saves_folder_path=self.saves_folder
            )
        for row_index, row_series in df.iterrows():
            if row_index in teleport_rows:
                if current_df.shape[0] > 0: split_dfs.append(current_df)
                current_df = DataFrame()
            if verbose: print(row_index); display(row_series); display(nu.convert_to_df(row_index, row_series)); raise
            current_df = concat([current_df, nu.convert_to_df(row_index, row_series)], axis='index')
        if current_df.shape[0] > 0:
            split_dfs.append(current_df)
        
        return split_dfs
    
    
    def show_long_runs(self, df, column_name, milliseconds, delta_fn, description, frvrs_logs_df=None):
        """
        """
        delta = delta_fn(milliseconds)
        print(f'\nThese files have {description} than {delta}:')
        mask_series = (df[column_name] > milliseconds)
        session_uuid_list = df[mask_series].session_uuid.tolist()
        if frvrs_logs_df is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=self.data_folder,
                saves_folder_path=self.saves_folder
            )
            frvrs_logs_df = nu.load_object('frvrs_logs_df')
        mask_series = frvrs_logs_df.session_uuid.isin(session_uuid_list)
        logs_folder = '../data/logs'
        import csv
        from datetime import datetime
        for old_file_name in frvrs_logs_df[mask_series].file_name.unique():
            old_file_path = osp.join(logs_folder, old_file_name)
            with open(old_file_path, 'r') as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                for values_list in reader:
                    date_str = values_list[2]
                    break
                try: date_obj = datetime.strptime(date_str, '%m/%d/%Y %H:%M')
                except ValueError: date_obj = datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p')
                new_file_name = date_obj.strftime('%y.%m.%d.%H%M.csv')
                new_sub_directory = old_file_name.split('/')[0]
                new_file_path = new_sub_directory + '/' + new_file_name
                print(f'{old_file_name} (or {new_file_path})')
    
    
    def replace_consecutive_rows(self, df, element_column, element_value, time_diff_column='time_diff', consecutive_cutoff=500):
        """
        Replaces consecutive rows in a list created from the element column with a count of how many there are in a row.
        
        Parameters:
            element_column: An element column from which the list of elements is created.
            element_value: The element to replace consecutive occurrences of.
            time_diff_column: the time diff column to check if the elements are close enough in time to consider "consecutive".
            consecutive_cutoff: the number of time units the time diff column must show or less.
        
        Returns:
            A DataFrame with the rows of consecutive elements replaced with one row where the
            element_value now has appended to it a count of how many rows were deleted plus one.
        """
        result_df = DataFrame([], columns=df.columns); row_index = 0; row_series = pd.Series([]); count = 0
        for row_index, row_series in df.iterrows():
            column_value = row_series[element_column]
            time_diff = row_series[time_diff_column]
            if (column_value == element_value) and (time_diff <= consecutive_cutoff):
                count += 1
                previous_row_index = row_index
                previous_row_series = row_series
            else:
                result_df.loc[row_index] = row_series
                if (count > 0):
                    result_df.loc[previous_row_index] = previous_row_series
                    result_df.loc[previous_row_index, element_column] = f'{element_value} x{str(count)}'
                count = 0
        
        # Handle the last element
        result_df.loc[row_index] = row_series
        if (count > 0): result_df.loc[row_index, element_column] = f'{element_value} x{str(count)}'
        
        return(result_df)