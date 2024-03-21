
#!/usr/bin/env python
# Utility Functions to manipulate FRVRS logger data.
# Dave Babbitt <dave.babbitt@gmail.com>
# Author: Dave Babbitt, Data Scientist
# coding: utf-8

# Soli Deo gloria

from . import nu
from datetime import datetime, timedelta
from pandas import DataFrame, concat, to_datetime, Series
import csv
import humanize
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas as pd
import random
import re
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

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
        
        # Create the data folder if it doesn't exist
        if data_folder_path is None: self.data_folder = '../data'
        else: self.data_folder = data_folder_path
        os.makedirs(self.data_folder, exist_ok=True)
        if verbose: print('data_folder: {}'.format(osp.abspath(self.data_folder)), flush=True)
        
        # Create the saves folder if it doesn't exist
        if saves_folder_path is None: self.saves_folder = '../saves'
        else: self.saves_folder = saves_folder_path
        os.makedirs(self.saves_folder, exist_ok=True)
        if verbose: print('saves_folder: {}'.format(osp.abspath(self.saves_folder)), flush=True)
        
        # FRVRS log constants
        self.data_logs_folder = osp.join(self.data_folder, 'logs'); os.makedirs(name=self.data_logs_folder, exist_ok=True)
        self.scene_groupby_columns = ['session_uuid', 'scene_id']
        self.patient_groupby_columns = self.scene_groupby_columns + ['patient_id']
        self.injury_groupby_columns = self.patient_groupby_columns + ['injury_id']
        self.modalized_columns = [
            'patient_id', 'injury_id', 'location_id', 'patient_sort', 'patient_pulse', 'patient_salt', 'patient_hearing', 'patient_breath', 'patient_mood', 'patient_pose', 'injury_severity',
            'injury_required_procedure', 'injury_body_region', 'tool_type'
        ]
        
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
        self.responder_negotiations_list = ['PULSE_TAKEN', 'PATIENT_ENGAGED', 'INJURY_TREATED', 'TAG_APPLIED', 'TOOL_APPLIED']
        
        # List of columns that contain only boolean values
        self.boolean_columns_list = [
            'injury_record_injury_treated_with_wrong_treatment', 'injury_record_injury_treated',
            'injury_treated_injury_treated_with_wrong_treatment', 'injury_treated_injury_treated'
        ]
        
        # List of columns that contain patientIDs
        self.patient_id_columns_list = [
            'patient_demoted_id', 'patient_record_id', 'injury_record_patient_id', 's_a_l_t_walk_if_can_patient_id',
            's_a_l_t_walked_patient_id', 's_a_l_t_wave_if_can_patient_id', 's_a_l_t_waved_patient_id', 'patient_engaged_id',
            'pulse_taken_patient_id', 'injury_treated_patient_id', 'tool_applied_patient_id', 'tag_applied_patient_id',
            'player_gaze_patient_id'
        ]
        
        # List of columns that contain locationIDs
        self.location_id_columns_list = [
            'teleport_location', 'patient_demoted_position', 'patient_record_position', 'injury_record_injury_injury_locator',
            's_a_l_t_walk_if_can_sort_location', 's_a_l_t_walked_sort_location', 's_a_l_t_wave_if_can_sort_location',
            's_a_l_t_waved_sort_location', 'patient_engaged_position', 'bag_access_location', 'injury_treated_injury_injury_locator',
            'bag_closed_location', 'tag_discarded_location', 'tool_discarded_location', 'player_location_location',
            'player_gaze_location'
        ]
        
        # List of columns with injuryIDs
        self.injury_id_columns_list = ['injury_record_id', 'injury_treated_id']
        
        # Patient SORT designations
        self.sort_columns_list = ['patient_demoted_sort', 'patient_record_sort', 'patient_engaged_sort']
        self.patient_sort_order = ['still', 'waver', 'walker']
        self.sort_category_order = pd.CategoricalDtype(categories=self.patient_sort_order, ordered=True)
        
        # Patient SALT designations
        self.salt_columns_list = ['patient_demoted_salt', 'patient_record_salt', 'patient_engaged_salt']
        self.salt_types = ['DEAD', 'EXPECTANT', 'IMMEDIATE', 'DELAYED', 'MINIMAL']
        self.salt_category_order = pd.CategoricalDtype(categories=self.salt_types, ordered=True)
        
        # Tag colors
        self.tag_columns_list = ['tag_selected_type', 'tag_applied_type', 'tag_discarded_type']
        self.tag_colors = ['black', 'gray', 'red', 'yellow', 'green', 'Not Tagged']
        self.colors_category_order = pd.CategoricalDtype(categories=self.tag_colors, ordered=True)
        
        # Patient pulse designations
        self.pulse_columns_list = ['patient_demoted_pulse', 'patient_record_pulse', 'patient_engaged_pulse']
        self.patient_pulse_order = ['none', 'faint', 'fast', 'normal']
        self.pulse_category_order = pd.CategoricalDtype(categories=self.patient_pulse_order, ordered=True)
        
        # Patient breath designations
        self.breath_columns_list = ['patient_demoted_breath', 'patient_record_breath', 'patient_engaged_breath']
        self.patient_breath_order = ['none', 'collapsedRight', 'restricted', 'fast', 'normal']
        self.breath_category_order = pd.CategoricalDtype(categories=self.patient_breath_order, ordered=True)
        
        # Patient hearing designations
        self.hearing_columns_list = ['patient_record_hearing', 'patient_engaged_hearing']
        self.patient_hearing_order = ['none', 'limited', 'normal']
        self.hearing_category_order = pd.CategoricalDtype(categories=self.patient_hearing_order, ordered=True)
        
        # Patient mood designations
        self.mood_columns_list = ['patient_demoted_mood', 'patient_record_mood', 'patient_engaged_mood']
        self.patient_mood_order = ['dead', 'unresponsive', 'agony', 'upset', 'calm']
        self.mood_category_order = pd.CategoricalDtype(categories=self.patient_mood_order, ordered=True)
        
        # Patient pose designations
        self.pose_columns_list = ['patient_demoted_pose', 'patient_record_pose', 'patient_engaged_pose']
        self.patient_pose_order = ['supine', 'fetal', 'sittingGround', 'kneeling', 'recovery', 'standing']
        self.pose_category_order = pd.CategoricalDtype(categories=self.patient_pose_order, ordered=True)
        
        # Delayed is yellow per Nick
        self.salt_to_tag_dict = {'DEAD': 'black', 'EXPECTANT': 'gray', 'IMMEDIATE': 'red', 'DELAYED': 'yellow', 'MINIMAL': 'green'}
        self.sort_to_color_dict = {'still': 'black', 'waver': 'red', 'walker': 'green'}
        
        # Reordered per Ewart so that the display is from left to right as follows: dead, expectant, immediate, delayed, minimal, not tagged
        self.error_table_df = pd.DataFrame([
            {'DEAD': 'Exact', 'EXPECTANT': 'Critical', 'IMMEDIATE': 'Critical', 'DELAYED': 'Critical', 'MINIMAL': 'Critical'},
            {'DEAD': 'Over',  'EXPECTANT': 'Exact',    'IMMEDIATE': 'Critical', 'DELAYED': 'Critical', 'MINIMAL': 'Critical'},
            {'DEAD': 'Over',  'EXPECTANT': 'Over',     'IMMEDIATE': 'Exact',    'DELAYED': 'Over',     'MINIMAL': 'Over'},
            {'DEAD': 'Over',  'EXPECTANT': 'Over',     'IMMEDIATE': 'Under',    'DELAYED': 'Exact',    'MINIMAL': 'Over'},
            {'DEAD': 'Over',  'EXPECTANT': 'Over',     'IMMEDIATE': 'Under',    'DELAYED': 'Under',    'MINIMAL': 'Exact'}
        ], columns=self.salt_types, index=self.tag_colors[:-1])
        
        # Define the custom categorical orders
        self.error_values = ['Exact', 'Critical', 'Over', 'Under']
        self.errors_category_order = pd.CategoricalDtype(categories=self.error_values, ordered=True)
        
        # Hemorrhage control procedures list
        self.hemorrhage_control_procedures_list = ['tourniquet', 'woundpack']
        
        # Injury required procedure designations
        self.required_procedure_columns_list = ['injury_record_required_procedure', 'injury_treated_required_procedure']
        self.injury_required_procedure_order = ['tourniquet', 'gauzePressure', 'decompress', 'woundpack', 'airway', 'none']
        self.required_procedure_category_order = pd.CategoricalDtype(categories=self.injury_required_procedure_order, ordered=True)
        self.required_procedure_to_tool_type_dict = {
            'tourniquet': 'Tourniquet',
            'gauzePressure': 'Gauze_Dressing',
            'decompress': 'Needle',
            'airway': 'Nasal Airway',
            'woundpack': 'Gauze_Pack'
        }
        
        # Injury severity designations
        self.severity_columns_list = ['injury_record_severity', 'injury_treated_severity']
        self.injury_severity_order = ['high', 'medium', 'low']
        self.severity_category_order = pd.CategoricalDtype(categories=self.injury_severity_order, ordered=True)
        
        # Injury body region designations
        self.body_region_columns_list = ['injury_record_body_region', 'injury_treated_body_region']
        self.injury_body_region_order = ['head', 'neck', 'chest', 'abdomen', 'leftLeg', 'rightLeg', 'rightArm', 'leftArm']
        self.body_region_category_order = pd.CategoricalDtype(categories=self.injury_body_region_order, ordered=True)
        
        # Pulse name designations
        self.pulse_name_order = ['pulse_none', 'pulse_faint', 'pulse_fast', 'pulse_normal']
        self.pulse_name_category_order = pd.CategoricalDtype(categories=self.pulse_name_order, ordered=True)
        
        # Tool type designations
        self.tool_type_columns_list = ['tool_hover_type', 'tool_selected_type', 'tool_applied_type', 'tool_discarded_type']
        self.tool_type_order = ['Tourniquet', 'Gauze_Pack', 'Needle', 'Naso', 'Nasal Airway', 'Gauze_Dressing']
        self.tool_type_category_order = pd.CategoricalDtype(categories=self.tool_type_order, ordered=True)
        self.tool_type_to_required_procedure_dict = {
            'Tourniquet': 'tourniquet',
            'Gauze_Dressing': 'gauzePressure',
            'Needle': 'decompress',
            'Naso': 'airway',
            'Nasal Airway': 'airway',
            'Gauze_Pack': 'woundpack'
        }
        
        # Tool data designations
        self.tool_data_order = ['right_chest', 'left_chest', 'right_underarm', 'left_underarm']
        self.tool_data_category_order = pd.CategoricalDtype(categories=self.tool_data_order, ordered=True)

    ### String Functions ###
    
    
    @staticmethod
    def format_timedelta(timedelta, minimum_unit='seconds'):
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
        elif (minimum_unit == 'minutes'): formatted_string = f'{minutes}:{seconds:02}'
        else: raise ValueError('minimum_unit must be either seconds or minutes.')
        
        return formatted_string
    
    
    ### List Functions ###
    
    
    @staticmethod
    def replace_consecutive_elements(actions_list, element='PATIENT_ENGAGED'):
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
    
    
    @staticmethod
    def get_new_file_name(old_file_name):
        """
        Generate a new file name based on the action tick extracted from the old file.
        
        Parameters
        ----------
        old_file_name : str
            The name of the old log file.

        Returns
        -------
        str
            The new file name with the updated action tick.
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
    
    
    @staticmethod
    def get_session_groupby(grouped_df, mask_series=None, extra_column=None):
        """
        Group the FRVRS logs DataFrame by session UUID, with optional additional grouping by an extra column,
        based on the provided mask and extra column parameters.

        Parameters
        ----------
        grouped_df : pd.DataFrame, optional
            DataFrame containing the FRVRS logs data.
        mask_series : Series, optional
            Boolean mask to filter rows of grouped_df, by default None.
        extra_column : str, optional
            Additional column for further grouping, by default None.

        Returns
        -------
        pd.DataFrameGroupBy
            GroupBy object grouped by session UUID, and, if provided, the extra column.
        """
        
        # Apply grouping based on the provided parameters
        if (mask_series is None) and (extra_column is None):
            gb = grouped_df.sort_values(['action_tick']).groupby(['session_uuid'])
        elif (mask_series is None) and (extra_column is not None):
            gb = grouped_df.sort_values(['action_tick']).groupby(['session_uuid', extra_column])
        elif (mask_series is not None) and (extra_column is None):
            gb = grouped_df[mask_series].sort_values(['action_tick']).groupby(['session_uuid'])
        elif (mask_series is not None) and (extra_column is not None):
            gb = grouped_df[mask_series].sort_values(['action_tick']).groupby(['session_uuid', extra_column])
        
        # Return the grouped data
        return gb
    
    
    @staticmethod
    def get_is_a_one_triage_file(session_df, file_name=None, logs_df=None, file_stats_df=None, scene_stats_df=None, verbose=False):
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
        needed_set = set(['scene_type', 'is_scene_aborted', 'file_name'])
        if 'is_a_one_triage_file' in session_df.columns: is_a_one_triage_file = session_df.is_a_one_triage_file.unique().item()
        elif not all(map(lambda x: x in session_df.columns, needed_set)):
            
            # Merge scene_type and is_scene_aborted and file_name into sessions_df
            logs_columns_set = set(logs_df.columns)
            file_columns_set = set(file_stats_df.columns)
            scene_columns_set = set(scene_stats_df.columns)
            
            # Merge in the file stats columns
            on_columns = sorted(logs_columns_set.intersection(file_columns_set))
            file_stats_columns = on_columns + sorted(needed_set.difference(scene_columns_set))
            merge_df = logs_df.merge(file_stats_df[file_stats_columns], on=on_columns)

            # Merge in the scene stats columns
            on_columns = sorted(logs_columns_set.intersection(scene_columns_set))
            scene_stats_columns = on_columns + sorted(needed_set.difference(file_columns_set))
            session_df = merge_df.merge(scene_stats_df[scene_stats_columns], on=on_columns)
        
        # Filter out the triage files in this file name
        mask_series = (session_df.scene_type == 'Triage') & (session_df.is_scene_aborted == False)
        if file_name is not None: mask_series &= (session_df.file_name == file_name)
        
        # Get whether the file has only one triage run
        triage_scene_count = len(session_df[mask_series].groupby('scene_id').groups)
        is_a_one_triage_file = bool(triage_scene_count == 1)
        if verbose:
            print(f'triage_scene_count={triage_scene_count}')
            display(session_df[mask_series].groupby('scene_id').size())
            raise
        
        # Return True if the file has only one triage run, False otherwise
        return is_a_one_triage_file
    
    
    @staticmethod
    def get_file_name(session_df, verbose=False):
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
    
    
    @staticmethod
    def get_logger_version(session_df_or_file_path, verbose=False):
        """
        Retrieve the unique logger version associated with the given session DataFrame.

        Parameters
        ----------
        session_df_or_file_path : pd.DataFrame or str
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
    def get_is_duplicate_file(session_df, verbose=False):
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
    
    
    @staticmethod
    def get_session_file_date(logs_df, uuid, session_df=None, verbose=False):
        if session_df is None:
            mask_series = (logs_df.session_uuid == uuid)
            session_df = logs_df[mask_series]
        file_date_str = session_df.event_time.min().strftime('%B %d, %Y')
        
        return file_date_str
    
    
    def get_distance_deltas_data_frame(self, logs_df, verbose=False):
        rows_list = []
        columns_list = ['patient_id', 'engagement_start', 'location_tuple', 'patient_sort', 'predicted_priority', 'injury_severity']
        import math
        for (session_uuid, scene_id), scene_df in logs_df.groupby(self.scene_groupby_columns):
            row_dict = {}
            for cn in self.scene_groupby_columns: row_dict[cn] = eval(cn)
            
            actual_engagement_order = self.get_engagement_starts_order(scene_df, verbose=False)
            
            # Get last still engagement and subtract the scene start
            df = DataFrame(actual_engagement_order, columns=columns_list)
            mask_series = (df.patient_sort == 'still')
            last_still_engagement = df[mask_series].engagement_start.max()
            mask_series = True
            for cn in self.scene_groupby_columns: mask_series &= (logs_df[cn] == eval(cn))
            row_dict['last_still_engagement'] = last_still_engagement - self.get_scene_start(logs_df[mask_series])
            
            # Actual
            actual_engagement_distance = sum([
                math.sqrt(
                    (first_tuple[2][0] - last_tuple[2][0])**2 + (first_tuple[2][1] - last_tuple[2][1])**2
                ) for first_tuple, last_tuple in zip(actual_engagement_order[:-1], actual_engagement_order[1:])
            ])
            row_dict['actual_engagement_distance'] = actual_engagement_distance
            
            # Ideal
            ideal_engagement_order = self.get_ideal_engagement_order(scene_df, verbose=False)
            ideal_engagement_distance = sum([
                math.sqrt(
                    (first_tuple[2][0] - last_tuple[2][0])**2 + (first_tuple[2][1] - last_tuple[2][1])**2
                ) for first_tuple, last_tuple in zip(ideal_engagement_order[:-1], ideal_engagement_order[1:])
            ])
            row_dict['ideal_engagement_distance'] = ideal_engagement_distance
            
            # Calculate the R-squared adjusted value as a measure of ideal ordering
            X, y = Series([t[1] for t in ideal_engagement_order]).values.reshape(-1, 1), Series([t[1] for t in actual_engagement_order]).values.reshape(-1, 1)
            if X.shape[0]:
                X1 = sm.add_constant(X)
                try: measure_of_ideal_ordering = sm.OLS(y, X1).fit().rsquared_adj
                except: measure_of_ideal_ordering = np.nan
            else: measure_of_ideal_ordering = np.nan
            row_dict['measure_of_ideal_ordering'] = measure_of_ideal_ordering
            
            # Distracted
            distracted_engagement_order = self.get_distracted_engagement_order(scene_df, tuples_list=actual_engagement_order, verbose=False)
            distracted_engagement_distance = sum([
                math.sqrt(
                    (first_tuple[2][0] - last_tuple[2][0])**2 + (first_tuple[2][1] - last_tuple[2][1])**2
                ) for first_tuple, last_tuple in zip(distracted_engagement_order[:-1], distracted_engagement_order[1:])
            ])
            row_dict['distracted_engagement_distance'] = distracted_engagement_distance
            
            # Calculate the R-squared adjusted value as a measure of distracted ordering
            X, y = Series([t[1] for t in distracted_engagement_order]).values.reshape(-1, 1), Series([t[1] for t in actual_engagement_order]).values.reshape(-1, 1)
            if X.shape[0]:
                X1 = sm.add_constant(X)
                try: measure_of_distracted_ordering = sm.OLS(y, X1).fit().rsquared_adj
                except: measure_of_distracted_ordering = np.nan
            else: measure_of_distracted_ordering = np.nan
            row_dict['measure_of_distracted_ordering'] = measure_of_distracted_ordering
            
            # Calculate the measure of right ordering
            row_dict['measure_of_right_ordering'] = self.get_measure_of_right_ordering(scene_df, verbose=verbose)
            
            row_dict['actual_ideal_delta'] = actual_engagement_distance - ideal_engagement_distance
            row_dict['actual_distracted_delta'] = actual_engagement_distance - distracted_engagement_distance
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
    
    
    @staticmethod
    def get_last_engagement(scene_df, verbose=False):
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
    def get_player_location(scene_df, action_tick, verbose=False):
        mask_series = (scene_df.action_type == 'PLAYER_LOCATION')
        df = scene_df[mask_series]
        df['action_delta'] = df.action_tick.map(lambda x: abs(action_tick, x))
        player_location = eval(df.sort_values('action_delta').iloc[0].location_id)
        
        return player_location
    
    
    @staticmethod
    def get_scene_type(scene_df, verbose=False):
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
    
    
    @staticmethod
    def get_injury_correctly_treated_count(scene_df, verbose=False):
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
        for injury_id, injury_df in scene_df.groupby('injury_id'):
            
            # Filter for injuries that have been treated and not wrong
            treated_mask_series = ~injury_df.injury_treated_required_procedure.isnull()
            treated_mask_series &= (injury_df.injury_treated_injury_treated_with_wrong_treatment == False)
            
            if injury_df[treated_mask_series].shape[0]: injury_correctly_treated_count += 1
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Number of records where injuries were correctly treated: {injury_correctly_treated_count}')
            display(scene_df)
        
        # Return the count of records where injuries were correctly treated
        return injury_correctly_treated_count
    
    
    @staticmethod
    def get_injury_wrongly_treated_count(scene_df, verbose=False):
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
    
    
    def get_ideal_engagement_order(self, scene_df, tuples_list=None, verbose=False):
        
        # Create the patient sort info
        engagement_starts_df = DataFrame(self.get_engagement_starts_order(scene_df), columns=[
            'patient_id', 'engagement_start', 'location_tuple', 'patient_sort', 'predicted_priority', 'injury_severity'
        ])
        engagement_starts_df.patient_sort = engagement_starts_df.patient_sort.astype(self.sort_category_order)
        engagement_starts_df.injury_severity = engagement_starts_df.injury_severity.astype(self.severity_category_order)
        
        # Get initial player location
        mask_series = (scene_df.action_type == 'PLAYER_LOCATION')
        if mask_series.any(): player_location = eval(scene_df[mask_series].sort_values('action_tick').iloc[0].location_id)
        else: player_location = (0.0, 0.0, 0.0)
        player_location = (player_location[0], player_location[2])
        
        # Go from nearest neighbor to nearest neighbor by high severity first, by still/waver/walker, then the rest by still/waver/walker
        ideal_engagement_order = []
        mask_series = (engagement_starts_df.injury_severity == 'high')
        for gb in [engagement_starts_df[mask_series].groupby('patient_sort'), engagement_starts_df[~mask_series].groupby('patient_sort')]:
            for patient_sort, patient_sort_df in gb:
                
                # Get locations list
                locations_list = patient_sort_df.location_tuple.tolist()
                
                # Pop the nearest neighbor off the locations list and add it to the engagement order
                # Assume no patients are in the exact same spot
                while locations_list:
                    nearest_neighbor = nu.get_nearest_neighbor(player_location, locations_list)
                    nearest_neighbor = locations_list.pop(locations_list.index(nearest_neighbor))
                    mask_series = (engagement_starts_df.location_tuple == nearest_neighbor)
                    if mask_series.any():
                        patient_sort_tuple = tuple(list(engagement_starts_df[mask_series].T.to_dict(orient='list').values())[0])
                        ideal_engagement_order.append(patient_sort_tuple)
                    player_location = nearest_neighbor
        
        if verbose: print(f'\n\nideal_engagement_order: {ideal_engagement_order}')
        
        return ideal_engagement_order
    
    
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
    
    
    def get_actual_and_ideal_sequences(self, scene_df, verbose=False):
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
        
        Notes:
            Only SORT categories included in `self.patient_sort_order` are considered.
            None values in the resulting lists indicate missing interactions.
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
                    patient_actions_df = scene_df[mask_series]
                    action_list.append(self.get_first_patient_interaction(patient_actions_df))
                
                # Sort the list of first interactions
                if verbose: display(sort, action_list)
                sort_dict[sort] = sorted([action for action in action_list if action is not None])
        
        # Get the whole ideal and actual sequences
        ideal_sequence = []
        for sort in self.patient_sort_order: ideal_sequence.extend(sort_dict.get(sort, []))
        ideal_sequence = Series(data=ideal_sequence)
        actual_sequence = ideal_sequence.sort_values(ascending=True)
        
        return actual_sequence, ideal_sequence, sort_dict
    
    
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
        
        # Calculate the R-squared adjusted value as a measure of right ordering
        actual_sequence, ideal_sequence, _ = self.get_actual_and_ideal_sequences(scene_df, verbose=verbose)
        X, y = ideal_sequence.values.reshape(-1, 1), actual_sequence.values.reshape(-1, 1)
        if X.shape[0]:
            X1 = sm.add_constant(X)
            try: measure_of_right_ordering = sm.OLS(y, X1).fit().rsquared_adj
            except: measure_of_right_ordering = np.nan
        
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
        procedure_count = 0; hemorrhage_count = 0; controlled_count = 0
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            for injury_id, injury_df in patient_df.groupby('injury_id'):
            
                # Count any hemorrhage-related injuries that have been recorded or treated
                mask_series = injury_df.injury_required_procedure.isin(self.hemorrhage_control_procedures_list)
                if mask_series.any(): procedure_count += 1
                
                # Count any injuries requiring hemorrhage control procedures
                mask_series = injury_df.injury_record_required_procedure.isin(self.hemorrhage_control_procedures_list)
                if mask_series.any(): hemorrhage_count += 1
                
                # Count any hemorrhage-related injuries that have been treated, and not wrong, and not counted twice
                if self.get_is_hemorrhage_controlled(injury_df, patient_df, verbose=False): controlled_count += 1
        
        if verbose: print(f'Hemorrhage-related injuries that have been recorded or treated: {procedure_count}')
        if verbose: print(f'Injuries requiring hemorrhage control procedures: {hemorrhage_count}')
        if verbose: print(f"Hemorrhage-related injuries that have been treated: {controlled_count}")
        
        # Calculate the percentage of controlled hemorrhage-related injuries
        try: percent_controlled = 100 * controlled_count / hemorrhage_count
        except ZeroDivisionError: percent_controlled = np.nan
        
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
            
            # Check if the patient is hemorrhaging
            if self.get_is_patient_hemorrhaging(patient_df):
                
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
    
    
    def get_triage_priority_data_frame(self, scene_df, verbose=False):
        input_features = [
            'injury_id', 'injury_severity', 'injury_required_procedure', 'patient_salt', 'patient_sort', 'patient_pulse', 'patient_breath',
            'patient_hearing', 'patient_mood', 'patient_pose'
            ]
        columns_list = self.scene_groupby_columns + input_features
        triage_priority_df = scene_df[columns_list].sort_values(['injury_severity', 'patient_sort'], ascending=[True, True])
        
        return triage_priority_df
    
    
    def get_engagement_starts_order(self, scene_df, verbose=False):
        """
        Get the chronological order of engagement starts for each patient in a scene.
        
        Parameters:
            - scene_df (pd.DataFrame): DataFrame containing scene data, including patient IDs, action types,
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
                
                # Add engagement information to the list
                engagement_tuple = (patient_id, engagement_start, location_tuple, patient_sort, predicted_priority, injury_severity)
                engagement_starts_list.append(engagement_tuple)
        
        # Sort the starts list chronologically
        engagement_order = sorted(engagement_starts_list, key=lambda x: x[1], reverse=False)
        
        return engagement_order
    
    
    def get_distracted_engagement_order(self, scene_df, tuples_list=None, verbose=False):

        # Create the patient sort tuples list
        if tuples_list is None: tuples_list = self.get_engagement_starts_order(scene_df, verbose=verbose)

        # Get initial player location
        mask_series = (scene_df.action_type == 'PLAYER_LOCATION')
        if mask_series.any(): player_location = eval(scene_df[mask_series].sort_values('action_tick').iloc[0].location_id)
        else: player_location = (0.0, 0.0, 0.0)
        player_location = (player_location[0], player_location[2])

        # Go from nearest neighbor to nearest neighbor
        distracted_engagement_order = []

        # Get locations list
        locations_list = [x[2] for x in tuples_list]

        # Pop the nearest neighbor off the locations list and add it to the engagement order
        # Assume no patients are in the exact same spot
        while locations_list:
            nearest_neighbor = nu.get_nearest_neighbor(player_location, locations_list)
            nearest_neighbor = locations_list.pop(locations_list.index(nearest_neighbor))
            for patient_sort_tuple in tuples_list:
                if (patient_sort_tuple[2] == nearest_neighbor):
                    distracted_engagement_order.append(patient_sort_tuple)
                    break
            player_location = nearest_neighbor

        if verbose: print(f'\n\ndistracted_engagement_order: {distracted_engagement_order}')

        return distracted_engagement_order
    
    
    ### Patient Functions ###
    
    
    @staticmethod
    def get_is_correct_bleeding_tool_applied(patient_df, verbose=False):
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
            bool or np.nan: True if the patient is considered dead, False if not, and np.nan if unknown.
        """
        
        # Handle missing values in both patient_record_salt and patient_engaged_salt
        if patient_df.patient_record_salt.isnull().all() and patient_df.patient_engaged_salt.isnull().all(): is_patient_dead = np.nan
        else:
            
            # Check the patient_record_salt field
            mask_series = ~patient_df.patient_record_salt.isnull()
            if mask_series.any():
                patient_record_salt = patient_df[mask_series].patient_record_salt.iloc[0]
                is_patient_dead = patient_record_salt in ['DEAD', 'EXPECTANT']
            
            # Check the patient_engaged_salt field if patient_record_salt is not available
            else:
                
                # Check 'patient_engaged_salt' for patient status if 'patient_record_salt' is empty
                mask_series = ~patient_df.patient_engaged_salt.isnull()
                if mask_series.any():
                    patient_engaged_salt = patient_df[mask_series].patient_engaged_salt.iloc[0]
                    is_patient_dead = patient_engaged_salt in ['DEAD', 'EXPECTANT']
                
                # If both columns are empty, the result is unknown
                else: is_patient_dead = np.nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Is patient considered dead: {is_patient_dead}')
            display(patient_df)
        
        # Return True if the patient is considered dead, False if not, and np.nan if unknown
        return is_patient_dead
    
    
    @staticmethod
    def get_is_patient_still(patient_df, verbose=False):
        """
        Determines whether a patient is considered still based on the presence of 'still' in their patient_record_sort or patient_engaged_sort fields.
        
        The function examines the 'patient_record_sort' and 'patient_engaged_sort' columns to
        determine if the patient is categorized as 'still'. If both columns are empty,
        the result is considered unknown (NaN).
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool or np.nan: True if the patient is marked as 'still', False if not, and np.nan if unknown.
        """
        
        # Handle missing values in both patient_record_sort and patient_engaged_sort
        if patient_df.patient_record_sort.isnull().all() and patient_df.patient_engaged_sort.isnull().all(): is_patient_still = np.nan
        else:
            
            # Check the patient_record_sort field
            mask_series = ~patient_df.patient_record_sort.isnull()
            if mask_series.any():
                patient_record_sort = patient_df[mask_series].patient_record_sort.iloc[0]
                is_patient_still = (patient_record_sort == 'still')
            
            # Check the patient_engaged_sort field if patient_record_sort is not available
            else:
                
                # Check 'patient_engaged_sort' for patient status if 'patient_record_sort' is empty
                mask_series = ~patient_df.patient_engaged_sort.isnull()
                if mask_series.any():
                    patient_engaged_sort = patient_df[mask_series].patient_engaged_sort.iloc[0]
                    is_patient_still = (patient_engaged_sort == 'still')
                
                # If both columns are empty, the result is unknown
                else: is_patient_still = np.nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Is patient considered still: {is_patient_still}')
            display(patient_df)
        
        # Return True if the patient is marked as 'still', False if not, and np.nan if unknown
        return is_patient_still
    
    
    @staticmethod
    def get_max_salt(patient_df, session_uuid=None, scene_id=None, random_patient_id=None, verbose=False):
        """
        Get the maximum salt value from the patient data frame.
        
        Parameters:
            patient_df (pandas.DataFrame, optional): DataFrame containing patient-specific data with relevant columns.
            session_uuid (str, optional): UUID of the session to load FRVRS logs. Required if 'patient_df' is None.
            scene_id (int, optional): Index of the scene to load FRVRS logs. Required if 'patient_df' is None.
            random_patient_id (int, optional): Random patient ID to use if 'patient_df' is None. Default is None.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int or tuple: The maximum salt value for the patient, or a tuple containing the patient ID and maximum salt value if `session_uuid` is provided.
        """
        
        # Get the max salt value
        # mask_series = patient_df.patient_record_salt.isnull()
        # try: max_salt = patient_df[~mask_series].patient_record_salt.max()
        try:
            mask_series = ~patient_df.patient_salt.isnull()
            max_salt = patient_df[mask_series].sort_values('action_tick').patient_salt.tolist()[-1]
        except Exception: max_salt = np.nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'max_salt={max_salt}, session_uuid={session_uuid}, scene_id={scene_id}, random_patient_id={random_patient_id}')
            display(patient_df)
        
        # Return a tuple (random_patient_id, max_salt) if 'session_uuid' is provided
        if session_uuid is not None:
            if random_patient_id is None: random_patient_id = list(patient_df.groupby('patient_id').groups.keys())[0]
            return random_patient_id, max_salt
        
        # Return the max salt value if 'session_uuid' is None
        else: return max_salt
    
    
    @staticmethod
    def get_last_tag(patient_df, verbose=False):
        """
        Retrieves the last tag applied to a patient in a given scene DataFrame.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            str or np.nan: The last tag applied to the patient, or np.nan if no tags have been applied.
        """
        
        # Get the last tag value
        mask_series = patient_df.tag_applied_type.isnull()
        try: last_tag = patient_df[~mask_series].tag_applied_type.iloc[-1]
        except Exception: last_tag = np.nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Last tag applied: {last_tag}')
            display(patient_df)
        
        # Return the last tag value or np.nan if no data is available
        return last_tag
    
    
    @staticmethod
    def get_patient_location(patient_df, action_tick, verbose=False):
        """
        Gets the patient location closest to the time of the action tick.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            action_tick (int): The time in milliseconds to locate the patient during.
        
        Returns:
            tuple: The coordinates of the patient.
        """
        patient_location = (0.0, 0.0, 0.0)
        mask_series = ~patient_df.location_id.isnull()
        if mask_series.any():
            df = patient_df[mask_series]
            df['action_delta'] = df.action_tick.map(lambda x: abs(action_tick - x))
            patient_location = eval(df.sort_values('action_delta').iloc[0].location_id)
        
        return patient_location
    
    
    def get_is_tag_correct(self, patient_df, verbose=False):
        """
        Determines whether the last tag applied to a patient in a given scene DataFrame matches the predicted tag based on the patient's record salt.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool or np.nan: Returns True if the tag is correct, False if incorrect, or np.nan if data is insufficient.
        """
        
        # Ensure both 'tag_applied_type' and 'patient_record_salt' each have at least one non-null value
        mask_series = ~patient_df.tag_applied_type.isnull()
        tag_applied_type_count = patient_df[mask_series].tag_applied_type.unique().shape[0]
        mask_series = ~patient_df.patient_record_salt.isnull()
        patient_record_salt_count = patient_df[mask_series].patient_record_salt.unique().shape[0]
        assert (tag_applied_type_count > 0) and (patient_record_salt_count > 0), "You need to have both a tag_applied_type and a patient_record_salt"
        
        # Get the last applied tag
        last_tag = self.get_last_tag(patient_df)
        
        # Get the maximum salt value for the patient
        max_salt = self.get_max_salt(patient_df)
        
        # Get the predicted tag based on the maximum salt value
        try: predicted_tag = self.salt_to_tag_dict.get(max_salt, np.nan)
        except Exception: predicted_tag = np.nan
            
        # Determine if the last applied tag matches the predicted tag
        try: is_tag_correct = bool(last_tag == predicted_tag)
        except Exception: is_tag_correct = np.nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Last tag value: {last_tag}')
            print(f'Predicted tag value based on max salt: {predicted_tag}')
            print(f'Is the tag correct? {is_tag_correct}')
            display(patient_df)
        
        # Return True if the tag is correct, False if incorrect, or np.nan if data is insufficient
        return is_tag_correct
    
    
    def get_is_patient_severely_hemorrhaging(self, patient_df, verbose=False):
        """
        Determines whether the patient has severe injuries.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool or np.nan: Returns True if the patient has severe injuries, False if the patient has no severe injuries.
        """
        is_patient_injured = False
        for injury_id, injury_df in patient_df.groupby('injury_id'):
            is_patient_injured = is_patient_injured or self.get_is_injury_severe(injury_df, verbose=verbose)
        
        return is_patient_injured
    
    
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
        else: engagement_start = np.nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'First patient interaction: {engagement_start}')
            display(patient_df[mask_series].dropna(axis='columns', how='all').T)
        
        # Return the action tick of the first patient interaction or np.nan if no data is available
        return engagement_start
    
    
    def get_last_patient_interaction(self, patient_df, verbose=False):
        """
        Get the action tick of the last patient interaction of specified types.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The action tick of the last patient interaction action, or None if no such action exists.
        """
        
        # Filter for actions involving patient interactions
        mask_series = patient_df.action_type.isin(self.action_types_list)
        
        # Include VOICE_COMMAND actions with specific message types
        mask_series |= ((patient_df.action_type == 'VOICE_COMMAND') & patient_df.voice_command_message.isin(self.command_messages_list))
        
        # If there are patient interaction actions, find the last action tick
        if mask_series.any(): engagement_end = patient_df[mask_series].action_tick.max()
        else: engagement_end = None
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Action tick of the last patient interaction: {engagement_end}')
            display(patient_df)
        
        # Return the action tick of the last patient interaction or np.nan if no data is available
        return engagement_end
    
    
    @staticmethod
    def get_is_patient_gazed_at(patient_df, verbose=False):
        """
        Determine whether the responder gazed at the patient at least once.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool: True if the responder gazed at the patient at least once, False otherwise.
        """
        
        # Define the mask for the 'PLAYER_GAZE' actions
        mask_series = (patient_df.action_type == 'PLAYER_GAZE')
        
        # Check if the responder gazed at the patient at least once
        gazed_at_patient = bool(patient_df[mask_series].shape[0])
        
        # If verbose is True, print additional information
        if verbose:
            print(f'The responder gazed at the patient at least once: {gazed_at_patient}')
            display(patient_df)
        
        # Return True if the responder gazed at the patient at least once, False otherwise
        return gazed_at_patient
    
    
    @staticmethod
    def get_wanderings(patient_df, verbose=False):
        """
        Extract the x and z dimensions of patient wanderings from relevant location columns.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            tuple: Two lists, x_dim and z_dim, representing the x and z coordinates of the patient's wanderings.
        """
        
        # Initialize empty lists to store x and z coordinates
        x_dim = []; z_dim = []
        
        # Define the list of location columns to consider
        location_cns_list = [
            'patient_demoted_position', 'patient_engaged_position', 'patient_record_position', 'player_gaze_location'
        ]
        
        # Create a melt DataFrame by filtering rows with valid location data
        mask_series = False
        for location_cn in location_cns_list: mask_series |= ~patient_df[location_cn].isnull()
        columns_list = ['action_tick'] + location_cns_list
        melt_df = patient_df[mask_series][columns_list].sort_values('action_tick')
        
        # Split the location columns into two columns using melt and convert the location into a tuple
        df = melt_df.melt(id_vars=['action_tick'], value_vars=location_cns_list).sort_values('action_tick')
        mask_series = ~df['value'].isnull()
        
        # If verbose is True, display the relevant part of the data frame
        if verbose: display(df[mask_series])
        
        # Extract x and z dimensions from the location tuples
        srs = df[mask_series]['value'].map(lambda x: eval(x))
        
        # Extract x and z coordinates and append them to corresponding lists
        x_dim.extend(srs.map(lambda x: x[0]).values)
        z_dim.extend(srs.map(lambda x: x[2]).values)
        
        # If verbose is True, print additional information
        if verbose:
            print(f'x_dim={x_dim}, z_dim={z_dim}')
            display(patient_df)
        
        return x_dim, z_dim
    
    
    @staticmethod
    def get_wrapped_label(patient_df, wrap_width=20, verbose=False):
        """
        Generate a wrapped label based on patient sorting information.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            wrap_width (int, optional): Number of characters with which to wrap the text in the label. Defaults to 20.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            str: A wrapped label for the patient based on the available sorting information.
        """
        
        # Pick from among the sort columns whichever value is not null and use that in the label
        columns_list = ['patient_demoted_sort', 'patient_engaged_sort', 'patient_record_sort']
        srs = patient_df[columns_list].apply(Series.notnull, axis='columns').sum()
        mask_series = (srs > 0)
        if mask_series.any():
            sort_cns_list = srs[mask_series].index.tolist()[0]
            sort_type = patient_df[sort_cns_list].dropna().iloc[-1]
        else: sort_type = 'no SORT info'
        
        # Generate a wrapped label
        patient_id = patient_df.patient_id.unique().item()
        label = patient_id.replace(' Root', ' (') + sort_type + ')'
        
        # Wrap the label to a specified width
        import textwrap
        label = '\n'.join(textwrap.wrap(label, width=wrap_width))
        
        # If verbose is True, print additional information
        if verbose:
            print(f'label={label}')
            display(patient_df)
        
        return label
    
    
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
        
        # Determine if the patient is hemorrhaging
        is_hemorrhaging = bool(patient_df[mask_series].shape[0])
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Is the patient hemorrhaging: {is_hemorrhaging}')
            display(patient_df)
        
        return is_hemorrhaging
    
    
    def get_time_to_hemorrhage_control(self, patient_df, scene_start=None, verbose=False):
        """
        Calculate the time it takes to control hemorrhage for the patient by getting the injury treatments
        where the responder is not confused from the bad feedback.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            scene_start (int, optional): The action tick of the first interaction with the patient.
                Defaults to None, in which case it will be calculated using `get_first_patient_interaction`.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The time it takes to control hemorrhage for the patient, in action ticks.
        """
        
        # If scene_start is not provided, determine it using the first patient interaction
        if scene_start is None: scene_start = self.get_first_patient_interaction(patient_df)
        
        # Initialize variables to track hemorrhage control time
        controlled_time = 0
        
        # Define columns for merging
        on_columns_list = ['injury_id']
        merge_columns_list = ['action_tick'] + on_columns_list
        
        # Loop through hemorrhage control procedures
        for control_procedure in self.hemorrhage_control_procedures_list:
            
            # Identify hemorrhage events for the current procedure
            mask_series = patient_df.injury_record_required_procedure.isin([control_procedure])
            hemorrhage_df = patient_df[mask_series][merge_columns_list]
            if verbose: display(hemorrhage_df)
            
            # Identify controlled hemorrhage events for the current procedure
            mask_series = patient_df.injury_treated_required_procedure.isin([control_procedure])
            controlled_df = patient_df[mask_series][merge_columns_list]
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
    def get_patient_engagement_count(patient_df, verbose=False):
        """
        Count the number of 'PATIENT_ENGAGED' actions in the patient's data.
        
        Parameters:
            patient_df (pandas.DataFrame): DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The number of times the patient has been engaged.
        """
        
        # Create a mask to filter 'PATIENT_ENGAGED' actions
        mask_series = (patient_df.action_type == 'PATIENT_ENGAGED')
        
        # Count the number of 'PATIENT_ENGAGED' actions
        engagement_count = patient_df[mask_series].shape[0]
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Number of times the patient has been engaged: {engagement_count}')
            display(patient_df)
        
        return engagement_count
    
    
    @staticmethod
    def get_maximum_injury_severity(patient_df, verbose=False):
        mask_series = ~patient_df.injury_severity.isnull()
        
        # The first of the high/medium/low category
        maximum_injury_severity = patient_df[mask_series].injury_severity.min()
        
        return maximum_injury_severity
    
    
    def get_is_life_threatened(self, patient_df, verbose=False):
        is_severity_high = (self.get_maximum_injury_severity(patient_df, verbose=verbose) == 'high')
        is_patient_hemorrhaging = self.get_is_patient_hemorrhaging(patient_df, verbose=verbose)
        
        return is_severity_high and is_patient_hemorrhaging
    
    
    ### Injury Functions ###
    
    
    @staticmethod
    def get_is_injury_correctly_treated(injury_df, verbose=False):
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
        
        # Create a mask to identify treated injuries
        mask_series = (injury_df.injury_treated_injury_treated == True)
        
        # Add to that mask to identify correctly treated injuries
        mask_series &= (injury_df.injury_treated_injury_treated_with_wrong_treatment == False)
        
        # Return True if there are correctly treated attempts, False otherwise
        is_correctly_treated = mask_series.any()
        
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
        
        # Check if the number of log entries indicating hemorrhage control is not zero
        is_hemorrhage = mask_series.any()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Is the injury a hemorrhage: {is_hemorrhage}')
            print('\n\n')
            display(injury_df.dropna(axis='columns', how='all').T)
        
        return is_hemorrhage
    
    
    def get_is_hemorrhage_tool_applied(self, injury_df, logs_df, verbose=False):
            
        # Get the entire patient record
        mask_series = ~injury_df.patient_id.isnull()
        patient_id = injury_df[mask_series].patient_id.tolist()[0]
        mask_series = (logs_df.patient_id == patient_id)
        patient_df = logs_df[mask_series]
        
        # See if there are any tools applied that are associated with the hemorrhage injuries
        applied_mask_series = Series([False] * patient_df.shape[0])
        record_mask_series = injury_df.injury_record_required_procedure.isin(self.hemorrhage_control_procedures_list)
        for required_procedure in injury_df[record_mask_series].injury_record_required_procedure.unique():
            tool_type = self.required_procedure_to_tool_type_dict[required_procedure]
            applied_mask_series |= (patient_df.tool_applied_type == tool_type)
        is_tool_applied_correctly = applied_mask_series.any()
        if verbose:
            
            # Get the injury ID
            mask_series = ~injury_df.injury_id.isnull()
            injury_id = injury_df[mask_series].injury_id.tolist()[0]
            
            print(f'A hemorrhage-related TOOL_APPLIED event can be associated with the injury ({injury_id}): {is_tool_applied_correctly}')
            print('\n\n')
            display(patient_df[applied_mask_series].dropna(axis='columns', how='all').T)
        
        return is_tool_applied_correctly
    
    
    def get_is_hemorrhage_controlled(self, injury_df, logs_df, verbose=False):
        """
        Determine if hemorrhage is controlled based on injury and log data.

        Parameters:
            injury_df (pandas.DataFrame): DataFrame containing injury-specific data with relevant columns.
            logs_df (pandas.DataFrame): DataFrame containing logs data.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool: True if hemorrhage is controlled, False otherwise.
        
        Note:
            The logs have instances of a TOOL_APPLIED but no INJURY_TREATED preceding it. But, we already know the injury
            that the patient has and the correct tool for every patient because we assigned those ahead of time.
        """
        
        # Check if an injury record or treatment exists for a hemorrhage-related procedure
        is_injury_hemorrhage = self.get_is_injury_hemorrhage(injury_df, verbose=verbose)
        if not is_injury_hemorrhage: is_controlled = np.nan
        else:
            
            # Check if the injury was treated correctly
            is_correctly_treated = self.get_is_injury_correctly_treated(injury_df, verbose=verbose)
            
            # See if there are any tools applied that are associated with the hemorrhage injuries
            is_tool_applied_correctly = self.get_is_hemorrhage_tool_applied(injury_df, logs_df, verbose=verbose)
            
            # Compute the is-controlled logic
            is_controlled = is_correctly_treated or is_tool_applied_correctly
            
        if verbose:
            mask_series = ~injury_df.injury_id.isnull()
            injury_id = injury_df[mask_series].injury_id.tolist()[0]
            print(f'Is hemorrhage controlled for the injury ({injury_id}): {is_controlled}')
            print('\n\n')
            display(injury_df.dropna(axis='columns', how='all').T)
        
        return is_controlled
    
    
    def get_is_injury_severe(self, injury_df, verbose=False):
        """
        Determine whether the given injury is severe based on the injury record or treatment record.
        
        Parameters:
            injury_df (pandas.DataFrame): DataFrame containing injury-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool: True if the injury is severe, False otherwise.
        """
        mask_series = (injury_df.injury_severity == 'high')
        
        return self.get_is_injury_hemorrhage(injury_df, verbose=verbose) and bool(injury_df[mask_series].shape[0])
    
    
    def get_is_bleeding_correctly_treated(self, injury_df, verbose=False):
        """
        Check if bleeding is correctly treated based on the provided injury DataFrame.
        
        Parameters:
            injury_df (pandas.DataFrame): DataFrame containing injury-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            bool: True if bleeding was correctly treated, False otherwise.
        
        Note:
            The FRVRS logger has trouble logging multiple tool applications,
            so injury_treated_injury_treated_with_wrong_treatment == True
            remains and another INJURY_TREATED is never logged, even though
            the right tool was applied after that.
        """
        
        # Create a mask to identify rows where the treatment record indicates successful bleeding treatment
        mask_series = injury_df.injury_treated_required_procedure.isin(self.hemorrhage_control_procedures_list)
        
        # Include injuries with successful treatment
        mask_series &= (injury_df.injury_treated_injury_treated == True)
        
        # Add to that mask to identify correctly treated injuries
        mask_series &= (injury_df.injury_treated_injury_treated_with_wrong_treatment == False)
        
        # Determine if bleeding was correctly treated
        bleeding_treated = bool(injury_df[mask_series].shape[0])
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Was bleeding correctly treated: {bleeding_treated}')
            display(injury_df)
        
        return bleeding_treated
    
    
    @staticmethod
    def get_injury_correctly_treated_time(injury_df, verbose=False):
        """
        Determine the action ticks it takes to correctly treat the given injury.
        
        Parameters:
            injury_df (pandas.DataFrame): DataFrame containing injury-specific data with relevant columns.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            int: The time it takes to correctly treat the injury, in action ticks.
        
        Note:
            The FRVRS logger has trouble logging multiple tool applications,
            so injury_treated_injury_treated_with_wrong_treatment == True
            remains and another INJURY_TREATED is never logged, even though
            the right tool was applied after that.
        """
        
        # Create a mask to identify rows where the treatment record indicates successful bleeding treatment
        mask_series = (injury_df.injury_treated_injury_treated == True)
        
        # Add to that mask to identify correctly treated injuries
        mask_series &= (injury_df.injury_treated_injury_treated_with_wrong_treatment == False)
        
        # Calculate the action tick when the injury is correctly treated
        bleeding_treated_time = injury_df[mask_series].action_tick.max()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Action tick when the injury is correctly treated: {bleeding_treated_time}')
            display(injury_df)
        
        return bleeding_treated_time
    
    
    ### Rasch Analysis Scene Functions ###
    
    
    def get_stills_value(self, scene_df, verbose=False):
        """
        0=All Stills not visited first, 1=All Stills visited first
        """
        actual_sequence, ideal_sequence, sort_dict = self.get_actual_and_ideal_sequences(scene_df, verbose=verbose)
        
        # Truncate both sequences to the stills length and compare them
        still_len = len(sort_dict.get('still', []))
        ideal_sequence = ideal_sequence.tolist()[:still_len]
        actual_sequence = actual_sequence.tolist()[:still_len]
        is_stills_visited_first = int(actual_sequence == ideal_sequence)
        
        return is_stills_visited_first
    
    
    def get_walkers_value(self, scene_df, verbose=False):
        """
        0=All Walkers not visited last, 1=All Walkers visited last
        """
        actual_sequence, ideal_sequence, sort_dict = self.get_actual_and_ideal_sequences(scene_df, verbose=verbose)
        
        # Truncate both sequences to the walkers length and compare them
        walker_len = len(sort_dict.get('walker', []))
        ideal_sequence = ideal_sequence.tolist()[-walker_len:]
        actual_sequence = actual_sequence.tolist()[-walker_len:]
        is_walkers_visited_last = int(actual_sequence == ideal_sequence)
        
        return is_walkers_visited_last
    
    
    @staticmethod
    def get_wave_value(scene_df, verbose=False):
        """
        0=No Wave Command issued, 1=Wave Command issued
        """
        mask_series = (scene_df.action_type == 'S_A_L_T_WAVE_IF_CAN')
        is_wave_command_issued = int(scene_df[mask_series].shape[0] > 0)
        
        return is_wave_command_issued
    
    
    @staticmethod
    def get_walk_value(scene_df, verbose=False):
        """
        0=No Walk Command issued, 1=Walk Command issued
        """
        mask_series = (scene_df.action_type == 'S_A_L_T_WALK_IF_CAN')
        is_walk_command_issued = int(scene_df[mask_series].shape[0] > 0)
        
        return is_walk_command_issued
    
    
    ### Rasch Analysis Patient Functions ###
    
    
    @staticmethod
    def get_treatment_value(patient_df, injury_id, verbose=False):
        """
        0=No Treatment or Wrong Treatment, 1=Correct Treatment
        """

        # Get required procedure
        mask_series = (patient_df.injury_id == injury_id) & ~patient_df.injury_record_required_procedure.isnull()
        if not mask_series.any(): return np.nan
        df = patient_df[mask_series]
        required_procedure = df.injury_record_required_procedure.squeeze()

        # Get first attempt
        mask_series = (patient_df.injury_id == injury_id) & ~patient_df.injury_treated_required_procedure.isnull()
        if not mask_series.any(): return 0
        df = patient_df[mask_series]
        first_procedure = df.sort_values(['action_tick']).injury_treated_required_procedure.tolist()[0]

        is_injury_treated = int(first_procedure == required_procedure)

        return is_injury_treated
    
    
    def get_tag_value(self, patient_df, verbose=False):
        """
        0=No Tag or Wrong Tag, 1=Correct Tag
        """
        try:
            is_tag_correct = self.get_is_tag_correct(patient_df, verbose=verbose)
            if np.isnan(is_tag_correct): is_tag_correct = 0
            else: is_tag_correct = int(is_tag_correct)
        except: is_tag_correct = 0

        return is_tag_correct
    
    
    @staticmethod
    def get_pulse_value(patient_df, verbose=False):
        """
        0=No Pulse Taken, 1=Pulse Taken
        """
        mask_series = (patient_df.action_type == 'PULSE_TAKEN')
        is_pulse_taken = int(patient_df[mask_series].shape[0] > 0)
        
        return is_pulse_taken
    
    
    ### Pandas Functions ###
    
    
    @staticmethod
    def get_statistics(describable_df, columns_list, verbose=False):
        """
        Calculates and returns descriptive statistics for a subset of columns in a Pandas DataFrame.
        
        Parameters:
            describable_df (pandas.DataFrame): The input DataFrame containing the data to analyze.
            columns_list (list): List of column names for which statistics should be calculated.
            verbose (bool): If True, display debug information.
        
        Returns:
            pandas.DataFrame: A DataFrame containing the descriptive statistics for the specified columns.
                The returned DataFrame includes the mean, mode, median, standard deviation (SD),
                minimum, 25th percentile, 50th percentile (median), 75th percentile, and maximum.
        """
        
        # Calculate basic descriptive statistics for the specified columns
        df = describable_df[columns_list].describe().rename(index={'std': 'SD'})
        
        # Check if the 'mode' statistic is present in the DataFrame
        if ('mode' not in df.index):
            
            # Create the mode row dictionary
            row_dict = {cn: describable_df[cn].mode().iloc[0] for cn in columns_list}
            
            # Convert the row dictionary to a data frame to match the df structure
            row_df = DataFrame([row_dict], index=['mode'])
            
            # Append the row data frame to the df data frame
            df = concat([df, row_df], axis='index', ignore_index=False)
        
        # Check if the 'median' statistic is present in the DataFrame
        if ('median' not in df.index):
            
            # Create the median row dictionary
            row_dict = {cn: describable_df[cn].median() for cn in columns_list}
            
            # Convert the row dictionary to a data frame to match the df structure
            row_df = DataFrame([row_dict], index=['median'])
            
            # Append the row data frame to the df data frame
            df = concat([df, row_df], axis='index', ignore_index=False)
        
        # Define a list of desired statistics indices
        index_list = ['mean', 'mode', 'median', 'SD', 'min', '25%', '50%', '75%', 'max']
        
        # Create a filter and reindexed data frame to select only the desired statistics
        mask_series = df.index.isin(index_list)
        statistics_df = df[mask_series].reindex(index_list)
        
        # If verbose is True, print additional information
        if verbose:
            print(f'columns_list: {columns_list}')
            display(describable_df)
            display(statistics_df)
        
        # Return the filtered DataFrame with the desired statistics
        return statistics_df
    
    
    def show_time_statistics(self, describable_df, columns_list):
        """
        Display summary statistics for time-related data in a readable format.
        
        Parameters:
            describable_df (pandas.DataFrame): The DataFrame containing time-related data.
            columns_list (list): List of column names for which time statistics should be calculated.
        
        This function calculates and displays summary statistics for time-related data,
        including mean, mode, median, and standard deviation (SD), in a human-readable format.
        
        The displayed DataFrame includes time-related statistics with formatted timedelta values.
        The standard deviation (SD) is presented as '±' followed by the corresponding value.
        
        Example usage:
        ```
        your_instance.show_time_statistics(your_dataframe, ['column1', 'column2'])
        ```
        
        Note: This function relies on the 'get_statistics' and 'format_timedelta' methods.
        """
        
        # Calculate basic descriptive statistics for time-related columns
        df = self.get_statistics(describable_df, columns_list)
        
        # Convert time values to timedelta objects and format them using format_timedelta()
        df = df.applymap(
            lambda x: self.format_timedelta(timedelta(milliseconds=int(x))),
            na_action='ignore'
        ).T
        
        # Format the standard deviation (SD) column to include the ± symbol
        df.SD = df.SD.map(lambda x: '±' + str(x))
        
        # Display the formatted time-related statistics
        display(df)
    
    
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
    
    
    @staticmethod
    def set_mcivr_metrics_types(action_type, df, row_index, row_series, verbose=False):
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
        
        # Attempt to read CSV file using pandas
        try: file_df = pd.read_csv(file_path, header=None, index_col=False)
        
        # If unsuccessful, try using a reader
        except:
            rows_list = []
            with open(file_path, 'r') as f:
                import csv
                reader = csv.reader(f, delimiter=',', quotechar='"')
                for values_list in reader:
                    if (values_list[-1] == ''): values_list.pop(-1)
                    rows_list.append({i: v for i, v in enumerate(values_list)})
            file_df = DataFrame(rows_list)
        
        # Ignore small files and return the subdirectory data frame unharmed
        if (file_df.shape[1] < 16): return sub_directory_df
        
        # Find columns containing only version numbers
        # VERSION_REGEX = re.compile(r'^\d\.\d$')
        # is_version_there = lambda x: bool(re.match(VERSION_REGEX, str(x)))
        # srs = file_df.applymap(is_version_there, na_action='ignore').sum()
        # columns_list = srs[srs == file_df.shape[0]].index.tolist()
        
        # Remove column 4 and rename all the numbered colums above that
        if (version_number > 1.0):
            file_df.drop(4, axis='columns', inplace=True)
            file_df.columns = list(range(file_df.shape[1]))
        
        # Add file name and logger version to the data frame
        file_dir_suffix = osp.abspath(sub_directory).replace(osp.abspath(self.data_logs_folder) + os.sep, '')
        file_df['file_name'] = '/'.join(file_dir_suffix.split(os.sep)) + '/' + file_name
        # if is_version_there(version_number): file_df['logger_version'] = float(version_number)
        # else: file_df['logger_version'] = 1.0
        file_df['logger_version'] = float(version_number)
        
        # Name the global columns
        columns_list = ['action_type', 'action_tick', 'event_time', 'session_uuid']
        file_df.columns = columns_list + file_df.columns.tolist()[len(columns_list):]
        
        # Parse the third column as a date column
        if ('event_time' in file_df.columns):
            if sub_directory.endswith('v.1.0'): file_df['event_time'] = to_datetime(file_df['event_time'], format='%m/%d/%Y %H:%M')
            
            # Attempt to infer the format automatically
            # else: file_df['event_time'] = to_datetime(file_df['event_time'], format='mixed')
            else: file_df['event_time'] = to_datetime(file_df['event_time'], infer_datetime_format=True)
        
        # Set the MCIVR metrics types
        for row_index, row_series in file_df.iterrows(): file_df = self.set_mcivr_metrics_types(row_series.action_type, file_df, row_index, row_series, verbose=verbose)
        
        # Section off player actions by session start and end
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
        for sub_directory, directories_list, files_list in os.walk(logs_folder):
            
            # Create a data frame to store the data for the current subdirectory
            sub_directory_df = DataFrame([])
            
            # Iterate over the files in the current subdirectory
            for file_name in files_list:
                
                # If the file is a CSV file, merge it into the subdirectory data frame
                if file_name.endswith('.csv'): sub_directory_df = self.process_files(sub_directory_df, sub_directory, file_name, verbose=verbose)
            
            # Append the data frame for the current subdirectory to the main data frame
            logs_df = pd.concat([logs_df, sub_directory_df], axis='index')
        
        # Convert event time to a datetime
        if ('event_time' in logs_df.columns): logs_df['event_time'] = pd.to_datetime(logs_df['event_time'], format='mixed')
        
        # Convert elapsed time to an integer
        if ('action_tick' in logs_df.columns):
            logs_df['action_tick'] = pd.to_numeric(logs_df['action_tick'], errors='coerce')
            logs_df['action_tick'] = logs_df['action_tick'].astype('int64')
        
        logs_df = logs_df.reset_index(drop=True)
        
        return logs_df
    
    
    @staticmethod
    def split_df_by_teleport(df, verbose=False):
        """
        Splits a DataFrame into multiple DataFrames based on teleport locations.
        
        Parameters:
            df (DataFrame): The DataFrame to split.
            nu (NotebookUtilities): An optional instance of NotebookUtilities.
            verbose (bool): Whether to print verbose output.
        
        Returns:
            List[DataFrame]: A list of DataFrames, each corresponding to a teleport location.
        """
        
        # Check for teleport locations
        if verbose:
            print(teleport_rows, df.index.tolist())
            raise # Consider removing the 'raise' statement after debugging.
        
        # Initialize variables
        split_dfs = []
        current_df = DataFrame()
        
        # Iterate over rows of the DataFrame
        for row_index, row_series in df.iterrows():
            
            # Check if the current row is a teleport location
            if row_index in teleport_rows:
                
                # If the current DataFrame is not empty, append it to the list of split DataFrames
                if current_df.shape[0] > 0: split_dfs.append(current_df)
                
                # Reset the current DataFrame
                current_df = DataFrame()
            
            # Print verbose output if verbose is True
            if verbose:
                print(row_index)
                display(row_series)
                display(nu.convert_to_df(row_index, row_series))
                raise
            
            # Append the current row to the current DataFrame
            current_df = concat([current_df, nu.convert_to_df(row_index, row_series)], axis='index')
        
        # If the current DataFrame is not empty, append it to the list of split DataFrames
        if current_df.shape[0] > 0: split_dfs.append(current_df)
        
        # Return the list of split DataFrames
        return split_dfs
    
    
    @staticmethod
    def show_long_runs(df, column_name, milliseconds, delta_fn, description, logs_df):
        """
        Display files with a specified duration in a given DataFrame.
        
        Parameters:
            df (pandas.DataFrame): DataFrame containing the relevant data.
            column_name (str): Name of the column in 'df' containing duration information.
            milliseconds (int): Threshold duration in milliseconds.
            delta_fn (function): Function to convert milliseconds to a time delta object.
            description (str): Description of the duration, e.g., 'longer' or 'shorter'.
            logs_df (pandas.DataFrame, optional): DataFrame containing additional log data, defaults to None.
        
        Returns:
            None

        Notes:
            This function prints the files with a duration greater than the specified threshold.
        """
        
        # Convert milliseconds to a time delta
        delta = delta_fn(milliseconds)
        
        # Display header
        print(f'\nThese files have {description} than {delta}:')
        
        # Filter sessions based on duration
        mask_series = (df[column_name] > milliseconds)
        session_uuid_list = df[mask_series].session_uuid.tolist()
        
        # Filter logs based on session UUID
        mask_series = logs_df.session_uuid.isin(session_uuid_list)
        
        # Specify logs folder path
        logs_folder = '../data/logs'
        
        # Process each unique file
        import csv
        from datetime import datetime
        for old_file_name in logs_df[mask_series].file_name.unique():
            old_file_path = osp.join(logs_folder, old_file_name)
            
            # Extract date from the log file
            with open(old_file_path, 'r') as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                for values_list in reader:
                    date_str = values_list[2]
                    break
                
                # Parse date string into a datetime object
                try: date_obj = datetime.strptime(date_str, '%m/%d/%Y %H:%M')
                except ValueError: date_obj = datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p')
                
                # Generate new file name based on the date
                new_file_name = date_obj.strftime('%y.%m.%d.%H%M.csv')
                new_sub_directory = old_file_name.split('/')[0]
                new_file_path = new_sub_directory + '/' + new_file_name
                
                # Display old and new file names
                print(f'{old_file_name} (or {new_file_path})')
    
    
    @staticmethod
    def replace_consecutive_rows(df, element_column, element_value, time_diff_column='time_diff', consecutive_cutoff=500):
        """
        Replaces consecutive rows in a list created from the element column with a count of how many there are in a row.
        
        Parameters:
            element_column: An element column from which the list of elements is created.
            element_value: The element to replace consecutive occurrences of.
            time_diff_column: the time diff column to check if the elements are close enough in time to consider consecutive.
            consecutive_cutoff: the number of time units the time diff column must show or less.
        
        Returns:
            A DataFrame with the rows of consecutive elements replaced with one row where the
            element_value now has appended to it a count of how many rows were deleted plus one.
        """
        result_df = DataFrame([], columns=df.columns); row_index = 0; row_series = Series([]); count = 0
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
    
    
    def get_elevens_data_frame(self, logs_df, file_stats_df, scene_stats_df, needed_columns=[], minimum_patient_count=11, verbose=False):
        
        # Get the column sets
        triage_columns = ['scene_type', 'is_scene_aborted']
        for cn in triage_columns:
            if not any(map(lambda df: cn in df.columns, [logs_df, file_stats_df, scene_stats_df])):
                raise ValueError(f'The {cn} column must be in either logs_df, file_stats_df, or scene_stats_df.')
        needed_set = set(triage_columns + needed_columns)
        logs_columns_set = set(logs_df.columns)
        file_columns_set = set(file_stats_df.columns)
        scene_columns_set = set(scene_stats_df.columns)
        
        # If some column's missing from just using the logs dataset...
        if bool(needed_set.difference(logs_columns_set)):
            
            # Merge in the file stats columns (what's still needed from using the scene and logs datasets together)
            on_columns = sorted(logs_columns_set.intersection(file_columns_set))
            file_stats_columns = on_columns + sorted(needed_set.difference(logs_columns_set).difference(scene_columns_set))
            merge_df = logs_df.merge(file_stats_df[file_stats_columns], on=on_columns)
            
            # Merge in the scene stats columns (what's still needed from using the file and logs datasets together)
            on_columns = sorted(logs_columns_set.intersection(scene_columns_set))
            scene_stats_columns = on_columns + sorted(needed_set.difference(logs_columns_set).difference(file_columns_set))
            merge_df = merge_df.merge(scene_stats_df[scene_stats_columns], on=on_columns)
        
        else: merge_df = logs_df
        
        # Get the triage scenes with at least eleven patients in them
        triage_mask = (merge_df.scene_type == 'Triage') & (merge_df.is_scene_aborted == False)
        elevens_df = merge_df[triage_mask].groupby(self.scene_groupby_columns).filter(lambda x: x.patient_id.nunique() >= minimum_patient_count)
        
        return elevens_df
    
    
    ### Plotting Functions ###
    
    
    def visualize_order_of_engagement(self, scene_df, engagement_order=None, color_dict=None, verbose=False):
        """
        Visualize the order of engagement of patients in a given scene.
        
        Parameters:
            scene_df (pandas.DataFrame): DataFrame containing scene data with relevant columns.
            engagement_order (list of tuples, optional): List of tuples of patient IDs,
                action ticks, and location coordinates. If not provided, a new list will be
                created using the patients in scene_df.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            None
        
        Raises:
            ValueError: If nu is not None and not an instance of NotebookUtilities.
        """
        
        # Get the engagement order
        if engagement_order is None: engagement_order = self.get_engagement_starts_order(scene_df, verbose=verbose)
        
        # Create a figure and add a subplot
        fig, ax = plt.subplots(figsize=(18, 9))
        
        # Get the color dictionary
        if color_dict is None:
            color_cycler = nu.get_color_cycler(scene_df.groupby('patient_id').size().shape[0])
            import matplotlib.colors as mcolors
            color_dict = {et[0]: mcolors.to_hex(fcd['color']) for et, fcd in zip(engagement_order, color_cycler())}
            if verbose:
                print('\nPrinting the colors in the color_cycler:')
                print(color_dict)
                print()
        color_dict = {et[0]: color_dict.get(et[0]) for et in engagement_order}
        
        # Show the positions of patients recorded and engaged at our scene
        for engagement_tuple in engagement_order:
            
            # Get the entire patient history
            patient_id = engagement_tuple[0]
            mask_series = (scene_df.patient_id == patient_id)
            patient_df = scene_df[mask_series]
            
            # Get all the wanderings
            x_dim, z_dim = self.get_wanderings(patient_df)
            
            # Generate a wrapped label
            label = self.get_wrapped_label(patient_df)
            
            # Plot the ball and chain
            face_color = color_dict[patient_id]
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
        title = f'Order-of-Engagement Map for UUID {scene_df.session_uuid.unique().item()} ({humanize.ordinal(scene_df.scene_id.unique().item()+1)} Scene)'
        ax.set_title(title)
        
        # Save figure to PNG
        file_path = osp.join(nu.saves_png_folder, re.sub(r'\W+', '_', str(title)).strip('_').lower() + '.png')
        if verbose: print(f'Saving figure to {file_path}')
        plt.savefig(file_path, bbox_inches='tight')
        
        return fig, ax
    
    
    def visualize_player_movement(self, logs_df, scene_mask, title=None, save_only=False, verbose=False):
        """
        Visualizes the player movement for the given session mask in a 2D plot.
    
        Parameters:
            scene_mask (pandas.Series): A boolean mask indicating which rows of the logs_df DataFrame belong to the current scene.
            title (str, optional): The title of the plot, if saving.
            save_only (bool, optional): Whether to only save the plot to a PNG file and not display it.
            logs_df (pandas.DataFrame): A DataFrame containing the FRVRS logs.
            verbose (bool, optional): Whether to print verbose output.
    
        Returns:
            None: The function either displays the plot or saves it to a file.
    
        Note:
            This function visualizes player movement based on data in the DataFrame `logs_df`.
            It can display player positions, locations, and teleportations.
            Use `scene_mask` to filter the data for a specific scene.
            Set `save_only` to True to save the plot as a PNG file with the specified `title`.
            Set `verbose` to True to enable verbose printing.
        """
    
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
            color_cycler = nu.get_color_cycler(logs_df[scene_mask].groupby('patient_id').size().shape[0])
            for (patient_id, patient_df), face_color_dict in zip(logs_df[scene_mask].sort_values('action_tick').groupby('patient_id'), color_cycler()):
                
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
            mask_series = (logs_df.action_type == 'PLAYER_LOCATION') & scene_mask
            locations_df = logs_df[mask_series]
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
            mask_series = (logs_df.action_type == 'TELEPORT') & scene_mask
            teleports_df = logs_df[mask_series]
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
        self, logs_df, df, sorting_column, mask_series=None, is_ascending=True, humanize_type='precisedelta',
        title_str='slowest action to control time', verbose=False
    ):
        """
        Get the time group with some edge case and visualize the player movement.
        
        Parameters:
            logs_df (pandas.DataFrame): A DataFrame containing the FRVRS logs.
            df (pandas.DataFrame): The input DataFrame containing player movement data.
            sorting_column (str): The column based on which the DataFrame will be sorted.
            mask_series (pandas.Series or None): Optional mask series to filter rows in the DataFrame.
            is_ascending (bool): If True, sort the DataFrame in ascending order; otherwise, in descending order.
            humanize_type (str): The type of humanization to be applied to the time values ('precisedelta', 'percentage', 'intword').
            title_str (str): Additional string to be included in the visualization title.
            nu (NotebookUtilities or None): An instance of NotebookUtilities or None. If None, a new instance will be created.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            None
        """
        
        # Set default mask_series if not provided
        if mask_series is None: mask_series = [True] * df.shape[0]
        
        # Get the row with extreme value based on sorting_column
        df1 = df[mask_series].sort_values(
            [sorting_column], ascending=[is_ascending]
        ).head(1)
        
        # Check if the DataFrame is not empty
        if df1.shape[0]:
            
            # Extract session_uuid and scene_id from the extreme row
            session_uuid = df1.session_uuid.squeeze()
            scene_id = df1.scene_id.squeeze()
            
            # Load logs DataFrame
            base_mask_series = (logs_df.session_uuid == session_uuid) & (logs_df.scene_id == scene_id)
            
            # Construct the title for the visualization
            title = f'Location Map for UUID {session_uuid} ({humanize.ordinal(scene_id+1)} Scene)'
            title += f' showing trainee with the {title_str} ('
            if is_ascending: column_value = df1[sorting_column].min()
            else: column_value = df1[sorting_column].max()
            if verbose: display(column_value)
            
            # Humanize the time value based on the specified type
            if (humanize_type == 'precisedelta'):
                title += humanize.precisedelta(timedelta(milliseconds=column_value)) + ')'
            elif (humanize_type == 'percentage'):
                title += str(100 * column_value) + '%)'
            elif (humanize_type == 'intword'):
                title += humanize.intword(column_value) + ')'
            
            # Visualize player movement
            self.visualize_player_movement(logs_df, base_mask_series, title=title)
    
    
    def show_timelines(self, logs_df, random_session_uuid=None, random_scene_index=None, color_cycler=None, verbose=False):
        """
        Display timelines for patient engagements in a random session and scene.
        
        Parameters:
            logs_df (pd.DataFrame): DataFrame containing FRVRS logs.
            random_session_uuid (str, optional): UUID of the random session. If not provided, a random session will be selected.
            random_scene_index (int, optional): Index of the random scene. If not provided, a random scene within the selected session will be chosen.
            color_cycler (callable, optional): A callable that returns a color for each patient engagement timeline. If not provided, it will be generated.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        
        Returns:
            Tuple of (random_session_uuid, random_scene_index).
        """
        
        # Get a random session if not provided
        if random_session_uuid is None: random_session_uuid = random.choice(logs_df.session_uuid.unique())
        
        # Get a random scene from within the session if not provided
        if random_scene_index is None:
            mask_series = (logs_df.session_uuid == random_session_uuid)
            random_scene_index = random.choice(logs_df[mask_series].scene_id.unique())
        
        # Get the scene mask
        scene_mask_series = (logs_df.session_uuid == random_session_uuid) & (logs_df.scene_id == random_scene_index)
        
        # Get the event time and elapsed time of each person engaged
        mask_series = scene_mask_series & logs_df.action_type.isin([
            'PATIENT_ENGAGED', 'INJURY_TREATED', 'PULSE_TAKEN', 'TAG_APPLIED', 'TOOL_APPLIED'
        ])
        columns_list = ['patient_id', 'action_tick']
        patient_engagements_df = logs_df[mask_series][columns_list].sort_values(['action_tick'])
        if verbose: display(patient_engagements_df)
        if patient_engagements_df.shape[0]:
            
            # For each patient, get a timeline of every reference on or before engagement
            if color_cycler is None: color_cycler = nu.get_color_cycler(len(patient_engagements_df.patient_id.unique()))
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
                mask_series = scene_mask_series & (logs_df.patient_id == random_patient_id)
                action_tick = patient_df.action_tick.max()
                mask_series &= (logs_df.action_tick <= action_tick)
                
                df1 = logs_df[mask_series].sort_values(['action_tick'])
                if verbose: display(df1)
                if df1.shape[0]:
                    
                    # Get the fine horizontal line parameters and plot dimensions
                    xmin = df1.action_tick.min(); hlinexmins_list.append(xmin);
                    if xmin < left_lim: left_lim = xmin
                    xmax = df1.action_tick.max(); hlinexmaxs_list.append(xmax);
                    if xmax > right_lim: right_lim = xmax
                    
                    # Get the vertical line parameters
                    mask_series = df1.action_type.isin(['SESSION_END', 'SESSION_START'])
                    for x in df1[mask_series].action_tick: vlinexs_list.append(x)
                    
                    # Get the action type annotation parameters
                    mask_series = df1.action_type.isin(['INJURY_TREATED', 'PATIENT_ENGAGED', 'PULSE_TAKEN', 'TAG_APPLIED', 'TOOL_APPLIED'])
                    for label, action_type_df in df1[mask_series].groupby('action_type'):
                        for x in action_type_df.action_tick:
                            annotation_tuple = (label.lower().replace('_', ' '), x, y)
                            hlineaction_types_list.append(annotation_tuple)
        
        # Create the subplot axis
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
        
        # Manually set xticklabels using humanized timedelta values
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
        Plot grouped box and whiskers using seaborn.
        
        Parameters:
            transformable_df: DataFrame, the data to be plotted.
            x_column_name: str, the column for x-axis.
            y_column_name: str, the column for y-axis.
            x_label: str, label for x-axis.
            y_label: str, label for y-axis.
            transformer_name: str, the name of the transformation function (default is 'min').
            is_y_temporal: bool, if True, humanize y-axis tick labels as temporal values.
        
        Returns:
            None
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
        
        # Humanize y tick labels if is_y_temporal is True
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
    
    
    def show_gaze_timeline(
        self, logs_df, random_session_uuid=None, random_scene_index=None, consecutive_cutoff=600, patient_color_dict=None, verbose=False
    ):
        """
        Display a timeline of player gaze events for a random session and scene.
        
        Parameters:
            logs_df (pandas.DataFrame): DataFrame containing logs data.
            random_session_uuid (str): UUID of the random session. If None, a random session will be selected.
            random_scene_index (int): Index of the random scene. If None, a random scene will be selected within the session.
            consecutive_cutoff (int): Time cutoff for consecutive rows.
            patient_color_dict (dict): Dictionary mapping patient IDs to colors.
            verbose (bool): If True, display additional information.
        
        Returns:
            tuple: Random session UUID and random scene index.
        """
        
        # Get a random session if not provided
        if random_session_uuid is None:
            mask_series = (logs_df.action_type.isin(['PLAYER_GAZE']))
            mask_series &= ~logs_df.player_gaze_patient_id.isnull()
            random_session_uuid = random.choice(logs_df[mask_series].session_uuid.unique())
        
        # Get a random scene within the session if not provided
        if random_scene_index is None:
            mask_series = (logs_df.session_uuid == random_session_uuid)
            mask_series &= (logs_df.action_type.isin(['PLAYER_GAZE']))
            mask_series &= ~logs_df.player_gaze_patient_id.isnull()
            random_scene_index = random.choice(logs_df[mask_series].scene_id.unique())
        
        # Get the scene mask
        scene_mask_series = (logs_df.session_uuid == random_session_uuid) & (logs_df.scene_id == random_scene_index)
        
        # Get the elapsed time of the player gaze
        mask_series = scene_mask_series & (logs_df.action_type.isin(['PLAYER_GAZE', 'SESSION_END', 'SESSION_START']))
        columns_list = ['action_tick', 'patient_id']
        patient_gazes_df = logs_df[mask_series][columns_list].sort_values(['action_tick'])
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
        for label, x, y in zip(hlinelabels_list, hlinexmins_list, hlineys_list):
            plt.annotate(label, (x, y), textcoords='offset points', xytext=(0, -8), ha='left')
        
        # Annotate the patients who have been gazed at along their timeline
        if patient_color_dict is None:
            scene_df = logs_df[scene_mask_series]
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
    
    
    @staticmethod
    def plot_sequence_by_scene_tuple(
        scene_tuple,
        sequence,
        logs_df,
        summary_statistics_df=None,
        actions_mask_series=None,
        highlighted_ngrams=[],
        color_dict={'SESSION_START': 'green', 'SESSION_END': 'red'},
        suptitle=None,
        verbose=False
    ):
        """
        Plots a given sequence for a specific scene from a VR simulation log.
        
        This function visualizes the sequence along with relevant information like the scene's entropy,
        turbulence, and complexity, while highlighting specific n-grams and coloring different action types.
        
        Parameters:
            scene_tuple (tuple): A tuple containing the session UUID and scene ID of the specific scene.
            sequence (list): The sequence of actions to be plotted (e.g., a list of action types).
            logs_df (pandas.DataFrame): DataFrame containing detailed logs of all VR simulation
                interactions.
            summary_statistics_df (pandas.DataFrame, optional): DataFrame containing summary statistics
                for all scenes. If not provided, tries to load from a pickle file.
            actions_mask_series (pandas.Series, optional): A mask defining which actions to include in the plot.
                If not provided, all actions are used.
            highlighted_ngrams (list, optional): A list of n-grams to highlight in the plot.
            color_dict (dict, optional): A dictionary mapping action types to color values.
            suptitle (str, optional): The title of the plot. If not provided, summary statistics are shown.
            verbose (bool, optional): Whether to print debug information during execution.
        
        Returns:
            matplotlib.figure.Figure: The figure containing the plot.
        
        Raises:
            Exception: If `summary_statistics_df` is missing and cannot be loaded from pickle files.
        """
        
        # Extract session_uuid and scene_id from the scene_tuple
        session_uuid, scene_id = scene_tuple
        
        # 1. Load summary_statistics_df from pickle if not provided
        if verbose: display(summary_statistics_df)
        if (summary_statistics_df is None):
            if nu.pickle_exists('summary_statistics_df'): summary_statistics_df = nu.load_object('summary_statistics_df')
            else: raise Exception('You need to provide summary_statistics_df')
        
        # 2. Get relevant actions from VR logs
        if verbose: display(logs_df)
        
        # Build a data frame of action types
        if (actions_mask_series is None): actions_mask_series = [True] * logs_df.shape[0]
        mask_series = (logs_df.session_uuid == session_uuid) & (logs_df.scene_id == scene_id) & actions_mask_series
        scene_df = logs_df[mask_series].sort_values('action_tick')
        
        # Build a list of the first of each action type
        actions_list = []
        for row_index, row_series in scene_df.iterrows():
            action_type = row_series.voice_command_message
            if pd.notnull(action_type) and (action_type not in actions_list): actions_list.append(action_type)
            action_type = row_series.action_type
            if pd.notnull(action_type) and (action_type != 'VOICE_COMMAND') and (action_type not in actions_list): actions_list.append(action_type)
        
        # Create a plot title
        if suptitle is None:
            mask_series = (summary_statistics_df.session_uuid == session_uuid) & (summary_statistics_df.scene_id == scene_id)
            df = summary_statistics_df[mask_series]
            entropy = df.sequence_entropy.squeeze()
            turbulence = df.sequence_turbulence.squeeze()
            complexity = df.sequence_complexity.squeeze()
            suptitle = f'entropy = {entropy:0.2f}, turbulence = {turbulence:0.2f}, complexity = {complexity:0.2f}'
        
        # Plot the sequence using notebook util's plot_sequence if sequence is not empty
        if verbose: print(sequence)
        if(sequence):
            fig, ax = nu.plot_sequence(
                sequence, highlighted_ngrams=highlighted_ngrams, color_dict=color_dict, suptitle=suptitle, alphabet_list=actions_list, verbose=verbose
            )
            
            return fig, ax