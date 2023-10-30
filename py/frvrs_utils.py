
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

    ### String Functions ###
    
    def format_timedelta(self, timedelta):
        """
        Formats a timedelta object to a string in the
        format '0 sec', '30 sec', '1 min', '1:30', '2 min', etc.
        
        Args:
          timedelta: A timedelta object.
        
        Returns:
          A string in the format '0 sec', '30 sec', '1 min',
          '1:30', '2 min', etc.
        """
        seconds = timedelta.total_seconds()
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        
        if minutes == 0: return f'{seconds} sec'
        elif seconds > 0: return f'{minutes}:{seconds:02}'
        else: return f'{minutes} min'
    
    
    ### List Functions ###
    
    
    def replace_consecutive_elements(self, actions_list, element='PATIENT_ENGAGED'):
        """
        Replaces consecutive elements in a list with a count of how many there are in a row.
        
        Args:
            list1: A list of elements.
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
        from datetime import datetime
        old_file_path = '../data/logs/' + old_file_name
        with open(old_file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for values_list in reader:
                date_str = values_list[2]
                break
            try: date_obj = datetime.strptime(date_str, '%m/%d/%Y %H:%M')
            except ValueError: date_obj = datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p')
            new_file_name = date_obj.strftime('%y.%m.%d.%H%M.csv')
            new_file_path = old_file_name.replace(old_file_name.split('/')[-1], new_file_name)
    
            return new_file_path
    
    ### Pandas Functions ###

    def get_statistics(self, describable_df, columns_list):
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
        df = self.get_statistics(describable_df, columns_list).applymap(lambda x: self.format_timedelta(timedelta(milliseconds=int(x))), na_action='ignore').T
        df.SD = df.SD.map(lambda x: '±' + str(x))
        display(df)
    
    def get_session_groupby(self, frvrs_logs_df=None, mask_series=None, extra_column=None):
        if frvrs_logs_df is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=self.data_folder,
                saves_folder_path=self.saves_folder
            )
            frvrs_logs_df = nu.load_object('frvrs_logs_df')
        if (mask_series is None) and (extra_column is None):
            gb = frvrs_logs_df.sort_values(['elapsed_time']).groupby(['session_uuid'])
        elif (mask_series is None) and (extra_column is not None):
            gb = frvrs_logs_df.sort_values(['elapsed_time']).groupby(['session_uuid', extra_column])
        elif (mask_series is not None) and (extra_column is None):
            gb = frvrs_logs_df[mask_series].sort_values(['elapsed_time']).groupby(['session_uuid'])
        elif (mask_series is not None) and (extra_column is not None):
            gb = frvrs_logs_df[mask_series].sort_values(['elapsed_time']).groupby(['session_uuid', extra_column])
    
        return gb
    
    def set_time_groups(self, df):
        """
        Section off player actions by session start and end.
        
        Args:
            df: A Pandas DataFrame containing the player action data with its index reset.
        
        Returns:
            A Pandas DataFrame with the `scene_index` column added.
        """
    
        # Set the whole file to zero first
        df = df.sort_values('elapsed_time')
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
    
        Args:
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
        columns_list = ['action_type', 'elapsed_time', 'event_time', 'session_uuid']
        file_df.columns = columns_list + file_df.columns.tolist()[len(columns_list):]
        
        # Parse the third column as a date column
        if ('event_time' in file_df.columns):
            if sub_directory.endswith('v.1.0'): file_df['event_time'] = to_datetime(file_df['event_time'], format='%m/%d/%Y %H:%M')
            # elif sub_directory.endswith('v.1.3'): file_df['event_time'] = to_datetime(file_df['event_time'], format='%m/%d/%Y %I:%M:%S %p')
            else: file_df['event_time'] = to_datetime(file_df['event_time'], format='mixed')
        
        # Set the MCIVR metrics types
        for row_index, row_series in file_df.iterrows(): file_df = self.set_mcivr_metrics_types(row_series.action_type, file_df, row_index, row_series)
        
        # Section off player actions by session start and end
        file_df = self.set_time_groups(file_df)
        
        # Append the data frame for the current file to the data frame for the current subdirectory
        sub_directory_df = concat([sub_directory_df, file_df], axis='index')
    
        return sub_directory_df
    
    def split_df_by_teleport(self, df, nu=None, verbose=False):
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
    
    def show_long_runs(self, df, column_name, milliseconds, delta_fn, description):
        delta = delta_fn(milliseconds)
        print(f'\nThese files have {description} than {delta}:')
        mask_series = (df[column_name] > milliseconds)
        session_uuid_list = df[mask_series].session_uuid.tolist()
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
    
    ### Plotting Functions ###
    
    def visualize_player_movement(self, session_mask, title=None, save_only=False, nu=None, verbose=False):
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
    
        # Load the notebook utilities if not provided.
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
            color_cycler = nu.get_color_cycler(frvrs_logs_df[session_mask].groupby('patient_id').size().shape[0])
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
                srs = df1[columns_list].apply(pd.Series.notnull, axis='columns').sum()
                mask_series = (srs > 0)
                cn = srs[mask_series].index.iloc[0]
                if (type(patient_id) == tuple): patient_id = patient_id[0]
    
                # Generate a wrapped label
                label = patient_id.replace(' Root', ' (') + df1[cn].dropna().iloc[-1] + ')'
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
        """
        Get time group with some edge case and visualize the player movement there
        """
        
        if mask_series is None: mask_series = [True] * df.shape[0]
        df1 = df[mask_series].sort_values(
            [sorting_column], ascending=[is_ascending]
        ).head(1)
        if df1.shape[0]:
            session_uuid = df1.session_uuid.squeeze()
            time_group = df1.time_group.squeeze()
            if frvrs_logs_df is None:
                from notebook_utils import NotebookUtilities
                nu = NotebookUtilities(
                    data_folder_path=self.data_folder,
                    saves_folder_path=self.saves_folder
                )
                frvrs_logs_df = nu.load_object('frvrs_logs_df')
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
    
    def show_timelines(self, random_session_uuid=None, random_time_group=None, captured_patient_id='Gary_3 Root', nu=None, verbose=False):
        
        # Get a random session
        if random_session_uuid is None:
            random_session_uuid = random.choice(self.frvrs_logs_df.session_uuid.unique())
        
        # Get a random scene from within the session
        if random_time_group is None:
            mask_series = (self.frvrs_logs_df.session_uuid == random_session_uuid)
            random_time_group = random.choice(self.frvrs_logs_df[mask_series].scene_index.unique())
        
        # Get the event time and elapsed time of each person engaged
        mask_series = (self.frvrs_logs_df.session_uuid == random_session_uuid) & (self.frvrs_logs_df.scene_index == random_time_group)
        mask_series &= self.frvrs_logs_df.action_type.isin([
            'PATIENT_ENGAGED', 'INJURY_TREATED', 'PULSE_TAKEN', 'TAG_APPLIED', 'TOOL_APPLIED'
        ])
        columns_list = ['patient_id', 'elapsed_time', 'event_time']
        patient_engagements_df = self.frvrs_logs_df[mask_series][columns_list].sort_values(['event_time', 'elapsed_time'])
        if verbose: display(patient_engagements_df)
        
        # For each patient, get a timeline of every reference on or before engagement
        if nu is None:
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities()
        color_cycler = nu.get_color_cycler(len(patient_engagements_df.patient_id.unique()))
        hlineys_list = []; hlinexmins_list = []; hlinexmaxs_list = []; hlinecolors_list = []; hlinelabels_list = []
        hlineaction_types_list = []; vlinexs_list = []
        left_lim = 999999; right_lim = -999999
        for (patient_id, df), (y, face_color_dict) in zip(patient_engagements_df.groupby('patient_id'), enumerate(color_cycler())):
        
            # Get the broad horizontal line parameters
            hlineys_list.append(y)
            face_color = face_color_dict['color']
            hlinecolors_list.append(face_color)
            hlinelabels_list.append(patient_id)
        
            # Create the filter for the first scene
            mask_series = (self.frvrs_logs_df.patient_id == patient_id)
            mask_series &= (self.frvrs_logs_df.session_uuid == random_session_uuid) & (self.frvrs_logs_df.scene_index == random_time_group)
            elapsed_time = df.elapsed_time.max()
            event_time = df.event_time.max()
            mask_series &= (self.frvrs_logs_df.elapsed_time <= elapsed_time) & (self.frvrs_logs_df.event_time <= event_time)
            
            df1 = self.frvrs_logs_df[mask_series].sort_values(['event_time', 'elapsed_time'])
            captured_patient_id_df = DataFrame([])
            if (patient_id == captured_patient_id): captured_patient_id_df = df1.copy()
        
            # Get the fine horizontal line parameters and plot dimensions
            xmin = df1.elapsed_time.min(); hlinexmins_list.append(xmin);
            if xmin < left_lim: left_lim = xmin
            xmax = df1.elapsed_time.max(); hlinexmaxs_list.append(xmax);
            if xmax > right_lim: right_lim = xmax
            
            # Get the vertical line parameters
            mask_series = df1.action_type.isin(['SESSION_END', 'SESSION_START'])
            for x in df1[mask_series].elapsed_time:
                vlinexs_list.append(x)
            
            # Get the action type annotation parameters
            mask_series = df1.action_type.isin(['INJURY_TREATED', 'PATIENT_ENGAGED', 'PULSE_TAKEN', 'TAG_APPLIED', 'TOOL_APPLIED'])
            for label, df2 in df1[mask_series].groupby('action_type'):
                for x in df2.elapsed_time:
                    annotation_tuple = (label.lower().replace('_', ' '), x, y)
                    hlineaction_types_list.append(annotation_tuple)
        
        ax = plt.figure(figsize=(18, 9)).add_subplot(1, 1, 1)
        
        # Add the timelines to the figure subplot axis
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
        ax.set_title(f'Multi-Patient Timeline for UUID {random_session_uuid} and Scene {random_time_group}')
        ax.set_xlabel('Elapsed Time since Scene Start')
        
        tick_labels = ax.get_xticklabels()
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
    
        return random_session_uuid, random_time_group, captured_patient_id_df