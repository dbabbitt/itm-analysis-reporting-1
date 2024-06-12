#!/usr/bin/env python
# Utility Functions to manipulate FRVRS logger data.
# Dave Babbitt <dave.babbitt@bigbear.ai>
# Author: Dave Babbitt, Machine Learning Engineer
# coding: utf-8

# Soli Deo gloria

from . import (
    nu, nan, isnan, listdir, makedirs, osp, remove, sep, walk, CategoricalDtype, DataFrame, Index, NaT, Series, concat, isna,
    notnull, read_csv, read_excel, read_pickle, to_datetime, math, np, re, warnings, display
)
from datetime import datetime, timedelta
from pandas import to_numeric
from re import sub
import csv
import humanize
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm

warnings.filterwarnings('ignore')

class FRVRSUtilities(object):
    """
    This class implements the core of the utility
    functions needed to manipulate FRVRS logger
    data.
    
    Examples:
        import sys
        import os.path as osp
        sys.path.insert(1, osp.abspath('../py'))
        from FRVRS import fu
    """
    
    
    def __init__(self, data_folder_path=None, saves_folder_path=None, verbose=False):
        
        # Create the data folder if it doesn't exist
        if data_folder_path is None: self.data_folder = '../data'
        else: self.data_folder = data_folder_path
        makedirs(self.data_folder, exist_ok=True)
        if verbose: print('data_folder: {}'.format(osp.abspath(self.data_folder)), flush=True)
        
        # Create the saves folder if it doesn't exist
        if saves_folder_path is None: self.saves_folder = '../saves'
        else: self.saves_folder = saves_folder_path
        makedirs(self.saves_folder, exist_ok=True)
        if verbose: print('saves_folder: {}'.format(osp.abspath(self.saves_folder)), flush=True)
        
        # FRVRS log constants
        self.data_logs_folder = osp.join(self.data_folder, 'logs'); makedirs(name=self.data_logs_folder, exist_ok=True)
        self.scene_groupby_columns = ['session_uuid', 'scene_id']
        self.patient_groupby_columns = self.scene_groupby_columns + ['patient_id']
        self.injury_groupby_columns = self.patient_groupby_columns + ['injury_id']
        self.modalized_columns = [
            'patient_id', 'injury_id', 'location_id', 'patient_sort', 'patient_pulse', 'patient_salt', 'patient_hearing', 'patient_breath', 'patient_mood', 'patient_pose', 'injury_severity',
            'injury_required_procedure', 'injury_body_region', 'tool_type'
        ]
        
        # List of action types to consider as simulation loggings that can't be directly read by the responder
        self.simulation_actions_list = ['INJURY_RECORD', 'PATIENT_RECORD', 'S_A_L_T_WALKED', 'TRIAGE_LEVEL_WALKED', 'S_A_L_T_WAVED', 'TRIAGE_LEVEL_WAVED']
        
        # List of action types that assume 1-to-1 interaction
        self.responder_negotiations_list = ['PULSE_TAKEN', 'TAG_APPLIED', 'TOOL_APPLIED']#, 'INJURY_TREATED', 'PATIENT_ENGAGED'
        
        # List of action types to consider as user actions
        self.known_mcivr_metrics = [
            'BAG_ACCESS', 'BAG_CLOSED', 'INJURY_RECORD', 'INJURY_TREATED', 'PATIENT_DEMOTED', 'PATIENT_ENGAGED', 'BREATHING_CHECKED', 'PATIENT_RECORD', 'SP_O2_TAKEN',
            'S_A_L_T_WALKED', 'TRIAGE_LEVEL_WALKED', 'S_A_L_T_WALK_IF_CAN', 'TRIAGE_LEVEL_WALK_IF_CAN', 'S_A_L_T_WAVED', 'TRIAGE_LEVEL_WAVED', 'S_A_L_T_WAVE_IF_CAN',
            'TRIAGE_LEVEL_WAVE_IF_CAN', 'TAG_DISCARDED', 'TAG_SELECTED', 'TELEPORT', 'TOOL_DISCARDED', 'TOOL_HOVER', 'TOOL_SELECTED', 'VOICE_CAPTURE',
            'VOICE_COMMAND', 'BUTTON_CLICKED', 'PLAYER_LOCATION', 'PLAYER_GAZE', 'SESSION_START', 'SESSION_END'
        ] + self.responder_negotiations_list
        self.action_types_list = [
            'TELEPORT', 'S_A_L_T_WALK_IF_CAN', 'TRIAGE_LEVEL_WALK_IF_CAN', 'S_A_L_T_WAVE_IF_CAN', 'TRIAGE_LEVEL_WAVE_IF_CAN', 'PATIENT_ENGAGED',
            'BAG_ACCESS', 'TOOL_HOVER', 'TOOL_SELECTED', 'INJURY_TREATED', 'TAG_SELECTED', 'BAG_CLOSED', 'TAG_DISCARDED', 'TOOL_DISCARDED'
        ] + self.responder_negotiations_list
        
        # According to the PatientEngagementWatcher class in the engagement detection code, this Euclidean distance, if the patient has been looked at, triggers enagement
        # The engagement detection code spells out the responder etiquette:
        # 1) if you don't want to trigger a patient walking by, don't look at them, 
        # 2) if you teleport to someone, you must look at them to trigger engagement
        patient_lookup_distance = 2.5
        
        # List of command messages to consider as user actions; added Open World commands 20240429
        self.command_columns_list = ['voice_command_message', 'button_command_message']
        self.command_messages_list = [
            'walk to the safe area', 'wave if you can', 'are you hurt', 'reveal injury', 'lay down', 'where are you',
            'can you hear', 'anywhere else', 'what is your name', 'hold still', 'sit up/down', 'stand up'
        ] + ['can you breathe', 'show me', 'stand', 'walk', 'wave']
        
        # List of columns that contain only boolean values
        self.boolean_columns_list = [
            'injury_record_injury_treated_with_wrong_treatment', 'injury_record_injury_treated',
            'injury_treated_injury_treated_with_wrong_treatment', 'injury_treated_injury_treated'
        ]
        
        # List of columns that contain patientIDs
        self.patient_id_columns_list = [
            'patient_demoted_patient_id', 'patient_record_patient_id', 'injury_record_patient_id', 's_a_l_t_walk_if_can_patient_id',
            's_a_l_t_walked_patient_id', 's_a_l_t_wave_if_can_patient_id', 's_a_l_t_waved_patient_id', 'patient_engaged_patient_id',
            'pulse_taken_patient_id', 'injury_treated_patient_id', 'tool_applied_patient_id', 'tag_applied_patient_id',
            'player_gaze_patient_id', 'breathing_checked_patient_id', 'sp_o2_taken_patient_id', 'triage_level_walked_patient_id'
        ]
        
        # List of columns that contain locationIDs
        self.location_id_columns_list = [
            'teleport_location', 'patient_demoted_position', 'patient_record_position', 'injury_record_injury_injury_locator',
            's_a_l_t_walk_if_can_sort_location', 's_a_l_t_walked_sort_location', 's_a_l_t_wave_if_can_sort_location',
            's_a_l_t_waved_sort_location', 'patient_engaged_position', 'bag_access_location', 'injury_treated_injury_injury_locator',
            'bag_closed_location', 'tag_discarded_location', 'tool_discarded_location', 'player_location_location',
            'player_gaze_location', 'triage_level_walked_location'
        ]
        
        # List of columns with injuryIDs
        self.injury_id_columns_list = ['injury_record_id', 'injury_treated_id']
        
        # Patient SORT designations
        self.patient_sort_columns_list = ['patient_demoted_sort', 'patient_record_sort', 'patient_engaged_sort']
        self.patient_sort_order = ['still', 'waver', 'walker']
        self.sort_category_order = CategoricalDtype(categories=self.patient_sort_order, ordered=True)
        
        # Patient SALT designations
        self.patient_salt_columns_list = ['patient_demoted_salt', 'patient_record_salt', 'patient_engaged_salt']
        self.patient_salt_order = ['DEAD', 'EXPECTANT', 'IMMEDIATE', 'DELAYED', 'MINIMAL']
        self.salt_category_order = CategoricalDtype(categories=self.patient_salt_order, ordered=True)
        
        # Tag colors
        self.tag_columns_list = ['tag_selected_type', 'tag_applied_type', 'tag_discarded_type']
        self.tag_color_order = ['black', 'gray', 'red', 'yellow', 'green', 'Not Tagged']
        self.colors_category_order = CategoricalDtype(categories=self.tag_color_order, ordered=True)
        
        # Patient pulse designations
        self.pulse_columns_list = ['patient_demoted_pulse', 'patient_record_pulse', 'patient_engaged_pulse']
        self.patient_pulse_order = ['none', 'faint', 'fast', 'normal']
        self.pulse_category_order = CategoricalDtype(categories=self.patient_pulse_order, ordered=True)
        
        # Patient breath designations
        self.breath_columns_list = ['patient_demoted_breath', 'patient_record_breath', 'patient_engaged_breath', 'patient_checked_breath']
        self.patient_breath_order = ['none', 'collapsedLeft', 'collapsedRight', 'restricted', 'fast', 'normal']
        self.breath_category_order = CategoricalDtype(categories=self.patient_breath_order, ordered=True)
        
        # Patient hearing designations
        self.hearing_columns_list = ['patient_record_hearing', 'patient_engaged_hearing']
        self.patient_hearing_order = ['none', 'limited', 'normal']
        self.hearing_category_order = CategoricalDtype(categories=self.patient_hearing_order, ordered=True)
        
        # Patient mood designations
        self.mood_columns_list = ['patient_demoted_mood', 'patient_record_mood', 'patient_engaged_mood']
        self.patient_mood_order = ['dead', 'unresponsive', 'agony', 'upset', 'calm', 'low', 'normal', 'none']
        self.mood_category_order = CategoricalDtype(categories=self.patient_mood_order, ordered=True)
        
        # Patient pose designations
        self.pose_columns_list = ['patient_demoted_pose', 'patient_record_pose', 'patient_engaged_pose']
        self.patient_pose_order = ['dead', 'supine', 'fetal', 'agony', 'sittingGround', 'kneeling', 'upset', 'standing', 'recovery', 'calm']
        self.pose_category_order = CategoricalDtype(categories=self.patient_pose_order, ordered=True)
        
        # Delayed is yellow per OSU
        self.salt_to_tag_dict = {'DEAD': 'black', 'EXPECTANT': 'gray', 'IMMEDIATE': 'red', 'DELAYED': 'yellow', 'MINIMAL': 'green'}
        self.sort_to_color_dict = {'still': 'black', 'waver': 'red', 'walker': 'green'}
        
        # Reordered per Ewart so that the display is from left to right as follows: dead, expectant, immediate, delayed, minimal, not tagged
        self.error_table_df = DataFrame([
            {'DEAD': 'Exact', 'EXPECTANT': 'Critical', 'IMMEDIATE': 'Critical', 'DELAYED': 'Critical', 'MINIMAL': 'Critical'},
            {'DEAD': 'Over',  'EXPECTANT': 'Exact',    'IMMEDIATE': 'Critical', 'DELAYED': 'Critical', 'MINIMAL': 'Critical'},
            {'DEAD': 'Over',  'EXPECTANT': 'Over',     'IMMEDIATE': 'Exact',    'DELAYED': 'Over',     'MINIMAL': 'Over'},
            {'DEAD': 'Over',  'EXPECTANT': 'Over',     'IMMEDIATE': 'Under',    'DELAYED': 'Exact',    'MINIMAL': 'Over'},
            {'DEAD': 'Over',  'EXPECTANT': 'Over',     'IMMEDIATE': 'Under',    'DELAYED': 'Under',    'MINIMAL': 'Exact'}
        ], columns=self.patient_salt_order, index=self.tag_color_order[:-1])
        
        # Define the custom categorical orders
        self.error_values = ['Exact', 'Critical', 'Over', 'Under']
        self.errors_category_order = CategoricalDtype(categories=self.error_values, ordered=True)
        
        # Hemorrhage control procedures list
        self.hemorrhage_control_procedures_list = ['tourniquet', 'woundpack']
        
        # Injury required procedure designations
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
        
        # Injury severity designations
        self.injury_severity_columns_list = ['injury_record_severity', 'injury_treated_severity']
        self.injury_severity_order = ['high', 'medium', 'low']
        self.severity_category_order = CategoricalDtype(categories=self.injury_severity_order, ordered=True)
        
        # Injury body region designations
        self.body_region_columns_list = ['injury_record_body_region', 'injury_treated_body_region']
        self.injury_body_region_order = ['head', 'neck', 'chest', 'abdomen', 'leftLeg', 'rightLeg', 'rightHand', 'rightArm', 'leftHand', 'leftArm']
        self.body_region_category_order = CategoricalDtype(categories=self.injury_body_region_order, ordered=True)
        
        # Pulse name designations
        self.pulse_name_order = ['pulse_none', 'pulse_faint', 'pulse_fast', 'pulse_normal']
        self.pulse_name_category_order = CategoricalDtype(categories=self.pulse_name_order, ordered=True)
        
        # Tool type designations
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
        
        # Tool data designations
        self.tool_applied_data_order = [
            'right_chest', 'left_chest', 'right_underarm', 'left_underarm', 'RightForeArm', 'SplintAttachPointRS',
            'SplintAttachPointRL', 'LeftForeArm', 'RightArm', 'SplintAttachPointLS', 'SplintAttachPointLL', 'LeftArm', 'PainMedsAttachPoint'
        ]
        self.tool_applied_data_category_order = CategoricalDtype(categories=self.tool_applied_data_order, ordered=True)
        
        # MCI-VR metrics types dictionary
        self.action_type_to_columns_dict = {
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
        # self.desert_patients_list += [c + ' Root' for c in self.desert_patients_list]
        self.jungle_patients_list = [
            'Open World Marine 1 Male', 'Open World Marine 2 Female', 'Open World Marine 3 Male', 'Open World Marine 4 Male'
        ]
        # self.jungle_patients_list += [c + ' Root' for c in self.jungle_patients_list]
        self.submarine_patients_list = ['Navy Soldier 1 Male', 'Navy Soldier 2 Male', 'Navy Soldier 3 Male', 'Navy Soldier 4 Female']
        # self.submarine_patients_list += [c + ' Root' for c in self.submarine_patients_list]
        self.urban_patients_list = ['Marine 1 Male', 'Marine 2 Male', 'Marine 3 Male', 'Marine 4 Male', 'Civilian 1 Female']
        # self.urban_patients_list += [c + ' Root' for c in self.urban_patients_list]
        self.ow_patients_list = self.desert_patients_list + self.jungle_patients_list + self.submarine_patients_list + self.urban_patients_list
        
        # TA1 patients as of 1:02 PM 5/21/2024
        self.ta1_patients_list = [
            'US Soldier 1', 'Local Soldier 1', 'NPC 1', 'NPC 2', 'NPC 3', 'NPC 4', 'Patient U', 'Patient V', 'Patient W', 'Patient X',
            'Civilian 1', 'Civilian 2', 'NPC', 'patient U', 'patient V', 'patient W', 'patient X', 'electrician', 'bystander',
            'Adept Shooter', 'Adept Victim'
        ]
        
        # Scenario initial teleport locations as of 1:29 PM 5/21/2024
        self.submarine_initial_teleport_location = (0.02, 0, -13.5)
        self.urban_initial_teleport_location = (13.126, 0, 21.61)
        self.jungle_initial_teleport_location = (0.7, 0, 5.45)
        self.desert_initial_teleport_location = (8.131, 0, -28.682)
    
    ### String Functions ###
    
    
    @staticmethod
    def format_timedelta_lambda(timedelta, minimum_unit='seconds'):
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
    
    
    ### File Functions ###
    
    
    @staticmethod
    def get_new_file_name(old_file_name):
        """
        Generate a new file name based on the action tick extracted from the old file.
        
        Parameters:
            old_file_name (str):
                The name of the old log file.

        Returns:
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

        Parameters:
            grouped_df (pandas.DataFrame, optional):
                DataFrame containing the FRVRS logs data.
            mask_series (Series, optional):
                Boolean mask to filter rows of grouped_df, by default None.
            extra_column (str, optional):
                Additional column for further grouping, by default None.
        
        Returns:
            pandas.DataFrameGroupBy
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

        Parameters:
            session_df (pandas.DataFrame):
                DataFrame containing session data for a specific file.
            file_name (str, optional):
                The name of the file to be checked, by default None.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the file contains only one triage run, False otherwise.
        """
        
        # Check if 'is_a_one_triage_file' column exists in the session data frame
        needed_set = set(['scene_type', 'is_scene_aborted', 'file_name'])
        if 'is_a_one_triage_file' in session_df.columns: is_a_one_triage_file = session_df.is_a_one_triage_file.unique().item()
        elif not all(map(lambda x: x in session_df.columns, needed_set)):
            
            # Merge scene_type and is_scene_aborted and file_name into sessions_df
            assert (logs_df is not None), "You don't have the needed set of scene_type, is_scene_aborted, and file_name columns in your session_df and therefore need to include a logs_df"
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
        
        Parameters:
            session_df (pandas.DataFrame):
                DataFrame containing session data.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
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
        
        Parameters:
            session_df_or_file_path (pandas.DataFrame or str):
                DataFrame containing session data or file path.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            str
                The unique logger version associated with the session DataFrame.
        """
        
        # Check if session_df_or_file_path is a string (and therefore a file path)
        if isinstance(session_df_or_file_path, str):
            
            # Assume there are only three versions possible to find in the file content
            with open(session_df_or_file_path, 'r', encoding=nu.encoding_type) as f:
                file_content = f.read()
                if ',1.3,' in file_content: logger_version = 1.3
                elif ',1.4,' in file_content: logger_version = 1.4
                else: logger_version = 1.0
        
        # Otherwise, extract the unique logger version from the session dataframe itself
        else:
            logger_version = session_df_or_file_path.logger_version.unique().item()
        
        # Print verbose output
        if verbose: print('Logger version: {}'.format(logger_version))
        
        # Return the unique logger version
        return logger_version
    
    
    @staticmethod
    def get_is_duplicate_file(session_df, verbose=False):
        """
        Check if a session DataFrame is a duplicate file, i.e., if there is more than one unique file name for the session UUID.
        
        Parameters:
            session_df (pandas.DataFrame):
                DataFrame containing session data for a specific session UUID.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
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
        """
        Get the date of the session file based on the earliest event time.
        
        This static method retrieves the date of the earliest file (based on the `event_time` 
        column) associated with a specific session (`uuid`) from a DataFrame (`logs_df`) 
        containing session logs.
        
        The function first checks if a `session_df` is provided. If not, it filters the 
        `logs_df` to obtain a DataFrame (`session_df`) containing only rows where the 
        `session_uuid` matches the provided `uuid`. Then, it extracts the minimum `event_time` 
        from `session_df` and formats it as a human-readable date string using `'%B %d, %Y'` 
        format specifiers (e.g., "October 26, 2023").
        
        Parameters:
            logs_df (pandas.DataFrame):
                A DataFrame containing session log data, including columns like `session_uuid` and 
                `event_time`.
            uuid (str):
                The unique identifier for the session.
            session_df (pandas.DataFrame, optional):
                A pre-populated DataFrame containing session information (defaults to None). If 
                provided, this DataFrame should include a column named 'session_uuid' to filter by.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            str
                A string representing the date of the first event in the session, formatted as 'Month Day, Year'.
        
        Raises:
            ValueError
                If no session is found for the provided `uuid` in either `logs_df` or `session_df`.
        """
        
        # Check if a pre-populated session dataframe is not provided
        if session_df is None:
            
            # Filter the logs dataframe to get the session dataframe
            mask_series = (logs_df.session_uuid == uuid)
            if not mask_series.any():
                raise ValueError(f"No session found for uuid: {uuid}")
            session_df = logs_df[mask_series]
        
        # Extract and format the date string from the minimum event time
        file_date_str = session_df.event_time.min().strftime('%B %d, %Y')
        
        # Return the formatted date string
        return file_date_str
    
    
    @staticmethod
    def get_last_still_engagement(actual_engagement_order, verbose=False):
        """
        A utility method to retrieve the timestamp of the last engagement of still patients.
        
        Parameters:
            actual_engagement_order (array_like): A 2D array containing engagement information for patients.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            last_still_engagement (float)
                The timestamp of the last engagement of still patients.
        
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
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
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
    
    
    def get_distance_deltas_dataframe(self, logs_df, verbose=False):
        """
        Compute various metrics related to engagement distances and ordering for scenes in the logs_df DataFrame.
        
        Parameters:
            logs_df (pandas DataFrame):
                Dataframe containing logs of engagement scenes.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            pandas.DataFrame
                DataFrame containing computed metrics for each scene.
        
        Notes:
            This function computes metrics such as patient count, engagement order, last still engagement, actual engagement distance,
            measure of right ordering, and adherence to SALT protocol for each scene in the logs DataFrame.
        """
        
        # Ensure all needed columns are present in logs_df
        needed_columns = set(self.scene_groupby_columns + ['patient_id', 'action_tick', 'patient_sort', 'injury_severity', 'action_type', 'location_id'])
        all_columns = set(logs_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from logs_df"
        
        # Loop through each scene in the logs dataframe
        rows_list = []
        for (session_uuid, scene_id), scene_df in logs_df.groupby(self.scene_groupby_columns):
            
            # Create a row dictionary to store scene/session information
            row_dict = {cn: eval(cn) for cn in self.scene_groupby_columns}
            
            # Add the patient_count to the row dictionary
            patient_count = self.get_patient_count(scene_df)
            row_dict['patient_count'] = patient_count
            
            # Get the chronological order of engagement starts for each patient in the scene
            actual_engagement_order = self.get_order_of_actual_engagement(scene_df, include_noninteracteds=True, verbose=False)
            assert len(actual_engagement_order) == patient_count, f"There are {patient_count} patients in this scene and only {len(actual_engagement_order)} engagement tuples:\n{scene_df[~scene_df.patient_id.isnull()].patient_id.unique().tolist()}\n{actual_engagement_order}"
            
            # Initialize the patient counts and loop through the engagement tuples
            unengaged_patient_count = 0; engaged_patient_count = 0
            for engagement_tuple in actual_engagement_order:
                
                # Check if the patient was unengaged
                if engagement_tuple[1] < 0:
                    
                    # If so, create an unengaged_patient column and increase the unengaged_patient_count
                    column_name = f'unengaged_patient{unengaged_patient_count:0>2}_metadata'
                    unengaged_patient_count += 1
                
                else:
                    
                    # Otherwise, create an engaged_patient column and increase the engaged_patient_count
                    column_name = f'engaged_patient{engaged_patient_count:0>2}_metadata'
                    engaged_patient_count += 1
                
                # Add the column to the row dictionary if the tuple wasn't NaN
                column_value = '|'.join([str(x) for x in list(engagement_tuple)])
                if not isna(column_value): row_dict[column_name] = column_value
            
            # Calculate the last still engagement and subtract the scene start and add that to the row dictionary
            last_still_engagement = self.get_last_still_engagement(actual_engagement_order, verbose=verbose)
            row_dict['last_still_engagement'] = last_still_engagement - self.get_scene_start(scene_df)
            
            # Calculate and add the actual engagement distance derived from the order of engagement starts
            row_dict['actual_engagement_distance'] = self.get_actual_engagement_distance(actual_engagement_order, verbose=verbose)
            
            # Calculate and add the measure of right ordering
            row_dict['measure_of_right_ordering'] = self.get_measure_of_right_ordering(scene_df, verbose=verbose)
            
            # Append the row dictionary to the list
            rows_list.append(row_dict)
        
        # Create the distance deltas dataframe from the list of row dictionaries
        distance_delta_df = DataFrame(rows_list)
        
        # Add the adherence to SALT protocol column
        mask_series = (distance_delta_df.measure_of_right_ordering == 1.0)
        distance_delta_df['adherence_to_salt'] = mask_series
        
        return distance_delta_df
    
    
    def get_is_tag_correct_dataframe(self, logs_df, groupby_column='responder_category', verbose=False):
        """
        Create a DataFrame indicating whether tags were applied correctly for patients in each group.
        
        This method iterates through the logs DataFrame, groups data by the specified column,
        and evaluates if the tag applied to each patient matches the predicted tag based on SALT values.
        
        Parameters:
            logs_df (pandas.DataFrame):
                DataFrame containing log data of sessions, scenes, and patients.
            groupby_column (str, optional):
                Column name to group by when evaluating tag correctness. Default is 'responder_category'.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            pandas.DataFrame:
                A DataFrame with information about tag correctness for each patient.
                - session_uuid (str): The session UUID.
                - scene_id (int): The scene ID.
                - patient_id (int): The patient ID.
                - groupby_column (str): The value of the groupby column.
                - patient_count (int): The number of occurrences for this patient (always 1).
                - last_tag (object): The last tag applied to the patient (can be NaN).
                - last_salt (object): The last salt value associated with the patient.
                - predicted_tag (object): The predicted tag based on the last salt value (can be NaN).
                - is_tag_correct (bool): Whether the predicted tag matches the last applied tag.
        
        Notes:
            The resulting DataFrame includes the custom categorical types for the tag, SALT, and predicted tag columns,
            and is sorted based on the custom categorical order of the predicted tag.
        """
        
        # Ensure all needed columns are present in logs_df
        needed_columns = set([groupby_column, 'action_tick', 'tag_applied_type', 'patient_salt'] + self.patient_groupby_columns)
        all_columns = set(logs_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from logs_df"
        
        # Iterate through groups based on the groupby column
        rows_list = []
        for groupby_value, groupby_df in logs_df.groupby(groupby_column):
            
            # Iterate through each session, scene, and patient
            for (session_uuid, scene_id, patient_id), patient_df in groupby_df.sort_values(['action_tick']).groupby(self.patient_groupby_columns):
                
                # Add the groupby columns and an account of the patient's existence to the row dictionary
                row_dict = {'session_uuid': session_uuid, 'scene_id': scene_id, 'patient_id': patient_id}
                row_dict[groupby_column] = groupby_value
                row_dict['patient_count'] = 1
                
                # Get the last tag, salt value, and predicted tag for the patient (handling exceptions)
                try:
                    last_tag = self.get_last_tag(patient_df)
                except Exception:
                    last_tag = nan
                row_dict['last_tag'] = last_tag
                
                # Add the PATIENT_RECORD SALT value for this patient
                last_salt = self.get_last_salt(patient_df)
                row_dict['last_salt'] = last_salt
                
                # Add the predicted tag value for this patient based on the SALT value
                try:
                    predicted_tag = self.salt_to_tag_dict.get(last_salt, nan)
                except Exception:
                    predicted_tag = nan
                row_dict['predicted_tag'] = predicted_tag
                
                # Add a flag indicating if the tagging was correct
                row_dict['is_tag_correct'] = bool(last_tag == predicted_tag)
                rows_list.append(row_dict)
        
        # Create the tag-to-SALT data frame from the list of row dictionaries
        is_tag_correct_df = DataFrame(rows_list)
        
        # Convert the tagged, SALT, and predicted tag columns to their custom categorical types
        is_tag_correct_df.last_tag = is_tag_correct_df.last_tag.astype(self.colors_category_order)
        is_tag_correct_df.last_salt = is_tag_correct_df.last_salt.astype(self.salt_category_order)
        is_tag_correct_df.predicted_tag = is_tag_correct_df.predicted_tag.astype(self.colors_category_order)
        
        # Sort the data frame based on the custom categorical order of the predicted tag
        is_tag_correct_df = is_tag_correct_df.sort_values('predicted_tag')
        
        return is_tag_correct_df
    
    
    def get_percentage_tag_correct_dataframe(self, is_tag_correct_df, groupby_column='responder_category', verbose=False):
        """
        Calculate the percentage of correct tags for each group in each scene.
        
        This function analyzes a DataFrame (`is_tag_correct_df`) containing information about 
        scene tags and responder categories. It calculates the percentage of correctly tagged 
        scenes for each scene and responder category combination.
        
        Parameters:
            is_tag_correct_df (pandas.DataFrame):
                A DataFrame containing scene tag information, including columns for 'session_uuid', 
                'scene_id', 'is_tag_correct' (indicating whether a tag is correct), 'patient_count' 
                (number of patients involved in the scene), and potentially a custom 'groupby_column' 
                used for grouping (defaults to 'responder_category').
            groupby_column (str, optional):
                The column to group the data by within each scene (defaults to 'responder_category').
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            pandas.DataFrame
                A DataFrame containing the following columns:
                - session_uuid: The unique identifier for the session.
                - scene_id: The unique identifier for the scene within the session.
                - <groupby_column>: The value of the column used for grouping (e.g., responder category).
                - percentage_tag_correct: The percentage of tags that were correct for this 
                scene/group combination (as a float).
        
        Raises:
            AssertionError
                If the required columns are not present in `is_tag_correct_df`.
        """
        
        # Ensure all needed columns are present in is_tag_correct_df
        groupby_columns = ['session_uuid', 'scene_id', groupby_column]
        needed_columns = set(groupby_columns + ['is_tag_correct', 'patient_count'])
        all_columns = set(is_tag_correct_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from is_tag_correct_df"
        
        # Initialize a list to store rows for the new DataFrame
        rows_list = []
        
        # Loop through each combination of session, scene, and group
        for (session_uuid, scene_id, groupby_value), groupby_df in is_tag_correct_df.groupby(groupby_columns):
            
            # Initialize dictionary to store row data
            row_dict = {'session_uuid': session_uuid, 'scene_id': scene_id, groupby_column: groupby_value}
            
            # Calculate the percentage of correct tags for each scene for each group
            row_dict['percentage_tag_correct'] = 100 * groupby_df.is_tag_correct.sum() / groupby_df.patient_count.sum()
            
            # Add the row dictionary to the list
            rows_list.append(row_dict)
        
        # Convert the list of dictionaries to a DataFrame
        percentage_tag_correct_df = DataFrame(rows_list)
        
        return percentage_tag_correct_df
    
    
    def get_correct_count_by_tag_dataframe(self, tag_to_salt_df, verbose=False):
        """
        Generate a DataFrame containing the correct count by predicted tag, scene, and session 
        for tagged and not-tagged patients.
        
        This function analyzes a DataFrame (`tag_to_salt_df`) containing patient tagging information
        and generates a new DataFrame that summarizes the number of correctly tagged scenes
        across different categories. The function differentiates between tagged and non-tagged patients
        based on the presence of values in the 'last_tag' and 'max_salt' columns.
        
        Parameters:
            tag_to_salt_df (pandas.DataFrame):
                The DataFrame containing patient tagging data. This DataFrame is expected to have 
                columns including 'session_uuid', 'scene_id', 'predicted_tag', 'is_tag_correct', and 
                'patient_count'.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            pandas.DataFrame
                A DataFrame containing the following columns:
                - session_uuid: The unique identifier for the session.
                - scene_id: The unique identifier for the scene within the session.
                - predicted_tag: The predicted tag for the scene.
                - is_scene_aborted: Whether the scene was aborted (obtained using 
                `get_is_scene_aborted`).
                - scene_type: The type of scene (obtained using `get_scene_type`).
                - correct_count: The total number of patients correctly tagged for this scene with the 
                predicted tag.
                - total_count: The total number of patients involved in the scene.
                - percentage_tag_correct: The percentage of patients correctly tagged for this scene 
                (as a float, potentially containing NaN values).
        """
        
        # Initialize an empty list to hold row dictionaries
        rows_list = []
        
        # Identify rows with missing tags (non-tagged patients who either don't have a SALT designation or a don't log a tagging event)
        tagged_mask_series = tag_to_salt_df.last_tag.isnull() | tag_to_salt_df.max_salt.isnull()
        
        # For the tagged patients, loop through each predicted tag of each scene of each session
        groupby_columns = ['session_uuid', 'scene_id', 'predicted_tag']
        for (session_uuid, scene_id, predicted_tag), df in tag_to_salt_df[~tagged_mask_series].groupby(groupby_columns):
            
            # Create a row dictionary to store scene/session/tag information
            row_dict = {cn: eval(cn) for cn in groupby_columns}
            
            # Add scene abort status and scene type to the row dictionary
            row_dict['is_scene_aborted'] = self.get_is_scene_aborted(df)
            row_dict['scene_type'] = self.get_scene_type(df)
            
            # Calculate and add correct and total counts for this tag/scene
            mask_series = (df.is_tag_correct == True)
            correct_count = df[mask_series].patient_count.sum()
            row_dict['correct_count'] = correct_count
            total_count = df.patient_count.sum()
            row_dict['total_count'] = total_count
            
            # Calculate and add percentage correct (handle potential division by zero)
            try:
                percentage_tag_correct = 100 * correct_count / total_count
            except ZeroDivisionError:
                percentage_tag_correct = nan
            row_dict['percentage_tag_correct'] = percentage_tag_correct
            
            # Append the row dictionary to the list
            rows_list.append(row_dict)
        
        # For the not-tagged patients, just loop through each scene of each session
        for (session_uuid, scene_id), df in tag_to_salt_df[tagged_mask_series].groupby(self.scene_groupby_columns):
            
            # Create a dictionary to store scene/session information
            row_dict = {cn: eval(cn) for cn in self.scene_groupby_columns}
            
            # Set predicted tag to 'Not Tagged'
            row_dict['predicted_tag'] = 'Not Tagged'
            
            # Add scene abort status and scene type to the row dictionary
            row_dict['is_scene_aborted'] = self.get_is_scene_aborted(df)
            row_dict['scene_type'] = self.get_scene_type(df)
            
            # Calculate and add correct (0) and total counts for this run
            mask_series = (df.is_tag_correct == True)
            correct_count = df[mask_series].patient_count.sum()
            row_dict['correct_count'] = correct_count
            total_count = df.patient_count.sum()
            row_dict['total_count'] = total_count
            
            # Calculate and add percentage that tag is correct (also zero)
            try:
                percentage_tag_correct = 100 * correct_count / total_count
            except ZeroDivisionError:
                percentage_tag_correct = nan
            row_dict['percentage_tag_correct'] = percentage_tag_correct
            
            # Append the row dictionary to the list
            rows_list.append(row_dict)
        
        # Create the correct count data frame from the list of row dictionaries
        correct_count_by_tag_df = DataFrame(rows_list)
        
        return correct_count_by_tag_df
    
    
    def get_is_expectant_treated(self, patient_df, verbose=False):
        """
        Determine whether a patient (who was expected to die before help arrives) has been treated by the participant.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing data for a specific patient.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the patient is expected to die before help arrives and was treated by the participant, False otherwise.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = {'patient_salt', 'action_tick', 'injury_treated_required_procedure', 'tool_applied_type'}
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        is_expectant_treated = False
        
        # Get the last recorded SALT designation for the patient
        last_salt = self.get_last_salt(patient_df, verbose=verbose)
        
        # Check if that designation was "EXPECTANT"
        if (last_salt == 'EXPECTANT'):
            
            # Check if there were any injuries treated or tools applied
            mask_series = ~patient_df.injury_treated_required_procedure.isnull() | ~patient_df.tool_applied_type.isnull()
            is_expectant_treated = mask_series.any()
        
        # Return whether or not the patient is expected to die before help arrives and was treated by the participant
        return is_expectant_treated
    
    
    def get_is_treating_expectants(self, scene_df, verbose=False):
        """
        Determine whether or not a participant is treating patients expected to die before help arrives in the scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing data for a specific scene.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the participant is treating patients expected to die before help arrives, False otherwise.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = set(['tool_applied_type', 'action_tick', 'patient_salt', 'injury_treated_required_procedure'])
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Find the expectant among all patients
        for (session_uuid, scene_id, patient_id), patient_df in scene_df.groupby(self.patient_groupby_columns):
            is_expectant_treated = self.get_is_expectant_treated(patient_df, verbose=verbose)
            
            # Return True if the participant is found treating one
            if is_expectant_treated: return True
        
        # Return False assuming the participant is NOT treating patients expected to die before help arrives
        return False
    
    
    def get_patient_stats_dataframe(self, logs_df, verbose=False):
        """
        Generate a DataFrame of patient statistics from a given DataFrame of logs.
        
        This function analyzes a DataFrame (`logs_df`) containing scene log data and generates 
        a new DataFrame (`patient_stats_df`) that summarizes various patient-centric metrics 
        for each scene. The function relies on several helper functions (assumed to be defined 
        elsewhere) to calculate specific metrics such as first/last interactions, tool 
        application correctness, and treatment values.
        
        Parameters:
            logs_df (pandas.DataFrame):
                The DataFrame containing log data for patients.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            pandas.DataFrame
                A DataFrame containing the calculated patient statistics for each scene in the input 
                DataFrame. The DataFrame may also include additional columns depending on the presence 
                of specific columns in the input `logs_df`.
        
        Notes:
            It is assumed that the input `logs_df` contains all the necessary columns for the 
            calculations, including:
            - 'injury_severity'
            - 'scene_id'
            - 'tool_applied_sender'
            - 'session_uuid'
            - 'action_tick'
            - 'patient_record_salt'
            - 'tool_applied_type'
            - 'patient_id'
            - 'patient_engaged_salt'
            - 'injury_record_required_procedure'
            - 'patient_record_sort'
            - 'injury_required_procedure'
            - 'tag_applied_type'
            - 'injury_treated_required_procedure'
            - 'injury_id'
            - 'patient_salt'
            - 'action_type'
            - 'patient_engaged_sort'
            - 'responder_category'
        """
        
        # Ensure all needed columns are present in logs_df
        needed_columns = set([
            'injury_severity', 'scene_id', 'tool_applied_sender', 'session_uuid', 'action_tick', 'patient_record_salt', 'tool_applied_type', 'patient_id',
            'patient_engaged_salt', 'injury_record_required_procedure', 'patient_record_sort', 'injury_required_procedure', 'tag_applied_type', 'injury_treated_required_procedure',
            'injury_id', 'patient_salt', 'action_type', 'patient_engaged_sort', 'responder_category'
        ])
        all_columns = set(logs_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from logs_df"
        
        # Initialize a list to hold the rows of the resulting DataFrame
        rows_list = []
        
        # Group by scene and iterate over each group
        for (session_uuid, scene_id), scene_df in logs_df.groupby(self.scene_groupby_columns):
            
            # Get the start time of the scene
            scene_start = self.get_scene_start(scene_df)
            
            # Iterate over priority group and patient sort
            for cn in ['priority_group', 'patient_sort']:
                if cn in scene_df.columns:
                    
                    # Get actual and ideal sequences
                    actual_sequence, ideal_sequence, sort_dict = eval(f'self.get_actual_and_ideal_{cn}_sequences(scene_df)')
                    unsort_dict = {v1: k for k, v in sort_dict.items() for v1 in v}
                    if verbose:
                        print(cn)
                        display(actual_sequence)
                        display(ideal_sequence)
                        print(sort_dict)
                        print(unsort_dict)
                    
                    # Calculate swaps to perfect order for priority_group and patient_sort
                    swaps_to_perfect_order_count = nu.count_swaps_to_perfect_order(
                        [unsort_dict[i] for i in ideal_sequence],
                        [unsort_dict[a] for a in actual_sequence]
                    )
                    exec(f'swaps_to_perfect_{cn}_order_count = swaps_to_perfect_order_count')
            
            # Group by encounter layout and patient id and iterate over each group
            for (encounter_layout, patient_id), patient_df in scene_df.groupby(['encounter_layout', 'patient_id']):
                row_dict = {'session_uuid': session_uuid, 'scene_id': scene_id, 'patient_id': patient_id, 'encounter_layout': encounter_layout}
                for cn in ['priority_group', 'patient_sort']:
                    if cn in scene_df.columns:
                        row_dict[f'swaps_to_perfect_{cn}_order_count'] = eval(f'swaps_to_perfect_{cn}_order_count')
                
                # Get all the FRVRS utils scalar patient values
                row_dict['first_patient_interaction'] = self.get_first_patient_interaction(patient_df)
                row_dict['first_patient_triage'] = self.get_first_patient_triage(patient_df)
                row_dict['is_correct_bleeding_tool_applied'] = self.get_is_correct_bleeding_tool_applied(patient_df)
                row_dict['is_patient_dead'] = self.get_is_patient_dead(patient_df)
                row_dict['is_patient_gazed_at'] = self.get_is_patient_gazed_at(patient_df)
                row_dict['is_patient_severely_hemorrhaging'] = self.get_is_patient_severely_injured(patient_df)
                row_dict['is_patient_still'] = self.get_is_patient_still(patient_df)
                row_dict['is_tag_correct'] = self.get_is_tag_correct(patient_df)
                row_dict['last_patient_interaction'] = self.get_last_patient_interaction(patient_df)
                row_dict['last_tag'] = self.get_last_tag(patient_df)
                row_dict['last_salt'] = self.get_last_salt(patient_df, session_uuid=session_uuid, scene_id=scene_id, random_patient_id=patient_id)
                row_dict['maximum_injury_severity'] = self.get_maximum_injury_severity(patient_df)
                row_dict['patient_engagement_count'] = self.get_patient_engagement_count(patient_df)
                row_dict['pulse_value'] = self.get_pulse_value(patient_df)
                row_dict['tag_value'] = self.get_tag_value(patient_df)
                row_dict['time_to_hemorrhage_control'] = self.get_time_to_hemorrhage_control(patient_df, scene_start)
                
                # Handle injury treatment value
                mask_series = ~patient_df.injury_id.isnull()
                if mask_series.any():
                    injury_id = patient_df[mask_series].injury_id.iloc[-1]
                    row_dict['treatment_value'] = self.get_treatment_value(patient_df, injury_id)
                
                # Handle tag correct value
                mask_series = ~patient_df.tag_applied_type.isnull()
                tag_applied_type_count = patient_df[mask_series].tag_applied_type.unique().shape[0]
                mask_series = ~patient_df.patient_record_salt.isnull()
                patient_record_salt_count = patient_df[mask_series].patient_record_salt.unique().shape[0]
                if (tag_applied_type_count > 0) and (patient_record_salt_count > 0):
                    row_dict['tag_correct'] = self.get_is_tag_correct(patient_df)
                else:
                    row_dict['tag_correct'] = nan
                
                # Count various types of actions
                mask_series = patient_df.action_type.isin(self.action_types_list)
                row_dict['action_count'] = mask_series.sum()
                
                mask_series = patient_df.action_type.isin(['PATIENT_ENGAGED', 'PULSE_TAKEN'])
                row_dict['assessment_count'] = mask_series.sum()
                
                mask_series = patient_df.action_type.isin(['INJURY_TREATED'])
                row_dict['treatment_count'] = mask_series.sum()
                
                mask_series = patient_df.action_type.isin(['TAG_APPLIED'])
                row_dict['tag_application_count'] = mask_series.sum()
                
                # Determine if the patient was expectant and treated
                is_expectant_treated = self.get_is_expectant_treated(patient_df, verbose=False)
                row_dict['is_expectant_treated'] = is_expectant_treated
                
                # Append the row dictionary to the rows list
                rows_list.append(row_dict)
        
        # Create a DataFrame from the rows list
        patient_stats_df = DataFrame(rows_list)
        patient_stats_df.last_salt = patient_stats_df.last_salt.astype(self.salt_category_order)
        patient_stats_df.last_tag = patient_stats_df.last_tag.astype(self.colors_category_order)
        
        # Return the patient statistics DataFrame
        return patient_stats_df
    
    
    ### Scene Functions ###
    
    
    @staticmethod
    def get_scene_start(scene_df, verbose=False):
        """
        Get the start time of the scene DataFrame run.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing data for a specific scene.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            float
                The start time of the scene in milliseconds.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_tick'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Find the minimum elapsed time to get the scene start time
        run_start = scene_df.action_tick.min()
        
        if verbose: print('Scene start: {}'.format(run_start))
        
        # Return the start time of the scene
        return run_start
    
    
    @staticmethod
    def get_last_engagement(scene_df, verbose=False):
        """
        Get the last time a patient was engaged in the given scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene-specific data.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
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
    def get_initial_location_of_player(scene_df, verbose=False):
        """
        Retrieve the initial location of a player/responder/participant.
        
        This function searches a DataFrame (`scene_df`) containing scene information for the 
        player's initial location. It considers only rows where the action type is 
        'PLAYER_LOCATION' and both `action_tick` and `location_id` columns have non-null 
        values.
        
        The function first filters the DataFrame to only include relevant rows. It then sorts 
        the filtered DataFrame by the action_tick column and retrieves the location from the 
        first row (the one with the earliest action tick). The location is evaluated from a 
        string representation and is directly evaluated into a tuple of three floats.
        
        Parameters:
            scene_df (pandas.DataFrame):
                A DataFrame containing scene information, including columns for 'action_type', 
                'action_tick', and 'location_id'.
            target_tick (int):
                The action tick at which to retrieve the player's location.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            tuple(float, float, float)
                A tuple representing the player's location (x, y, z) at the earliest action tick in 
                the scene, or (0.0, 0.0, 0.0) if no relevant location is found.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_type', 'action_tick', 'location_id'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Filter the dataframe to only include PLAYER_LOCATION rows
        mask_series = (scene_df.action_type == 'PLAYER_LOCATION')
        
        # If available, sort the filtered dataframe chronologically and retrieve the location from the first row
        if mask_series.any(): player_location = eval(scene_df[mask_series].sort_values('action_tick').iloc[0].location_id)
        
        # If not available, use the default location (origin)
        else: player_location = (0.0, 0.0, 0.0)
        
        return player_location
    
    
    @staticmethod
    def get_location_of_player(scene_df, target_tick, verbose=False):
        """
        Retrieve the location of a player in a given scene at a specific tick.
        
        This function searches a DataFrame (`scene_df`) containing scene information for the 
        player's location at a given action tick (`target_tick`). It considers only rows where 
        the action type is 'PLAYER_LOCATION' and both `action_tick` and `location_id` columns 
        have non-null values.
        
        The function first filters the DataFrame to only include relevant rows. It then 
        calculates the absolute difference between each action tick in the filtered DataFrame 
        and the provided `target_tick`. Finally, it sorts the filtered DataFrame by this 
        difference and retrieves the location from the first row (the one with the closest 
        action tick). The location is evaluated from a string representation (assuming it can 
        be directly evaluated into a tuple of three floats).
        
        Parameters:
            scene_df (pandas.DataFrame):
                A DataFrame containing scene information, including columns for 'action_type', 
                'action_tick', and 'location_id'.
            target_tick (int):
                The action tick at which to retrieve the player's location.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            tuple(float, float, float)
                A tuple representing the player's location (x, y, z) at the closest action tick before 
                or at `target_tick`, or (0.0, 0.0, 0.0) if no relevant location is found.
        """
        
        # Initialize player location to default (origin)
        player_location = (0.0, 0.0, 0.0)
        
        # Filter for relevant rows (PLAYER_LOCATION with valid action_tick and location_id)
        mask_series = (scene_df.action_type == 'PLAYER_LOCATION') & ~scene_df.action_tick.isnull() & ~scene_df.location_id.isnull()
        
        # Check if any 'PLAYER_LOCATION' actions exist
        if mask_series.any():
            
            # Apply the mask to the dataframe
            df = scene_df[mask_series]
            
            # Calculate the absolute time difference between action ticks of each row and the target tick
            df['action_delta'] = df.action_tick.map(lambda x: abs(target_tick - x))
            
            # Sort the DataFrame by action_delta (smallest delta = closest action tick) to find the closest action_tick
            closest_action_row = df.sort_values('action_delta').iloc[0]
            
            # Evaluate the location_id string to convert it to a tuple
            player_location = eval(closest_action_row.location_id)
        
        return player_location
    
    
    @staticmethod
    def get_scene_type(scene_df, verbose=False):
        """
        Get the type of a scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing data for a specific scene.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
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
        
        Parameters:
            scene_df (pandas.DataFrame):
                A Pandas DataFrame containing the scene data.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
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
        Calculate the number of unique patient IDs in a scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including 'patient_id' column.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                Number of unique patients in the scene DataFrame.
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
        Calculate the number of records where injury treatment attempts were logged for a given scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The number of records where injury treatment attempts were logged.
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
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The number of patients who have not received injury treatment.
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
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                Count of records where injuries were correctly treated in the scene DataFrame.
        
        Note:
            The FRVRS logger has trouble logging multiple tool applications,
            so injury_treated_injury_treated_with_wrong_treatment == True
            remains and another INJURY_TREATED is never logged, even though
            the right tool was applied after that.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = set([
            'patient_id', 'injury_id', 'injury_record_required_procedure', 'injury_treated_required_procedure', 
            'injury_treated_injury_treated', 'injury_treated_injury_treated_with_wrong_treatment', 'tool_applied_type'
        ])
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
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
        Calculate the number of patients whose injuries have been incorrectly treated in a scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                The DataFrame containing scene data.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The number of patients whose injuries have been incorrectly treated.
        
        Note:
            The FRVRS logger has trouble logging multiple tool applications,
            so injury_treated_injury_treated_with_wrong_treatment == True
            remains and another INJURY_TREATED is never logged, even though
            the right tool was applied after that.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = set(['injury_severity'])
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
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
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The number of PULSE_TAKEN actions in the scene DataFrame.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_type'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
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
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The number of TELEPORT actions in the scene DataFrame.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_type'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
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
        Calculate the number of VOICE_CAPTURE actions in a scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The number of VOICE_CAPTURE actions in the scene DataFrame.
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
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The number of walk to the safe area voice commands in the scene DataFrame.
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
        Calculate the number of wave if you can voice commands in a scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The number of wave if you can voice commands in the scene DataFrame.
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
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                Action tick of the first 'PATIENT_ENGAGED' action in the scene DataFrame.
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
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                Action tick of the first 'INJURY_TREATED' action in the scene DataFrame.
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
    
    
    def get_order_of_ideal_engagement(self, scene_df, tuples_list=None, verbose=False):
        
        # Create the patient sort info
        engagement_starts_df = DataFrame(self.get_order_of_actual_engagement(scene_df), columns=[
            'patient_id', 'engagement_start', 'location_tuple', 'patient_sort', 'predicted_priority', 'injury_severity'
        ])
        engagement_starts_df.patient_sort = engagement_starts_df.patient_sort.astype(self.sort_category_order)
        engagement_starts_df.injury_severity = engagement_starts_df.injury_severity.astype(self.severity_category_order)
        
        # Get initial player location from the first PLAYER_LOCATION action (or default to origin)
        player_location = self.get_initial_location_of_player(scene_df, verbose=verbose)
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
        Determine whether or not a scene is aborted.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing data for a specific scene.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the scene is aborted, False otherwise.
        """
        
        # Check if the is_scene_aborted column exists in the scene data frame, and get the unique value if it is
        if ('is_scene_aborted' in scene_df.columns) and (scene_df.is_scene_aborted.unique().shape[0] > 0):
            is_scene_aborted = scene_df.is_scene_aborted.unique().item()
        else: is_scene_aborted = False
        if not is_scene_aborted:
            
            # Otherwise, calculate the time duration between scene start and last engagement
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
        Calculate the triage time for a scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                Triage time in milliseconds.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_tick'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Get the scene start and end times
        time_start = self.get_scene_start(scene_df, verbose=verbose)
        time_stop = self.get_scene_end(scene_df, verbose=verbose)
        
        # Calculate the triage time
        triage_time = time_stop - time_start
        
        return triage_time
    
    
    def get_dead_patients(self, scene_df, verbose=False):
        """
        Get a list of unique patient IDs marked as DEAD or EXPECTANT in a scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            list
                List of unique patient IDs marked as DEAD or EXPECTANT in the scene DataFrame.
        """
        
        # Filter the treat-as-dead columns and combine them into a mask series
        mask_series = False
        for column_name in self.patient_salt_columns_list: mask_series |= scene_df[column_name].isin(['DEAD', 'EXPECTANT'])
        
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
        Get a list of unique patient IDs marked as 'still' in a scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            list
                List of unique patient IDs marked as 'still' in the scene DataFrame.
        """
        
        # Filter the '_sort' columns and combine them into a mask series
        mask_series = False
        for column_name in self.patient_sort_columns_list: mask_series |= (scene_df[column_name] == 'still')
        
        # Extract the list of still patients from the filtered mask series
        still_list = scene_df[mask_series].patient_id.unique().tolist()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'List of patients marked as still: {still_list}')
            display(scene_df)
        
        # Return the list of unique patient IDs marked as 'still'
        return still_list
    
    
    def get_total_actions_count(self, scene_df, verbose=False):
        """
        Calculate the total number of user actions within a given scene DataFrame,
        including voice commands with specific messages.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                Total number of user actions in the scene DataFrame.
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
        Extract the actual and ideal patient SORT sequences of first interactions from a scene dataframe.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing patient interactions with columns, including 'patient_sort' and 'patient_id'.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            tuple
                A tuple of three elements:
                - actual_sequence (pandas.Series): The actual sequence of first interactions, sorted.
                - ideal_sequence (pandas.Series): Series of ideal patient interactions based on SORT categories.
                - sort_dict (dict): Dictionary containing lists of first interactions for each SORT category.
        
        Note:
            Only SORT categories included in `self.patient_sort_order` are considered.
            None values in the resulting lists indicate missing interactions.
            When the walkers walk too close to the responder they can trigger the PATIENT_ENGAGED action
            which incorrectly assigns them as a patient that has been seen. 
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = set([
            'patient_sort', 'patient_id', 'action_type', 'action_tick'
        ])
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
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
            actual_sequence (array-like):
                The observed sequence of actions or events.
            ideal_sequence (array-like):
                The ideal or expected sequence of actions or events.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            float:
                The R-squared adjusted value as a measure of ordering. Returns NaN if model fitting 
                fails.
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
        Calculate a measure of right ordering for patients based on their SORT category and elapsed times.
        The measure of right ordering is an R-squared adjusted value, where a higher value indicates better right ordering.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            float
                The measure of right ordering for patients.
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
        Calculate the percentage of hemorrhage-related injuries that have been controlled in a scene.
        
        The percentage is based on the count of 'hemorrhage_control_procedures_list' procedures
        recorded in the 'injury_record_required_procedure' column compared to the count of those
        procedures being treated and marked as controlled in the 'injury_treated_required_procedure' column.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            float
                Percentage of hemorrhage cases successfully controlled.
        
        Note:
            The logs have instances of a TOOL_APPLIED but no INJURY_TREATED preceding it. But, we already know the injury
            that the patient has and the correct tool for every patient because we assigned those ahead of time.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = set([
            'injury_treated_injury_treated_with_wrong_treatment', 'injury_required_procedure', 'patient_record_salt', 'injury_treated_injury_treated',
            'action_type', 'injury_treated_required_procedure', 'injury_record_required_procedure', 'patient_id', 'patient_engaged_salt'
        ])
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Initialize hemorrhage and controlled counts and loop through each patient in the scene
        hemorrhage_count = 0; controlled_count = 0
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            
            # Check if the patient is not dead
            is_patient_dead = self.get_is_patient_dead(patient_df, verbose=verbose)
            if not is_patient_dead:
                
                # Loop through each injury, examining its required procedures and wrong treatments
                for injury_id, injury_df in patient_df.groupby('injury_id'):
                    
                    # Check if an injury record or treatment exists for a hemorrhage-related procedure
                    is_injury_hemorrhage = self.get_is_injury_hemorrhage(injury_df, verbose=verbose)
                    if is_injury_hemorrhage:
                        
                        # Count as hemorrhage any injuries requiring hemorrhage control procedures
                        hemorrhage_count += 1
                        
                        # Check if the injury was treated correctly
                        is_correctly_treated = self.get_is_injury_correctly_treated(injury_df, patient_df, verbose=verbose)
                        
                        # Check if there are any tools applied that are associated with the hemorrhage injuries
                        is_tool_applied_correctly = self.get_is_hemorrhage_tool_applied(injury_df, patient_df, verbose=verbose)
                        
                        # Count as controlled any hemorrhage-related injuries that have been treated, and not wrong, and not counted twice
                        if is_correctly_treated or is_tool_applied_correctly:
                            controlled_count += 1
        
        if verbose:
            print(f'Injuries requiring hemorrhage control procedures: {hemorrhage_count}')
            print(f"Hemorrhage-related injuries that have been treated: {controlled_count}")
        
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
        Calculate the time to the last hemorrhage-controlled event for patients in a scene.
        
        The time is determined based on the 'hemorrhage_control_procedures_list' procedures being treated
        and marked as controlled. The function iterates through patients in the scene to find the maximum
        time to hemorrhage control.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The time to the last hemorrhage control action, or 0 if no hemorrhage control actions exist.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = set([
            'injury_required_procedure', 'tool_applied_type', 'action_tick', 'patient_record_salt', 'injury_id', 'patient_engaged_salt',
            'injury_record_required_procedure', 'injury_treated_required_procedure'
        ])
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"

        
        # Get the start time of the scene (defined as the minimum action tick in the logs delimited by SESSION_ENDs)
        scene_start = self.get_scene_start(scene_df)
        
        # Initialize the last controlled time to 0
        last_controlled_time = 0
        
        # Iterate through patients in the scene
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            
            # Check if the patient is hemorrhaging (defined as the injury record requires the hemorrhage control procedures) and not dead
            if self.get_is_patient_hemorrhaging(patient_df, verbose=verbose) and not self.get_is_patient_dead(patient_df, verbose=verbose):
                
                # Get the time to hemorrhage control for the patient (defined as the maximum action tick of the controlled hemorrhage events)
                controlled_time = self.get_time_to_hemorrhage_control(patient_df, scene_start=scene_start, use_dead_alternative=False, verbose=verbose)
                
                # Update the last controlled time if the current controlled time is greater
                last_controlled_time = max(controlled_time, last_controlled_time)
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Time to last hemorrhage controlled: {last_controlled_time} milliseconds')
            display(scene_df)
        
        # Return the time to the last hemorrhage-controlled event
        return last_controlled_time
    
    
    def get_time_to_hemorrhage_control_per_patient(self, scene_df, verbose=False):
        """
        Calculate the time to hemorrhage control per patient for the scene.
        
        According to our research papers we define time to hemorrhage control per patient like this:
        Duration of time from when the patient was first approached by the participant until
        the time hemorrhage treatment was applied (with a tourniquet or wound packing).
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The time it takes to control hemorrhage for the scene, per patient, in action ticks.
        
        Note:
            If you trim off the action ticks at the beginning of the scene so that the
            first action tick ends up as the responder engaging the only hemorrhaging,
            non-dead patient, you will zero out the time_to_hemorrhage_control_per_patient
            for the scenes.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = set([
            'patient_id', 'patient_record_salt', 'patient_engaged_salt', 'action_tick', 'injury_id', 'injury_record_required_procedure',
            'injury_treated_required_procedure', 'tool_applied_type', 'action_type'
        ])
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Iterate through patients in the scene
        times_list = []
        patient_count = 0
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            
            # Check if the patient is hemorrhaging (defined as the injury record requires the hemorrhage control procedures) and not dead
            if self.get_is_patient_hemorrhaging(patient_df, verbose=verbose) and not self.get_is_patient_dead(patient_df, verbose=verbose):
                
                # Count the patient and add its hemorrhage control time to a list for averaging
                action_tick = self.get_time_to_hemorrhage_control(patient_df, scene_start=self.get_first_patient_interaction(patient_df), use_dead_alternative=False)
                times_list.append(action_tick)
                patient_count += 1
        
        # Calculate the hemorrhage control per patient by summing the control times and dividing by the patient count
        try: time_to_hemorrhage_control_per_patient = sum(times_list) / patient_count
        except ZeroDivisionError: time_to_hemorrhage_control_per_patient = nan
        
        return time_to_hemorrhage_control_per_patient
    
    
    def get_triage_priority_dataframe(self, scene_df, verbose=False):
        """
        Generate a DataFrame that prioritizes patients based on injury severity and sort order for a scene.
        
        This function filters and sorts the input DataFrame based on specified columns to 
        generate a triage priority DataFrame. It ensures that all necessary columns are 
        present in the input DataFrame and sorts the data by 'injury_severity' and 
        'patient_sort'.
        
        Parameters:
            scene_df (pandas.DataFrame):
                A DataFrame containing scene data. This DataFrame is expected to have columns 
                including those specified in `input_features` and potentially others. See the source 
                code for a list of expected columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            pandas.DataFrame
                A DataFrame containing triage priority data for the scene, including the specified 
                scene groupby columns and the defined input features. The DataFrame is sorted by 
                injury severity (ascending) and patient sort order (ascending).
        
        Raises:
            AssertionError
                If the input DataFrame is missing required columns.
        """
        
        # Define the input features needed for triage priority
        input_features = [
            'injury_id', 'injury_severity', 'injury_required_procedure', 'patient_salt', 
            'patient_sort', 'patient_pulse', 'patient_breath', 'patient_hearing', 'patient_mood', 
            'patient_pose'
        ]
        
        # Combine scene groupby columns and features into a list of required columns
        columns_list = self.scene_groupby_columns + input_features
        
        # Ensure all required columns are present in scene_df
        needed_columns = set(columns_list)
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Filter and sort the dataframe by injury severity and patient sort
        triage_priority_df = scene_df[columns_list].sort_values(['injury_severity', 'patient_sort'], ascending=[True, True])
        
        # Return the sorted DataFrame
        return triage_priority_df
    
    
    def get_order_of_actual_engagement(self, scene_df, include_noninteracteds=False, verbose=False):
        """
        Get the chronological order of engagement starts for each patient in a scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including patient IDs, action types, action ticks, 
                location IDs, patient sorts, and DTR triage priority model predictions.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            engagement_order (list):
                List of tuples containing engagement information ordered chronologically:
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
    
    
    def get_order_of_distracted_engagement(self, scene_df, tuples_list=None, verbose=False):
        """
        Calculate the theorectical order of patient engagement based on the participant's 
        proximity to the player.
        
        This function determines the order in which a player engages with different 
        locations in a scene, starting from the player's initial location and moving 
        to the nearest neighbor iteratively.
        
        Parameters:
            scene_df (pandas.DataFrame):
                The scene DataFrame containing the action type and location id.
            tuples_list (list of tuples, optional):
                A list of tuples representing engagement orders. Defaults to None, in which 
                case the function calls `get_order_of_actual_engagement` to generate this list.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            list
                A list of tuples representing the order of distracted engagement.
        """
        
        # Create the patient sort tuples list (use actual engagement order if not provided)
        if tuples_list is None:
            tuples_list = self.get_order_of_actual_engagement(scene_df, verbose=verbose)
        
        # Get initial player location from the first PLAYER_LOCATION action (or default to origin)
        player_location = self.get_initial_location_of_player(scene_df, verbose=verbose)
        
        # Use x and z coordinates for 2D proximity
        player_location = (player_location[0], player_location[2])
        
        # Initialize the list for the order of distracted engagement
        distracted_engagement_order = []
        
        # Extract a list of patient locations from the tuples (using third element as location)
        locations_list = [x[2] for x in tuples_list]
        
        # Iteratively find nearest neighbor and update order until no locations remain
        while locations_list:
            
            # Find the nearest neighbor patient location based on the current player location
            nearest_neighbor = nu.get_nearest_neighbor(player_location, locations_list)
            
            # Remove the nearest neighbor from the locations list
            nearest_neighbor = locations_list.pop(locations_list.index(nearest_neighbor))
            
            # Find the corresponding patient tuple and add it to the engagement order
            for patient_sort_tuple in tuples_list:
                if patient_sort_tuple[2] == nearest_neighbor:
                    distracted_engagement_order.append(patient_sort_tuple)
                    break
            
            # Update player location to the current nearest neighbor for next iteration
            player_location = nearest_neighbor
        
        # If verbose is True, print the distracted engagement order
        if verbose:
            print(f'\n\ndistracted_engagement_order: {distracted_engagement_order}')
        
        # Return the order of distracted engagement
        return distracted_engagement_order
    
    
    def get_actual_and_ideal_priority_group_sequences(self, scene_df, verbose=False):
        """
        Extract the actual and ideal priority group sequences of first interactions from a scene dataframe.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing patient interactions with columns, including 'priority_group' and 'patient_id'.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            tuple
                A tuple of three elements:
                - actual_sequence (pandas.Series): The actual sequence of first interactions, sorted.
                - ideal_sequence (pandas.Series): Series of ideal patient interactions based on SORT categories.
                - sort_dict (dict): Dictionary containing lists of first interactions for each SORT category.
        
        Notes:
            Only SORT categories included in `priority_group_order` are considered.
            None values in the resulting lists indicate missing interactions.
        """
        priority_group_order = [1, 2, 3]
        
        # Group patients by their priority group category and get lists of their elapsed times
        sort_dict = {}
        for sort, priority_group_df in scene_df.groupby('priority_group'):
            
            # Only consider SORT categories included in the priority_group_order
            if sort in priority_group_order:
                
                # Loop through the SORT patients to add their first interactions to the action list
                action_list = []
                for patient_id in priority_group_df.patient_id.unique():
                    mask_series = (scene_df.patient_id == patient_id)
                    patient_actions_df = scene_df[mask_series]
                    action_list.append(self.get_first_patient_triage(patient_actions_df))
                
                # Sort the list of first interactions
                if verbose: display(sort, action_list)
                sort_dict[sort] = sorted([action for action in action_list if action is not nan])
        
        # Get the whole ideal and actual sequences
        ideal_sequence = []
        for sort in priority_group_order: ideal_sequence.extend(sort_dict.get(sort, []))
        ideal_sequence = Series(data=ideal_sequence)
        actual_sequence = ideal_sequence.sort_values(ascending=True)
        
        return actual_sequence, ideal_sequence, sort_dict
    
    
    @staticmethod
    def get_tool_indecision_time(scene_df, verbose=False):
        """
        Calculate the time (between first-in-sequence TOOL_HOVER and last-in-sequence 
        TOOL_SELECTED) that responders take to select a tool after hovering over them.
        
        This method processes a DataFrame to find periods of indecision when hovering over tools 
        before selecting one. It calculates the time difference between the first tool hover and 
        the subsequent tool selection, and returns the mean of these times.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including action types and action ticks.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            float:
                The mean indecision time users take to select a tool after hovering over them (in the same unit as action_tick).
        
        Raises:
            AssertionError
                If the required columns ('action_type', 'action_tick') are not present in scene_df.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_type', 'action_tick'}
        all_columns = set(scene_df.columns)
        if verbose:
            print(f'all_columns = "{all_columns}"')
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Identify scene_df indices based on TOOL_SELECTED actions
        scene_df = scene_df.reset_index(drop=True)
        mask_series = (scene_df.action_type == 'TOOL_SELECTED')
        tool_selected_indices = scene_df[mask_series].index
        
        # Split the DataFrame at TOOL_SELECTED indices
        split_dfs = nu.split_df_by_iloc(scene_df, tool_selected_indices)
        
        # Loop over each split dataframe
        indecision_times = []
        for split_df in split_dfs:
            
            # Check for any TOOL_HOVER actions in the split dataframe
            mask_series = (split_df.action_type == 'TOOL_HOVER')
            if mask_series.any():
                tool_hover_df = split_df[mask_series]
                
                # Append time difference between TOOL_SELECTED action and first TOOL_HOVER action to the list
                time_difference = split_df.action_tick.max() - tool_hover_df.action_tick.min()
                indecision_times.append(time_difference)
        
        # Calculate the mean indecision time
        mean_tool_indecision_time = np.mean(indecision_times)
        
        return mean_tool_indecision_time
    
    
    def get_action_count(self, scene_df, verbose=False):
        """
        Calculate the total count of the participant's player actions of strictly the PULSE_TAKEN, 
        TOOL_APPLIED, and TAG_APPLIED action types performed within a scene DataFrame.
        
        This method counts the occurrences of specific actions in the responder negotiations list
        within the provided `scene_df`. It achieves this by filtering the DataFrame for rows where the 'action_type' column
        matches one of the listed actions. The count of these filtered rows represents the total scene action count.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including action types.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int:
                The total count of actions (PULSE_TAKEN, TOOL_APPLIED, TAG_APPLIED) in the scene.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_type'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Create a mask to filter for specific action types
        mask_series = scene_df.action_type.isin(self.responder_negotiations_list)
        
        # Calculate scene action count based on the mask
        scene_action_count = scene_df[mask_series].shape[0]
        
        return scene_action_count
    
    
    @staticmethod
    def get_discarded_count(scene_df, verbose=False):
        """
        Calculate the total count of the participant's "indecision metric" of TAG_DISCARDED 
        and TOOL_DISCARDED action types performed within a scene DataFrame.
        
        This static method counts the occurrences of TAG_DISCARDED and TOOL_DISCARDED action 
        types within the provided `scene_df`. It achieves this by filtering the DataFrame 
        for rows where the 'action_type' column matches one of the listed discardeds. The 
        count of these filtered rows represents the total scene discarded count.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including discarded types.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int:
                The total count of discardeds (TAG_DISCARDED, TOOL_DISCARDED) in the scene.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_type'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Create a mask to filter for specific discarded types
        mask_series = scene_df.action_type.isin(['TAG_DISCARDED', 'TOOL_DISCARDED'])
        
        # Calculate discarded count based on the mask
        discarded_count = scene_df[mask_series].shape[0]
        
        return discarded_count
    
    
    @staticmethod
    def get_patient_injuries_count(scene_df, verbose=False):
        """
        Calculate the total count of all the injuries of all the patients within a scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including patient and injury IDs.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int:
                The total count of patient_injuriess (TAG_DISCARDED, TOOL_DISCARDED) in the scene.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'patient_id', 'injury_id'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Get the count of all the patient injuries
        patient_injuries_count = 0
        for patient_id, patient_df in scene_df.groupby('patient_id'):
            patient_injuries_count += patient_df.injury_id.nunique()
        
        return patient_injuries_count
    
    
    @staticmethod
    def get_assessment_count(scene_df, verbose=False):
        """
        Calculate the total count of the participant's PATIENT_ENGAGED 
        and PULSE_TAKEN action types performed within a scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including action types.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int:
                The total count of assessments (PATIENT_ENGAGED, PULSE_TAKEN) in the scene.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_type'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Create a mask to filter for specific action types
        mask_series = scene_df.action_type.isin(['PATIENT_ENGAGED', 'PULSE_TAKEN'])
        
        # Calculate assessment count based on the mask
        assessment_count = scene_df[mask_series].shape[0]
        
        return assessment_count
    
    
    @staticmethod
    def get_treatment_count(scene_df, verbose=False):
        """
        Calculate the total count of the participant's INJURY_TREATED action types performed within a scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including action types.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int:
                The total count of treatments (INJURY_TREATED) in the scene.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_type'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Create a mask to filter for specific action types
        mask_series = scene_df.action_type.isin(['INJURY_TREATED'])
        
        # Calculate treatment count based on the mask
        treatment_count = scene_df[mask_series].shape[0]
        
        return treatment_count
    
    
    @staticmethod
    def get_tag_application_count(scene_df, verbose=False):
        """
        Calculate the total count of the participant's TAG_APPLIED action types performed within a scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including action types.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int:
                The total count of tag_applications (TAG_APPLIED) in the scene.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'action_type'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Create a mask to filter for specific action types
        mask_series = scene_df.action_type.isin(['TAG_APPLIED'])
        
        # Calculate tag_application count based on the mask
        tag_application_count = scene_df[mask_series].shape[0]
        
        return tag_application_count
    
    
    def get_percent_accurate_tagging(self, scene_df, verbose=False):
        """
        Calculate the percentage of correct tagging within a scene DataFrame.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including patient SALT.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int:
                The percentage of correct taggings in the scene.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = set(['participant_id', 'action_tick', 'tag_applied_type', 'patient_salt'] + self.patient_groupby_columns)
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Create the tag-to-SALT data frame using the scene_df
        tag_to_salt_df = self.get_is_tag_correct_dataframe(scene_df, groupby_column='participant_id')
        
        # Create the correct-count-by-tag data frame using the tag-to-SALT dataframe
        correct_count_by_tag_df = self.get_percentage_tag_correct_dataframe(tag_to_salt_df, groupby_column='participant_id')
        
        # Get the mean of the percentage_tag_correct column as the percentage percent accurate tagging for the scene
        percent_accurate_tagging = correct_count_by_tag_df.percentage_tag_correct.mean()
        
        return percent_accurate_tagging
    
    
    def get_percent_injury_correctly_treated(self, scene_df, verbose=False):
        """
        Calculate the percentage of correct injury treatment within a scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including patient SALT.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int:
                The percentage of correctly treated injuries in the scene.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = set([
            'patient_id', 'injury_id', 'injury_record_required_procedure', 'injury_treated_required_procedure', 
            'injury_treated_injury_treated', 'injury_treated_injury_treated_with_wrong_treatment', 'tool_applied_type'
        ])
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Get the count of all the patient injuries for the scene
        all_patient_injuries_count = self.get_patient_injuries_count(scene_df)
        
        # Get the count of all correctly treated injuries for the scene
        correctly_treated_count = self.get_injury_correctly_treated_count(scene_df)
        
        # Compute the percentage of correctly treated injuries for the scene
        try: percent_injury_correctly_treated = 100 * correctly_treated_count / all_patient_injuries_count
        
        # If you get a division-by-zero error, just leave it as NaN
        except ZeroDivisionError: percent_injury_correctly_treated = nan
        
        return percent_injury_correctly_treated
    
    
    def get_treated_expectant_count(self, scene_df, verbose=False):
        """
        Calculate the total number of instances where a participant treated a patient expected to die within a scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data, including patient SALT.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int:
                The total count of tag_applications (TAG_APPLIED) in the scene.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'patient_salt', 'action_tick', 'injury_treated_required_procedure', 'tool_applied_type'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
        # Loop through each patient to count the instances where a participant treated a patient expected to die
        treated_expectant_count = 0
        for (session_uuid, scene_id, patient_id), patient_df in scene_df.groupby(self.patient_groupby_columns):
            if self.get_is_expectant_treated(patient_df):
                treated_expectant_count += 1
        
        return treated_expectant_count
    
    
    ### Patient Functions ###
    
    
    @staticmethod
    def get_is_correct_bleeding_tool_applied(patient_df, verbose=False):
        """
        Determine whether the correct bleeding control tool (tourniquet or packing gauze) has been applied to a patient in a scene.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the correct bleeding control tool has been applied, False otherwise.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = {'tool_applied_sender'}
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
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
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool or numpy.nan
                True if the patient is considered dead, False if not, and numpy.nan if unknown.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = {'patient_record_salt', 'patient_engaged_salt'}
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
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
    
    
    @staticmethod
    def get_is_patient_still(patient_df, verbose=False):
        """
        Determine whether a patient is considered still based on the presence of 'still' in their patient_record_sort or patient_engaged_sort fields.
        
        The function examines the 'patient_record_sort' and 'patient_engaged_sort' columns to
        determine if the patient is categorized as 'still'. If both columns are empty,
        the result is considered unknown (NaN).
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool or numpy.nan
                True if the patient is marked as 'still', False if not, and numpy.nan if unknown.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = {'patient_record_sort', 'patient_engaged_sort'}
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Handle missing values in both patient_record_sort and patient_engaged_sort
        if patient_df.patient_record_sort.isnull().all() and patient_df.patient_engaged_sort.isnull().all(): is_patient_still = nan
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
                else: is_patient_still = nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Is patient considered still: {is_patient_still}')
            display(patient_df)
        
        # Return True if the patient is marked as 'still', False if not, and numpy.nan if unknown
        return is_patient_still
    
    
    @staticmethod
    def get_last_salt(patient_df, verbose=False):
        """
        Get the most recent SALT value from the patient DataFrame.
        
        Parameters:
            patient_df (pandas.DataFrame, optional):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The latest salt value for the patient.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['action_tick', 'patient_salt'])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Attempt to assign a legitimate value to last_salt
        try:
            
            # Filter out missing patient SALT values
            mask_series = ~patient_df.patient_salt.isnull()
            
            # Select the patient SALT with the highest action tick
            last_salt = patient_df[mask_series].sort_values('action_tick').patient_salt.iloc[-1]
        
        # If unsuccessful, assign NaN to the last_salt
        except Exception: last_salt = nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'last_salt={last_salt}')
            display(patient_df)
        
        # Return the last salt value
        return last_salt
    
    
    @staticmethod
    def get_max_salt(patient_df, verbose=False):
        """
        Get the highest order SALT value from the patient DataFrame ('DEAD' > 'EXPECTANT' > 'IMMEDIATE' > 'DELAYED' > 'MINIMAL').
        
        Parameters:
            patient_df (pandas.DataFrame, optional):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int or tuple
                The maximum salt value for the patient.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['action_tick', 'patient_salt'])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
            
        # Filter out missing patient SALT values
        mask_series = ~patient_df.patient_salt.isnull()
        
        # Create a series of all patient SALTs categorized with the salt_category_order
        patient_salts_srs = Series(patient_df[mask_series].patient_salt).astype(self.salt_category_order)
        
        # Get the minimum (by salt_category_order) value of that series and assign it to max_salt
        max_salt = patient_salts_srs.min()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'max_salt={max_salt}')
            display(patient_df)
        
        # Return the max salt value
        return max_salt
    
    
    @staticmethod
    def get_last_tag(patient_df, verbose=False):
        """
        Retrieve the last tag applied to a patient in a scene.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            str or numpy.nan
                The last tag applied to the patient, or numpy.nan if no tags have been applied.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = {'tag_applied_type', 'action_tick'}
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Get the last tag value
        mask_series = ~patient_df.tag_applied_type.isnull()
        try: last_tag = patient_df[mask_series].sort_values('action_tick').tag_applied_type.iloc[-1]
        except Exception: last_tag = nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Last tag applied: {last_tag}')
            display(patient_df)
        
        # Return the last tag value or numpy.nan if no data is available
        return last_tag
    
    
    @staticmethod
    def get_location_of_patient(patient_df, target_tick, verbose=False):
        """
        Get the patient location closest to the time of the target action tick.
        
        This function searches for the patient's location in `patient_df` that is closest (in 
        time) to the provided `target_tick`.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns, including:
                - location_id: The ID of the patient's location (expected to be a string that can be 
                evaluated to a tuple).
                - action_tick (int): The timestamp of an action associated with the patient.
            target_tick (int):
                The time in milliseconds to locate the patient during.
        
        Returns:
            tuple (float, float, float)
                The coordinates of the patient's location closest to the `target_tick`. The function 
                returns (0.0, 0.0, 0.0) if no valid location data is found.
        """
        
        # Initialize patient location to default (origin)
        patient_location = (0.0, 0.0, 0.0)
        
        # Check if the patient has location data
        mask_series = ~patient_df.location_id.isnull()
        if mask_series.any():
            
            # Filter for rows with location data
            df = patient_df[mask_series]
            
            # Calculate the absolute time difference between each action_tick and the target_tick
            df['action_delta'] = df.action_tick.map(lambda x: abs(target_tick - x))
            
            # Sort by the calculated time difference (ascending)
            df_sorted = df.sort_values('action_delta')
            
            # Get the location of the patient with the closest action_tick (assuming first row)
            patient_location = eval(df_sorted.iloc[0].location_id)
        
        # Return the patient's location coordinates
        return patient_location
    
    
    def get_is_tag_correct(self, patient_df, verbose=False):
        """
        Determines whether the last tag applied to a patient in a given scene DataFrame matches the predicted tag based on the patient's record SALT.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool or numpy.nan
                Returns True if the tag is correct, False if incorrect, or numpy.nan if data is insufficient.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = {'tag_applied_type', 'patient_record_salt', 'action_tick', 'patient_salt'}
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Ensure both 'tag_applied_type' and 'patient_record_salt' each have at least one non-null value
        mask_series = ~patient_df.tag_applied_type.isnull()
        tag_applied_type_count = patient_df[mask_series].tag_applied_type.unique().shape[0]
        mask_series = ~patient_df.patient_record_salt.isnull()
        patient_record_salt_count = patient_df[mask_series].patient_record_salt.unique().shape[0]
        if (tag_applied_type_count == 0) or (patient_record_salt_count == 0): return nan
        
        # Get the last applied tag
        last_tag = self.get_last_tag(patient_df)
        
        # Get the maximum SALT value for the patient
        max_salt = self.get_max_salt(patient_df)
        
        # Get the predicted tag based on the maximum SALT value
        try: predicted_tag = self.salt_to_tag_dict.get(max_salt, nan)
        except Exception: predicted_tag = nan
            
        # Determine if the last applied tag matches the predicted tag
        try: is_tag_correct = bool(last_tag == predicted_tag)
        except Exception: is_tag_correct = nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Last tag value: {last_tag}')
            print(f'Predicted tag value based on max SALT: {predicted_tag}')
            print(f'Is the tag correct? {is_tag_correct}')
            display(patient_df)
        
        # Return True if the tag is correct, False if incorrect, or numpy.nan if data is insufficient
        return is_tag_correct
    
    
    def get_is_patient_severely_injured(self, patient_df, verbose=False):
        """
        Determine whether the patient has severe injuries.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool or numpy.nan
                Returns True if the patient has severe injuries, False if the patient has no severe injuries.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = {'injury_id', 'injury_severity', 'injury_required_procedure'}
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Initialize that the patient is not severely injured
        is_patient_injured = False
        
        # Loop through each injury
        for injury_id, injury_df in patient_df.groupby('injury_id'):
            
            # If the injury is severe, recompute that they are
            is_patient_injured = is_patient_injured or self.get_is_injury_severe(injury_df, verbose=verbose)
            
            # If they are severely injured, break out of the loop
            if is_patient_injured:
                break
        
        return is_patient_injured
    
    
    def get_first_patient_triage(self, patient_df, verbose=False):
        """
        Get the action tick of the first patient triage of the tool applied or tag applied type.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The action tick of the first responder negotiation action, or None if no such action exists.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['action_type', 'action_tick'])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Filter for actions involving responder negotiations (TAG_APPLIED and INJURY_TREATED)
        mask_series = patient_df.action_type.isin(['TAG_APPLIED', 'INJURY_TREATED'])
        
        # If there are responder negotiation actions, find the first action tick
        if mask_series.any(): engagement_start = patient_df[mask_series]['action_tick'].min()
        else: engagement_start = nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'First patient triage: {engagement_start}')
            display(patient_df[mask_series].dropna(axis='columns', how='all').T)
        
        # Return the action tick of the first patient triage or numpy.nan if no data is available
        return engagement_start
    
    
    def get_first_patient_interaction(self, patient_df, verbose=False):
        """
        Get the action tick of the first patient interaction of a specific type.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The action tick of the first responder negotiation action, or None if no such action exists.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['action_type', 'action_tick'])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
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
    
    
    def get_last_patient_interaction(self, patient_df, verbose=False):
        """
        Get the action tick of the last patient interaction involving responder negotiations.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The action tick of the last responder negotiation action, or numpy.nan if no such action exists.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['action_type', 'action_tick'])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Filter for actions involving responder negotiations
        mask_series = patient_df.action_type.isin(self.responder_negotiations_list)
        
        # If there are responder negotiation actions, find the last action tick
        if mask_series.any(): engagement_end = patient_df[mask_series].action_tick.max()
        else: engagement_end = nan
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Action tick of the last patient interaction: {engagement_end}')
            display(patient_df[mask_series].dropna(axis='columns', how='all').T)
        
        # Return the action tick of the last patient interaction or numpy.nan if no data is available
        return engagement_end
    
    
    @staticmethod
    def get_is_patient_gazed_at(patient_df, verbose=False):
        """
        Determine whether the responder gazed at the patient at least once.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the responder gazed at the patient at least once, False otherwise.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = {'action_type'}
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
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
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            tuple
                Two lists, x_dim and z_dim, representing the x and z coordinates of the patient's wanderings.
        """
        
        # Initialize empty lists to store x and z coordinates
        x_dim = []; z_dim = []
        
        # Define the list of location columns to consider
        location_cns_list = [
            'patient_demoted_position', 'patient_engaged_position', 'patient_record_position', 'player_gaze_location'
        ]
        
        # Create a melt dataframe by filtering rows with valid location data
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
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            wrap_width (int, optional):
                Number of characters with which to wrap the text in the label. Defaults to 20.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            str
                A wrapped label for the patient based on the available sorting information.
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
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the patient has an injury record indicating hemorrhage, False otherwise.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['injury_required_procedure'])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Create the mask for injury records requiring hemorrhage control procedures
        mask_series = patient_df.injury_required_procedure.isin(self.hemorrhage_control_procedures_list)
        
        # Compute whether this patient has any
        is_hemorrhaging = mask_series.any()
        
        # If verbose is True, print additional information
        if verbose:
            print(f'Is the patient hemorrhaging: {is_hemorrhaging}')
            display(patient_df)
        
        return is_hemorrhaging
    
    
    def get_time_to_hemorrhage_control(self, patient_df, scene_start, use_dead_alternative=False, verbose=False):
        """
        Calculate the time it takes to control hemorrhage for the patient by getting the injury treatments
        where the responder is not confused from any bad feedback.
        
        According to the paper we define time to hemorrhage control like this:
        Duration of time from the scene start time until the time that the last patient who requires life
        threatening bleeding has been treated with hemorrhage control procedures (when the last
        tourniquet or wound packing was applied).
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            scene_start (int):
                The action tick of the first interaction with the patient.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The time it takes to control hemorrhage for the patient, in action ticks.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set([
            'patient_record_salt', 'patient_engaged_salt', 'action_tick', 'injury_id', 'injury_record_required_procedure',
            'injury_treated_required_procedure', 'tool_applied_type'
        ])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Initialize the controlled time to zero
        controlled_time = 0
        
        # Compute if the patient is dead
        is_patient_dead = self.get_is_patient_dead(patient_df, verbose=verbose)
        
        # Check if we are using the alternative to ignoring dead patients
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
            
            # Update the maximum hemorrhage control time with the time to correct triage tag (black of gray) an alternative measure of proper treatment
            if predicted_tag in ['black', 'gray']:
                controlled_time = max(controlled_time, action_tick - scene_start)
                if verbose: print(f'controlled_time = {controlled_time}')
        
        # Check if the patient is not dead
        elif not is_patient_dead:
            
            # If not dead, define columns for merging
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
    def get_patient_engagement_count(patient_df, verbose=False):
        """
        Count the number of 'PATIENT_ENGAGED' actions in the patient's data.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The number of times the patient has been engaged.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['action_type'])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
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
        """
        Compute the maximum injury severity from a DataFrame of patient data.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient data, including a column 'injury_severity'
                which indicates the severity of the injury.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
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
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['injury_severity'])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Filter out rows where injury severity is missing
        mask_series = ~patient_df.injury_severity.isnull()
        
        # Find the minimum severity among valid values
        # since higher values indicate more severe injuries
        maximum_injury_severity = patient_df[mask_series].injury_severity.min()
        
        return maximum_injury_severity
    
    
    def get_is_life_threatened(self, patient_df, verbose=False):
        """
        Determine whether a patient's life is threatened based on injury severity and hemorrhaging.
        
        This function analyzes a pandas DataFrame (`patient_df`) containing patient 
        information to assess their life-threatening status. It assumes the DataFrame includes 
        columns for injury severity (`injury_severity`) and whether a hemorrhaging condition 
        is present (`injury_required_procedure` - assumed to indicate procedures needed for 
        bleeding control).
        
        The function first verifies that the DataFrame contains all required columns using an 
        assertion. It then calls other functions (assumed to be implemented elsewhere) to 
        retrieve the maximum injury severity and hemorrhaging status. Finally, it combines 
        these factors using a logical AND operation to determine the life-threatening 
        condition.
        
        Parameters:
            patient_df (pandas.DataFrame):
                DataFrame containing patient data with columns 'injury_severity' and 
                'injury_required_procedure'.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the patient's life is threatened, False otherwise.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['injury_severity', 'injury_required_procedure'])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Determine if the maximum injury severity is 'high'
        is_severity_high = (self.get_maximum_injury_severity(patient_df, verbose=verbose) == 'high')
        
        # Determine if the patient is hemorrhaging
        is_patient_hemorrhaging = self.get_is_patient_hemorrhaging(patient_df, verbose=verbose)
        
        # Return True if both conditions are met, otherwise return False
        return is_severity_high and is_patient_hemorrhaging
    
    
    ### Injury Functions ###
    
    
    def get_is_injury_correctly_treated(self, injury_df, patient_df=None, verbose=False):
        """
        Determine whether the given injury was correctly treated.
        
        Parameters:
            injury_df (pandas.DataFrame):
                DataFrame containing injury-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the injury was correctly treated, False otherwise.
        
        Note:
            The FRVRS logger has trouble logging multiple tool applications,
            so injury_treated_injury_treated_with_wrong_treatment == True
            remains and another INJURY_TREATED is never logged, even though
            the right tool was applied after that.
        """
        
        # Ensure all needed columns are present in injury_df
        needed_columns = set([
            'injury_record_required_procedure', 'injury_treated_required_procedure', 'injury_treated_injury_treated', 'injury_treated_injury_treated_with_wrong_treatment', 'action_type'
        ])
        all_columns = set(injury_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from injury_df"
        
        # Get the required procedure for the injury
        mask_series = ~injury_df.injury_record_required_procedure.isnull()
        required_procedure = injury_df[mask_series].injury_record_required_procedure.mode().squeeze()
        if isinstance(required_procedure, Series):
            mask_series = ~injury_df.injury_treated_required_procedure.isnull()
            required_procedure = injury_df[mask_series].injury_treated_required_procedure.mode().squeeze()
        assert not isinstance(required_procedure, Series), "The injury has no required procedures"
        
        # Check if the required procedure isn't "none" and the patient_df is None
        is_correctly_treated = (required_procedure == 'none')
        if (not is_correctly_treated) and (patient_df is None):
            
            # Create a mask to identify treated injuries
            mask_series = (injury_df.injury_treated_injury_treated == True)
            
            # Add to that mask to identify correctly treated injuries
            mask_series &= (injury_df.injury_treated_injury_treated_with_wrong_treatment == False)
            
            # is_correctly_treated is True if there are correctly treated attempts, False otherwise
            is_correctly_treated = mask_series.any()
        
        # Check if the patient_df isn't None
        elif patient_df is not None:
            
            # Ensure all needed columns are present in patient_df
            needed_columns = set([
                'action_tick', 'action_type', 'tool_applied_type'
            ])
            all_columns = set(patient_df.columns)
            assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
            
            # Look for a TOOL_APPLIED within 3 milliseconds of an INJURY_TREATED
            millisecond_threshold = 3
            mask_series = (injury_df.action_type == 'INJURY_TREATED')
            action_ticks_list = sorted(injury_df[mask_series].action_tick.unique())
            for action_tick in action_ticks_list:
                mask_series = patient_df.action_tick.map(
                    lambda ts: abs(ts - action_tick) < millisecond_threshold
                ) & patient_df.action_type.isin(['TOOL_APPLIED'])
                
                # If you found any, check if it's the correct tool type for our injury
                if mask_series.any():
                    
                    # is_correctly_treated is True if tool applied to our injury was the correct one, False otherwise
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
            injury_df (pandas.DataFrame):
                DataFrame containing injury-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the injury is a hemorrhage, False otherwise.
        """
        
        # Ensure all needed columns are present in injury_df
        needed_columns = set(['injury_required_procedure'])
        all_columns = set(injury_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from injury_df"
        
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
        Check if a hemorrhage control tool was applied for a given injury.
        
        Parameters:
            injury_df (pandas.DataFrame):
                DataFrame containing injury data. Must include columns for
                - patient_id (int): Unique identifier for the patient.
                - injury_id (int): Unique identifier for the injury. (Optional)
                - injury_record_required_procedure (str): Required procedure for the injury record.
            logs_df (pandas.DataFrame):
                DataFrame containing log records. Must include columns for
                - patient_id (int): Unique identifier for the patient.
                - tool_applied_type (str): Type of tool applied during the procedure.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if a hemorrhage control tool was applied for the injury, False otherwise.
        """
        
        # Ensure all needed columns are present in injury_df
        needed_columns = set([
            'patient_id', 'injury_record_required_procedure'
        ])
        all_columns = set(injury_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from injury_df"
        
        # Ensure all needed columns are present in logs_df
        needed_columns = set([
            'patient_id', 'tool_applied_type'
        ])
        all_columns = set(logs_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from logs_df"
        
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
        
        # Get the injury ID if verbose
        if verbose:
            mask_series = ~injury_df.injury_id.isnull()
            injury_id = injury_df[mask_series].injury_id.tolist()[0]
            
            print(f'A hemorrhage-related TOOL_APPLIED event can be associated with the injury ({injury_id}): {is_tool_applied_correctly}')
        
        return is_tool_applied_correctly
    
    
    def get_is_hemorrhage_controlled(self, injury_df, logs_df, verbose=False):
        """
        Determine if hemorrhage is controlled based on injury and log data.

        Parameters:
            injury_df (pandas.DataFrame):
                DataFrame containing injury-specific data with relevant columns.
            logs_df (pandas.DataFrame):
                DataFrame containing logs data.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if hemorrhage is controlled, False otherwise.
        
        Note:
            The logs have instances of a TOOL_APPLIED but no INJURY_TREATED preceding it. But, we already know the injury
            that the patient has and the correct tool for every patient because we assigned those ahead of time.
        """
        
        # Ensure all needed columns are present in injury_df
        needed_columns = set([
            'injury_treated_required_procedure', 'patient_id', 'action_type', 'injury_treated_injury_treated_with_wrong_treatment', 'injury_required_procedure',
            'injury_treated_injury_treated', 'injury_record_required_procedure'
        ])
        all_columns = set(injury_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from injury_df"
        
        # Ensure all needed columns are present in logs_df
        needed_columns = set([
            'patient_id', 'tool_applied_type'
        ])
        all_columns = set(logs_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from logs_df"
        
        # Check if an injury record or treatment exists for a hemorrhage-related procedure
        is_injury_hemorrhage = self.get_is_injury_hemorrhage(injury_df, verbose=verbose)
        if not is_injury_hemorrhage: is_controlled = nan
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
            injury_df (pandas.DataFrame):
                DataFrame containing injury-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if the injury is severe, False otherwise.
        """
        
        # Ensure all needed columns are present in injury_df
        needed_columns = set(['injury_required_procedure'])
        all_columns = set(injury_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from injury_df"
        
        # Filter injury_df for rows with high injury severity
        mask_series = (injury_df.injury_severity == 'high')
        
        # Compute the is_injury_severe logic by checking if the injury is a hemorrhage AND if there are any filtered rows
        is_injury_severe = self.get_is_injury_hemorrhage(injury_df, verbose=verbose) and bool(injury_df[mask_series].shape[0])
        
        return is_injury_severe
    
    
    def get_is_bleeding_correctly_treated(self, injury_df, verbose=False):
        """
        Check if bleeding is correctly treated based on the provided injury DataFrame.
        
        Parameters:
            injury_df (pandas.DataFrame):
                DataFrame containing injury-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            bool
                True if bleeding was correctly treated, False otherwise.
        
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
            injury_df (pandas.DataFrame):
                DataFrame containing injury-specific data with relevant columns.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            int
                The time it takes to correctly treat the injury, in action ticks.
        
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
        Score a patient with a 0 for All Stills not visited first, or a 1 for All Stills visited first, given a patient DataFrame.
        """
        
        # Ensure all needed columns are present in scene_df
        needed_columns = {'patient_sort', 'patient_id', 'action_type', 'action_tick'}
        all_columns = set(scene_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from scene_df"
        
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
        Score a patient with a 0 for All Walkers not visited last, or a 1 for All Walkers visited last, given a patient DataFrame.
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
        Score a patient with a 0 for No Wave Command issued, or a 1 for Wave Command issued, given a patient DataFrame.
        """
        
        # Check in the scene if there are any WAVE_IF_CAN actions
        mask_series = scene_df.action_type.isin(['S_A_L_T_WAVE_IF_CAN', 'TRIAGE_LEVEL_WAVE_IF_CAN'])
        
        # If there are, output a 1 (Wave Command issued), if not, output a 0 (No Wave Command issued)
        is_wave_command_issued = int(scene_df[mask_series].shape[0] > 0)
        
        return is_wave_command_issued
    
    
    @staticmethod
    def get_walk_value(scene_df, verbose=False):
        """
        Score a patient with a 0 for No Walk Command issued, or a 1 for Walk Command issued, given a patient DataFrame.
        """
        
        # Check in the scene if there are any WALK_IF_CAN actions
        mask_series = scene_df.action_type.isin(['S_A_L_T_WALK_IF_CAN', 'TRIAGE_LEVEL_WALK_IF_CAN'])
        
        # If there are, output a 1 (Walk Command issued), if not, output a 0 (No Walk Command issued)
        is_walk_command_issued = int(scene_df[mask_series].shape[0] > 0)
        
        return is_walk_command_issued
    
    
    ### Rasch Analysis Patient Functions ###
    
    
    @staticmethod
    def get_treatment_value(patient_df, injury_id, verbose=False):
        """
        Score a patient with a 0 for No Treatment or Wrong Treatment, or a 1 for Correct Treatment, given a patient DataFrame.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['injury_id', 'injury_record_required_procedure', 'injury_treated_required_procedure', 'action_tick'])
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Get required procedure
        mask_series = (patient_df.injury_id == injury_id) & ~patient_df.injury_record_required_procedure.isnull()
        if not mask_series.any(): return nan
        df = patient_df[mask_series]
        required_procedure = df.injury_record_required_procedure.squeeze()
        
        # Get first attempt
        mask_series = (patient_df.injury_id == injury_id) & ~patient_df.injury_treated_required_procedure.isnull()
        if not mask_series.any(): return 0
        df = patient_df[mask_series]
        first_procedure = df.sort_values(['action_tick']).injury_treated_required_procedure.tolist()[0]
        
        # If the first attempt was the required procedure, output a 1 (Correct Treatment), if not, output a 0 (No Treatment or Wrong Treatment)
        is_injury_treated = int(first_procedure == required_procedure)

        return is_injury_treated
    
    
    def get_tag_value(self, patient_df, verbose=False):
        """
        Score a patient with a 0 for No Tag or Wrong Tag, or a 1 for Correct Tag, given a patient DataFrame.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = set(['responder_category', 'action_tick', 'tag_applied_type', 'patient_salt'] + self.patient_groupby_columns)
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        try:
            
            # Find out whether a patient's tag is correct
            is_tag_correct = self.get_is_tag_correct(patient_df, verbose=verbose)
            
            # If it's NaN, score it as zero
            if isnan(is_tag_correct): is_tag_correct = 0
            
            # Otherwise, score it as the integer of the boolean
            else: is_tag_correct = int(is_tag_correct)
        except: is_tag_correct = 0
        
        return is_tag_correct
    
    
    @staticmethod
    def get_pulse_value(patient_df, verbose=False):
        """
        Score a patient with a 0 for No Pulse Taken or a 1 for Pulse Taken, given a patient DataFrame.
        """
        
        # Ensure all needed columns are present in patient_df
        needed_columns = {'action_type'}
        all_columns = set(patient_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from patient_df"
        
        # Filter for PULSE_TAKEN action types
        mask_series = (patient_df.action_type == 'PULSE_TAKEN')
        
        # Compute the score based on a non-zero count of rows in the filter
        is_pulse_taken = int(mask_series.any())
        
        return is_pulse_taken
    
    
    ### Pandas Functions ###
    
    
    def show_time_statistics(self, describable_df, columns_list):
        """
        Display summary statistics for time-related data in a readable format.
        
        Parameters:
            describable_df (pandas.DataFrame):
                The DataFrame containing time-related data.
            columns_list (list):
                List of column names for which time statistics should be calculated.
        
        This function calculates and displays summary statistics for time-related data,
        including mean, mode, median, and standard deviation (SD), in a human-readable format.
        
        The displayed DataFrame includes time-related statistics with formatted timedelta values.
        The standard deviation (SD) is presented as '±' followed by the corresponding value.
        
        Example usage:
        ```
        your_instance.show_time_statistics(your_dataframe, ['column1', 'column2'])
        ```
        
        Note:
            This function relies on the 'get_statistics' and 'format_timedelta_lambda' methods.
        """
        
        # Calculate basic descriptive statistics for time-related columns
        df = nu.get_statistics(describable_df, columns_list)
        
        # Convert time values to timedelta objects and format them using format_timedelta_lambda()
        df = df.applymap(
            lambda x: self.format_timedelta_lambda(timedelta(milliseconds=int(x))),
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
            file_df:
                A Pandas DataFrame containing the player action data with its index reset.
        
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
            action_type:
                The action type.
            df:
                The DataFrame containing the MCI-VR metrics with the logger version column removed.
            row_index:
                The index of the row in the DataFrame to set the metrics for.
            row_series:
                The row series containing the MCI-VR metrics.
    
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
            df.loc[row_index, 'patient_demoted_patient_id'] = row_series[6] # patientId
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
            df.loc[row_index, 'patient_engaged_patient_id'] = row_series[6] # patientId
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
            df.loc[row_index, 'breathing_checked_patient_id'] = row_series[5] # patientId
        elif (action_type == 'PATIENT_RECORD'): # PatientRecord
            df.loc[row_index, 'patient_record_health_level'] = row_series[4] # healthLevel
            df.loc[row_index, 'patient_record_health_time_remaining'] = row_series[5] # healthTimeRemaining
            df.loc[row_index, 'patient_record_patient_id'] = row_series[6] # patientId
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
            df.loc[row_index, 'sp_o2_taken_patient_id'] = row_series[5] # patientId
        elif (action_type == 'S_A_L_T_WALKED'): # SALTWalked
            df.loc[row_index, 's_a_l_t_walked_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_walked_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_walked_patient_id'] = row_series[6]
        elif (action_type == 'TRIAGE_LEVEL_WALKED'):
            df.loc[row_index, 'triage_level_walked_location'] = row_series[4]
            df.loc[row_index, 'triage_level_walked_command_text'] = row_series[5]
            df.loc[row_index, 'triage_level_walked_patient_id'] = row_series[6] # patientId
        elif (action_type == 'S_A_L_T_WALK_IF_CAN'): # SALTWalkIfCan
            df.loc[row_index, 's_a_l_t_walk_if_can_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_walk_if_can_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_walk_if_can_patient_id'] = row_series[6] # patientId
        elif (action_type == 'TRIAGE_LEVEL_WALK_IF_CAN'):
            df.loc[row_index, 'triage_level_walk_if_can_location'] = row_series[4]
            df.loc[row_index, 'triage_level_walk_if_can_command_text'] = row_series[5]
            df.loc[row_index, 'triage_level_walk_if_can_patient_id'] = row_series[6] # patientId
        elif (action_type == 'S_A_L_T_WAVED'): # SALTWave
            df.loc[row_index, 's_a_l_t_waved_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_waved_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_waved_patient_id'] = row_series[6] # patientId
        elif (action_type == 'TRIAGE_LEVEL_WAVED'):
            df.loc[row_index, 'triage_level_waved_location'] = row_series[4]
            df.loc[row_index, 'triage_level_waved_command_text'] = row_series[5]
            df.loc[row_index, 'triage_level_waved_patient_id'] = row_series[6] # patientId
        elif (action_type == 'S_A_L_T_WAVE_IF_CAN'): # SALTWaveIfCan
            df.loc[row_index, 's_a_l_t_wave_if_can_sort_location'] = row_series[4] # sortLocation
            df.loc[row_index, 's_a_l_t_wave_if_can_sort_command_text'] = row_series[5] # sortCommandText
            df.loc[row_index, 's_a_l_t_wave_if_can_patient_id'] = row_series[6] # patientId
        elif (action_type == 'TRIAGE_LEVEL_WAVE_IF_CAN'):
            df.loc[row_index, 'triage_level_wave_if_can_location'] = row_series[4]
            df.loc[row_index, 'triage_level_wave_if_can_command_text'] = row_series[5]
            df.loc[row_index, 'triage_level_wave_if_can_patient_id'] = row_series[6] # patientId
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
            df.loc[row_index, 'tool_applied_patient_id'] = row_series[4] # patientId
            df.loc[row_index, 'tool_applied_type'] = row_series[5] # type
            df.loc[row_index, 'tool_applied_attachment_point'] = row_series[6] # attachmentPoint
            df.loc[row_index, 'tool_applied_tool_location'] = row_series[7] # toolLocation
            df.loc[row_index, 'tool_applied_data'] = row_series[8] # data
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
            df.loc[row_index, 'player_gaze_location'] = row_series[4] # Location (x,y,z)
            df.loc[row_index, 'player_gaze_patient_id'] = row_series[5] # patientId
            df.loc[row_index, 'player_gaze_distance_to_patient'] = row_series[6] # Distance to Patient
            df.loc[row_index, 'player_gaze_direction_of_gaze'] = row_series[7] # Direction of Gaze (vector3)
        elif action_type not in self.action_type_to_columns_dict:
            if (action_type == 'Participant ID'):
                df = df.drop(index=row_index)
                return df
            elif (action_type in ['SESSION_START', 'SESSION_END']):
                return df
            else:
                raise Exception(f"\n\n{action_type} not found in self.action_type_to_columns_dict:\n{row_series}")
        
        return df
    
    
    def process_files(self, sub_directory_df, sub_directory, file_name, verbose=False):
        """
        Process files and update the subdirectory dataframe.
        
        This function takes a Pandas DataFrame representing the subdirectory and the name of the file to process.
        It then reads the file, parses it into a DataFrame, and appends the DataFrame to the subdirectory DataFrame.
        
        Parameters:
            sub_directory_df (pandas.DataFrame):
                The DataFrame for the current subdirectory.
            sub_directory (str):
                The path of the current subdirectory.
            file_name (str):
                The name of the file to process.
        
        Returns:
            pandas.DataFrame
                The updated subdirectory DataFrame.
        """
        
        # Construct the full path to the file and get the logger version
        file_path = osp.join(sub_directory, file_name)
        version_number = self.get_logger_version(file_path, verbose=False)
        
        # Read CSV file using a CSV reader
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
        
        # Set the MCIVR metrics types
        for row_index, row_series in file_df.iterrows(): file_df = self.set_mcivr_metrics_types(row_series.action_type, file_df, row_index, row_series, verbose=verbose)
        
        # Section off player actions by session start and end
        file_df = self.set_scene_indices(file_df.reset_index(drop=True))
        
        # Append the data frame for the current file to the data frame for the current subdirectory
        sub_directory_df = concat([sub_directory_df, file_df], axis='index')
        
        return sub_directory_df
    
    
    def concatonate_logs(self, logs_folder=None, verbose=False):
        """
        Concatenate all the CSV files in the given logs folder into a single data frame.
        
        Parameters:
            logs_folder (str, optional):
                The path to the folder containing the CSV files. The default value is the data logs folder.
        
        Returns:
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
        
        # Convert elapsed time (action_tick) to an integer
        if ('action_tick' in logs_df.columns):
            logs_df.action_tick = to_numeric(logs_df.action_tick, errors='coerce')
            mask_series = ~logs_df.action_tick.isnull()
            logs_df = logs_df[mask_series]
            logs_df.action_tick = logs_df.action_tick.astype('int64')
        
        logs_df = logs_df.reset_index(drop=True)
        
        return logs_df
    
    
    @staticmethod
    def show_long_runs(df, column_name, milliseconds, delta_fn, description, logs_df):
        """
        Display files with a specified duration in a given DataFrame.
        
        Parameters:
            df (pandas.DataFrame):
                DataFrame containing the relevant data.
            column_name (str):
                Name of the column in 'df' containing duration information.
            milliseconds (int):
                Threshold duration in milliseconds.
            delta_fn (function):
                Function to convert milliseconds to a time delta object.
            description (str):
                Description of the duration, e.g., 'longer' or 'shorter'.
            logs_df (pandas.DataFrame, optional):
                DataFrame containing additional log data, defaults to None.
        
        Returns:
            None

        Note:
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
    
    
    def get_elevens_dataframe(self, logs_df, file_stats_df, scene_stats_df, needed_columns=[], filter_lambda=None, verbose=False):
        """
        Merge and filter a set of DataFrames to create a DataFrame containing triage scenes 
        with at least eleven patients.
        
        This function merges data from logs, file stats, and scene stats DataFrames
        and filters for triage scenes that have not been aborted and contain at least
        eleven patients.
        
        Parameters:
            logs_df (pandas.DataFrame):
                A DataFrame containing scene log data.
            file_stats_df (pandas.DataFrame):
                A DataFrame containing file statistics.
            scene_stats_df (pandas.DataFrame):
                A DataFrame containing scene statistics.
            needed_columns (list of str, optional):
                Additional columns needed for the output DataFrame. Default is an empty list.
            filter_lambda (function, optional):
                A function used to filter scenes based on patient count (defaults to a function that 
                requires at least eleven unique patient IDs). This function should accept a DataFrame 
                representing a scene and return True if the scene meets the patient count criteria.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            pandas.DataFrame
                A DataFrame containing information about triage scenes with at least eleven patients. 
                The DataFrame includes columns specified in `needed_columns` and potentially 
                additional columns from the input DataFrames.
        """
        
        # Get the column sets and ensure the columns are in one or more of the input dataframes
        triage_columns = ['scene_type', 'is_scene_aborted']
        needed_set = set(triage_columns + list(needed_columns))
        for cn in needed_set:
            if not any(map(lambda df: cn in df.columns, [logs_df, file_stats_df, scene_stats_df])):
                raise ValueError(f'The {cn} column must be in either logs_df, file_stats_df, or scene_stats_df.')
        
        logs_columns_set = set(logs_df.columns)
        file_columns_set = set(file_stats_df.columns)
        scene_columns_set = set(scene_stats_df.columns)
        
        # Check if some columns are missing from just using the logs dataset
        if bool(needed_set.difference(logs_columns_set)):
            
            # Merge in the file stats columns (what's still needed from using the scene and logs datasets together)
            on_columns = sorted(logs_columns_set.intersection(file_columns_set))
            file_stats_columns = on_columns + sorted(needed_set.difference(logs_columns_set).difference(scene_columns_set))
            merge_df = logs_df.merge(file_stats_df[file_stats_columns], on=on_columns)
            
            # Merge in the scene stats columns (what's still needed from using the file and logs datasets together)
            on_columns = sorted(logs_columns_set.intersection(scene_columns_set))
            scene_stats_columns = on_columns + sorted(needed_set.difference(logs_columns_set).difference(file_columns_set))
            merge_df = merge_df.merge(scene_stats_df[scene_stats_columns], on=on_columns)
        
        # If not, just use the logs_df dataframe
        else:
            merge_df = logs_df
        
        # Apply the filter lambda to get scenes with at least eleven patients
        if filter_lambda is None:
            filter_lambda = lambda scene_df: (scene_df.patient_id.nunique() >= 11)
        elevens_df = merge_df.groupby(self.scene_groupby_columns).filter(filter_lambda)
        
        return elevens_df
    
    
    def convert_column_to_categorical(self, categorical_df, column_name, verbose=False):
        """
        Convert a specified column in a DataFrame to a categorical type based on predefined 
        order and category attributes.
        
        This function takes a DataFrame (`categorical_df`), a column name (`column_name`), and 
        an optional verbosity flag (`verbose`). It aims to convert the specified column to a 
        categorical format if the column exists in the DataFrame.
        
        The function first checks if the `column_name` exists in the DataFrame. If it does, it 
        attempts to find two class attributes:
            - Order attribute: This attribute should be named following the format 
            "{column_name_parts[i:]}_order" (e.g., "country_order" for "country" column). It 
            should contain a list or tuple defining the expected order of categories.
            - Category attribute: This attribute should be named following the format 
            "{column_name_parts[i:]}_category_order" (e.g., "country_category_order" for "country" 
            column). It should contain a data type (e.g., "category") suitable for representing 
            the categorical data.
        
        The function verifies that these attributes exist and that the unique values in the 
        column are a subset of the allowed values defined in the order attribute. If all 
        conditions are met, the column is converted to a categorical type using the category 
        labels retrieved from the class attribute. Optionally, descriptive information about 
        the conversion can be printed based on the `verbose` flag.
        
        Parameters:
            categorical_df (pandas.DataFrame):
                The DataFrame containing the column to be converted.
            column_name (str):
                The name of the column to convert to categorical format.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            pandas.DataFrame
                The DataFrame with the specified column converted to a categorical type.
        
        Raises:
            AssertionError
                If the unique values in the column are not a subset of the allowed values defined in 
                the order attribute (found using a specific naming convention).
            AttributeError
                If the expected class attributes for order or category information are not found.
        
        Notes:
            This function assumes that the class has attributes defining the order and category 
            order for the columns to be converted.
        """
        
        # Check if the column exists in the DataFrame
        if (column_name in categorical_df.columns):
            
            # Initialize flag for displaying results
            display_results = False
            
            # Split the column name into parts
            name_parts_list = column_name.split('_')
            
            # Find the order attribute
            attribute_name = 'XXXX'
            for i in range(3):
                
                # If the attribute does not exist, construct its name
                if not hasattr(self, attribute_name):
                    attribute_name = f"{'_'.join(name_parts_list[i:])}_order"
                    
                    # If verbose is True, print it
                    if verbose:
                        print(f"Finding {attribute_name} as the order attribute")
                else:
                    break
            
            # Check if the order attribute exists
            if hasattr(self, attribute_name):
                
                # Check for missing elements in the column
                mask_series = ~categorical_df[column_name].isnull()
                feature_set = set(categorical_df[mask_series][column_name].unique())
                order_set = set(eval(f"self.{attribute_name}"))
                assert feature_set.issubset(order_set), f"You're missing {feature_set.difference(order_set)} from self.{attribute_name}"
                
                # Find the category attribute
                attribute_name = 'XXXX'
                for i in range(3):
                    
                    # If the attribute does not exist, construct its name
                    if not hasattr(self, attribute_name):
                        attribute_name = f"{'_'.join(name_parts_list[i:])}_category_order"
                        
                        # Print it if verbose is True
                        if verbose:
                            print(f"Finding {attribute_name} as the category attribute")
                    else:
                        break
                
                # Check if the category attribute exists
                if hasattr(self, attribute_name):
                    if verbose:
                        print(f"\nConvert {column_name} column to categorical")
                    
                    # Convert the column to the categorical type defined by the category attribute
                    categorical_df[column_name] = categorical_df[column_name].astype(eval(f"self.{attribute_name}"))
                    
                    # Set flag to display results
                    display_results = True
                else:
                    if verbose:
                        print(f"AttributeError: 'FRVRSUtilities' object has no attribute '{attribute_name}'")
            
            # Display results if verbose is True and display_results flag is set
            if verbose and display_results:
                print(f"{column_name} has {categorical_df[column_name].nunique()} unique categories")
                display(categorical_df.groupby(column_name).size().to_frame().rename(columns={0: 'record_count'}).sort_values('record_count', ascending=False).head(20))
        
        # Return the modified DataFrame
        return categorical_df
    
    
    def add_modal_column_to_dataframe(self, new_column_name, modal_df, is_categorical=True, verbose=False):
        """
        Add a modal column to a DataFrame.
        
        This function checks if a column with the given name exists in the DataFrame. If it 
        does not exist, it attempts to create it by combining existing columns as specified by 
        the attribute name. The new column can be optionally converted to a categorical type.
        
        Parameters:
            new_column_name (str):
                The name of the new column to be added.
            modal_df (pandas.DataFrame):
                The DataFrame to which the new column will be added.
            is_categorical (bool, optional):
                Whether to convert the new column to a categorical type. Default is True.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            pandas.DataFrame
                The modified DataFrame with the new modal column (if applicable).
        """
        
        # Check if the new column already exists
        if new_column_name not in modal_df.columns:
            
            # Split the column name for attribute name construction
            name_parts_list = new_column_name.split('_')
            
            # Optional verbose message
            if verbose:
                print(f"\nModalize into one {' '.join(name_parts_list)} column if possible")
            
            # Initialize the attribute name to search for columns list
            attribute_name = 'XXXX'
            
            # Iterate to find the correct attribute name
            for i in range(3):
                
                # Check if the attribute exists in the class instance
                if not hasattr(self, attribute_name):
                    
                    # Construct the attribute name by joining parts of the column name
                    attribute_name = f"{'_'.join(name_parts_list[i:])}_columns_list"
                
                # Break the loop if the attribute is found
                else:
                    break
            
            # Modalize the columns in the dataframe using the found attribute name
            modal_df = nu.modalize_columns(modal_df, eval(f"self.{attribute_name}"), new_column_name)
            
            # Handle categorical conversion
            if is_categorical:
                modal_df = self.convert_column_to_categorical(modal_df, new_column_name, verbose=verbose)
            
            # Display a sample of the new column if verbose
            elif verbose:
                print(modal_df[new_column_name].nunique())
                display(modal_df.groupby(new_column_name).size().to_frame().rename(columns={0: 'record_count'}).sort_values('record_count', ascending=False).head(20))
        
        # Return the modified dataframe
        return modal_df
    
    
    ### Plotting Functions ###
    
    
    def visualize_order_of_engagement(self, scene_df, engagement_order=None, color_dict=None, verbose=False):
        """
        Visualize the order of engagement of patients in a given scene.
        
        Parameters:
            scene_df (pandas.DataFrame):
                DataFrame containing scene data with relevant columns.
            engagement_order (list of tuples, optional):
                List of tuples of patient IDs, action ticks, and location coordinates. If not 
                provided, a new list will be created using the patients in scene_df.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            None
        
        Raises:
            ValueError:
                If nu is not None and not an instance of NotebookUtilities.
        """
        
        # Get the engagement order
        if engagement_order is None: engagement_order = self.get_order_of_actual_engagement(scene_df, verbose=verbose)
        
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
        file_path = osp.join(nu.saves_png_folder, sub(r'\W+', '_', str(title)).strip('_').lower() + '.png')
        if verbose: print(f'Saving figure to {file_path}')
        plt.savefig(file_path, bbox_inches='tight')
        
        return fig, ax
    
    
    def visualize_player_movement(self, logs_df, scene_mask, title=None, save_only=False, verbose=False):
        """
        Visualizes the player movement for the given session mask in a 2D plot.
    
        Parameters:
            scene_mask (pandas.Series):
                A boolean mask indicating which rows of the logs_df DataFrame belong to the current scene.
            title (str, optional):
                The title of the plot, if saving.
            save_only (bool, optional):
                Whether to only save the plot to a PNG file and not display it.
            logs_df (pandas.DataFrame):
                A DataFrame containing the FRVRS logs.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
    
        Returns:
            None
                The function either displays the plot or saves it to a file.
    
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
            file_path = osp.join(self.saves_folder, 'png', sub(r'\W+', '_', str(title)).strip('_').lower() + '.png')
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
            logs_df (pandas.DataFrame):
                A DataFrame containing the FRVRS logs.
            df (pandas.DataFrame):
                The input DataFrame containing player movement data.
            sorting_column (str):
                The column based on which the DataFrame will be sorted.
            mask_series (pandas.Series or None):
                Optional mask series to filter rows in the DataFrame.
            is_ascending (bool):
                If True, sort the DataFrame in ascending order; otherwise, in descending order.
            humanize_type (str):
                The type of humanization to be applied to the time values ('precisedelta', 'percentage', 'intword').
            title_str (str):
                Additional string to be included in the visualization title.
            nu (NotebookUtilities or None):
                An instance of NotebookUtilities or None. If None, a new instance will be created.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
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
        Display timelines for patient engagements in an arbitrary session and scene.
        
        Parameters:
            logs_df (pandas.DataFrame):
                DataFrame containing FRVRS logs.
            random_session_uuid (str, optional):
                UUID of the random session. If not provided, a random session will be selected.
            random_scene_index (int, optional):
                Index of the random scene. If not provided, a random scene within the selected session will be chosen.
            color_cycler (callable, optional):
                A callable that returns a color for each patient engagement timeline. If not provided, it will be generated.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
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
            'PATIENT_ENGAGED', 'INJURY_TREATED'
        ] + self.responder_negotiations_list)
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
                    mask_series = df1.action_type.isin(['INJURY_TREATED', 'PATIENT_ENGAGED'] + self.responder_negotiations_list)
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
            transformable_df:
                DataFrame, the data to be plotted.
            x_column_name:
                str, the column for x-axis.
            y_column_name:
                str, the column for y-axis.
            x_label:
                str, label for x-axis.
            y_label:
                str, label for y-axis.
            transformer_name:
                str, the name of the transformation function (default is 'min').
            is_y_temporal:
                bool, if True, humanize y-axis tick labels as temporal values.
        
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
        transformed_df[y_column_name] = transformed_df[y_column_name].map(lambda x: float(x))
        
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
            for text_obj in ax.get_yticklabels():
                text_obj.set_text(
                    humanize.precisedelta(timedelta(milliseconds=text_obj.get_position()[1])).replace(', ', ',\n').replace(' and ', ' and\n')
                )
                yticklabels_list.append(text_obj)
            ax.set_yticklabels(yticklabels_list);
        
        plt.show()
        
        return plt
    
    
    def show_gaze_timeline(
        self, logs_df, random_session_uuid=None, random_scene_index=None, consecutive_cutoff=600, patient_color_dict=None, verbose=False
    ):
        """
        Display a timeline of player gaze events for an arbitrary session and scene.
        
        Parameters:
            logs_df (pandas.DataFrame):
                DataFrame containing logs data.
            random_session_uuid (str):
                UUID of the arbitrary session. If None, a random session will be selected.
            random_scene_index (int):
                Index of the arbitrary scene. If None, a random scene will be selected within the session.
            consecutive_cutoff (int):
                Time cutoff for consecutive rows.
            patient_color_dict (dict):
                Dictionary mapping patient IDs to colors.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            tuple
                Session UUID and scene index.
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
        
        # Create the scene mask
        scene_mask_series = (logs_df.session_uuid == random_session_uuid) & (logs_df.scene_id == random_scene_index)
        
        # Calculate the elapsed time of the player gaze
        mask_series = scene_mask_series & (logs_df.action_type.isin(['PLAYER_GAZE', 'SESSION_END', 'SESSION_START']))
        columns_list = ['action_tick', 'patient_id']
        patient_gazes_df = logs_df[mask_series][columns_list].sort_values(['action_tick'])
        patient_gazes_df['time_diff'] = patient_gazes_df.action_tick.diff()
        for patient_id in patient_gazes_df.patient_id.unique():
            patient_gazes_df = nu.replace_consecutive_rows(
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
            
            # Loop over each row of the player gaze dataframe
            for row_index, row_series in patient_gazes_df.iterrows():
                action_tick = row_series.action_tick
                patient_id = row_series.patient_id
                time_diff = row_series.time_diff
                
                # Get the player gaze annotation parameters
                if not isna(patient_id):
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
        Plot a given sequence for a specific scene.
        
        This function visualizes the sequence along with relevant information like the scene's entropy,
        turbulence, and complexity, while highlighting specific n-grams and coloring different action types.
        
        Parameters:
            scene_tuple (tuple):
                A tuple containing the session UUID and scene ID of the specific scene.
            sequence (list):
                The sequence of actions to be plotted (e.g., a list of action types).
            logs_df (pandas.DataFrame):
                DataFrame containing detailed logs of all VR simulation interactions.
            summary_statistics_df (pandas.DataFrame, optional):
                DataFrame containing summary statistics for all scenes. If not provided, tries to 
                load from a pickle file.
            actions_mask_series (pandas.Series, optional):
                A mask defining which actions to include in the plot. If not provided, all actions 
                are used.
            highlighted_ngrams (list, optional):
                A list of n-grams to highlight in the plot.
            color_dict (dict, optional):
                A dictionary mapping action types to color values.
            suptitle (str, optional):
                The title of the plot. If not provided, summary statistics are shown.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            matplotlib.figure.Figure:
                The figure containing the plot.
        
        Raises:
            Exception:
                If `summary_statistics_df` is missing and cannot be loaded from pickle files.
        """
        
        # Extract session_uuid and scene_id from the scene_tuple
        session_uuid, scene_id = scene_tuple
        
        # Load summary_statistics_df from pickle if not provided
        if verbose: display(summary_statistics_df)
        if (summary_statistics_df is None):
            if nu.pickle_exists('summary_statistics_df'): summary_statistics_df = nu.load_object('summary_statistics_df')
            else: raise Exception('You need to provide summary_statistics_df')
        
        # Display the logs dataframe if verbose
        if verbose: display(logs_df)
        
        # Create the scene history filtered by the action mask series, if provided
        if (actions_mask_series is None): actions_mask_series = [True] * logs_df.shape[0]
        mask_series = (logs_df.session_uuid == session_uuid) & (logs_df.scene_id == scene_id) & actions_mask_series
        scene_df = logs_df[mask_series].sort_values('action_tick')
        
        # Build the alphabet list of the first of each action type
        actions_list = []
        for row_index, row_series in scene_df.iterrows():
            action_type = row_series.voice_command_message
            if notnull(action_type) and (action_type not in actions_list): actions_list.append(action_type)
            action_type = row_series.action_type
            if notnull(action_type) and (action_type != 'VOICE_COMMAND') and (action_type not in actions_list): actions_list.append(action_type)
        
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
        if sequence:
            fig, ax = nu.plot_sequence(
                sequence, highlighted_ngrams=highlighted_ngrams, color_dict=color_dict, suptitle=suptitle, alphabet_list=actions_list, verbose=verbose
            )
            
            return fig, ax
    
    
    ### Open World Functions ###
    
    
    def add_encounter_layout_column_to_json_stats(self, csv_stats_df, json_stats_df, verbose=False):
        """
        Add a new column to the JSON statistics DataFrame indicating the environment of each encounter.
        
        Parameters:
            csv_stats_df (pandas.DataFrame):
                DataFrame containing statistics from CSV files.
            json_stats_df (pandas.DataFrame):
                DataFrame containing statistics from JSON files.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            None
        """
        
        # Print the shape of the CSV stats DataFrame if verbose
        if verbose:
            print(csv_stats_df.shape)
        
        # Use the patients lists from the March 25th ITM BBAI Exploratory analysis email
        new_column_name = 'encounter_layout'
        
        # Loop through each session and scene in the CSV stats dataset
        for (session_uuid, scene_id), scene_df in csv_stats_df.groupby(self.scene_groupby_columns):
            
            # Print the unique patient IDs for each scene if verbose
            if verbose:
                print(scene_df.patient_id.unique().tolist())
            
            # Loop through each environment and get the patients list for that environment
            for env_str in ['desert', 'jungle', 'submarine', 'urban']:
                patients_list = eval(f'self.{env_str}_patients_list')
                
                # Check if all patients are in that scene
                if all(map(lambda patient_id: patient_id in scene_df.patient_id.unique().tolist(), patients_list)):
                    
                    # If so, find the corresponding session in the JSON stats dataset and add that environment to it as a new column
                    mask_series = (json_stats_df.session_uuid == session_uuid)
                    json_stats_df.loc[mask_series, new_column_name] = env_str.title()
        
        if verbose:
            
            # Print the shape of the JSON stats DataFrame if verbose
            print(json_stats_df.shape) # (43, 3541)
            
            # Display the count of records for each environment if verbose
            display(json_stats_df.groupby(new_column_name, dropna=False).size().to_frame().rename(columns={0: 'record_count'}))
        
        return json_stats_df
    
    
    def add_medical_role_column_to_anova_dataframe(self, json_stats_df, anova_df, verbose=False):
        """
        Add a medical role column to the dataframe by merging with JSON stats.
        
        Parameters:
            json_stats_df (pandas.DataFrame):
                Dataframe containing JSON statistics.
            anova_df (pandas.DataFrame):
                Dataframe to merge with JSON statistics dataframe.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
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
            
            if verbose:
                print(anova_df.shape)
                print(anova_df.columns.tolist())
                display(anova_df.groupby(column_name).size().to_frame().rename(columns={0: 'record_count'}).sort_values(
                    'record_count', ascending=False
                ).head(5))
        
        return anova_df
    
    
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
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            float
                The average difficulty level.
        """
        
        # Find the scenarioData dictionary within the configData dictionary of the JSON data for that session and participant
        
        # Inside, you will find three keys: description, difficulty, and name. The difficulty key is semi-continously numeric, and you can average it for whatever grouping you need
        pass
    
    
    def add_prioritize_severity_column_to_merge_dataframe(self, merge_df, new_column_name='prioritize_high_injury_severity_patients', verbose=False):
        """
        Add a new column to the DataFrame indicating if a row (patient) has the highest injury severity among engaged patients (engaged_patientXX_injury_severity).
        
        Parameters:
            merge_df (pandas.DataFrame):
                The DataFrame to add the new column to.
            new_column_name (str, optional):
                The name of the new column to be added. Defaults to 'prioritize_high_injury_severity_patients'.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
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
        
        # Check if new_column_name already exists
        if new_column_name in merge_df.columns:
            raise ValueError(f"Column '{new_column_name}' already exists in the dataframe.")
        
        # Check if engaged_patient00_injury_severity doesn't exist
        if 'engaged_patient00_injury_severity' not in merge_df.columns:
            
            # Loop through each scene in the merge dataframe
            rows_list = []
            for (session_uuid, scene_id), scene_df in merge_df.groupby(self.scene_groupby_columns):
            
                # Create a dictionary to store scene/session information
                row_dict = {cn: eval(cn) for cn in self.scene_groupby_columns}
                
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
    
    
    def get_action_tick_by_encounter_layout(self, session_df, encounter_layout=None, verbose=False):
        """
        Calculate the action tick and Euclidean distance for the first teleport action closest to the encounter layout's initial teleport location.
        
        This function determines the action tick and Euclidean distance from the initial teleport 
        location to the nearest neighbor location based on the encounter layout provided or found 
        in the session DataFrame.
        
        Parameters:
            self (object):
                The instance of the class containing the initial teleport location attributes for each encounter layout.
            session_df (pandas.DataFrame):
                DataFrame containing session data, including action types and locations.
            encounter_layout (str, optional):
                The encounter layout to use for determining the initial teleport location. If None, 
                the encounter layout is inferred from the session DataFrame. Default is None.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            tuple
                A tuple containing two elements:
                - action_tick (float or nan): The action tick for the first teleport closest to the base point, or nan if not found.
                - euclidean_distance (float or nan): The Euclidean distance to the first teleport closest to the base point, or nan if not found.
        
        Raises:
            AssertionError
                If 'encounter_layout' column is missing from session_df when encounter_layout is None.
                If session_df is missing any of the required columns: 'action_type', 'location_id', 'action_tick'.
        """
        
        # Ensure all needed columns are present in session_df
        needed_columns = {'action_type', 'location_id', 'action_tick'}
        all_columns = set(session_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from session_df"
        
        action_tick = nan
        euclidean_distance = nan
        
        # Determine the encounter layout if not provided
        if encounter_layout is None:
            assert 'encounter_layout' in session_df.columns, "You need to supply an encounter_layout column in session_df or as a parameter"
            
            # Create a mask for non-null encounter_layout values
            mask_series = ~session_df.encounter_layout.isnull()
            
            # If there are any non-null encounter_layout values, use the most frequent one
            if mask_series.any():
                encounter_layout = session_df[mask_series].encounter_layout.value_counts().head(1).index.item()
                if verbose: print(f'encounter_layout = "{encounter_layout}"')
        
        # Get the base point for the initial teleport location
        base_point = eval('self.' + encounter_layout.lower() + '_initial_teleport_location')
        if verbose: print(f'base_point = "{base_point}"')
        
        # Create a mask for TELEPORT actions with non-null location_id values
        mask_series = (session_df.action_type == 'TELEPORT') & ~session_df.location_id.isnull()
        
        # If there are any such actions, proceed to find the nearest neighbor
        if mask_series.any():
            neighbors_list = [eval(location_id) for location_id in session_df[mask_series].location_id.unique()]
            if verbose: print(f'neighbors_list = "{neighbors_list}"')
            
            # Get the nearest neighbor to the base point
            nearest_neighbor = nu.get_nearest_neighbor(base_point, neighbors_list)
            if verbose: print(f'nearest_neighbor = "{nearest_neighbor}"')
            
            # Calculate the Euclidean distance to the nearest neighbor
            euclidean_distance = nu.get_euclidean_distance(base_point, nearest_neighbor)
            if verbose: print(f'euclidean_distance = "{euclidean_distance}"')
            
            # Find the minimum action tick for the nearest neighbor location
            mask_series = session_df.location_id.isin([str(nearest_neighbor)])
            if mask_series.any():
                action_tick = session_df[mask_series].action_tick.min()
                if verbose: print(f'action_tick = "{action_tick}"')
        
        return action_tick, euclidean_distance
    
    
    @staticmethod
    def add_triage_error_rate_columns_to_row(groupby_df, row_dict, verbose=False):
        """
        Calculate and add over-triage, under-triage, and critical-triage error rate columns to a
        dictionary representing a new row in a triage error rates dataframe being built.
        
        This function takes a pandas DataFrame grouped by 'error_type' with a 'patient_count' 
        column, a dictionary to store the results, and an optional verbosity flag. It 
        calculates the error rates as percentages of the total patient count for each error 
        type ('Over', 'Under', 'Critical') and adds them to the `row_dict` dictionary with keys 
        'over_triage_error_rate', 'under_triage_error_rate', and 
        'critical_triage_error_rate'. If the total patient count is zero, the error rate is 
        set to NaN.
        
        Parameters:
            groupby_df (pandas.DataFrame):
                DataFrame with columns 'error_type' and 'patient_count' to calculate error rates from.
            row_dict (dict):
                Dictionary to which the dataframe row of calculated error rates will be added.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            dict
                The updated row dictionary with the added error rate columns.
        
        Raises:
            AssertionError
                If the 'groupby_df' is missing any of the required columns ('error_type', 'patient_count').
        """
        
        # Check if required columns are present
        needed_columns = set(['error_type', 'patient_count'])
        all_columns = set(groupby_df.columns)
        assert needed_columns.issubset(all_columns), f"groupby_df is missing these columns: {needed_columns.difference(all_columns)}"
        
        # Calculate total patient count and error type counts
        df = groupby_df.groupby('error_type').patient_count.sum().reset_index(drop=False)
        total_patient_count = df.patient_count.sum()
        
        # Convert the result to a dictionary for easier lookup
        error_dict = df.set_index('error_type').patient_count.to_dict()
        
        # Calculate over-triage error rate
        over_patient_count = error_dict.get('Over', 0)
        try: over_triage_error_rate = round(100*over_patient_count/total_patient_count, 1)
        except ZeroDivisionError: over_triage_error_rate = nan
        row_dict['over_triage_error_rate'] = over_triage_error_rate
        
        # Calculate under-triage error rate
        under_patient_count = error_dict.get('Under', 0)
        try: under_triage_error_rate = round(100*under_patient_count/total_patient_count, 1)
        except ZeroDivisionError: under_triage_error_rate = nan
        row_dict['under_triage_error_rate'] = under_triage_error_rate
        
        # Calculate critical-triage error rate
        critical_patient_count = error_dict.get('Critical', 0)
        try: critical_triage_error_rate = round(100*critical_patient_count/total_patient_count, 1)
        except ZeroDivisionError: critical_triage_error_rate = nan
        row_dict['critical_triage_error_rate'] = critical_triage_error_rate
        
        if verbose: print(
            f"over-triage error rate {over_triage_error_rate:.1f}%, under-triage error rate {under_triage_error_rate:.1f}%,"
            f" critical-triage error rate {critical_triage_error_rate:.1f}%"
        )
        
        return row_dict
    
    
    def create_triage_error_rates_dataframe(self, error_types_df, groupby_columns, verbose=False):
        """
        Create a DataFrame containing triage error rates calculated from an error types 
        DataFrame.
        
        This function iterates through groups in the `error_types_df` defined by the 
        `groupby_columns` and calculates triage error rates for each group. The calculated 
        rates are then stored in a new DataFrame with columns from `groupby_columns` and 
        additional columns containing the error rates.
        
        Parameters:
            error_types_df (pandas.DataFrame):
                The DataFrame containing error types data.
            groupby_columns (list or str):
                Column(s) to group by when calculating triage error rates.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
        
        Returns:
            pandas.DataFrame
                A new DataFrame containing triage error rates for each group.
        
        Raises:
            AssertionError
                If any of the columns specified in groupby_columns are not in error_types_df.
        """
        
        # Ensure groupby_columns is a list
        if not isinstance(groupby_columns, list): groupby_columns = [groupby_columns]
        
        # Ensure all needed columns are present in error_types_df
        needed_columns = set(['patient_count', 'error_type'] + groupby_columns)
        all_columns = set(error_types_df.columns)
        assert needed_columns.issubset(all_columns), f"You're missing {needed_columns.difference(all_columns)} from error_types_df"

        
        # Group by the specified columns and process each group
        rows_list = []
        for groupby_tuple, groupby_df in error_types_df.groupby(groupby_columns):
            
            # Create a row dictionary for the current group
            row_dict = {column_name: column_value for column_name, column_value in zip(groupby_columns, groupby_tuple)}
        
            if verbose: print(f"{groupby_tuple}: ", end='')
            
            # Add error rate columns to the row dictionary
            row_dict = self.add_triage_error_rate_columns_to_row(groupby_df, row_dict, verbose=verbose)
            rows_list.append(row_dict)
        
        # Create a DataFrame from the list of rows
        triage_error_rates_df = DataFrame(rows_list)
        
        return triage_error_rates_df

# print(r'\b(' + '|'.join(dir()) + r')\b')