import json
import numpy as np
import pandas as pd

import random
from sklearn.preprocessing import MinMaxScaler
# from collections import Counter

import time

## -------------------- create useful dictionaries
# asset cat dictionary
asset_cat_dict = dict(FREE=1, AUC=2, CONF=3)
# intent dictionary - relative to different intent
suppress_int = dict(suppress=1, neutralize=2, destroy=3)
neutralize_int = dict(neutralize=1, destroy=2, suppress=3)
destroy_int = dict(destroy=1, neutralize=2, suppress=3)
intent_dict = {
    'suppress':suppress_int,
    'neutralize':neutralize_int,
    'destroy':destroy_int
}
# intent rank dictionary - relative to different intent
suppress_rank = {1:1, 2:2, 3:3}
neutralize_rank = {1:2, 2:3, 3:1}
destroy_rank = {1:3, 2:2, 3:1}
intent_rank_dict = {
    'suppress':suppress_rank,
    'neutralize':neutralize_rank,
    'destroy':destroy_rank
}
# weight dictionary - across column
weight_dict = dict(
    intent=100,
    cmdsup = 10,
    derived_time = 1,
    frange = 0.1,
    asset_cat = 0.01
)

## -------------------- create relevant dataframes

class RB:
    def __init__(self, acc_info, unit_info, time_info):
        self.acc_info = acc_info
        self.unit_info = unit_info
        self.time_info = time_info
        self.df_hptl = pd.DataFrame(self.acc_info).T
        self.df_unit_master = pd.DataFrame(self.unit_info).T
        self.df_time = pd.DataFrame(self.time_info)
        self.df_cap = pd.DataFrame()
        self.status_check = 0 # decide phase

    def update_df(self):
        # HTPL input
        self.df_hptl.loc[:, "status"] =  self.df_hptl.loc[:, "status"].apply(lambda x: 0 if x=='decide' else 1)
        self.status_check = int(self.df_hptl['status'].unique())

        # unit info input
        self.df_unit_master.loc[:, "asset_cat"] =  self.df_unit_master.loc[:, "asset_cat"].apply(lambda x: asset_cat_dict[x])

        mm = MinMaxScaler(feature_range=(1, 3)) # standardize across all columns
        self.df_unit_master['frange'] = mm.fit_transform(np.array(self.df_unit_master['frange'].values.reshape(-1, 1)))
        self.df_unit_master['frange'] = self.df_unit_master['frange'].apply(lambda x: round(x,3))

        # tune info input
        self.df_time = pd.DataFrame(mm.fit_transform(self.df_time), index=self.df_time.index, columns=self.df_time.columns)

        # capability dataframe
        self.df_cap['category'] = self.df_hptl['category']
        for k, v in self.acc_info.items():
            temp_d = v['weapon_target']
            for intent, assetType in temp_d.items():
                for i in assetType:
                    self.df_cap.loc[k, assetType] = intent
        self.df_cap.fillna(0, inplace=True)
        return self.df_hptl, self.df_unit_master, self.df_time, self.df_cap

    ## -------------------- main
    def main(self):
        # get a copy of unit_info dataframe:
        df_unit_info = self.df_unit_master.copy(deep=True)
        unit_deployed = list()
        unit_deployed_count = dict()
        warning_dict = dict()

        asset_list = self.df_cap.columns[1:]
        prev_unit = 'NIL'
        # starttime = time.time()

        for idx, row in self.df_cap.iterrows():
            if len(df_unit_info)==0: # if no more assets left, break out of the loop
                self.target_unit_list = list(self.df_hptl.index)
                t_unit_index = self.target_unit_list.index(idx)
                for _ in self.target_unit_list[t_unit_index:]:
                    unit_deployed.append('NS')
                print('breaking here!')
                break
            else:
                # print(f'target-unit: {idx}')
                dummy = pd.DataFrame()
                for asset in asset_list:
                    if row[asset] != 0:
                        df_sub = df_unit_info[df_unit_info['asset_type']==asset] # the df
                        if len(df_sub)==0:
                            continue
                        else:
                            sub_unit_list = list(df_sub.index)
                            df_sub.loc[:, 'intent'] = row[asset]
                            df_sub.loc[:, 'target_unit'] = idx
                            df_sub.loc[:, 'derived_time'] = self.df_time.loc[sub_unit_list, idx]
                            dummy = pd.concat([dummy, df_sub])
                if len(dummy)==0:
                    # no assets available for this target
                    # print('no suitable asset')
                    selected_unit = 'NS'
                    unit_deployed.append(selected_unit)
                else:
                    right_intent = self.df_hptl.loc[idx,'intent'] # suppress, neutralize, destroy
                    # print(f'right intent: {right_intent}')
                    right_intent_rank = intent_dict[right_intent][right_intent]
                    dummy.loc[:, "intent"] =  dummy.loc[:, "intent"].apply(lambda x: intent_dict[right_intent][x])

                    df_temp = dummy.copy(deep=True)
                    for item, penalty in weight_dict.items():
                        df_temp.loc[:, item] = df_temp.loc[:, item].apply(lambda x: x*penalty)
                        dummy.loc[:, 'score'] = df_temp['intent']+df_temp['cmdsup']+df_temp['frange']+df_temp['derived_time']+df_temp['asset_cat']
                    sorted_df = dummy.sort_values(by=['score'])
                    # print(sorted_df)
                    if len(sorted_df) == 0:
                        unit_deployed.append('NS')
                    else:
                        score_list = sorted(list(sorted_df['score'].unique()))
                        min_score = score_list.pop(0)
                        sub_asset_list = list(sorted_df[sorted_df['score']==min_score].index)
                        selected_unit = random.sample(sub_asset_list, 1)[0]
                        # print(f"initial selected unit: {selected_unit}")
                        sub_asset_list.remove(selected_unit)
                        if self.status_check==1: # detect phase
                            unit_deployed.append(selected_unit)
                            df_unit_info.drop([selected_unit], inplace=True)
                            # print(f'detect phase selected unit: {selected_unit}')
                        else: # decide phase
                            # check if selected unit is organic assets
                            organic_check = self.df_unit_master.loc[selected_unit, "cmdsup"]
                            if organic_check == 1 and selected_unit == prev_unit: # organic + consecutive deployment
                                if len(sub_asset_list) == 0 and len(score_list)==0:
                                    # end of the list/filtered dataframe
                                    # print(f'no other assets available: consecutive deployment of {selected_unit}')
                                    pass
                                elif len(sub_asset_list) == 0 and len(score_list)>0:
                                    min_score2 = score_list.pop(0) # next best score
                                    intent_check2 = int(sorted_df[sorted_df['score']==min_score2]['intent'].unique())
                                    intent_check2_rank = intent_rank_dict[right_intent][intent_check2]
                                    organic_check2 = int(sorted_df[sorted_df['score']==min_score2]['cmdsup'].unique())
                                    if organic_check2 == 1 and intent_check2_rank>=right_intent_rank: # assets with same or higher intent
                                        sub_asset_list2 = list(sorted_df[sorted_df['score']==min_score2].index)
                                        selected_unit = random.sample(sub_asset_list2, 1)[0]
                                elif len(sub_asset_list) == 1:
                                    selected_unit = sub_asset_list[0]
                                elif len(sub_asset_list) >= 2:
                                    selected_unit = random.sample(sub_asset_list, 1)[0]
                                # print(f'decide phase selected unit - second org choice: {selected_unit}')
                            elif organic_check == 1 and selected_unit != prev_unit:
                                # good to deploy
                                # print(f'decide phase selected unit - first org choice: {selected_unit}')
                                pass
                            else: # allocated assets - deploying only once
                                df_unit_info.drop([selected_unit], inplace=True)
                                # print(f'decide phase selected unit - allocated: {selected_unit}')
                                try:
                                    warning_dict[idx][selected_unit].append('allocated_asset')
                                except:
                                    warning_dict[idx] = {f"{selected_unit}":['allocated_asset']}
                            
                            # keep track of deployment count -- decide phase
                            if selected_unit != 'NS':
                                try:
                                    unit_deployed_count[selected_unit] += 1
                                    if unit_deployed_count[selected_unit] >= 3:
                                        df_unit_info.drop([selected_unit], inplace=True)
                                except KeyError:
                                    unit_deployed_count[selected_unit] = 1

                            # unit deployed for this target
                            unit_deployed.append(selected_unit)
                        
                        selected_unit_assType = df_unit_master.loc[selected_unit,'asset_type']
                        selected_unit_intent = df_cap.loc[idx, selected_unit_assType]
                        selected_unit_rank = intent_dict[right_intent][selected_unit_intent] # assigned rank
                        selected_unit_c = intent_rank_dict[right_intent][selected_unit_rank] # checking rank
                        if selected_unit_c < intent_rank_dict[right_intent][right_intent_rank]:
                            try:
                                warning_dict[idx][selected_unit].append('intent_lowered')
                            except:
                                warning_dict[idx] = {f"{selected_unit}":['intent_lowered']}
                        # 2. timeliness check
                        if time_info[idx][selected_unit] > acc_info[idx]['timeliness']:
                            try:
                                warning_dict[idx][selected_unit].append('timeliness_violation')
                            except:
                                warning_dict[idx] = {f"{selected_unit}":['timeliness_violation']}
                prev_unit = selected_unit
        # print(f'WARNING: {warning_dict}\n')
        return unit_deployed, warning_dict

if __name__ == "__main__":

    ## -------------------- read in JSON files
    # with open("json_files/data5_trial.json", 'r') as j: # 20 cases
    # with open("json_files/data5_100.json", 'r') as j: # 120 cases
    with open("json_files/data_500.json", 'r') as j: # 500 cases
        acc_info = json.loads(j.read())

    # with open("json_files/unit_info5_trial.json", 'r') as j: # not enough units
    with open("json_files/unit_info_100.json", 'r') as j: # 120 units
    # with open("json_files/unit_info_500.json", 'r') as j: # 500 units
    # with open("json_files/unit_info_300.json", 'r') as j: # 300 units
        unit_info = json.loads(j.read())

    # with open("json_files/time.json", 'r') as j:
    with open("json_files/time_300.json", 'r') as j: # 300 units 500 cases
    # with open("json_files/time_500.json", 'r') as j: # 500 units 500 cases
    # with open("json_files/time_same.json", 'r') as j: # same derived time accross different targets
        time_info= json.loads(j.read())

    rb_trial = RB(acc_info, unit_info, time_info)
    df_hptl, df_unit_master, df_time, df_cap = rb_trial.update_df()

    starttime = time.time()
    for i in range(3):
        output, warning_dict = rb_trial.main()
        print(f"output{i+1}: {output}")
        # print(f"counter check: {[(k,v) for k,v in Counter(output).items() if v>1]}")
        if len(warning_dict)>0:
            print(f"\n WARNING: {warning_dict}")
        print()
    endtime = time.time()
    print(f'total runtime: {round(endtime-starttime, 4)}s')