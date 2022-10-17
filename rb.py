import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import json
import random
from sklearn.preprocessing import MinMaxScaler

from shapely.geometry import Polygon, Point
from haversine import haversine

def safe_division(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        result = 0
    return result


## -------------------- create useful dictionaries + tools
# intend dictionary - relative to different intend
suppress_int = dict(Suppress=1, Neutralize=2, Destroy=3)
neutralize_int = dict(Neutralize=1, Destroy=2, Suppress=3)
destroy_int = dict(Destroy=1, Neutralize=2, Suppress=3)
intend_dict = {
    "Suppress": suppress_int,
    "Neutralize": neutralize_int,
    "Destroy": destroy_int,
}

# intend rank dictionary - relative to different intend
suppress_rank = {1: 1, 2: 2, 3: 3}
neutralize_rank = {1: 2, 2: 3, 3: 1}
destroy_rank = {1: 3, 2: 2, 3: 1}
intend_rank_dict = {
    "Suppress": suppress_rank,
    "Neutralize": neutralize_rank,
    "Destroy": destroy_rank,
}

mm = MinMaxScaler(feature_range=(1, 3))
twothree_scaler = MinMaxScaler(feature_range=(2, 3))

## -------------------- create relevant dataframes
class RB:
    def __init__(self, excel_filename):
        self.excel_filename = excel_filename
        self.df_hptl = pd.read_excel(self.excel_filename, sheet_name="High Payoff Target List").fillna(0)
        self.df_asset_ml = pd.read_excel(self.excel_filename, sheet_name="Asset Master List")
        self.df_asset_c = pd.read_excel(self.excel_filename, sheet_name="Asset Capability Table")
        self.df_sector = pd.read_excel(self.excel_filename, sheet_name="Sector")
        self.df_target_sel = pd.read_excel(self.excel_filename, sheet_name="Target Selection Standards")
        self.sectors = dict()
        self.target_unit_list = list()
        self.status_flag = 0 # default: decide phase

    def tidy_up(self):

        if "Detected" in self.df_hptl["Status"].unique():
            self.status_flag = 1

        # 1. High Payoff Target List
        self.df_hptl.set_index("Target Designation", inplace=True)
        self.df_hptl.drop(columns=['Decide Phase Ans.'], inplace=True) # temporary, this column does not exist in real case
        self.target_unit_list = list(self.df_hptl.index)

        # 2. Asset Master List
        self.df_asset_ml.set_index("Unit", inplace=True)
        self.df_asset_ml["CMD/SUP"] = self.df_asset_ml["CMD/SUP"].apply(lambda x: 1 if x == "Organic" else 2)
        self.df_asset_ml.drop(self.df_asset_ml.loc[self.df_asset_ml['Assignment']=='RESERVED'].index, inplace=True) # drop
        self.df_asset_ml['Assignment'] = self.df_asset_ml['Assignment'].apply(lambda x: 1 if x=='FREE' else 2 if x=='AUC' else 3)
        self.df_asset_ml["Sector_A"] = self.df_asset_ml["Coverage"].apply(lambda x: 1 if "A" in x else 0)
        self.df_asset_ml["Sector_B"] = self.df_asset_ml["Coverage"].apply(lambda x: 1 if "B" in x else 0)
        self.df_asset_ml["Sector_C"] = self.df_asset_ml["Coverage"].apply(lambda x: 1 if "C" in x else 0)
        

        # 3. Asset Capability Table
        self.df_asset_c.dropna(inplace=True)
        self.df_asset_c.set_index("Category", inplace=True)

        # 4. Sectors
        for i in range(len(self.df_sector)):
            all_points = list()
            for j in range(1, 9, 2):
                all_points.append(list(self.df_sector.values[i, j : j + 2]))
            self.sectors[self.df_sector.values[i, 0]] = Polygon(all_points)

        # 5. Target Selection Standards
        self.df_target_sel.set_index("Category", inplace=True)

        return


def get_weight_dict(json_input):
    with open(json_input, "r") as f:
        weight_dict = json.loads(f.read())
    
    return weight_dict


def run(rb_case, weight_dict, view=True):

    warning_dict = dict()
    output = dict()
    
    ## ---------- get all assets deploy count
    asset_unit_list = rb_case.df_asset_ml.index
    asset_dep_count = dict(
        zip(asset_unit_list, [0 for _ in range(len(asset_unit_list))])
    )

    df_hptl_md = rb_case.df_hptl.copy(deep=True)
    if rb_case.status_flag == 1: # detect phase
        df_hptl_md = df_hptl_md.loc[(df_hptl_md['Status']=='Detected') & (df_hptl_md['How']==0)]
        target_unit_taken = dict()
        rb_case.target_unit_list = list(df_hptl_md.index)
    
    for i, (idx, row) in enumerate(df_hptl_md.iterrows()):
        print(f"target_unit {i+1}: {idx}")
        ## ---------- get basic info of the target unit
        acc_cat = row["Category"]
        print(f"incident category: {acc_cat}")
        right_intend = row["Intend"]
        print(f"incident right intend: {right_intend}")
        acc_location = (row["Latitude"], row["Longitude"])
        print(f"incident location: {acc_location}")
        acc_timeliness = rb_case.df_target_sel.at[acc_cat, "Timeliness (Mins)"] * 60 # convert to seconds
        print(f"incident timeliness: {acc_timeliness} secs")
        acc_priority = row["Priority"]
        print(f"incident priority: {acc_priority}")
        acc_t_sensitive = row["Time-Sensitive"]
        print(f"incident t sensitive: {acc_t_sensitive}")

        df_unit_info = rb_case.df_asset_ml.copy(deep=True)
        df_unit_info.loc[:, "DeployCount"] = list(asset_dep_count.values())
        ## ---------- filter asset solution space based on count deployed - wrt decide and detect phase
        # decide phase - each organic asset can be deployed up to 3 targets
        if rb_case.status_flag == 0:
            organic_units = df_unit_info.loc[((df_unit_info['DeployCount']<3) & (df_unit_info['CMD/SUP']==1)), :]
            allocated_units = df_unit_info.loc[((df_unit_info['DeployCount']<1) & (df_unit_info['CMD/SUP']==2)), :]
            df_unit_info = pd.concat([organic_units, allocated_units])
        # detect phase - each asset can only be deployed up ONCE
        else:
            df_unit_info = df_unit_info.loc[df_unit_info['DeployCount']<1, :]
            ## ----------- time sensitive related
            if acc_t_sensitive == 0:
                df_unit_info = df_unit_info.loc[df_unit_info['Assignment'] < 3, :]
        
        if len(df_unit_info) == 0:
            t_unit_index = rb_case.target_unit_list.index(idx)
            for r in rb_case.target_unit_list[t_unit_index:]:
                output[r] = "NS-NA"
                warning_dict[r] = "NS-NA"
            print("NO SOLUTION: no assets left!")
            print()
            break
        else:
            # reassigning asset deploy count to value of 1 to 3
            df_unit_info.loc[:, "DeployCount"] = df_unit_info.loc[:, "DeployCount"].apply(
                lambda x: 1 if x==0 else 2 if x==1 else 3
            )
            # target location - check target falls in which sector
            acc_point = Point([row["Latitude"], row["Longitude"]])
            acc_coverage = [
                sector for sector, poly in rb_case.sectors.items() if acc_point.within(poly)
            ]
            print(f"coverage includes sectors: {acc_coverage}")
            # get all assets within the sectors involved
            idx_list = list()
            for c in acc_coverage:
                idx_list += list(
                    df_unit_info.loc[df_unit_info[f"Sector_{c}"] > 0].index
                )
            idx_list = list(set(idx_list))
            df_unit_sub = df_unit_info.loc[idx_list, :]

            if len(df_unit_sub) == 0:
                selected_unit = "NS-NC"
                output[idx] = selected_unit
                warning_dict[idx] = "NS-NC"
                print(f"NO SOLUTION: no assets in coverage!")
                print()
                continue
            else:
                ## ---------- FRange related
                # dist of incident to asset location
                df_unit_sub["Distance"] = df_unit_sub.apply(lambda x: haversine(acc_location, (x["Latitude"], x["Longitude"])),axis=1,)
                print(f"out of range units: {list(df_unit_sub[df_unit_sub['Distance']>df_unit_sub['Effective Radius (km)']].index)}")
                # filter out assets within range only
                df_unit_sub = df_unit_sub.loc[df_unit_sub["Distance"] < df_unit_sub["Effective Radius (km)"]]
                # if no asset within range, continue to the next target unit
                if len(df_unit_sub) == 0:
                    selected_unit = "NS-OR"
                    output[idx] = selected_unit
                    warning_dict[idx] = "NS-OR"
                    print("NO SOLUTION: all assets out of range!")
                    print()
                    continue
                else:
                    # scaling of range
                    df_unit_sub.loc[:, "FRange"] = mm.fit_transform(
                        np.array(
                            df_unit_sub["Effective Radius (km)"].values.reshape(-1, 1)
                        )
                    )

                    ## ----------- time related
                    df_unit_sub.loc[:, "Time (s)"] = df_unit_sub.apply(lambda x: safe_division(x["Distance"], x["Speed (Km/h)"]) * 3600, axis=1,)  # in seconds
                    exceed_t_temp = df_unit_sub.loc[df_unit_sub["Time (s)"] > acc_timeliness, :]
                    # if assets beyond timeliness, will be scaled to value ranging from 2-3
                    if (len(exceed_t_temp) > 0):
                        timeliness_flag = 1
                        df_unit_sub.loc[
                            exceed_t_temp.index, "DerivedTime"
                        ] = twothree_scaler.fit_transform(
                            np.array(exceed_t_temp["Time (s)"].values.reshape(-1, 1))
                        )
                        df_unit_sub.loc[
                            ~df_unit_sub.index.isin(exceed_t_temp.index), "DerivedTime"
                        ] = float(1)
                    # else all assets are within timeliness - all assets derived time is set to 1 (no preference)
                    else:
                        timeliness_flag = 0
                        df_unit_sub.loc[:, "DerivedTime"] = float(1)

                    ## ---------- status related
                    conf_asset_list = df_unit_sub.loc[df_unit_sub['Status']>0, :].index
                    if len(conf_asset_list)>0:
                        df_unit_sub.loc[conf_asset_list, 'Status'] = twothree_scaler.fit_transform(df_unit_sub.loc[conf_asset_list, 'Status'].values.reshape(-1,1))
                    df_unit_sub.loc[df_unit_sub['Status']==0, 'Status'] = 1
                    
                    ## ---------- intend related
                    right_intend_rank = intend_dict[right_intend][right_intend]
                    df_unit_sub.loc[:, "Intend"] = df_unit_sub.apply(lambda x: rb_case.df_asset_c.loc[acc_cat, x["Asset Type"]], axis=1)
                    df_unit_sub.loc[:, "Intend"] = df_unit_sub.loc[:, "Intend"].apply(lambda x: intend_dict[right_intend][x])
                    df_unit_sub = df_unit_sub.drop(
                        columns=[
                            "Qty",
                            "Configuration",
                            "Effective Radius (km)",
                            "Coverage",
                            "Latitude",
                            "Longitude",
                            "Speed (Km/h)",
                        ]
                    )

                    ## ---------- get the score for each asset
                    df_temp = pd.DataFrame()
                    for item, weight in weight_dict.items():
                        df_temp.loc[:, item] = df_unit_sub.loc[:, item].apply(
                            lambda x: x * weight
                        )
                    df_unit_sub.loc[:, "score"] = df_temp.sum(axis=1)
                    sorted_df = df_unit_sub.sort_values(by=["score"])
                    if view:
                        print(sorted_df)

                    ## ---------- Decide Phase asset assignment ---------- ##
                    if rb_case.status_flag ==0:
                        score_list = sorted(list(sorted_df["score"].unique()))
                        min_score = score_list.pop(0)
                        sub_asset_list = list(sorted_df.loc[sorted_df["score"] == min_score, :].index)
                        selected_unit = random.sample(sub_asset_list, 1)[0]
                    
                    ## ---------- Detect Phase asset assignment - V2 ---------- ##
                    else:
                        df_hptl_below = rb_case.df_hptl.loc[
                            (rb_case.df_hptl['How'].isin(sorted_df.index)) &
                            (rb_case.df_hptl['Priority']>acc_priority), :]
                        if len(df_hptl_below) <= 0: # all possible solutions are above current rank
                            print('All possible solution is above current priority')
                            # check if there's FREE assets available in sorted df -- above: irregardless of other criteria
                            slice_free = sorted_df.loc[sorted_df['Assignment']==1,:]
                            if len(slice_free) > 0:
                                selected_unit = slice_free.index[0]
                                selected_target_unit = 'FA'
                            else:
                                df_hptl_above = rb_case.df_hptl.loc[
                                    (rb_case.df_hptl['How'].isin(sorted_df.index)), :]
                                df_hptl_above = df_hptl_above.sort_values(by=['Priority'])
                                print(df_hptl_above)
                                selected_unit = df_hptl_above.iloc[-1,-1]
                                # get the target unit where the AUC asset is deployed from
                                selected_target_unit = df_hptl_above.loc[df_hptl_above['How']==selected_unit, :].index[-1]
                        else: # there are assets below current rank
                            asset_below = set(df_hptl_below.loc[:, 'How'])
                            print(f'There are possible solutions below current priority: {asset_below}')
                            print(f'All units in current solution space: {sorted_df.index}')
                            for unit in sorted_df.index:
                                if sorted_df.loc[unit,'Assignment']==1: # free asset available as next best option
                                    selected_unit = unit
                                    selected_target_unit = 'FA'
                                    break
                                elif unit in asset_below:
                                    selected_unit = unit
                                    selected_target_unit = df_hptl_below.loc[df_hptl_below['How']==selected_unit, :].index[-1]
                                    break
                        target_unit_taken[idx] = selected_target_unit
                    
                    if not selected_unit.startswith("NS"):
                        asset_dep_count[selected_unit] += 1

                    ## ---------- get all the warnings
                    # 1. allocated asset warning
                    organic_check = df_unit_info.at[selected_unit, "CMD/SUP"]
                    if organic_check == 2:  # allocated asset
                        try:
                            warning_dict[idx][selected_unit].append("allocated_asset")
                        except:
                            warning_dict[idx] = {f"{selected_unit}": ["allocated_asset"]}

                    # 2. lower intent warning
                    selected_unit_assType = rb_case.df_asset_ml.at[selected_unit, "Asset Type"]
                    selected_unit_intend = rb_case.df_asset_c.at[acc_cat, selected_unit_assType]

                    selected_unit_rank = intend_dict[right_intend][selected_unit_intend]  # assigned rank
                    selected_unit_check = intend_rank_dict[right_intend][selected_unit_rank]  # checking rank

                    if (selected_unit_check< intend_rank_dict[right_intend][right_intend_rank]):
                        try:
                            warning_dict[idx][selected_unit].append("intent_lowered")
                        except:
                            warning_dict[idx] = {f"{selected_unit}": ["intent_lowered"]}

                    # 3. timeliness warning
                    if timeliness_flag == 1 and selected_unit in exceed_t_temp.index:
                        try:
                            warning_dict[idx][selected_unit].append("timeliness_violation")
                        except:
                            warning_dict[idx] = {f"{selected_unit}": ["timeliness_violation"]}

        # output.append(selected_unit)
        output[idx] = selected_unit
        print(f"Selected Unit: {selected_unit}")
        if rb_case.status_flag == 1:
            print(f"Taken from: {selected_target_unit}")
        print()

    final_output = dict()
    final_output['unit_deployed'] = output
    final_output['warning'] = warning_dict
    if rb_case.status_flag == 1:
        final_output['target_unit_taken'] = target_unit_taken

    return final_output