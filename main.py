import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

from shapely.geometry import Polygon, Point
from haversine import haversine

import time
import argparse

def safe_division(a,b):
    try:
        result = a/b
    except ZeroDivisionError:
        result = 0
    return result

## -------------------- create useful dictionaries + tools
# asset cat dictionary
asset_cat_dict = dict(FREE=1, AUC=2, CONF=3)

# intend dictionary - relative to different intend
suppress_int = dict(Suppress=1, Neutralize=2, Destroy=3)
neutralize_int = dict(Neutralize=1, Destroy=2, Suppress=3)
destroy_int = dict(Destroy=1, Neutralize=2, Suppress=3)
intend_dict = {
    'Suppress':suppress_int,
    'Neutralize':neutralize_int,
    'Destroy':destroy_int
}

# deploy count dictionary
dep_count_dict = {
    0: 1,
    1: 2,
    2: 3
}

# intend rank dictionary - relative to different intend
suppress_rank = {1:1, 2:2, 3:3}
neutralize_rank = {1:2, 2:3, 3:1}
destroy_rank = {1:3, 2:2, 3:1}
intend_rank_dict = {
    'Suppress':suppress_rank,
    'Neutralize':neutralize_rank,
    'Destroy':destroy_rank
}

## -------------------- create relevant dataframes
class RB:
    def __init__(self, excel_filename):
        self.excel_filename = excel_filename
        self.df_hptl = pd.read_excel(self.excel_filename, sheet_name='High Payoff Target List')
        self.df_asset_ml = pd.read_excel(self.excel_filename, sheet_name='Asset Master List').drop([13,14], axis=0)
        self.df_asset_c = pd.read_excel(self.excel_filename, sheet_name='Asset Capability Table').drop([1], axis=0).reset_index(drop=True)
        self.df_sector = pd.read_excel(self.excel_filename, sheet_name='Sector').iloc[:3, :]
        self.df_target_sel = pd.read_excel(self.excel_filename, sheet_name='Target Selection Standards')
        self.sectors = dict()
        self.target_unit_list = list()
        self.mm = MinMaxScaler(feature_range=(1, 3))
        self.status_check = self.df_hptl['Status'].unique()

    def update_df(self):
        # 1. High Payoff Target List
        self.df_hptl.set_index('Target Unit', inplace=True)
        self.target_unit_list = list(self.df_hptl.index)

        # 2. Asset Master List
        self.df_asset_ml.set_index('Unit', inplace=True)
        self.df_asset_ml['Sector_A'] = self.df_asset_ml['Coverage'].apply(lambda x: 1 if 'A' in x else 0)
        self.df_asset_ml['Sector_B'] = self.df_asset_ml['Coverage'].apply(lambda x: 1 if 'B' in x else 0)
        self.df_asset_ml['Sector_C'] = self.df_asset_ml['Coverage'].apply(lambda x: 1 if 'C' in x else 0)
        
        # 3. Asset Capability Table
        self.df_asset_c.fillna(0, inplace=True)
        self.df_asset_c.set_index('Category', inplace=True)

        # 4. Sectors
        for i in range(len(self.df_sector)):
            all_points = list()
            for j in range(1,9,2):
                all_points.append(list(self.df_sector.values[i, j:j+2]))
            self.sectors[self.df_sector.values[i, 0]] = Polygon(all_points)

        # 5. Target Selection Standards
        self.df_target_sel.set_index('Category', inplace=True)

        return

def get_weight_dict(intent_w, cmdsup_w, depcount_w, frange_w, derivet_w):
    weight_dict = {
        'Intend':intent_w,
        'CMD/SUP':cmdsup_w,
        'DeployCount':depcount_w,
        'FRange':frange_w,
        'DerivedTime':derivet_w
        # 'AssetCat': assetcat_w
    }

    return weight_dict


def run(rb, weight_dict):
    
    warning_dict = dict()
    unit_deployed = list()

    # update organic asset deployment count
    organic_assets = rb.df_asset_ml.loc[rb.df_asset_ml['CMD/SUP']=='Organic'].index
    organic_dep_count = dict(zip(organic_assets, [0 for _ in range(len(organic_assets))])) # initialize

    df_unit_info = rb.df_asset_ml.copy(deep=True)
    df_unit_info['CMD/SUP'] = df_unit_info['CMD/SUP'].apply(lambda x: 1 if x=='Organic' else 2)

    for i, (idx,row) in enumerate(rb.df_hptl.iterrows()):
        print(f'target_unit {i+1}: {idx}')
        if len(df_unit_info) == 0:
            t_unit_index = rb.target_unit_list.index(idx)
            for _ in rb.target_unit_list[t_unit_index:]:
                unit_deployed.append('NS')
            print('NO SOLUTION: no assets left!')
            print()
            break
        else:
            # remove organic deployment > 3
            df_unit_info.loc[:, 'DeployCount'] = 0
            for unit, c in organic_dep_count.items():
                if c < 3:
                    df_unit_info.loc[unit, 'DeployCount'] = c
                else:
                    try:
                        df_unit_info = df_unit_info.drop(unit, axis=0)
                    except KeyError: # already dropped
                        pass
            df_unit_info.loc[:,'DeployCount'] = df_unit_info.loc[:,'DeployCount'].apply(lambda x: dep_count_dict[x])

            ## ---------- coverage
            acc_point = Point([row['Latitude'], row['Longitude']])
            acc_coverage = [sector for sector,poly in rb.sectors.items() if acc_point.within(poly)]
            print(f'coverage includes sectors: {acc_coverage}')
            idx_list = list()
            for c in acc_coverage:
                idx_list += list(df_unit_info.loc[df_unit_info[f'Sector_{c}']>0].index)
            idx_list = list(set(idx_list))
            df_unit_sub = df_unit_info.loc[idx_list,:]
            if len(df_unit_sub)== 0:
                selected_unit = 'NS'
                unit_deployed.append(selected_unit)
                print(f'NO SOLUTION: no assets in coverage!')
                print()
                continue
            else:
                ## ---------- basic info
                acc_cat = row['Category']
                print(f'incident category: {acc_cat}')
                right_intend = row['Intend']
                print(f'incident right intend: {right_intend}')
                acc_location = (row['Latitude'], row['Longitude'])
                print(f'incident location: {acc_location}')
                acc_timeliness = rb.df_target_sel.loc[acc_cat, 'Timeliness (Mins)']
                print(f'incident timeliness: {acc_timeliness}mins')

                ## ---------- FRange related
                df_unit_sub['Distance'] = df_unit_sub.apply(lambda x: haversine(acc_location, (x['Latitude'],x['Longitude'])), axis=1) # dist of incident to asset location
                print(f"out of range units: {list(df_unit_sub[df_unit_sub['Distance']>df_unit_sub['Effective Radius (km)']].index)}")
                df_unit_sub = df_unit_sub.loc[df_unit_sub['Distance']<df_unit_sub['Effective Radius (km)']] # filter out assets within range only
                if len(df_unit_sub)==0:
                    print('NO SOLUTION: all assets out of range!')
                    selected_unit = 'NS'
                    unit_deployed.append(selected_unit)
                else:
                    df_unit_sub.loc[:, 'FRange'] = rb.mm.fit_transform(np.array(df_unit_sub['Effective Radius (km)'].values.reshape(-1, 1))) # scaling of range

                    ## ----------- time related
                    df_unit_sub.loc[:, 'Time (m)'] = df_unit_sub.apply(lambda x: safe_division(x['Distance'],x['Speed (Km/h)'])*60, axis=1) # in minutes
                    exceed_t_temp = df_unit_sub.loc[df_unit_sub['Time (m)'] > acc_timeliness, :]
                    if len(exceed_t_temp) > 0: # assets more than timeliness limit allowed
                        time_scaler = MinMaxScaler(feature_range=(2,3))
                        df_unit_sub.loc[exceed_t_temp.index, "DerivedTime"] = time_scaler.fit_transform(np.array(exceed_t_temp['Time (m)'].values.reshape(-1, 1)))
                        df_unit_sub.loc[~df_unit_sub.index.isin(exceed_t_temp.index), "DerivedTime"] = float(1)
                    else:
                        df_unit_sub.loc[:, 'DerivedTime'] = float(1)
                    
                    ## ---------- intend related
                    right_intend_rank= intend_dict[right_intend][right_intend]
                    df_unit_sub.loc[:, 'Intend'] = df_unit_sub.apply(lambda x: rb.df_asset_c.loc[acc_cat, x['Asset Type']], axis=1)
                    df_unit_sub.loc[:, "Intend"] =  df_unit_sub.loc[:, "Intend"].apply(lambda x: intend_dict[right_intend][x])
                    df_unit_sub = df_unit_sub.drop(columns=['Qty','Configuration','Effective Radius (km)','Coverage','Latitude','Longitude','Speed (Km/h)'])

                    df_temp = df_unit_sub.copy(deep=True)
                    for item, penalty in weight_dict.items():
                        df_temp.loc[:, item] = df_temp.loc[:, item].apply(lambda x: x*penalty)
                        df_unit_sub.loc[:, 'score'] = df_temp['Intend']+df_temp['CMD/SUP']+df_temp['FRange']+df_temp['DerivedTime']+df_temp['DeployCount']
                    sorted_df = df_unit_sub.sort_values(by=['score'])
                    print(sorted_df)

                    score_list = sorted(list(sorted_df['score'].unique()))
                    min_score = score_list.pop(0)
                    sub_asset_list = list(sorted_df.loc[sorted_df['score']==min_score,:].index)
                    selected_unit = random.sample(sub_asset_list, 1)[0]
                    print(f"initial selected unit: {selected_unit}")
                    
                    sub_asset_list.remove(selected_unit)

                    if rb.status_check != 'Unknown': # detect phase
                        unit_deployed.append(selected_unit)
                        df_unit_info.drop([selected_unit], inplace=True)
                        print(f'detect phase selected unit: {selected_unit}')
                    else: # decide phase
                        organic_check = df_unit_info.loc[selected_unit, "CMD/SUP"]
                        if organic_check == 2: # allocated asset
                            df_unit_info.drop([selected_unit], inplace=True)
                            print(f'decide phase selected unit - allocated: {selected_unit}')
                            try:
                                warning_dict[idx][selected_unit].append('allocated_asset')
                            except:
                                warning_dict[idx] = {f"{selected_unit}":['allocated_asset']}

                        # keep track of deployment count (only applicable to decide phase)
                        if selected_unit != 'NS':
                            try:
                                organic_dep_count[selected_unit] += 1
                            except KeyError:
                                pass

                        unit_deployed.append(selected_unit)

                    ## ---------- getting all the warnings
                    # 1. lower intent warnings
                    selected_unit_assType = rb.df_asset_ml.loc[selected_unit,'Asset Type']
                    selected_unit_intend = rb.df_asset_c.loc[acc_cat, selected_unit_assType]

                    selected_unit_rank = intend_dict[right_intend][selected_unit_intend] # assigned rank
                    selected_unit_check = intend_rank_dict[right_intend][selected_unit_rank] # checking rank

                    if selected_unit_check < intend_rank_dict[right_intend][right_intend_rank]:
                        try:
                            warning_dict[idx][selected_unit].append('intent_lowered')
                        except:
                            warning_dict[idx] = {f"{selected_unit}":['intent_lowered']}

                    # 2. timeliness warnings
                    if selected_unit in exceed_t_temp.index:
                        try:
                            warning_dict[idx][selected_unit].append('timeliness_violation')
                        except:
                            warning_dict[idx] = {f"{selected_unit}":['timeliness_violation']}
        
        print()

    return unit_deployed, warning_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./csv_files/DMS_target_sample.xlsx")
    args = parser.parse_args()

    filename = args.input_path

    rb = RB(filename)
    rb.update_df()

    weight_v1 = get_weight_dict(intent_w=1000, cmdsup_w=100, depcount_w=10, frange_w=1, derivet_w=0.1)
    weight_v2 = get_weight_dict(intent_w=1000, cmdsup_w=10, depcount_w=100, frange_w=1, derivet_w=0.1)

    start_time = time.time()

    solv1, warning1 = run(rb, weight_v1)
    solv2, warning2 = run(rb, weight_v2)
    solv1, warning1 = run(rb, weight_v1)

    end_time = time.time()

    print(f'Total runtime: {round(end_time-start_time, 4)}s')
    print(f'sol_v1: \n{solv1}')
    if len(warning1)>0:
        print(f'sol_v1 warning: \n{warning1}')
    
    print(f'\nsol_v2: \n{solv2}')
    if len(warning2)>0:
        print(f'sol_v2 warning: \n{warning2}')