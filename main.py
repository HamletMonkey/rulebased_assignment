import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

from shapely.geometry import Polygon, Point
from haversine import haversine

import time
import argparse


def safe_division(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        result = 0
    return result


## -------------------- create useful dictionaries + tools
# asset cat dictionary - TBC
asset_cat_dict = dict(FREE=1, AUC=2, CONF=3)

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

## -------------------- create relevant dataframes
class RB:
    def __init__(self, excel_filename):
        self.excel_filename = excel_filename
        self.df_hptl = pd.read_excel(self.excel_filename, sheet_name="High Payoff Target List")
        self.df_asset_ml = pd.read_excel(self.excel_filename, sheet_name="Asset Master List").drop([13, 14], axis=0)
        self.df_asset_c = (pd.read_excel(self.excel_filename, sheet_name="Asset Capability Table").drop([1], axis=0).reset_index(drop=True))
        self.df_sector = pd.read_excel(self.excel_filename, sheet_name="Sector").iloc[:3, :]
        self.df_target_sel = pd.read_excel(self.excel_filename, sheet_name="Target Selection Standards")
        self.sectors = dict()
        self.target_unit_list = list()
        self.mm = MinMaxScaler(feature_range=(1, 3))
        self.status_check = self.df_hptl["Status"].unique()

    def update_df(self):
        # 1. High Payoff Target List
        self.df_hptl.set_index("Target Unit", inplace=True)
        self.target_unit_list = list(self.df_hptl.index)

        # 2. Asset Master List
        self.df_asset_ml.set_index("Unit", inplace=True)
        self.df_asset_ml["Sector_A"] = self.df_asset_ml["Coverage"].apply(lambda x: 1 if "A" in x else 0)
        self.df_asset_ml["Sector_B"] = self.df_asset_ml["Coverage"].apply(lambda x: 1 if "B" in x else 0)
        self.df_asset_ml["Sector_C"] = self.df_asset_ml["Coverage"].apply(lambda x: 1 if "C" in x else 0)
        self.df_asset_ml["CMD/SUP"] = self.df_asset_ml["CMD/SUP"].apply(lambda x: 1 if x == "Organic" else 2)

        # 3. Asset Capability Table
        self.df_asset_c.fillna(0, inplace=True)
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


def get_weight_dict(intent_w, cmdsup_w, depcount_w, frange_w, derivedt_w):
    weight_dict = {
        "Intend": intent_w,
        "CMD/SUP": cmdsup_w,
        "DeployCount": depcount_w,
        "FRange": frange_w,
        "DerivedTime": derivedt_w
        # 'AssetCat': assetcat_w
    }
    return weight_dict


def run(rb, weight_dict, view=True):

    warning_dict = dict()
    unit_deployed = list()

    # get all assets deploy count
    asset_unit_list = rb.df_asset_ml.index
    asset_dep_count = dict(
        zip(asset_unit_list, [0 for _ in range(len(asset_unit_list))])
    )

    for i, (idx, row) in enumerate(rb.df_hptl.iterrows()):
        print(f"target_unit {i+1}: {idx}")

        df_unit_info = rb.df_asset_ml.copy(deep=True)
        df_unit_info.loc[:, "DeployCount"] = list(asset_dep_count.values())
        ## ---------- filter asset solution space based on count deployed - wrt decide and detect phase
        # decide phase - each organic asset can be deployed up to 3 targets
        if rb.status_check == 'Unknown':
            organic_units = df_unit_info.loc[((df_unit_info['DeployCount']<3) & (df_unit_info['CMD/SUP']==1)), :]
            allocated_units = df_unit_info.loc[((df_unit_info['DeployCount']<1) & (df_unit_info['CMD/SUP']==2)), :]
            df_unit_info = pd.concat([organic_units, allocated_units])
        # detect phase - each asset can only be deployed up ONCE
        else:
            df_unit_info = df_unit_info.loc[df_unit_info['DeployCount']<1, :]

        # if no assets left, break for loop
        if len(df_unit_info) == 0:
            t_unit_index = rb.target_unit_list.index(idx)
            for _ in rb.target_unit_list[t_unit_index:]:
                unit_deployed.append("NS-NA")
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
                sector for sector, poly in rb.sectors.items() if acc_point.within(poly)
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
            # if no assets within sectors, continue to the next target unit
            if len(df_unit_sub) == 0:
                selected_unit = "NS-NC"
                unit_deployed.append(selected_unit)
                print(f"NO SOLUTION: no assets in coverage!")
                print()
                continue
            else:
                ## ---------- get basic info of the target unit
                acc_cat = row["Category"]
                print(f"incident category: {acc_cat}")
                right_intend = row["Intend"]
                print(f"incident right intend: {right_intend}")
                acc_location = (row["Latitude"], row["Longitude"])
                print(f"incident location: {acc_location}")
                acc_timeliness = rb.df_target_sel.at[acc_cat, "Timeliness (Mins)"]
                print(f"incident timeliness: {acc_timeliness}mins")

                ## ---------- FRange related'
                # dist of incident to asset location
                df_unit_sub["Distance"] = df_unit_sub.apply(lambda x: haversine(acc_location, (x["Latitude"], x["Longitude"])),axis=1,)
                print(f"out of range units: {list(df_unit_sub[df_unit_sub['Distance']>df_unit_sub['Effective Radius (km)']].index)}")
                # filter out assets within range only
                df_unit_sub = df_unit_sub.loc[df_unit_sub["Distance"] < df_unit_sub["Effective Radius (km)"]]

                # if no asset within range, continue to the next target unit
                if len(df_unit_sub) == 0:
                    selected_unit = "NS-OR"
                    unit_deployed.append(selected_unit)
                    print("NO SOLUTION: all assets out of range!")
                    print()
                    continue
                else:
                    df_unit_sub.loc[:, "FRange"] = rb.mm.fit_transform(
                        np.array(
                            df_unit_sub["Effective Radius (km)"].values.reshape(-1, 1)
                        )
                    ) # scaling of range

                    ## ----------- time related
                    df_unit_sub.loc[:, "Time (m)"] = df_unit_sub.apply(lambda x: safe_division(x["Distance"], x["Speed (Km/h)"]) * 60,axis=1,)  # in minutes
                    exceed_t_temp = df_unit_sub.loc[df_unit_sub["Time (m)"] > acc_timeliness, :]
                    # if assets beyond timeliness, will be scaled to value ranging from 2-3
                    if (len(exceed_t_temp) > 0):
                        timeliness_flag = 1
                        time_scaler = MinMaxScaler(feature_range=(2, 3))
                        df_unit_sub.loc[
                            exceed_t_temp.index, "DerivedTime"
                        ] = time_scaler.fit_transform(
                            np.array(exceed_t_temp["Time (m)"].values.reshape(-1, 1))
                        )
                        df_unit_sub.loc[
                            ~df_unit_sub.index.isin(exceed_t_temp.index), "DerivedTime"
                        ] = float(1)
                    # else all assets are within timeliness - all assets derived time is set to 1 (no preference)
                    else:
                        timeliness_flag = 0
                        df_unit_sub.loc[:, "DerivedTime"] = float(1)

                    ## ---------- intend related
                    right_intend_rank = intend_dict[right_intend][right_intend]
                    df_unit_sub.loc[:, "Intend"] = df_unit_sub.apply(lambda x: rb.df_asset_c.loc[acc_cat, x["Asset Type"]], axis=1)
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
                    df_temp = df_unit_sub.copy(deep=True)
                    for item, penalty in weight_dict.items():
                        df_temp.loc[:, item] = df_temp.loc[:, item].apply(
                            lambda x: x * penalty
                        )
                        df_unit_sub.loc[:, "score"] = (
                            df_temp["Intend"]
                            + df_temp["CMD/SUP"]
                            + df_temp["FRange"]
                            + df_temp["DerivedTime"]
                            + df_temp["DeployCount"]
                        )
                    sorted_df = df_unit_sub.sort_values(by=["score"])
                    if view:
                        print(sorted_df)

                    score_list = sorted(list(sorted_df["score"].unique()))
                    min_score = score_list.pop(0)
                    sub_asset_list = list(sorted_df.loc[sorted_df["score"] == min_score, :].index)
                    selected_unit = random.sample(sub_asset_list, 1)[0]
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
                    selected_unit_assType = rb.df_asset_ml.at[selected_unit, "Asset Type"]
                    selected_unit_intend = rb.df_asset_c.at[acc_cat, selected_unit_assType]

                    selected_unit_rank = intend_dict[right_intend][selected_unit_intend]  # assigned rank
                    selected_unit_check = intend_rank_dict[right_intend][selected_unit_rank]  # checking rank

                    if (selected_unit_check< intend_rank_dict[right_intend][right_intend_rank]):
                        try:
                            warning_dict[idx][selected_unit].append("intent_lowered")
                        except:
                            warning_dict[idx] = {f"{selected_unit}": ["intent_lowered"]}

                    # 2. timeliness warnings
                    if timeliness_flag == 1 and selected_unit in exceed_t_temp.index:
                        try:
                            warning_dict[idx][selected_unit].append("timeliness_violation")
                        except:
                            warning_dict[idx] = {f"{selected_unit}": ["timeliness_violation"]}

        unit_deployed.append(selected_unit)
        print(f"Selected Unit: {selected_unit}")
        print()

    return unit_deployed, warning_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default="./csv_files/DMS_target_sample.xlsx"
    )
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--intend_w", type=int, default=1000)
    parser.add_argument("--cmdsup_w", type=int, default=10)
    parser.add_argument("--depcount_w", type=int, default=100)
    parser.add_argument("--frange_w", type=int, default=1)
    parser.add_argument("--derivedt_w", type=int, default=0.1)
    parser.add_argument(
        "--view",
        default=False,
        action="store_true",
        help="view capable assets score breakdown for each target",
    )
    parser.add_argument("--var_sol", type=int, default=3)
    args = parser.parse_args()

    filename = args.input_path
    output_path = args.output_path

    rb = RB(filename)
    rb.update_df()

    weight_v1 = get_weight_dict(
        intent_w=args.intend_w,
        cmdsup_w=args.cmdsup_w,
        depcount_w=args.depcount_w,
        frange_w=args.frange_w,
        derivedt_w=args.derivedt_w,
    )

    start_time = time.time()

    sol = dict()
    for i in range(1, args.var_sol+1):
        solv1, warning1 = run(rb, weight_v1, args.view)
        sol[i] = solv1

    end_time = time.time()

    print(f"Total runtime: {round(end_time-start_time, 4)}s")

    print(f"Final Deployment: {solv1}")
    if len(warning1) > 0:
        print(f"\nWARNING: {warning1}")

    df_sol = pd.DataFrame(sol)
    print("\nsaving results...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_sol.to_csv(os.path.join(output_path, "results.csv"))
    print("DONE! :^)")