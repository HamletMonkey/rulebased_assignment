import os

import polars as pl
import numpy as np
import json

import random
from sklearn.preprocessing import MinMaxScaler

from shapely.geometry import Polygon, Point
from haversine import haversine

import time
from datetime import datetime
import argparse

pl.cfg.Config.set_tbl_cols(-1)

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

## -------------------- create relevant lazyframes
class RB:
    def __init__(self, csv_files):
        self.csv_files = csv_files
        self.lz_hptl = pl.scan_csv(os.path.join(self.csv_files, "hptl.csv"))
        self.lz_sector = pl.scan_csv(os.path.join(self.csv_files, "sector.csv"))
        self.lz_asset_ml = pl.scan_csv(os.path.join(self.csv_files, "asset_master_list.csv"))
        self.lz_asset_c = pl.scan_csv(os.path.join(self.csv_files, "asset_capability_table.csv"))
        self.lz_target_sel = pl.scan_csv(os.path.join(self.csv_files, "target_selection_standards.csv"))
        self.status_check = "Unknown"  # default
        self.target_unit_list = list()
        self.sectors = dict()
        self.iea_dict = dict()
        self.mm = MinMaxScaler(feature_range=(1, 3))
        self.time_scaler = MinMaxScaler(feature_range=(2, 3))
        self.q_unit_info = self.lz_asset_ml

    def tidy_up(self):
        self.q_unit_info = (
            self.q_unit_info.select(
                [
                    pl.col("*"),
                    pl.col("Coverage").apply(lambda x: 1 if "A" in x else 0).alias("Sector A"),
                    pl.col("Coverage").apply(lambda x: 1 if "B" in x else 0).alias("Sector B"),
                    pl.col("Coverage").apply(lambda x: 1 if "C" in x else 0).alias("Sector C"),
                ]
            )
            .with_column(
                pl.when(pl.col("CMD/SUP") == "Organic").then(1).otherwise(2).keep_name()
            )
            .drop_nulls()
        )
        self.status_check = self.lz_hptl.select("Status").unique().collect().row(0)[0]
        self.target_unit_list = (self.lz_hptl.select("Target Unit").collect().to_series().to_list())

        for i in range(len(self.lz_sector.collect())):
            all_points = list()
            for j in range(1, 9, 2):
                all_points.append(list(self.lz_sector.collect().rows()[i][j : j + 2]))
            self.sectors[self.lz_sector.collect().rows()[i][0]] = Polygon(all_points)

        intend_dict_master = (
            self.lz_asset_c.filter(pl.col("Category") != "0")
            .collect()
            .to_dict(as_series=False)
        )
        cat_count = intend_dict_master["Category"]
        for i, cat in enumerate(cat_count):
            temp = {ass: intd[i]for ass, intd in intend_dict_master.items()if ass != "Category"}
            self.iea_dict[cat] = temp
        return


def get_weight_dict(intent_w, cmdsup_w, depcount_w, frange_w, derivedt_w):
    weight_dict = {
        "Intend": intent_w,
        "CMD/SUP": cmdsup_w,
        "DeployCount": depcount_w,
        "FRange": frange_w,
        "DerivedTime": derivedt_w
        # 'AssetCat': assetcat_w - TBC
    }
    return weight_dict


def run(rb, weight_dict, view=True):

    warning_dict = dict()
    unit_deployed = list()

    asset_unit_list = rb.q_unit_info.select("Unit").collect().to_series().to_list()
    asset_dep_count = dict(zip(asset_unit_list, [0 for _ in range(len(asset_unit_list))]))

    for idx, tunit in enumerate(rb.target_unit_list):
        print(f"target unit {idx+1}: {tunit}")
        ## ---------- filter asset solution space based on count deployed - wrt decide and detect phase
        # decide phase - each organic asset can be deployed up to 3 targets
        if rb.status_check == "Unknown":
            df_unit_info = (
                rb.q_unit_info.with_column(
                    pl.Series(name="DeployCount", values=list(asset_dep_count.values()))
                )
                .filter(
                    ((pl.col("CMD/SUP") == 1) & (pl.col("DeployCount") < 3))
                    | ((pl.col("CMD/SUP") == 2) & (pl.col("DeployCount") < 1))
                )
                .collect()
            )
        # detect phase - each asset can only be deployed up ONCE
        else:
            df_unit_info = (
                rb.q_unit_info.with_column(
                    pl.Series(name="DeployCount", values=list(asset_dep_count.values()))
                )
                .filter((pl.col("DeployCount") < 1))
                .collect()
            )
        # if no assets left, break for loop
        if len(df_unit_info) == 0:
            t_unit_index = rb.target_unit_list.index(tunit)
            for _ in rb.target_unit_list[t_unit_index:]:
                unit_deployed.append("NS-NA")
            print("NO SOLUTION: no assets left!")
            print()
            break
        else:
            # reassigning asset deploy count to value of 1 to 3
            df_unit_info = (
                df_unit_info.lazy()
                .with_column(
                    pl.when(pl.col("DeployCount") == 0).then(1).when(pl.col("DeployCount") == 1).then(2).otherwise(3).keep_name()
                )
                .collect()
            )
            # target location - check target falls in which sector
            acc_point = (
                rb.lz_hptl.filter((pl.col("Target Unit") == tunit))
                .select(["Latitude", "Longitude"])
                .collect()
                .row(0)
            )
            acc_point = Point(list(acc_point))
            acc_coverage = [sector for sector, poly in rb.sectors.items() if acc_point.within(poly)]
            print(f"coverage includes sectors: {acc_coverage}")
            # get all assets within the sectors involved
            dummy = pl.DataFrame()
            for sec in acc_coverage:
                temp = (df_unit_info.lazy().filter(pl.col(f"Sector {sec}") == 1).collect())
            dummy = pl.concat([dummy, temp])
            # if no assets within sectors, continue to the next target unit
            if len(dummy) == 0:
                selected_unit = "NS-NC"
                unit_deployed.append(selected_unit)
                print(f"NO SOLUTION: no assets in coverage!")
                print()
                continue
            else:
                ## ---------- get basic info of the target unit from df_hptl lazyframe - dictionary
                hptl_info = (
                    rb.lz_hptl.filter(pl.col("Target Unit") == tunit)
                    .select(
                        [
                            pl.col("Category"),
                            pl.col("Intend"),
                            pl.col("Latitude"),
                            pl.col("Longitude"),
                        ]
                    )
                    .collect()
                    .to_dicts()[0]
                )
                print(f"acc hptl info: {hptl_info}")

                ## ---------- get timeliness of each target unit from df_target_sel lazyframe
                acc_timeliness = (
                    rb.lz_target_sel.filter(pl.col("Category") == hptl_info["Category"])
                    .select([pl.col("Timeliness (Mins)")])
                    .collect()
                    .to_numpy()[0, 0]
                )
                print(f"acc timeliness: {acc_timeliness}mins")

                ## ---------- get distance between target unit and asset then filter based on Effective Radius
                acc_points = (hptl_info["Latitude"], hptl_info["Longitude"])
                dummy = (
                    dummy.lazy()
                    .select(
                        [
                            pl.col("*"),
                            pl.struct(["Latitude", "Longitude"])
                            .apply(lambda x: haversine(acc_points, (x["Latitude"], x["Longitude"])))
                            .alias("Distance"),
                        ]
                    )
                    .filter(pl.col("Distance") < pl.col("Effective Radius (km)"))
                    .collect()
                )
                # if no asset within range, continue to the next target unit
                if len(dummy) == 0:
                    selected_unit = "NS-OR"
                    unit_deployed.append(selected_unit)
                    print("NO SOLUTION: all assets out of range!")
                    print()
                    continue
                else:
                    ## ---------- get derived time + scaling of Effective Radius column --> FRange column
                    dummy = (
                        dummy.lazy()
                        .select(
                            [
                                pl.col("*"),
                                (pl.col("Distance") / pl.col("Speed (Km/h)") * 60).alias("Time (mins)"),
                            ]
                        )
                        .with_columns(
                            [
                                (
                                    pl.when(pl.col("Time (mins)").is_infinite())
                                    .then(0)
                                    .otherwise(pl.col("Time (mins)"))
                                    .keep_name()
                                ),
                                (
                                    pl.Series(
                                        rb.mm.fit_transform(
                                            np.array(
                                                dummy.select(
                                                    pl.col("Effective Radius (km)")
                                                ).to_series()
                                            ).reshape(-1, 1)
                                        ).reshape(-1)
                                    ).alias("FRange")
                                ),
                            ]
                        )
                        .collect()
                    )

                    ## --------- get time in scale, anything within timeliness is 1, beyond will be scaled between 2 and 3 --> DerivedTime column
                    # if all assets are within timeliness - all assets derived time is set to 1 (no preference)
                    if len(dummy.filter(pl.col("Time (mins)") > acc_timeliness)) < 1:
                        timeliness_flag = 0
                        dummy = dummy.with_column(
                            pl.lit(1).alias("DerivedTime")  # resukt in integer
                        )
                    # else, those beyond timeliness, will be scaled to value ranging from 2-3
                    else:
                        timeliness_flag = 1
                        # dt is a dictionary
                        dt = (
                            dummy.lazy()
                            .filter(pl.col("Time (mins)") > acc_timeliness)
                            .select([pl.col("Unit"), pl.col("Time (mins)")])
                            .collect()
                            .to_dict(as_series=False)
                        )
                        exceed_t_units = list(
                            dt.keys()
                        )  # list of units exceed timeliness
                        # dt_scale is a list - scaled time of assets beyond timeliness
                        dt_scale = list(
                            rb.time_scaler.fit_transform(
                                np.array(dt["Time (mins)"]).reshape(-1, 1)
                            ).reshape(-1)
                        )
                        # reassign
                        dt["Time (mins)"] = dt_scale
                        dt_scale_df = pl.DataFrame(dt)
                        # joining of 2 dfs --- check for better approach for this
                        dummy = dummy.join(
                            dt_scale_df, how="left", on="Unit"
                        ).fill_null(1)
                        dummy = dummy.rename({"Time (mins)_right": "DerivedTime"})

                    ## ---------- get intend of each asset with respect to each target unit + score
                    # also dropping columns
                    right_intend_rank = intend_dict[hptl_info["Intend"]][hptl_info["Intend"]]
                    dummy = (
                        dummy.lazy()
                        .select(
                            [
                                pl.exclude(
                                    [
                                        "Qty",
                                        "Configuration",
                                        "Effective Radius (km)",
                                        "Coverage",
                                        "Latitude",
                                        "Longitude",
                                        "Speed (Km/h)",
                                    ]
                                ),
                                pl.col("Asset Type")
                                .apply(
                                    lambda x: intend_dict[
                                        rb.iea_dict[hptl_info["Category"]][x]
                                    ][rb.iea_dict[hptl_info["Category"]][x]]
                                )
                                .alias("Intend"),
                            ]
                        )
                        .select(
                            [
                                pl.col("*"),
                                # penalty for FREE AUC CONF - tbc
                                pl.struct(
                                    [
                                        "Intend",
                                        "CMD/SUP",
                                        "FRange",
                                        "DerivedTime",
                                        "DeployCount",
                                    ]
                                )
                                .apply(
                                    lambda x: x["Intend"] * weight_dict["Intend"]
                                    + x["CMD/SUP"] * weight_dict["CMD/SUP"]
                                    + x["FRange"] * weight_dict["FRange"]
                                    + x["DerivedTime"] * weight_dict["DerivedTime"]
                                    + x["DeployCount"] * weight_dict["DeployCount"]
                                )
                                .alias("Score"),
                            ]
                        )
                        .sort("Score")
                        .collect()
                    )

                    ## ---------- get asset to deploy
                    # list of scores for all capable assets
                    score_list = sorted(
                        dummy.lazy()
                        .select(pl.col("Score"))
                        .collect()
                        .to_series()
                        .to_list()
                    )
                    min_score = score_list.pop(0)
                    # sub asset list for assets with the same (lowest) score
                    sub_asset_list = (
                        dummy.lazy()
                        .filter(pl.col("Score") == min_score)
                        .select(pl.col("Unit"))
                        .collect()
                        .to_series()
                        .to_list()
                    )

                    selected_unit = random.sample(sub_asset_list, 1)[0]
                    if not selected_unit.startswith("NS"):
                        asset_dep_count[selected_unit] += 1

                    # glimpse of dataframe
                    if view:
                        print(dummy)

                    ## ---------- getting all the warnings
                    # 1. allocated asset warning
                    organic_check = (
                        dummy.lazy()
                        .filter(pl.col("Unit") == selected_unit)
                        .select(pl.col("CMD/SUP"))
                        .collect()
                        .row(0)[0]
                    )
                    if organic_check == 2:  # allocated asset
                        try:
                            warning_dict[tunit][selected_unit].append("allocated_asset")
                        except:
                            warning_dict[tunit] = {f"{selected_unit}": ["allocated_asset"]}

                    # 2. lower intend warning
                    selected_unit_assType = (
                        rb.lz_asset_ml.filter(pl.col("Unit") == selected_unit)
                        .select(pl.col("Asset Type"))
                        .collect()
                        .row(0)[0]
                    )
                    selected_unit_intend = (
                        rb.lz_asset_c.filter(
                            pl.col("Category") == hptl_info["Category"]
                        )
                        .select(pl.col(selected_unit_assType))
                        .collect()
                        .row(0)[0]
                    )

                    selected_unit_rank = intend_dict[hptl_info["Intend"]][selected_unit_intend]  # assigned rank
                    selected_unit_check = intend_rank_dict[hptl_info["Intend"]][selected_unit_rank]  # checking rank

                    if (selected_unit_check< intend_rank_dict[hptl_info["Intend"]][right_intend_rank]):
                        try:
                            warning_dict[tunit][selected_unit].append("intent_lowered")
                        except:
                            warning_dict[tunit] = {f"{selected_unit}": ["intent_lowered"]}

                    # 3. timeliness warning
                    if timeliness_flag == 1 and selected_unit in exceed_t_units:
                        try:
                            warning_dict[tunit][selected_unit].append("timeliness_violation")
                        except:
                            warning_dict[tunit] = {f"{selected_unit}": ["timeliness_violation"]}

        unit_deployed.append(selected_unit)
        print(f"Selected Unit: {selected_unit}")
        print()

    return unit_deployed, warning_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="csv_files")
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--intend_w", type=int, default=1000)
    parser.add_argument("--cmdsup_w", type=int, default=100)
    parser.add_argument("--depcount_w", type=int, default=10)
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
    rb.tidy_up()

    weight = get_weight_dict(
        intent_w=args.intend_w,
        cmdsup_w=args.cmdsup_w,
        depcount_w=args.depcount_w,
        frange_w=args.frange_w,
        derivedt_w=args.derivedt_w,
    )

    start_time = time.time()

    combined_output = dict()
    combined_warning = dict()
    for i in range(1, args.var_sol+1):
        output, warning = run(rb, weight, args.view)
        combined_output[str(i)] = output
        combined_warning[str(i)] = warning

    end_time = time.time()

    print(f"Total Runtime: {round(end_time-start_time, 4)}s")

    print(f"Final Deployment (combined output): {combined_output}")
    
    print(combined_warning)
    df_sol = pl.DataFrame(combined_output)
    print("\nsaving results...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_sol.write_csv(os.path.join(output_path, f"output_polars_{time_stamp}.csv"))
    with open(os.path.join(output_path, f"warning_polars_{time_stamp}.json"), "w") as outfile:
        json.dump(combined_warning, outfile)
    print("DONE! :^)")