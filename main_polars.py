import os

import polars as pl
import json

from shapely.geometry import Polygon, Point
from haversine import haversine

import time
import argparse

pl.cfg.Config.set_tbl_cols(-1)

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

## -------------------- create relevant lazyframes
class RB:
    def __init__(self, excel_filename):
        self.excel_filename = excel_filename
        self.df_hptl = pl.read_excel(self.excel_filename, sheet_name="High Payoff Target List").select(
            pl.exclude("Decide Phase Ans.")
        ) # temporary
        self.df_asset_ml = pl.read_excel(self.excel_filename, sheet_name="Asset Master List").drop_nulls()
        self.df_asset_c = pl.read_excel(self.excel_filename, sheet_name="Asset Capability Table").drop_nulls()
        self.df_sector = pl.read_excel(self.excel_filename, sheet_name="Sector")
        self.df_target_sel = pl.read_excel(self.excel_filename, sheet_name="Target Selection Standards")
        self.status_flag = 0 # default for Decide Phase
        self.target_unit_list = list()
        self.sectors = dict()
        self.iea_dict = dict()
        self.q_unit_info = self.df_asset_ml.lazy() # lazy df

    def tidy_up(self):
        # get q_unit_info - add in priority and assigned target designation for AUC and CONF assets
        conf_units = self.q_unit_info.filter(pl.col("Assignment")=="CONF").select(pl.col("Unit")).collect().to_series().to_list()
        auc_units = self.q_unit_info.filter(pl.col("Assignment")=="AUC").select(pl.col("Unit")).collect().to_series().to_list()
        conf_sub_df = self.df_hptl.lazy().filter(
            (pl.col("How").is_in(conf_units)) &
            (pl.col("Status")=="Detected")
        ).groupby("How").agg([
            pl.all().sort_by('Priority').last(),
        ]).select(
            [
                pl.col("How"),
                pl.col("Priority"),
                pl.col("Target Designation")
            ]
        )
        auc_sub_df = self.df_hptl.lazy().filter(
            (pl.col("How").is_in(auc_units))
        ).groupby("How").agg([
            pl.all().sort_by('Priority').last(),
        ]).select(
            [
                pl.col("How"),
                pl.col("Priority"),
                pl.col("Target Designation")
            ]
        )
        conf_auc_df = conf_sub_df.collect().vstack(auc_sub_df.collect())

        self.q_unit_info = (
            self.q_unit_info.join(
                conf_auc_df.lazy(), left_on=["Unit"], right_on=["How"], how='left'
            ).filter(
                pl.col("Assignment") != "RESERVED"
            ).with_columns(
                [
                    pl.when(pl.col("CMD/SUP") == "Organic").then(1).otherwise(2).keep_name(),
                    pl.when(pl.col('Assignment')=='FREE').then(1).when(pl.col('Assignment')=='AUC').then(2).otherwise(3).keep_name(),
                    pl.col("Priority").fill_null(0).keep_name(),
                    pl.col("Target Designation").fill_null('FA').keep_name()
                ]
            ).with_column(
                pl.when(pl.col("Priority")==0).then(0).otherwise(pl.col("Priority").max()+1 - pl.col("Priority")).keep_name()
            ).select(
                [
                    pl.col("*"),
                    pl.col("Coverage").apply(lambda x: 1 if "A" in x else 0).alias("Sector A"),
                    pl.col("Coverage").apply(lambda x: 1 if "B" in x else 0).alias("Sector B"),
                    pl.col("Coverage").apply(lambda x: 1 if "C" in x else 0).alias("Sector C"),
                ]
            ).rename({'Target Designation':'Taken From'})
        )
        
        # get target unit list
        self.status_check = self.df_hptl.select("Status").unique().to_numpy()
        if 'Detected' in self.status_check:
            self.status_flag = 1# Detect Phase
        self.target_unit_list = (self.df_hptl.select("Target Designation").to_series().to_list())

        # create iea dictionary
        for i in range(len(self.df_sector)):
            all_points = list()
            for j in range(1, 9, 2):
                all_points.append(list(self.df_sector.rows()[i][j : j + 2]))
            self.sectors[self.df_sector.rows()[i][0]] = Polygon(all_points) 
        intend_dict_master = (
            self.df_asset_c.filter(pl.col("Category") != "0")
            .to_dict(as_series=False)
        )
        cat_count = intend_dict_master["Category"]
        for i, cat in enumerate(cat_count):
            temp = {ass: intd[i]for ass, intd in intend_dict_master.items()if ass != "Category"}
            self.iea_dict[cat] = temp
        return


def get_weight_dict(json_input):
    with open(json_input, "r") as f:
        weight_dict = json.loads(f.read())
    return weight_dict

def run(rb, weight, view=True):

    warning_dict = dict()
    output = dict()

    asset_unit_list = rb.q_unit_info.select("Unit").collect().to_series().to_list()
    asset_dep_count = dict(zip(asset_unit_list, [0 for _ in range(len(asset_unit_list))]))

    if rb.status_flag ==1:
        df_hptl_md = rb.df_hptl.lazy().filter(
            (pl.col('Status')=='Detected') & (pl.col('How').is_null())
        )
        target_unit_taken = dict()
        rb.target_unit_list = df_hptl_md.select("Target Designation").collect().to_series().to_list()

    for idx, tunit in enumerate(rb.target_unit_list):
        print(f"Target Designation {idx+1}: {tunit}")
        ## ---------- get basic info of the target unit
        hptl_info = (
            rb.df_hptl.lazy().filter(pl.col("Target Designation") == tunit)
            .select(
                [
                    pl.col("Category"),
                    pl.col("Intend"),
                    pl.col('Time-Sensitive'),
                    pl.col("Priority"),
                    pl.col("Latitude"),
                    pl.col("Longitude"),
                ]
            )
            .collect()
            .to_dicts()[0]
        )
        print(f"target hptl info: {hptl_info}")
        print(f"target time_sensitive: {hptl_info['Time-Sensitive']}")
        print(f"target priority: {hptl_info['Priority']}")
        
        ## ---------- filter asset solution space based on count deployed - wrt decide and detect phase
        # decide phase - each organic asset can be deployed up to 3 targets
        if rb.status_flag == 0: # decide phase
            df_unit_info = (
                rb.q_unit_info.with_column(
                    pl.Series(name="DeployCount", values=list(asset_dep_count.values()))
                )
                .filter(
                    ((pl.col("CMD/SUP") == 1) & (pl.col("DeployCount") < 3))
                    | ((pl.col("CMD/SUP") == 2) & (pl.col("DeployCount") < 1))
                )
            )
        # detect phase - each asset can only be deployed up ONCE
        else:
            df_unit_info = (
                rb.q_unit_info.with_column(
                    pl.Series(name="DeployCount", values=list(asset_dep_count.values()))
                )
                .filter((pl.col("DeployCount") < 1))
            )
            ## ----------- Time Sensitive related
            if hptl_info['Time-Sensitive'] == 0:
                df_unit_info = df_unit_info.filter(
                pl.col('Assignment')<3
            )
        # if no assets left, break for-loop
        if df_unit_info.collect().is_empty() :
            t_unit_index = rb.target_unit_list.index(tunit)
            for r in rb.target_unit_list[t_unit_index:]:
                output[r] = "NS-NA"
                warning_dict[r] = "NS-NA"
            print("NO SOLUTION: no assets left!")
            print()
            break
        else:
            # reassigning asset deploy count to value of 1 to 3
            df_unit_info = df_unit_info.with_column(
                    pl.when(pl.col("DeployCount") == 0).then(1).when(pl.col("DeployCount") == 1).then(2).otherwise(3).keep_name()
            )
            # target location - check target falls in which sector
            acc_point = (
                rb.df_hptl.lazy().filter((pl.col("Target Designation") == tunit))
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
                temp = (df_unit_info.filter(pl.col(f"Sector {sec}") == 1).collect())
                dummy = pl.concat([dummy, temp])
            # if no assets within sectors, continue to the next target designation
            if dummy.is_empty():
                selected_unit = "NS-NC"
                output[tunit] = selected_unit
                warning_dict[idx] = "NS-NC"
                print(f"NO SOLUTION: no assets in coverage!")
                print()
                continue
            else:
                ## ---------- get timeliness of each target designation from df_target_sel lazyframe
                acc_timeliness = (
                    rb.df_target_sel.lazy().filter(pl.col("Category") == hptl_info["Category"])
                    .select([pl.col("Timeliness (Mins)")])
                    .collect()
                    .to_numpy()[0, 0]
                ) *60 # to seconds
                print(f"acc timeliness: {acc_timeliness} secs")

                ## ---------- get distance between target designation and asset then filter based on Effective Radius
                acc_points = (hptl_info["Latitude"], hptl_info["Longitude"])
                dummy = dummy.lazy().select(
                        [
                            pl.col("*"),
                            pl.struct(["Latitude", "Longitude"])
                            .apply(lambda x: haversine(acc_points, (x["Latitude"], x["Longitude"])))
                            .alias("Distance"),
                        ]
                    ).filter(pl.col("Distance") < pl.col("Effective Radius (km)")).collect()

                # if no asset within range, continue to the next target designation
                if dummy.is_empty():
                    selected_unit = "NS-OR"
                    output[tunit] = selected_unit
                    warning_dict[tunit] = "NS-OR"
                    print("NO SOLUTION: all assets out of range!")
                    print()
                    continue
                else:
                    ## ---------- get derived time if assets beyond timeliness, keep derived time value
                    dummy = dummy.select(
                        [
                            pl.col("*"),
                            (pl.col("Distance") / pl.col("Speed (Km/h)") * 3600 + pl.col("Status")).alias("Time (secs)"),
                        ]
                    ).with_columns(
                        [
                            pl.when(pl.col("Time (secs)").is_infinite()).then(0).otherwise(pl.col("Time (secs)")).keep_name()
                        ]
                    ).fill_nan(1)

                    # if all assets are within timeliness - all assets derived time is set to 1 (no preference)
                    if dummy.filter(pl.col("Time (secs)") > acc_timeliness).is_empty():
                        timeliness_flag = 0
                        dummy = dummy.with_column(
                            pl.lit(1).alias("DerivedTime")  # result in integer
                        )
                    # else, those beyond timeliness, will be having its original derived time value
                    else:
                        timeliness_flag = 1
                        timeliness_pred = pl.col('Time (secs)')>acc_timeliness
                        dummy = dummy.with_column(
                            pl.when(timeliness_pred).then(pl.col('Time (secs)')).otherwise(1).alias('DerivedTime')
                        ).fill_nan(1)
                        exceed_t_units = dummy.filter(pl.col("DerivedTime")>1).select("Unit").to_series()

                    ## ---------- get intend of each asset with respect to each target designation
                    # also dropping columns
                    right_intend_rank = intend_dict[hptl_info["Intend"]][hptl_info["Intend"]]
                    right_intend_rank = intend_dict[hptl_info["Intend"]][hptl_info["Intend"]]
                    dummy = dummy.select(
                        [
                            (pl.col("Asset Type").apply(
                                lambda x: intend_dict[hptl_info["Intend"]][rb.iea_dict[hptl_info["Category"]][x]]
                            ).alias("Intend")),
                            (pl.exclude(["Qty","Configuration","Coverage","Latitude","Longitude","Speed (Km/h)"]))
                        ]
                    ).sort([pl.col(x) for x in weight])

                    if view:
                        print(dummy)

                    ## ---------- assigning of asset to deploy
                    selected_unit = dummy.select(pl.col('Unit')).row(0)[0]
                    ## ---------- Decide Phase asset assignment ---------- ##
                    if rb.status_flag == 1:      
                        selected_target_unit = dummy.select(pl.col('Taken From')).row(0)[0]
                        target_unit_taken[tunit] = selected_target_unit     
                    
                    if not selected_unit.startswith("NS"):
                        asset_dep_count[selected_unit] += 1

                    ## ---------- getting all the warnings
                    # 1. allocated asset warning
                    organic_check = (
                        dummy.lazy().filter(pl.col("Unit") == selected_unit).select(pl.col("CMD/SUP")).collect().row(0)[0]
                    )
                    if organic_check == 2:  # allocated asset
                        try:
                            warning_dict[tunit][selected_unit].append("allocated_asset")
                        except:
                            warning_dict[tunit] = {f"{selected_unit}": ["allocated_asset"]}

                    # 2. lower intend warning
                    selected_unit_assType = (
                        rb.df_asset_ml.lazy().filter(pl.col("Unit") == selected_unit).select(pl.col("Asset Type")).collect().row(0)[0]
                    )
                    selected_unit_intend = (
                        rb.df_asset_c.lazy().filter(
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

        output[tunit] = selected_unit
        print(f"Selected Unit: {selected_unit}")
        if rb.status_flag == 1:
            print(f"Selected Target Unit: {selected_target_unit}")
        print()

    final_output = dict()
    final_output['unit_deployed'] = output
    final_output['warning'] = warning_dict
    if rb.status_flag == 1:
        final_output['target_unit_taken'] = target_unit_taken
    
    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # "--input_path", type=str, default="./xlsx_files/decide_phase/DMS_target_sample_v2.xlsx"
        "--input_path", type=str, default="./xlsx_files/detect_phase/DMS_target_sample_det1_tsensitive.xlsx"
    )
    parser.add_argument("--output_path", type=str, default="trial")
    # parser.add_argument("--weight_path", type=str, default="./weight_decide_ns.json")
    parser.add_argument("--weight_path", type=str, default="./weight_detect_ns.json")
    parser.add_argument(
        "--view",
        default=False,
        action="store_true",
        help="view capable assets score breakdown for each target",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="save output",
    )
    args = parser.parse_args()

    filename = args.input_path
    output_path = args.output_path

    case = RB(filename)
    case.tidy_up()
    weight = get_weight_dict(args.weight_path)

    start_time = time.time()

    combined_output = dict()
    for k, w in weight.items():
        f_output = run(case, w, args.view)
        combined_output[k] = f_output

    end_time = time.time()

    print(f"Total Runtime: {round(end_time-start_time, 4)}s")

    for k in weight.keys():
        print(f"Final Deployment {int(k)} (combined output): {combined_output[k]['unit_deployed']}\n")
        if len(combined_output[k]['warning']) > 0:
            print(f"Warning {int(k)}: {combined_output[k]['warning']}\n")
        if 'target_unit_taken' in combined_output[k]:
            print(f"Asset deployed from Target Unit {int(k)}: {combined_output[k]['target_unit_taken']}\n")

    if args.save:
        print("\nsaving results...")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        df_hptl_final = case.df_hptl.to_pandas().set_index("Target Designation")
        for k,v in combined_output.items():
            # 1. unit deployed
            how_col = v['unit_deployed']
            df_hptl_final.loc[list(how_col.keys()), "How"] = list(how_col.values())
            # 2. detect phase only: target unit AUC asset taken from
            if case.status_flag == 1:
                tu_taken_col = v['target_unit_taken']
                df_hptl_final.loc[list(tu_taken_col.keys()), "Target Unit-Decide"] = list(tu_taken_col.values())
            # 3. warning message
            warn_col = v['warning']
            if len(warn_col)!=0:
                df_hptl_final.loc[list(warn_col.keys()), "Warning"] = list(warn_col.values())
            df_hptl_final.to_excel(os.path.join(output_path, f"polars_result_{k}.xlsx"))
        
        with open(os.path.join(output_path, f"polars_output.json"), "w") as outfile:
            json.dump(combined_output, outfile)

    print("DONE! :^)")