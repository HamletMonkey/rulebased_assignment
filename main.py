
import os
import time
import argparse

import json
from rb import RB
from rb import get_weight_dict, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default="./xlsx_files/decide_phase/DMS_target_sample_v2.xlsx"
        # "--input_path", type=str, default="./xlsx_files/detect_phase/DMS_target_sample_det3_tsensitive.xlsx"
    )
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--weight_path", type=str, default="./weight_decide.json")
    # parser.add_argument("--weight_path", type=str, default="./weight_detect.json")
    parser.add_argument(
        "--view",
        default=False,
        action="store_true",
        help="view capable assets score breakdown for each target",
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

    print(f"Total runtime: {round(end_time-start_time, 4)}s\n")

    for k in weight.keys():
        print(f"Final Deployment {int(k)} (combined output): {combined_output[k]['unit_deployed']}\n")
        if len(combined_output[k]['warning']) > 0:
            print(f"Warning {int(k)}: {combined_output[k]['warning']}\n")
        if 'target_unit_taken' in combined_output[k]:
            print(f"Asset deployed from Target Unit {int(k)}: {combined_output[k]['target_unit_taken']}\n")

    print("\nsaving results...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for k,v in combined_output.items():
        # 1. unit deployed
        how_col = v['unit_deployed']
        case.df_hptl.loc[list(how_col.keys()), "How"] = list(how_col.values())
        # 2. detect phase only: target unit AUC asset taken from
        if case.status_flag == 1:
            tu_taken_col = v['target_unit_taken']
            case.df_hptl.loc[list(tu_taken_col.keys()), "Target Unit-Decide"] = list(tu_taken_col.values())
        # 3. warning message
        warn_col = v['warning']
        if len(warn_col)!=0:
            case.df_hptl.loc[list(warn_col.keys()), "Warning"] = list(warn_col.values())
        case.df_hptl.to_excel(os.path.join(output_path, f"result_{k}_dec1.xlsx"))
    
    with open(os.path.join(output_path, f"output_{k}_dec1.json"), "w") as outfile:
        json.dump(combined_output, outfile)

    print("DONE! :^)")