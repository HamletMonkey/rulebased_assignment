# Optimisation (Rule Based)

## Background

This case study is built as Decision Making System (DMS) for Military Case use.

***Goal***: To provide an optimal development solution to despatch controller given the situation and constraints

## Optimisation Problem

- To assign the right equipment/asset to the incident site to achieve the required outcome within the stipulated time.

- Each incident in the list based on priority must be assigned the correct equipment/asset with the intended outcome.

- The asset assigned must be available to carry out the mission within the stipulated time.

- If the equipment with the intended outcome cannot be meet, the next best intended outcome shall be recommended with the penalty indicated.

- The time baselines assume requests no larger than 500 targets.

## General Rules

- Situation shall be handled based on the priority allocated - highest priority on top of the list, in descending order

- In the event that the intended outcome such as destroy, neutralize etc. cannot be achieved (asset time or availability), the next level such as control shall be used as an outcome to manage the situation till the required asset is available.

- Intent is allowed to be lowered up to 2 levels: e.g. from `destroy` to `suppress` in the case where asset with `neutralize` capability is not available

```python
rank_of_intent = {
    'destroy':1, # highest
    'neutralize':2,
    'suppress':3
}
```

- The algorithm will only assign one type of unit per target.

- Assets which exceed timeliness criteria will still be part of the solution space, given an output warning to the user.

- Present at least 2-3 solutions if possible rather than just the optimal ones.

- Warning should be provided to alert user in the case of: intent lowered, timeliness violation etc.

### Decide Phase

- An organic asset unit is allowed to be assigned to up to 3 different target units.

- Only FREE assets (assets not assigned to any targets yet) will be provided in this phase.

### Detect Phase

- All asset unit is only allowed to be assigned to 1 target unit.

- In the case where FREE assets are not available, AUC assets (asset assigned to an unconfirmed target in decide phase) can be deployed to detected target in this phase.

- Once an asset unit is assigned to a detected target unit, the asset unit is categorized as CONF assets (asset assigned to confirmed targets).

- **Time Sensitive Target**: in the case where Time-Sensitive Target is passed in, all asset categories (FREE, AUC, CONF) are allowed to be used; whereas for Non Time-Sensitive Target only FREE or AUC assets can be deployed.

> *A target which has no solution should be penalized the most irrespective of the priority.*

> *In the event to satisfy targets during detect phase, organic AUC asset (asset assigned to unconfirmed target) is preferred over a FREE allocated asset.*

### Sector/ Coverage

- Battalion boundaries (sectors) are unique, no overlapping between 2 different sectors.

- Assets coverage area can overlap, i.e. an asset can have multiple areas of responsibility.

- During the "decide phase", assets within the right sectors are pre-filtered and based on their "attack range" to decide which assets are assigned first.

- Asset with shorter range is preferred, and to be assigned first.

## Setup

1. Create new environment
`conda create --name opt_rulebased python==3.9`

2. Install dependencies
`pip install -r requirements.txt`

3. Run the code
`python main.py`

### Current Example of CSV input (Software Output)

> *subject to change*

#### 1. High Payoff Target List (HPTL):

contains the information on all target passed in, the 2 tables below shows HPTL input in different phases

*Decide Phase:*

| Priority | Category    | HPT             | Target Designation | Size    | Latitude    | Longitude   | Intend  | When        | Phase | Status  | Time-Sensitive | How |
|----------|-------------|-----------------|--------------------|---------|-------------|-------------|---------|-------------|-------|---------|----------------|-----|
| 1        | Engineer    | Cbt Eng Vehicle | 10_12_30_1_1       | Team    | 24.56555556 | 51.20416667 | Destroy | As Acquired |       | Unknown | 0              |     |
| 2        | Air Defence | ADA Radar       | 10_12_40_4         | Platoon | 24.32333333 | 51.23888889 | Destroy | As Acquired |       | Unknown | 0              |     |

*Detect Phase:*
   - `How` column contains asset assigned for each target as the DM algo output from Decide Phase
   - The asset assigned in Decide Phase (solution) is checked for its validity whenever any targets is detected in Detect Phase
   - If the asset assigned is still valid (see Priority 1 target below), the target will not be passed into the DM algo
   - However if the asset assigned in the Decide Phase is no longer valid (see Priority 2 target below), it will be passed into the DM algo to obtain the right asset for it

| Priority | Category       | HPT             | Target Designation | Size    | Latitude    | Longitude   | Intend  | When          | Phase | Status   | Time-Sensitive | How     |
|----------|----------------|-----------------|--------------------|---------|-------------|-------------|---------|---------------|-------|----------|----------------|---------|
| 1        | Engineer       | Cbt Eng Vehicle | 10_12_30_1_1       | Team    | 24.56555556 | 51.20416667 | Destroy | As Acquired   |       | Detected | 1              | 1_1_4_A |
| 2        | Air Defence    | ADA Radar       | 10_12_40_4         | Platoon | 24.32333333 | 51.23888889 | Destroy | As Acquired   |       | Detected | 0              |         |
| 3        | Fire Support   | 155-SPG         | 10_12_4_A          | Company | 24.37194444 | 51.11722222 | Destroy | As Acquired   |       | Unknown  | 0              | 1_5_1_B |
| 4        | Fire Support   | 155-SPG         | 10_11_4_C          | Company | 24.52277778 | 50.66555556 | Destroy | As   Acquired |       | Unknown  | 0              | NS-NA   |


#### 2. Asset Master List:
- `Status` column represents the preparation time required for each asset: 0 indicates that the asset is available immediately
- `Speed` column represents the traveling speed for each asset: 0 indicates that the asset has a fixed location (non-movable)
- `Assignment` column represent the different category of each asset:
   1. FREE: immediately available
   2. AUC: Assigned to an unconfirmed target in Decide Phase
   3. CONF: Asset that is on engaged in Detect Phase
   4. RESERVED: Asset that is reserved in Detect Phase, should not be included in the solution space

|   Category   | Asset Type |   Unit  | Qty |  CMD/SUP  |  Configuration  | Effective Radius (km) | Coverage | Status |   Latitude  |  Longitude  | Speed (Km/h) | Assignment |
|--------------|------------|---------|-----|-----------|-----------------|-----------------------|----------|--------|-------------|-------------|--------------|------------|
| Fire Support | 155-SPG    | 1_1_4_A | 1   | Organic   | 155 gun         | 30                    | A        | 0      | 24.72638889 | 50.94138889 | 0            | RESERVED   |
| Fire Support | 155-SPG    | 1_1_4_B | 1   | Organic   | 155 gun         | 30                    | B        | 0      | 24.6625     | 51.05722222 | 0            | FREE       |
| Fire Support | 155-SPG    | 1_1_4_C | 1   | Organic   | 155 gun         | 30                    | C        | 0      | 24.68111111 | 51.2725     | 0            | AUC        |
| Attack Heli  | AH         | 220_1_3 | 1   | Allocated | LGM/Rockets/Gun | 150                   | A, B, C  | 650    | 25.27722222 | 51.525      | 185.2        | CONF       |


#### 3. Target Selection Standard:

|   Category   | Timeliness (Mins) | Accuracy (m) |
|--------------|-------------------|--------------|
| Engineer     | 30                | 50           |
| Fire Support | 60                | 100          |


#### 4. Asset Capability Table:

|   Category   |        AH       |   FGA   | 155-SPG | MLRS-LR | MLRS-SR |
|--------------|-----------------|---------|---------|---------|---------|
|              | LGM/Rockets/Gun | LGB/CB  | 155 gun | PGM     | Rocket  |
| Engineer     | Destroy         | Destroy | Destroy | Destroy | Destroy |
| Fire Support | Destroy         | Destroy | Destroy | Destroy | Destroy |


#### 5. Weapon Target Table:

|      Category     |     Size     | AH | FGA | 155-SPG | MLRS-LR | MLRS-SR |
|-------------------|--------------|----|-----|---------|---------|---------|
| Command & Control | Team         | 1  | 1   | 1       | 1       | 1       |
| Command & Control | Command Post | 1  | 1   | 1       | 1       | 1       |
| Air Defence       | Platoon      | 1  | 1   | 1       | 1       | 1       |


#### 6. Sector:

| Sector Name | P1 Latitude | P1 Longitude | P2 Latitude | P2 Longitude | P3 Latitude | P3 Longitude | P4 Latitude | P4 Longitude |
|-------------|-------------|--------------|-------------|--------------|-------------|--------------|-------------|--------------|
| A           | 24.86722222 | 50.83611111  | 24.55388889 | 50.36972222  | 24.16666667 | 50.8075      | 24.93444444 | 51.03944444  |
| B           | 24.93444444 | 51.03944444  | 24.16666667 | 50.8075      | 24.17361111 | 51.20055556  | 24.90194444 | 51.10472222  |
| C           | 24.90194444 | 51.10472222  | 24.17361111 | 51.20055556  | 24.25722222 | 51.60444444  | 24.70944444 | 51.45222222  |

### Current Output Solution

For target unit that has no solution, it will be indicated as:
- `NS-NA`: no asset available
- `NS-NC`: no asset within coverage (sector)
- `NS-OR`: target is out of all assets' attack range

Warning Message included:
- `allocated_asset`: whenever an allocated asset is deployed to a target
- `intend_lowered`: whenever an asset with lower level of capability is deployed instead of the original intend
- `timeliness_violation`: whenever an asset which derived time is beyond the target's timeliness standard

Asset unit selected for each target unit is represented differently.

*Decide Phase:*

```python
# {target_unit: asset_unit_deployed}
sample_unit_deployed = {
    '10_13_40_3': '230_1_1',
    '10_12_CP': '230_1_3',
    '10_11_CP': 'NS-NC'
}

# {target_unit: {asset_unit_deployed: list of relevant warning messages}}
sample_warning = {
    '10_13_40_3': {'230_1_1':['allocated_asset']},
    '10_12_CP': {'230_1_1':['allocated_asset','intend_lowered']},
    '10_13_20_1_1': 'NS-NA'
}
```

*Detect Phase:*

There would be additional output for deploying AUC assets in Detect Phase. This is to indicate which Target Unit will the AUC asset (assigned in Decide Phase) be deployed from.

```python
# {target_unit: target_unit_AUC_asset_from}
sample_target_unit_taken = {
    '10_12_40_4': '10_11_40_4',
    '10_12_40_2': '10_11_40_2',
    '10_12_40_1': '10_13_40_4'
}
```

> *running `main.py` saves output in a JSON file (containing these 3 dictionaries) and a CSV file*

### How do we score each asset?

The weight used in different phases is passed into the DM algo by specifying the JSON file path via argument `--weight`. We can also passed in more than one set of "weights" in the same JSON file to have the DM algo to produce different outputs in the same run.

```json
{
    "1":
    {
        "Intend": 1000,
        "CMD/SUP": 100,
        "DeployCount": 10,
        "FRange": 1,
        "DerivedTime": 0.1
    },
    "2":
    {
        "Intend": 1000,
        "DeployCount": 100,
        "CMD/SUP": 10,
        "FRange": 1,
        "DerivedTime": 0.1
    }
}
```

**Updates on Weight Assignment:**

*27 Sept 2022*:

1. Priority
2. Intent/ Capability (e.g. if intent can be fulfilled with allocated assets, we will pick it as solution)
3. Organic/ Allocated asset type
4. Timeliness
5. Non-consecutive asset deployment (only applicable for organic assets)
7. FREE/ AUC/ CONF asset cat

*30 Sept 2022*:

- Solution Ver. 1

1. Priority
2. Intent/ Capability (e.g. if intent can be fulfilled with allocated assets, we will pick it as solution)
3. Organic/ Allocated asset type
4. Deploy Count (Non-consecutive asset deployment - only applicable for organic assets)
5. FRange (Effective Range)
6. Timeliness (Derived Time = Prep Time + Travel Time)
7. FREE/ AUC/ CONF asset cat

- Solution Ver. 2

1. Priority
2. Intent/ Capability (e.g. if intent can be fulfilled with allocated assets, we will pick it as solution)
3. Deploy Count (Non-consecutive asset deployment - applicable across Organic and Allocated assets)
4. Organic/ Allocated asset type
5. FRange (Scaled effective range, distance between asset and target < asset's effective range)
6. Timeliness (Derived Time = Prep Time + Travel Time)
7. FREE/ AUC/ CONF asset cat

*10 Oct 2022*:

- Detect Phase

1. Priority
2. Intend
3. Organic/ Allocated asset type
4. FREE/ AUC/ CONF asset cat
5. Status (Prep Time, value only for CONF assets)
6. FRange (Scaled effective range, distance between asset and target < asset's effective range)
7. Timeliness (Derived Time = Prep Time + Travel Time)