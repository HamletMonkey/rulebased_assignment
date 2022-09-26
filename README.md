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

- In the event that the intended outcome such as destroy, neutralize etc. cannot be achieve (asset time or availability), the next level such as control shall be used as an outcome to manage the situation till the required asset is available.

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

- **Time Sensitive Target**: in the case where Time Sensitive Target is passed in, all asset categories (FREE, AUC, CONF) are allowed to be used.

*Note:*
*1. A target which has no solution should be penalized the most irrespective of the priority.*

*2. In the event to satisfy targets during detect phase, organic AUC asset (asset assigned to unconfirmed target) is preferred over a FREE allocated asset.*

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

### Current Example of JSON input (Software Output)

***+ Note: subject to change +***

#### 1. acc_info:
- represents the HPTL (High Priority Target List) passed in

```json
{
    "1A": {
        "category": "MANOEUVRE_ARMOUR",
        "size": "BATTALION",
        "intent": "destroy",
        "status": "decide",
        "timeliness": 120,
        "weapon_target":{
            "neutralize":[
                "AMX F3 STD-HE",
                "PANZER2000",
                "BM21 M-21OF-M"
            ],
            "destroy":[
                "AH-64E ROCKETS"
            ]
        },
        "intention_rank":{
            "destroy":1,
            "neutralize":2,
            "suppress":3
        }
    }
}
```

#### 2. unit_info:
- `cmdsup` is the asset type where 1 represents Organic and 2 represents Allocated asset
- `frange` is the firing/ attack range of each asset unit
- `asset_cat` is the asset category where 1 represents FREE, 2 represents AUC and 3 represents CONF asset

```json
{
    "A1": {
        "asset_type": "AMX F3 STD-HE",
        "cmdsup": 1,
        "frange":15,
        "asset_cat": "FREE"
    },
    "A2": {
        "asset_type": "PANZER2000",
        "cmdsup": 1,
        "frange":20,
        "asset_cat": "FREE"
    }
}
```

#### 3. time:
- depending on the latitude and longitude coordinates of each asset unit, the derived time with respect to each target unit is different

```json
{
    "1A": {
        "A1":5,
        "A2":20,
    },
    "2A": {
        "A1":15,
        "A2":11,
    }
}
```


### Output Solution

Asset unit selected for each target unit is represented as:

```python
sample_individual = [
    A11, A5, A65, A23, A33, A9
]
```

### How do we score each asset?

**Weight Assignment:**

1. Priority
2. Intent/ Capability (e.g. if intent can be fulfilled with allocated assets, we will pick it as solution)
3. Organic/ Allocated asset type
4. Timeliness
5. Non-consecutive asset deployment (only applicable for organic assets)
7. FREE/ AUC/ CONF asset cat