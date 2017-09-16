## Load and Fix. 

1) Load `vin` and `epa` datasets from files:
	- `vin` gets loaded from `vin_file` which contains decoded vin data that has been joined with vtyp3 data.
	- `epa` gets loaded from `epa_file` contains mpg data. 

2) Fix errors in datasets: e.g. wrong displacment for ford expedition, typo in model solstice (spelled solistice).

3) Drop all records that are missing the make, the model or the year. 

4) Replace missing fuel with `gasoline`. 

5) Make all fields lower case and trim white spaces. 

6) Drop all vehicles that are in the following categories: `incomplete`, `trailer`, `motorcycle`, `bus`, `low speed vehicle (lsv)`.

7) Get rid of duplicates fields: e.g. `gasoline, gasoline` becomes `gasoline`.

8) Drop records that are trucks based on a list of makes:
	- volvo truck
	- western star
	- whitegmc
	- winnebago
	- winnebago industries, inc.
	- workhorse
	- ai-springfield
	- autocar industries
	- capacity of texas
	- caterpillar
	- e-one
	- freightliner
	- kenworth
	- mack
	- navistar
	- peterbilt
	- pierce manufacturing
	- spartan motors chassis
	- terex advance mixer
	- the vehicle production group
	- utilimaster motor corporation
	- international.

9) Replace fields based on the following mapping:

```
		'vin': {
			'fuelType1':	{
				'compressed natural gas (cng)':									'natural gas',
				'liquefied petroleum gas (propane or lpg)':						'natural gas',
				'liquefied natural gas (lng)':									'natural gas',
				'gasoline, diesel':												'gasoline',
				'diesel, gasoline':												'gasoline',
				'ethanol (e85)':												'ethanol',
				'compressed natural gas (cng), gasoline':						'gasoline',
				'gasoline, compressed natural gas (cng)':						'gasoline',
				'compressed hydrogen / hydrogen':								'hydrogen',
				'fuel cell':													'hydrogen',
				},
			'drive':	{
				'4x2':																'two',
				'6x6':																'all',
				'6x2':																'two',
				'8x2':																'two',
				'rwd/ rear wheel drive':											'two',
				'fwd/front wheel drive':											'two',
				'4x2, rwd/ rear wheel drive':										'two',
				'4x2, fwd/front wheel drive':										'two',
				'rwd/ rear wheel drive, 4x2':										'two',
				'fwd/front wheel drive, 4x2':										'two',
				'4wd/4-wheel drive/4x4':											'all',
				'awd/all wheel drive':												'all',
				},
			'transmission_type':	{
				'manual/standard': 												'manu',
				'automated manual transmission (amt)': 							'manu',
				'manual/standard, manual/standard': 							'manu',
				'dual-clutch transmission (dct)': 								'manu',
				'continuously variable transmission (cvt)': 					'auto',
				'automatic': 													'auto',
				'automatic, continuously variable transmission (cvt)': 			'auto',
				}
			},
		'epa': {
			'fuelType1':	{
				'regular gasoline':			'gasoline',
				'premium gasoline':			'gasoline',
				'midgrade gasoline':		'gasoline',
				},
			'drive':	{
				'rear-wheel drive':				'two',
				'front-wheel drive':			'two',
				'2-wheel drive':				'two',
				'all-wheel drive':				'all',
				'4-wheel drive':				'all',
				'4-wheel or all-wheel drive':	'all',
				'part-time 4-wheel drive':		'all',
				},
			}
		}

```

e.g. for `vin`, where `fuelType1` is `compressed natural gas (cng)`, it's replaced with `natural gas`; where `drive` is `4x2`, it becomes `two`.

10) Modify fuel type to identify flexible fuel vehicles and electric vehicles. 
	10) a) Flexible fuel vehicles. 
	- In both `epa` and `vin`,  whenever `fuelType1` or `model` or `fuelType2` contains any of the following: `ffv`, `flexible`, `ethanol`, `e85`, or `natural gas`, `fuelType1_mod` becomes `ffv`. 
	- In `epa`, whenever the variable `atvType` contains `bi` or `ffv`, or `eng_dscr` contains `ffv`, `fuelType1_mod` is set to `ffv`.

## Split and Expand.

In `vin`, split models using the regex `[\w -]+`, e.g. `575 m maranello/575 m maranello f1` becomes `575 m maranello` and `575 m maranello f1`; `sl2, sw2` becomes `sl2` and `sw2`.
In `epa`, split any model name that contains the string `'/|,'`; e.g. `b2000/b2200/b2600` becomes `b2000`, `b2200`, and `b2600`; and use regex magic to catch strings that need to be duplicated `rally g15/25 2wd (passenger)` becomes `rally g15 2wd (passenger)`, `rally g25 2wd (passenger)`.

## Modify Datasets. 

Modify datasets such that the fields that are being matched correspond and add custom variables. 

1) Extract displacement, transmission speeds, type, 
2) Add tonnage variable (named `type`). 1/4 and 1/2 ton has a type of `15`, 3/4 ton is `25`, and 1 ton is `35`. 
3) Add `weight` variable based on `GVWR`. We're extracting the upper bound of the range `GVWR`.
4) Modify models and makes so they correspond; e.g. `pathfinder armada` becomes `armada`; `accord crosstour` becomes `crosstour`. 

## Merging.

1) Merge using `make`, `model_mod`, `year`, `fuelType1_mod`, `type` and all possible combinations of the following: `drive_mod`, `displ_mod`, `cylinders`, `transmission_type_mod`, `transmission_speeds_mod`, while dropping 1 field, then 2, etc. and eventually all fields, successively; i.e.
	a) match on all fields first: `make`, `model_mod`, `year`, `fuelType1_mod`, `type`, `drive_mod`, `displ_mod`, `cylinders`, `transmission_type_mod`, `transmission_speeds_mod`; 
	b) drop 1 field:
		- drop `transmission_speeds_mod`, match on `make`, `model_mod`, `year`, `fuelType1_mod`, `type`, `drive_mod`, `displ_mod`, `cylinders`, `transmission_type_mod`;
		- drop `transmission_type_mod`, match on `make`, `model_mod`, `year`, `fuelType1_mod`, `type`, `drive_mod`, `displ_mod`, `cylinders`, `transmission_speeds_mod`;
		- ...
	c) drop 2 fields:
		- drop `transmission_type_mod` and `tranmission_speeds_mod`, match on `make`, `model_mod`, `year`, `fuelType1_mod`, `type`, `drive_mod`, `displ_mod`, `cylinders`;
		- drop `displ_mod`, `cylinders`, match on `make`, `model_mod`, `year`, `fuelType1_mod`, `type`, `drive_mod`, `transmission_type_mod`, `transmission_speeds_mod`
		- ...
	d) ...

2) Merge using 'make', 'model_mod', 'year', 'type' (same as 1) but drop the fuel type) and use the same logic as 1) with the rest of the fields. 

3) For all models that haven't been matched, if `weight` is above 8,000, tag as heavy and take out of the merging process. 

4) Merge using 'make', 'model_mod', 'year', 'fuelType1_mod' only.

Results of the merge:

```
Merging datasets
**************************************************
('Merging using:', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type'])
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'displ_mod', 'cylinders', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 1.31%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'displ_mod', 'cylinders', 'transmission_type_mod'])
Weighted match fraction: 1.63%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'displ_mod', 'cylinders', 'transmission_speeds_mod'])
Weighted match fraction: 1.65%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'displ_mod', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 1.88%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'cylinders', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 1.88%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'displ_mod', 'cylinders', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 6.65%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'displ_mod', 'cylinders'])
Weighted match fraction: 48.72%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'displ_mod', 'transmission_type_mod'])
Weighted match fraction: 48.92%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'displ_mod', 'transmission_speeds_mod'])
Weighted match fraction: 48.92%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'cylinders', 'transmission_type_mod'])
Weighted match fraction: 48.92%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'cylinders', 'transmission_speeds_mod'])
Weighted match fraction: 48.92%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 48.92%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'displ_mod', 'cylinders', 'transmission_type_mod'])
Weighted match fraction: 50.98%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'displ_mod', 'cylinders', 'transmission_speeds_mod'])
Weighted match fraction: 50.99%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'displ_mod', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 51.29%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'cylinders', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 51.33%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'displ_mod'])
Weighted match fraction: 56.45%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'cylinders'])
Weighted match fraction: 58.11%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'transmission_type_mod'])
Weighted match fraction: 58.16%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod', 'transmission_speeds_mod'])
Weighted match fraction: 58.16%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'displ_mod', 'cylinders'])
Weighted match fraction: 80.25%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'displ_mod', 'transmission_type_mod'])
Weighted match fraction: 80.38%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'displ_mod', 'transmission_speeds_mod'])
Weighted match fraction: 80.38%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'cylinders', 'transmission_type_mod'])
Weighted match fraction: 80.46%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'cylinders', 'transmission_speeds_mod'])
Weighted match fraction: 80.46%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 80.66%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'drive_mod'])
Weighted match fraction: 82.12%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'displ_mod'])
Weighted match fraction: 86.44%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'cylinders'])
Weighted match fraction: 87.13%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'transmission_type_mod'])
Weighted match fraction: 87.33%
('Matching on ', ['make', 'model_mod', 'year', 'fuelType1_mod', 'type', 'transmission_speeds_mod'])
Weighted match fraction: 87.33%
**************************************************
('Merging using:', ['make', 'model_mod', 'year', 'type'])
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'displ_mod', 'cylinders', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 87.33%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'displ_mod', 'cylinders', 'transmission_type_mod'])
Weighted match fraction: 87.33%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'displ_mod', 'cylinders', 'transmission_speeds_mod'])
Weighted match fraction: 87.33%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'displ_mod', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 87.33%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'cylinders', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 87.33%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'displ_mod', 'cylinders', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 87.35%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'displ_mod', 'cylinders'])
Weighted match fraction: 88.14%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'displ_mod', 'transmission_type_mod'])
Weighted match fraction: 88.14%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'displ_mod', 'transmission_speeds_mod'])
Weighted match fraction: 88.14%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'cylinders', 'transmission_type_mod'])
Weighted match fraction: 88.14%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'cylinders', 'transmission_speeds_mod'])
Weighted match fraction: 88.14%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 88.14%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'displ_mod', 'cylinders', 'transmission_type_mod'])
Weighted match fraction: 88.67%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'displ_mod', 'cylinders', 'transmission_speeds_mod'])
Weighted match fraction: 88.67%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'displ_mod', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 88.67%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'cylinders', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 88.68%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'displ_mod'])
Weighted match fraction: 88.76%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'cylinders'])
Weighted match fraction: 89.10%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'transmission_type_mod'])
Weighted match fraction: 89.10%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod', 'transmission_speeds_mod'])
Weighted match fraction: 89.10%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'displ_mod', 'cylinders'])
Weighted match fraction: 89.48%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'displ_mod', 'transmission_type_mod'])
Weighted match fraction: 89.48%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'displ_mod', 'transmission_speeds_mod'])
Weighted match fraction: 89.48%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'cylinders', 'transmission_type_mod'])
Weighted match fraction: 89.48%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'cylinders', 'transmission_speeds_mod'])
Weighted match fraction: 89.48%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'transmission_type_mod', 'transmission_speeds_mod'])
Weighted match fraction: 89.48%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'drive_mod'])
Weighted match fraction: 89.59%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'displ_mod'])
Weighted match fraction: 89.63%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'cylinders'])
Weighted match fraction: 89.70%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'transmission_type_mod'])
Weighted match fraction: 89.71%
('Matching on ', ['make', 'model_mod', 'year', 'type', 'transmission_speeds_mod'])
Weighted match fraction: 89.71%
**************************************************
('Matching on', ['make', 'model_mod', 'year', 'fuelType1_mod'])
Weighted match fraction: 99.35%
**************************************************
```

