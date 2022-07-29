# Comparing Rebalancing Strategies in Car Sharing Systems
This is the code for semester project ***Comparing Rebalancing Strategies in Car Sharing Systems*** done in TRANS-OR lab at EPFL. Please refer to `Semester_project_Sharing_car_rebalancing_xinlingli.pdf` for the detail of the project.

----
## Files
`model/` contains the following files:
- `model1.py`: Implementation of the model from [this paper](https://www.sciencedirect.com/science/article/pii/S0305048317302803) based on cplex.
- `model2.py`: Implementation of the model from [this paper](https://www.sciencedirect.com/science/article/pii/S0191261517302965) based on cplex.
- `model3.py`: Implementation of a model adpoted from model1 based on cplex.

Replace "NUMBER_OF_VEHICLE" and "NUMBER_OF_STAFF" in the model files with expected values to run them. The result will be output as a .json file in the same folder.

All the data files needed for running models are included in `data/`:
- `Torino_bookings.csv`: Original dataset downloaded from [here](https://data.mendeley.com/datasets/drtn5499j2/1). You can find the description of the original data in the link.
- `data_processed.csv`: Filter `Torino_booings.csv` by only keeping trip duration between 5 minutes and 2 hours. You can run the ***Data cleaning*** section in `preprocessing.ipynb` to get this file.
- `distance_10.csv`: Distance between stations from Google Maps API. You can run the ***Station clustering***  and ***Distance calculation*** sections in `preprocessing.ipynb` to get this file. Remember to replace "YOUR_KEY" with your Google Maps API key.
- `model_net_oneday_10_25`: Input file for `model2.py`. You can run the ***Input for model 2*** section in `preprocessing.ipynb` to get this file.
- `model1_3/`: Input files for `model1.py` and `model3.py`. You can run the ***Input for model 1*** and ***Input for model 3*** sections in `preprocessing.ipynb` to get these file.