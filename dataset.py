import torch
from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, target_colname: str, numerical_columns: list, cat_columns: list):
        self.race_groups = list(dataframe.groupby(["season", "round"]))
        
        self.cat_columns = cat_columns
        self.numerical_columns = numerical_columns
        self.target_colname = target_colname
        
        self.numerical_features = dataframe[numerical_columns]
        self.cat_features = dataframe[cat_columns]

        # broj unikatnih vrednosti za svaki kategoricki feature 
        self.num_of_cat_features = {
            col: len(self.cat_features[col].unique())
            for col in cat_columns
        }

    def __len__(self) -> int:
        return len(self.race_groups)
    

    def __getitem__(self, index: int):
        race_df = self.race_groups[index][1]

        x_cat_dict = {
            col: torch.tensor(race_df[col].values, dtype=torch.long).unsqueeze(1)
            for col in self.cat_columns
        }

        x_num = torch.tensor(race_df[self.numerical_columns].values, dtype=torch.float)
        y = torch.tensor(race_df[self.target_colname].values, dtype=torch.float)

        # Dodatni podaci za ispis (drivers_num, position_quali)
        drivers_num = torch.tensor(race_df['drivers_num'].values, dtype=torch.long)
        position_quali = torch.tensor(race_df['position_quali'].values, dtype=torch.long)

        return x_cat_dict, x_num, y, drivers_num, position_quali