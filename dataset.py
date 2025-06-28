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


#     def __getitem__(self, index: int):
#         race_df = self.race_groups[index][1]  

#         x_cat_dict = {
#             col: torch.tensor(race_df[col].values, dtype=torch.long).unsqueeze(1)  # shape [20,1]
#             for col in self.cat_columns
#         }

#         x_num = torch.tensor(race_df[self.numerical_columns].values, dtype=torch.float)

#         y = torch.tensor(race_df[self.target_colname].values, dtype=torch.float)

#         return x_cat_dict, x_num, y











# # import torch
# # from torch.utils.data import Dataset, DataLoader
# # import pandas as pd


# # class CustomDataset(Dataset):
# #     def __init__(self, dataframe: pd.DataFrame, target_colname: str, numerical_columns: list, cat_columns: list):
# #         #self.X = torch.tensor(dataframe[feature_colnames].values, dtype=torch.float32)
# #         self.race_groups = list(dataframe.groupby(["season", "round"]))
# #         numerical_features = dataframe[numerical_columns]
# #         cat_features = dataframe[cat_columns]
# #         #broj unkata vrednosti za svaki kategoricki feature
# #         num_of_cat_features = dict()
# #         for col in cat_features.columns:
# #             num_of_cat_features[col] = len(list(set(cat_features[col])))
        
# #         self.num_of_cat_features = num_of_cat_features
# #         self.cat_dit_tensors = {
# #             col: torch.tensor(dataframe[col].values, dtype=torch.long)
# #             for col in cat_columns
# #         }

# #         self.race_groups = list(dataframe.groupby(["season", "round"]))

        
# #         self.y = torch.tensor(dataframe[target_colname].values, dtype=torch.float64)
# #         self.num_tensor = torch.tensor(numerical_features, dtype=torch.float64)
        

# #     def __len__(self) -> int:
# #         return len(self.y)

# #     def __getitem__(self, index: int):
# #         #promenti skrozzzzzzzz
# #         # return self.num_tensor[index * 20 : index * 20 + 20], self.cat_dict_tensor[index * 20 : index * 20 + 20], self.y[index * 20 : index * 20 + 20]
        
# #         race_df = self.race_groups[index][1]  # [1] gives the DataFrame, [0] would be ("season", round)

# #         # Categorical features — dict of tensors [20]
# #         x_cat_dict = {
# #             col: torch.tensor(race_df[col].values, dtype=torch.long).unsqueeze(1)  # → [20, 1]
# #             for col in self.cat_columns
# #         }

# #         # Numerical features — tensor [20, num_numerical]
# #         x_num = torch.tensor(race_df[self.numerical_columns].values, dtype=torch.float)

# #         # Target — tensor [20]
# #         y = torch.tensor(race_df[self.target_colname].values, dtype=torch.float)

# #         return x_cat_dict, x_num, y