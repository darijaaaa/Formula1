import torch
import torch.nn as nn
import torch.nn.functional as F



class DnnWOTransformer(nn.Module):
    def __init__(self, embeddings_dict, num_numerical, embedding_dim=8, hidden_dim=136, dropout_p=0.4):
        super(DnnWOTransformer, self).__init__() 
        #sloj za embedovanje
        self.embeddings = nn.ModuleDict({
            name : nn.Embedding(num_categories, embedding_dim)
            for name, num_categories in embeddings_dict.items()}
        )


        #broj feature-a po vozacu
        self.dim_features = len(embeddings_dict) * embedding_dim + num_numerical
        
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.dnn = nn.Sequential(
            nn.Linear(self.dim_features, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, 1)
        )

    #ja modelu prosledjujem dict tipa {feature_name : num_of_dif_values_of_feature}, categorijske feature, i numericke feature
    def forward(self, cat_features, num_tensor):
        
        num_drivers, batch_size  = cat_features[list(cat_features.keys())[0]].shape
        cat_input_flat = {
            name: cat_features[name].view(batch_size * num_drivers)
            for name in cat_features
        }


        #cat features treba da bude dictonary feature-a kao {'ime feature-a': svi featuri}
        embedded = [self.embeddings[name](cat_input_flat[name]) for name in self.embeddings]
        cat_tensor_embeded = torch.cat(embedded, dim=1)
        

        num_tensor_flat = num_tensor.view(batch_size * num_drivers, -1)


        x = torch.cat([cat_tensor_embeded, num_tensor_flat], dim=1)

        out = self.dnn(x)
        return out.view(num_drivers, batch_size, 1)

        