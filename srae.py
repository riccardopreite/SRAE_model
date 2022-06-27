from copy import deepcopy
import math
import random
import torch, json
import pandas as pd
from multiple_optimizer import MultipleOptimizer
from torch_optimizer import Lookahead
'''Sustainable regressive Auto Encoder model'''
class SRAE(torch.nn.Module):
    train_cols = [
        "iteration", "year", "index_loss", "ae_loss",
        "ground_truth_index", "index_predicted", "index_error",
    ]
    test_cols = ["Year", "index_loss", "ae_loss"]
    train_loss_df = pd.DataFrame([], columns=train_cols)
    test_loss_df = pd.DataFrame([], columns=test_cols)
    def __init__(self, indicator_number=1, task_number=1, macro_number=1):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.activation = self.tanh

        self.fc0 = torch.nn.Linear(
            in_features=indicator_number,
            out_features=task_number,
        )
        self.fc1 = torch.nn.Linear(
            in_features=task_number,
            out_features=macro_number,
        )

        self.fc_regressive = torch.nn.Linear(
            in_features=macro_number,
            out_features=1,
        )

        self.fc0_decoder = torch.nn.Linear(
            in_features=macro_number,
            out_features=task_number,
        )
        self.fc1_decoder = torch.nn.Linear(
            in_features=task_number,
            out_features=indicator_number,
        )
            
        self.encoder = torch.nn.Sequential(
            self.fc0,
            self.activation,
            self.fc1,
            self.activation
        )
		
        self.regressive = torch.nn.Sequential(
            self.fc_regressive,
			self.sigmoid
        )

        self.decoder = torch.nn.Sequential(
            self.fc0_decoder,
            self.activation,
            self.fc1_decoder,
            self.sigmoid
        )
        self.lr = 0.022
        self.regressive_loss = torch.nn.MSELoss()
        self.autoencoder_loss = torch.nn.MSELoss()
        self.ae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.ae_optimizer = torch.optim.Adam(self.ae_params, lr=self.lr)
        self.ae_optimizer = Lookahead(optimizer=self.ae_optimizer,k=5,alpha=0.5)
        self.regressive_optimizer = torch.optim.SGD(self.regressive.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = MultipleOptimizer(self.regressive_optimizer, self.ae_optimizer)
        self.loss = 0
    
    def forward(self, ground_truth_input, ground_truth_index, train=False, validation=False, predict=False):
        regressive_loss = None
        ae_loss = None

        '''Forward pass'''
        encoded = self.encoder(ground_truth_input)
        index_predicted = self.regressive(encoded)
        input_predicted = self.decoder(encoded)

        '''Computing loss if net is training and not predicting or validating'''
        if not validation and train:
            self.optimizer.zero_grad()
        
        if not predict:
            ground_truth_index = torch.flatten(ground_truth_index,start_dim=-1,end_dim=0)
            ae_loss = self.autoencoder_loss(input_predicted.float(),ground_truth_input.float()).float()
            regressive_loss = self.regressive_loss(index_predicted.float(),ground_truth_index.float()).float()
            self.loss = regressive_loss + ae_loss
        
        if not validation and train:
            self.loss.backward() 
            self.optimizer.step()
            
        return index_predicted.detach().clone().item(), input_predicted, regressive_loss, ae_loss

    def dump_df(self, output_name):
        self.test_loss_df.to_csv(output_name+"_test_df.csv",header=True, index=False)
        self.train_loss_df.to_csv(output_name+"_train_df.csv",header=True, index=False)


    def rebuild_input(self, input):
        keys = list(self.max_min.keys())
        for index, _ in enumerate(input):
            multiplier = self.max_min[keys[index]]['max']-self.max_min[keys[index]]['min']
            input[index] = (input[index] * multiplier) + self.max_min[keys[index]]['min']

        return input

    def max_min_init(self, max_min):
        self.max_min = max_min
        return 
    
    def normalize_json(self, j_input):
        for _, key in enumerate(list(j_input.keys())):
            if j_input[key] == None:
                j_input[key] = random.random()

            elif not math.isnan(j_input[key]):
                if j_input[key] > self.max_min[key]['max']:
                    self.max_min[key]['max'] = j_input[key]

                if j_input[key] < self.max_min[key]['min']:
                    self.max_min[key]['min'] = j_input[key]
                divider = self.max_min[key]['max']-self.max_min[key]['min']
                j_input[key] = (j_input[key] - self.max_min[key]['min']) / divider
        return j_input

    def normalize_input(self, received):
        if torch.is_tensor(received):
            input = received.detach().clone()
        else:
            input = deepcopy(received)
        keys = list(self.max_min.keys())
        for index, _ in enumerate(input):
            if input[index] == None:
                print("default val for none value")
                input[index] = random.random()
            elif not math.isnan(input[index]):
                if input[index] > self.max_min[keys[index]]['max']:
                    print("update max for",keys[index])
                    self.max_min[keys[index]]['max'] = input[index]
                if input[index] < self.max_min[keys[index]]['min']:
                    print("update min for",keys[index])
                    self.max_min[keys[index]]['min'] = input[index]
                divider = self.max_min[keys[index]]['max']-self.max_min[keys[index]]['min']

                print("for key:",keys[index])
                print("\tmax",self.max_min[keys[index]]['max'])
                print("\tmin",self.max_min[keys[index]]['min'])
                print("\tval",input[index])
                input[index] = (input[index] - self.max_min[keys[index]]['min']) / divider
                print("\tnew val",input[index])
        return input
    
    def dump_max_min(self, output_name):
        with open(output_name+'.json', 'w') as outfile:
            json.dump(self.max_min, outfile)
            
    def load_max_min(self, input_name):
        with open(input_name+'.json') as json_file:
            self.max_min = json.load(json_file)
    
    