import math
import os
import sys
from time import sleep
import numpy as np
import pandas as pd
import torch
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
MODEL_DIR_NAME = "SRAE_model"
sys.path.insert(1, os.path.join(os.getcwd(), "..",MODEL_DIR_NAME,"preprocess"))
from create_bidimensional_data import split_indicators

sys.path.insert(1, './srae')
SUSTAINABLE_PATH = os.path.join(os.getcwd(), "..",MODEL_DIR_NAME)

MAX_EPOCH = 3
from srae import SRAE


device = torch.device("cpu")
def build_torch():
    torch.autograd.set_detect_anomaly(True)
    global device
    if torch.cuda.is_available():
        print("USING CUDA")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    torch.manual_seed(50)

def build_model(indicator_number, task_number, macro_number):
    srae = SRAE(indicator_number, task_number, macro_number)
    srae = srae.to(device)
    return srae
    
def normalize(dataset):
    max_min = {}
    if dataset.empty:
        return dataset, max_min
    for col in dataset.columns:
        if col != "GeoAreaName" and col != "year":
            if not (dataset[col] == dataset[col].iloc[0]).all():
                max_min[col] = {}
                max_min[col]["max"] = dataset[col].max()
                max_min[col]["min"] = dataset[col].min()
                max_min[col]["divider"] = dataset[col].max()-dataset[col].min()
                dataset[col] = (dataset[col]-dataset[col].min()) / (dataset[col].max() - dataset[col].min())
            else:
                max_min[col] = {}
                max_min[col]["max"] = dataset[col].max()
                max_min[col]["min"] = dataset[col].min()
                max_min[col]["divider"] = 0.00001
                dataset[col] = 1

    return dataset, max_min


def validation(model, valid_df, full_target):

    model.eval()
    loss_total = 0
    with torch.no_grad():
        for index, full_input in valid_df.iterrows():
            geoname = full_input["GeoAreaName"]
            year = full_input["year"]
            full_input.drop("GeoAreaName", inplace=True)
            full_input.drop("year", inplace=True)
            ground_truth_input = torch.tensor(full_input).to(device)
            target = full_target[full_target["Country"]==geoname][str(year)].values[0]
            if not math.isnan(target):
                ground_truth_index = torch.tensor(target).to(device)

            index_predicted, input_predicted, regressive_loss, ae_loss = model(ground_truth_input, ground_truth_index, True, True)
            loss_total += (regressive_loss+ ae_loss).item()
        return loss_total / len(valid_df)

def test(model, test_df, full_target, name):

    model.eval()
    loss_total = 0
    with torch.no_grad():
        for index, full_input in test_df.iterrows():
            geoname = full_input["GeoAreaName"]
            year = full_input["year"]
            full_input.drop("GeoAreaName", inplace=True)
            full_input.drop("year", inplace=True)
            ground_truth_input = torch.tensor(full_input).to(device)
            target = full_target[full_target["Country"]==geoname][str(year)].values[0]
            if not math.isnan(target):
                ground_truth_index = torch.tensor(target).to(device)

            index_predicted, input_predicted, regressive_loss, ae_loss = model(ground_truth_input, ground_truth_index, True, True)
            loss_total += (regressive_loss+ ae_loss).item()
            writer.add_scalar(name+": Loss/test", (regressive_loss+ ae_loss).item(), index)
    return loss_total / len(test_df)



def train(input_dir, output_path, data_type, is_early):
    
    build_torch()
    model = torch.load(output_path+".pt")
    name = os.path.split(output_path)[-1]
    indicator = len(model['fc0.weight'][0])
    MAX_EPOCH = indicator*6
    task = len(model['fc1.weight'][0])
    macro = len(model['fc_regressive.weight'][0])
    batch_size = 16
    srae = SRAE(
        indicator_number=indicator, 
        task_number=task, 
        macro_number=macro
    )
    srae.load_state_dict(model)
    print(srae)
    srae.to(device)
    years_path = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    full_target = pd.read_csv(os.path.join(SUSTAINABLE_PATH, 'data', 'index', data_type+'sdi.csv'))
    data = []

    '''Building unique dataset'''
    for train_path in years_path:
        dataset = pd.read_csv(train_path)
        year_target = os.path.split(train_path)[1].replace('.csv', '')
        dataset["year"] = int(year_target)
        data.append(dataset)
    full_dataset = pd.concat(data,ignore_index=True)
    full_dataset, max_min = normalize(full_dataset)
    srae.max_min_init(max_min)
    '''END'''

    '''Removing sdi that are not in train dataset'''
    country = full_dataset["GeoAreaName"].unique().tolist()
    full_target = pd.DataFrame([row for _, row in full_target.iterrows() if row["Country"] in country], columns=full_target.columns)
    '''END'''
    
    '''Splitting train/validation/test set'''
    train_validation_set = full_dataset.sample(n = int(full_dataset.shape[0]*0.8))
    
    test_set = full_dataset.drop(train_validation_set.index)
    train_set = train_validation_set.sample(n = int(train_validation_set.shape[0]*0.8))
    validation_set = train_validation_set.drop(train_set.index)

    train_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 
    early_stopping = EarlyStopping(patience=5, verbose=True, path=output_path+".pt")
    for epoch in range(1, MAX_EPOCH+1):
        srae.train()
        srae.float()
        compute_loss = False
        batch_index = 1
        train_set = train_set.sample(frac = 1)
        for _, full_input in train_set.iterrows():
            geoname = full_input["GeoAreaName"]
            year = full_input["year"]
            full_input.drop("GeoAreaName", inplace=True)
            full_input.drop("year", inplace=True)
            ground_truth_input = torch.tensor(full_input).to(device)
            target = full_target[full_target["Country"]==geoname][str(year)].values[0]

            if not math.isnan(target):

                ground_truth_index = torch.tensor(target).to(device)
                if (batch_index % batch_size) == 0 or (batch_index+1) == len(train_set):
                    compute_loss = True
                index_predicted, input_predicted, regressive_loss, ae_loss = srae(ground_truth_input, ground_truth_index, compute_loss, False)
                if compute_loss:
                    train_losses.append((regressive_loss + ae_loss).item())
                    
                compute_loss = False
                batch_index += 1
        train_loss = np.average(train_losses)
        writer.add_scalar(name+": Loss/train", train_loss, epoch)
        valid_loss = validation(srae, validation_set, full_target)
        writer.add_scalar(name+": Loss/validation", valid_loss, epoch)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(MAX_EPOCH))
        
        print_msg = (f'For model: {name} [{epoch:>{epoch_len}}/{MAX_EPOCH:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        train_losses = []
        early_stopping(valid_loss, srae)
        
        if early_stopping.early_stop and is_early:
            print("Early stopping")
            break

    srae.load_state_dict(torch.load(output_path+".pt"))
    srae.dump_max_min(output_path)
    test_loss = test(srae,test_set,full_target, name)
    print("Test Loss:",test_loss)
    return "Training process finished."



def train_template(output_path, indicators, tasks, data_type,is_early=False):
    split_indicators(output_path, indicators, tasks, data_type)
    bidimensional_data_path = os.path.join(output_path,"years")
    print("Data preprocessing ended...\nStarting building model")
    sleep(1)
    return train(bidimensional_data_path, output_path, data_type, is_early)


def create_template(output_path, indicators, tasks, macros):

    srae = SRAE(
        indicator_number=len(indicators), 
        task_number=len(tasks), 
        macro_number=len(macros),
    )
    
    srae = srae.to(device)
    torch.save(srae.state_dict(), output_path+".pt")
    print("Model builded:")
    print(srae)
    return srae




# def train_template_no_create_indicators(output_path, indicators, tasks, data_type):

#     bidimensional_data_path = os.path.join(output_path,"years")
#     print("Data preprocessing ended...\nStarting building model")
#     sleep(1)
#     return train(bidimensional_data_path, output_path, data_type)

# def init_model(output_path, indicators_number, tasks_number, macro_number):
#     build_torch()
#     srae = build_model(indicators_number, tasks_number, macro_number)
#     torch.save(srae.state_dict(), output_path+".pt")
#     print("Model builded:")
#     print(srae)
#     return srae

    


    



    
    
# def train(year_target, srae, train_data_tensor, train_target_tensor):
#     tot_loss_regressive = 0
#     tot_loss_ae = 0
#     for epoch in range(0, MAX_EPOCH):

#         index_running_loss = 0.0
#         ae_running_loss = 0.0
#         i = 0
    
#         for _, full_input in train_data_tensor.iterrows():

#             geoname = full_input["GeoAreaName"]
#             full_input.drop("GeoAreaName", inplace=True)
#             ground_truth_input = torch.tensor(full_input)
#             ground_truth_input = ground_truth_input.cuda()
            
#             target = train_target_tensor[train_target_tensor["Country"]==geoname].values[0][1]
#             if not math.isnan(target):

#                 ground_truth_index = torch.tensor(target) # next(targetit)
#                 ground_truth_index = ground_truth_index.cuda()
#                 index_predicted, input_predicted, regressive_loss, ae_loss = srae(ground_truth_input, ground_truth_index, True)
                
                        
#                 index_running_loss += regressive_loss.item()
#                 ae_running_loss += ae_loss.item()

#                 if i % 50 == 49:
#                     index_error = abs(ground_truth_index - index_predicted)
#                     input_error = [round(n, 5) for n in abs(ground_truth_input - input_predicted).detach().cpu().numpy().tolist()]

#                     input_predicted = [round(n, 5) for n in input_predicted.detach().cpu().numpy().tolist()]
#                     ground_truth_input = [round(n, 5) for n in ground_truth_input.cpu().numpy().tolist()]
#                     ground_truth_index = ground_truth_index.item()
                    
#                     tmp_loss1 = index_running_loss / 50
#                     tmp_loss2 = ae_running_loss / 50
#                     os.system('cls||clear')
#                     new_desc = str(
#                         bcolors.HEADER  + f'At iteration {i+1} of epoch:{epoch + 1} for year: {year_target}:\nindex loss: {tmp_loss1:.4f}, ae loss: {tmp_loss2:.4f}'+ bcolors.ENDC+
#                         bcolors.OKGREEN + f"\n\tWith input: {ground_truth_input}"        + bcolors.ENDC+
#                         bcolors.OKCYAN  + f"\n\tInput predicted: {input_predicted}"      + bcolors.ENDC+
#                         bcolors.FAIL    + f"\n\t\tError input(BCE):{input_error}"        + bcolors.ENDC+
#                         bcolors.OKGREEN + f"\n\tWith output: {ground_truth_index:.5f}"   + bcolors.ENDC+
#                         bcolors.OKCYAN  + f"\n\tOutput predicted: {index_predicted:.5f}" + bcolors.ENDC+
#                         bcolors.FAIL    + f"\n\t\tError index(MSE): {index_error:.5f}"   + bcolors.ENDC
#                         )
#                     print(new_desc)

#                     tot_loss_regressive += index_running_loss
#                     tot_loss_ae += ae_running_loss
#                     index_running_loss = 0.0
#                     ae_running_loss = 0.0

#                 i+=1
#     print(f'Training process of merged net has finished. Total loss: {tot_loss_regressive / (len(train_data_tensor)*MAX_EPOCH):.10f}, ae loss: {tot_loss_ae / (len(train_data_tensor)*MAX_EPOCH):.10f}')
#     sleep(3)
#     return srae

