# import my libs
from advanced_vgg16_to_classify_hymenoptera.image_transform import ImageTransform
from advanced_vgg16_to_classify_hymenoptera.my_dataset import HymenopteraDataset
from advanced_vgg16_to_classify_hymenoptera.utils import MyUtils

# import libs for torch
import torch
from torch import nn
from torch import optim
import torch.utils.data as data
import torchvision
from torchvision import models

# import common libs 
from tqdm import tqdm

def configure_gpu(enable_benchmark = True):
    ret_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = enable_benchmark
    return ret_device

if __name__ == "__main__":
    print("hoge")

    # prepare train ----
    ## configure dataset & dataloader.
    param_dict =  {"resize":224, "mean":(0.485, 0.456, 0.406), "std":(0.229, 0.224, 0.225)}
    image_transform = ImageTransform(param_dict["resize"], param_dict["mean"], param_dict["std"])

    file_list_dict = {}
    dataset_dict = {}
    dataloader_dict = {}

    user_batch_size = 10
    shuffle_on_dict = {"train": True, "val": False}

    for phase in ["train", "val"]:
        curr_file_list = MyUtils.make_datapath_list(phase)
        file_list_dict[phase] = curr_file_list

        curr_dataset = HymenopteraDataset(curr_file_list, image_transform, phase)
        dataset_dict[phase] = curr_dataset
        dataloader_dict[phase] = data.DataLoader(curr_dataset, user_batch_size, shuffle_on_dict[phase])

    ## import vgg model & re-configure output's layer of vgg.
    vgg_net = models.vgg16(pretrained=True)

    ### re-configure
    print(vgg_net.classifier[6])
    vgg_net.classifier[6] = nn.Linear(in_features = 4096, out_features = 2)
    print(vgg_net.classifier[6])

    ## configure criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    
    ### extract params from my-vgg_net for optimizer 
    keys_of_param_to_update = ["classifier.6.weight", "classifier.6.bias"]
    extracted_params = []

    ### search specified name from vgg_net.named_parameters
    for name, param in vgg_net.named_parameters():
        print(name, param)

        if (name in keys_of_param_to_update):
            extracted_params.append(param)

    optimizer = optim.SGD(params = extracted_params, lr = 0.001, momentum = 0.9)

    # print(optimizer)
    # ---- prepare train 

    num_epochs = 3

    device = configure_gpu()
    vgg_net.to(device)

    print()

    for curr_epoch in tqdm(range(num_epochs)):
        for phase in ["train", "val"]:
            if (curr_epoch == 0) and (phase == "train"):
                continue

            sum_loss_of_curr_epoch = 0.0
            num_accuracy_of_curr_epoch = 0

            for inputs, labels in dataloader_dict[phase]:
                # print(inputs.size())
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = vgg_net(inputs)     
                    curr_loss = criterion(outputs, labels)  ## this is averaged using the size of the current batch.
                    # print(curr_loss)
                    _, preds = torch.max(outputs, 1)
                    # print(preds)

                    if (phase == "train"):
                       curr_loss.backward()
                       optimizer.step()

                    sum_loss_of_curr_epoch += curr_loss * inputs.size(0)
                    num_accuracy_of_curr_epoch += torch.sum(preds == labels)

            # print()
            # print(num_accuracy)
            acc_rate = num_accuracy_of_curr_epoch.double() / len(dataloader_dict[phase].dataset)
            loss_of_objective_func = sum_loss_of_curr_epoch.double() / len(dataloader_dict[phase].dataset)
            print("phase:", phase, ", accuracy:", acc_rate, ", total loss (of objective function):", loss_of_objective_func)

