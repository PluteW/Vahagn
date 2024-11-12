import os

import torch
from torch import optim
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")


from libs.models import network
from libs.models.Space.SpaceAttentionNet import SpaceAttentionNet
from libs.utils import data_loader

params = {}
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train_net(params):

    writer = SummaryWriter(log_dir=params['save_dir']+f"/logs/{params['run_time']}")
    # dummy_input = [torch.zeros(1,3,480,640) for i in range(8)]
    # Create network
    slip_detection_model = SpaceAttentionNet(base_network=params['cnn'], pretrained=params['pretrained'],
                                                          use_gpu=params['use_gpu'],
                                                          dropout=params['dropout'])
    if params['use_gpu']:
        slip_detection_model = slip_detection_model.to(device)
    # Some Warnings in there.
    # writer.add_graph(slip_detection_model, input_to_model=(dummy_input, dummy_input))

    if 'net_params' in params.keys():
        assert params['net_params'].endswith('.pth'), "Wrong model path {}".format(params['net_params'])
        net_params_state_dict = torch.load(params['net_params'])
        slip_detection_model.load_state_dict(net_params_state_dict)

    # Init optimizer & loss func.
    optimizer = optim.Adam(slip_detection_model.parameters(), lr=params['lr'])
    loss_function = nn.CrossEntropyLoss()

    # Dataloader
    train_dataset = data_loader.Tactile_Vision_dataset(data_path=params['train_data_dir'], transform=True)
    train_data_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                   num_workers=params['num_workers'])
    test_dataset = data_loader.Tactile_Vision_dataset(data_path=params['test_data_dir'], transform=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=params['num_workers'])
    #   vaild_model
    vaild_dataset = data_loader.Tactile_Vision_dataset(data_path=params['valid_data_dir'], transform=True)
    vaild_data_loader = DataLoader(vaild_dataset, batch_size=1, shuffle=True, num_workers=params['num_workers'])
    
    # To record training procession
    train_loss = []
    train_acc = []

    # Start training
    for epoch in range(params['epochs']):
        # Start
        total_loss = 0.0
        total_acc = 0.0
        total = 0.0
        for i, data in enumerate(train_data_loader):
            # one iteration
            rgb_imgs, tactile_imgs, label = data
            output = slip_detection_model(rgb_imgs, tactile_imgs)
            if params['use_gpu']:
                label = label.to(device)
            loss = loss_function(output, label)

            # Backward & optimize
            slip_detection_model.zero_grad()
            loss.backward()
            optimizer.step()

            # cal training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == label).sum().item()
            total_loss += float(loss.data)
            total += len(label)
        train_loss.append(total_loss/total)
        train_acc.append(total_acc/total)

        writer.add_scalar('train_loss', train_loss[epoch],)
        writer.add_scalar('train_acc', train_acc[epoch],)
        if epoch%params['print_interval'] == 0:
            print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f'
                  % (epoch, params['epochs'], train_loss[epoch], train_acc[epoch],))
        if (epoch + 1)%params['test_interval'] == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                for rgb_imgs, tactile_imgs, labels in vaild_data_loader:
                    outputs = slip_detection_model(rgb_imgs, tactile_imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    if params['use_gpu']:
                        labels = labels.to(device)
                    correct += (predicted == labels).sum().item()
                test_acc = 100 * correct / total
                writer.add_scalar('test_acc', test_acc,)
                print('Valid Accuracy of the model on the {} Valid images: {} %'.format(total, test_acc))

            with torch.no_grad():
                correct = 0
                total = 0
                for rgb_imgs, tactile_imgs, labels in test_data_loader:
                    outputs = slip_detection_model(rgb_imgs, tactile_imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    if params['use_gpu']:
                        labels = labels.to(device)
                    correct += (predicted == labels).sum().item()
                vaild_acc = 100 * correct / total
                
                print('>>>>>> Test Accuracy of the model on the {} Test images: {} % <<<<<<'.format(total, vaild_acc))


        # Save 5 different model
        if epoch%(int(params['epochs']/5)) == 0:
            if 'save_dir' in params.keys():
                model_path = os.path.join(params['save_dir'], 'slip_detection_network_{:0>5}.pth'.format(epoch))
                if not os.path.exists(params['save_dir']):
                    os.makedirs(params['save_dir'])
                torch.save(slip_detection_model.state_dict(), model_path)


    
    

    if 'save_dir' in params.keys():
        model_path = os.path.join(params['save_dir'], 'slip_detection_network_{:0>6}.pth'.format(epoch))
        
        if not os.path.exists(params['save_dir']):
            os.makedirs(params['save_dir'])
            
        torch.save(slip_detection_model.state_dict(), model_path)
    writer.close()



if __name__ == '__main__':
    # No modification is recommended.
    params['rnn_input_size'] = 64
    params['rnn_hidden_size'] = 64
    params['num_classes'] = 2
    params['num_layers'] = 1
    params['use_gpu'] = False
    # if torch.cuda.is_available():
    params['use_gpu'] = True
    # Customer params setting.
    params['epochs'] = 10
    params['print_interval'] = 1
    params['test_interval'] = 1
    params['batch_size'] = 20
    params['num_workers'] = 1
    params['lr'] = 5e-4
    params['dropout'] = 0.8
    # C:/WorkSpace/TouchVisionGrab/Slip/Slip_small_dataset/Dataset
    params['train_data_dir'] = './data/training/'
    params['valid_data_dir'] = './data/validation/'
    params['test_data_dir'] = './data/testing/'
    params['frame'] = 8
    
    bone_model_name = 'vgg_16'
    params['run_time'] = "Spital"
    # Use Alextnet to debug.
    # You can choose vgg_16, vgg_19 or inception_v3(unreliable). Poor MBP
    params['cnn'] = bone_model_name
    params['pretrained'] = True # CNN is pretrained by ImageNet or not
    # params['net_params'] = 'model/pretrained_net/'

    params['save_dir'] = f'./model/{bone_model_name}'
    
    # Start train
    train_net(params)
