# Importing needed packages
import argparse
import os
import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
import torchvision.transforms as transform
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
from torchsummary import summary
import matplotlib.pyplot as plt

# Importing functions from our other files
import network
import loss
import lr_schedule
from galaxy_utils import EarlyStopping, image_classification_test, distance_classification_test, domain_cls_accuracy, visualizePerformance
from import_and_normalize import array_to_tensor, update
from visualize import plot_grad_flow, plot_learning_rate_scan

# Optimizer options. We use Adam.
optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}

def unnormalize(predictions, sfr = True):
    
    if sfr:
        return np.e**(predictions * (4.42656013 + 7.82404601) + -7.82404601)
    else:
        return np.e**(predictions*(28.25348346 - 21.87513544) + 21.87513544)
    
    return predictions

def remove_outliers(p, l):
    std = np.std(p)
    mean = np.mean(p)
    n = 2
    p_rv = p[(p >= mean - n * std) & (p <= mean + n * std)]
    l_rv = l[(p >= mean - n * std) & (p <= mean + n * std)]
    return p_rv, l_rv

def train(config):
    # Fix random seed and unable deterministic calcualtions
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    
    # Set up summary writer
    writer = SummaryWriter(config['output_path'])
    class_num = config["network"]["params"]["class_num"]
    
    class_criterion = nn.MSELoss()
    transfer_criterion = config["loss"]["name"]
    center_criterion = config["loss"]["discriminant_loss"](num_classes=class_num)
    loss_params = config["loss"]

    # Prepare image data. Image shuffling is fixed with the random seed choice.
    # Train:validation:test = 70:10:20
    dsets = {}
    dset_loaders = {}
    pristine_indices = torch.randperm(len(pristine_x))
    # Train sample

    pristine_x_train = pristine_x[pristine_indices[:int(np.floor(.7*len(pristine_x)))]]
    pristine_y_train = pristine_y[pristine_indices[:int(np.floor(.7*len(pristine_x)))]]
    # Validation sample --- gets passed into test functions in train file
    
    pristine_x_valid = pristine_x[pristine_indices[int(np.floor(.7*len(pristine_x))) : int(np.floor(.8*len(pristine_x)))]]
    
    pristine_y_valid = pristine_y[pristine_indices[int(np.floor(.7*len(pristine_x))) : int(np.floor(.8*len(pristine_x)))]]
    
    # Test sample for evaluation file
    pristine_x_test = pristine_x[pristine_indices[int(np.floor(.8*len(pristine_x))):]]
    pristine_y_test = pristine_y[pristine_indices[int(np.floor(.8*len(pristine_x))):]]
    noisy_indices = torch.randperm(len(noisy_x))
    # Train sample
    noisy_x_train = noisy_x[noisy_indices[:int(np.floor(.7*len(noisy_x)))]]
    noisy_y_train = noisy_y[noisy_indices[:int(np.floor(.7*len(noisy_x)))]]
    # Validation sample --- gets passed into test functions in train file
    noisy_x_valid = noisy_x[noisy_indices[int(np.floor(.7*len(noisy_x))) : int(np.floor(.8*len(noisy_x)))]]
    noisy_y_valid = noisy_y[noisy_indices[int(np.floor(.7*len(noisy_x))) : int(np.floor(.8*len(noisy_x)))]]
    # Test sample for evaluation file
    noisy_x_test = noisy_x[noisy_indices[int(np.floor(.8*len(noisy_x))):]]
    noisy_y_test = noisy_y[noisy_indices[int(np.floor(.8*len(noisy_x))):]]


    dsets["source"] = TensorDataset(pristine_x_train, pristine_y_train)
    dsets["target"] = TensorDataset(noisy_x_train, noisy_y_train)

    dsets["source_valid"] = TensorDataset(pristine_x_valid, pristine_y_valid)
    dsets["target_valid"] = TensorDataset(noisy_x_valid, noisy_y_valid)

    dsets["source_test"] = TensorDataset(pristine_x_test, pristine_y_test)
    dsets["target_test"] = TensorDataset(noisy_x_test, noisy_y_test)

    dset_loaders["source"] = DataLoader(dsets["source"], batch_size = 128, shuffle = True, num_workers = 1)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size = 128, shuffle = True, num_workers = 1)

    dset_loaders["source_valid"] = DataLoader(dsets["source_valid"], batch_size = 64, shuffle = True, num_workers = 1)
    dset_loaders["target_valid"] = DataLoader(dsets["target_valid"], batch_size = 64, shuffle = True, num_workers = 1)

    dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size = 64, shuffle = True, num_workers = 1)
    dset_loaders["target_test"] = DataLoader(dsets["target_test"], batch_size = 64, shuffle = True, num_workers = 1)

    config['out_file'].write("dataset sizes: source={}, target={}\n".format(
        len(dsets["source"]), len(dsets["target"])))

    # Set number of epochs, and logging intervals
    config["num_iterations"] = len(dset_loaders["source"])*config["epochs"]+1
    config["test_interval"] = len(dset_loaders["source"])
    config["snapshot_interval"] = len(dset_loaders["source"])*config["epochs"]*.25
    config["log_iter"] = len(dset_loaders["source"])

    # Print the configuration you are using
    config["out_file"].write("config: {}\n".format(config))
    config["out_file"].flush()

    # Set up early stop
    early_stop_engine = EarlyStopping(config["early_stop_patience"])
    
    # Set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])

    #Loading trained model if we want:
    if config["ckpt_path"] is not None:
        print('load model from {}'.format(config['ckpt_path']))
        ckpt = torch.load(config['ckpt_path']+'/best_model.pth.tar')
        base_network.load_state_dict(ckpt['base_network'])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('use_gpu')
        base_network = base_network.cuda()
    else:
        print('not use gpu')
        
    #summary(base_network, (3, 100, 100))
    
    # Collect parameters for the chosen network to be trained
    if "DeepMerge" in args.net:
            parameter_list = [{"params":base_network.parameters(), "lr_mult":1, 'decay_mult':2}]
    elif net_config["params"]["new_cls"]:
        if net_config["params"]["use_bottleneck"]:
            parameter_list = [{"params":base_network.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":base_network.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":base_network.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":base_network.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":base_network.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":base_network.parameters(), "lr_mult":10, 'decay_mult':2}]

    # Class weights in case we need them, here we have balanced sample so weights are 1.0
    class_weight = torch.from_numpy(np.array([1.0] * class_num))
    if use_gpu:
        class_weight = class_weight.cuda()

    parameter_list.append({"params":center_criterion.parameters(), "lr_mult": 10, 'decay_mult':1})
 
    # Set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, \
                    **(optimizer_config["optim_params"]))

    # Set learning rate scheduler
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    scan_lr = []
    scan_loss = []
    ###################
    ###### TRAIN ######
    ###################
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    len_valid_source = len(dset_loaders["source_valid"])
    len_valid_target = len(dset_loaders["target_valid"])
    
    best_loss = None
    plot_loss_epochs = np.array([])
    plot_loss_classifier = np.array([])
    plot_loss_transfer = np.array([])
    print('trade off', loss_params["trade_off"])
    for i in range(config["num_iterations"]):
        epoch = int(float(i/len(dset_loaders["source"])))

        ## Train one iteration
        base_network.train(True)
        
        #Optimizer
        if i % config["log_iter"] == 0:
            optimizer = lr_scheduler(param_lr, optimizer, i, config["log_iter"], config["frozen lr"], config["cycle_length"], **schedule_param)

        if config["optimizer"]["lr_type"] == "one-cycle":
            optimizer = lr_scheduler(param_lr, optimizer, i, config["log_iter"], config["frozen lr"], config["cycle_length"], **schedule_param)

        if config["optimizer"]["lr_type"] == "linear":
            optimizer = lr_scheduler(param_lr, optimizer, i, config["log_iter"], config["frozen lr"], config["cycle_length"], **schedule_param)

        optim = optimizer.state_dict()
        scan_lr.append(optim['param_groups'][0]['lr'])

        optimizer.zero_grad()

        #Set Up for Training Loss
        try:
            inputs_source, labels_source = iter(dset_loaders["source"]).next()
            inputs_target, labels_target = iter(dset_loaders["target"]).next()

        except StopIteration:
            iter(dset_loaders["source"])
            iter(dset_loaders["target"])

        if use_gpu:
            inputs_source, inputs_target, labels_source = \
                Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), \
                Variable(labels_source).cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), \
                Variable(inputs_target), Variable(labels_source)
           
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        source_batch_size = inputs_source.size(0)

        # Distance type. We use cosine.
        if config['loss']['ly_type'] == 'cosine':
            features, logits = base_network(inputs)
            source_logits = logits.narrow(0, 0, source_batch_size)
        elif config['loss']['ly_type'] == 'euclidean':
            features, _ = base_network(inputs)
            logits = -1.0 * loss.distance_to_centroids(features, center_criterion.centers.detach())
            source_logits = logits.narrow(0, 0, source_batch_size)

        # Transfer loss - MMD
        transfer_loss_train = transfer_criterion(features[:source_batch_size], features[source_batch_size:])

        # Source domain classification task loss
        classifier_loss_train = class_criterion(source_logits, labels_source.float())
        
        train_loss = loss_params["trade_off"] * transfer_loss_train + classifier_loss_train

        scan_loss.append(train_loss.cpu().float().item())

        train_loss.backward()
        
        optimizer.step()
        
        if i % config["log_iter"] == 0:

            # Logging:
            config['out_file'].write('epoch {}: train total loss={:0.4f}, train transfer loss={:0.4f}, train classifier loss={:0.4f}\n'.format(
                epoch, train_loss.data.cpu().float().item(), transfer_loss_train.data.cpu().float().item(), classifier_loss_train.data.cpu().float().item()))

            config['out_file'].flush()
            print('epoch {}: train total loss={:0.4f}, train transfer loss={:0.4f}, train classifier loss={:0.4f}'.format(
                epoch, train_loss.data.cpu().float().item(), transfer_loss_train.data.cpu().float().item(), classifier_loss_train.data.cpu().float().item()))

            # Logging for tensorboard
            writer.add_scalar("training total loss", train_loss.data.cpu().float().item(), epoch)
            writer.add_scalar("training classifier loss", classifier_loss_train.data.cpu().float().item(), epoch)
            writer.add_scalar("training transfer loss", transfer_loss_train.data.cpu().float().item(), epoch)

            #################
            # Validation step
            #################
            valid_loss = 0
            this_transfer_loss = 0
            for j in range(0, len(dset_loaders["source_valid"])):
                base_network.train(False)
                with torch.no_grad():

                    try:
                        inputs_valid_source, labels_valid_source = iter(dset_loaders["source_valid"]).next()
                        inputs_valid_target, labels_valid_target = iter(dset_loaders["target_valid"]).next()
                    except StopIteration:
                        iter(dset_loaders["source_valid"])
                        iter(dset_loaders["target_valid"])

                    if use_gpu:
                        inputs_valid_source, inputs_valid_target, labels_valid_source = \
                            Variable(inputs_valid_source).cuda(), Variable(inputs_valid_target).cuda(), \
                            Variable(labels_valid_source).cuda()
                    else:
                        inputs_valid_source, inputs_valid_target, labels_valid_source = Variable(inputs_valid_source), \
                            Variable(inputs_valid_target), Variable(labels_valid_source)
                       
                    valid_inputs = torch.cat((inputs_valid_source, inputs_valid_target), dim=0)
                    valid_source_batch_size = inputs_valid_source.size(0)

                    # Distance type. We use cosine.
                    if config['loss']['ly_type'] == 'cosine':
                        features, logits = base_network(valid_inputs)
                        source_logits = logits.narrow(0, 0, valid_source_batch_size)
                    elif config['loss']['ly_type'] == 'euclidean':
                        features, _ = base_network(valid_inputs)
                        logits = -1.0 * loss.distance_to_centroids(features, center_criterion.centers.detach())
                        source_logits = logits.narrow(0, 0, valid_source_batch_size)


                    # Transfer loss - MMD
                    transfer_loss_valid = transfer_criterion(features[:valid_source_batch_size], features[valid_source_batch_size:])

                    # Source domain classification task loss
                    classifier_loss_valid = class_criterion(source_logits, labels_valid_source.float())

                    # Final loss in case we do not want to add Fisher loss and Entropy minimization
                    valid_loss_j = loss_params["trade_off"] * transfer_loss_valid + classifier_loss_valid
                    valid_loss += valid_loss_j

                    this_transfer_loss += transfer_loss_valid
            # Logging:
            config['out_file'].write('epoch {}: valid total loss={:0.4f}, valid transfer loss={:0.4f}, valid classifier loss={:0.4f}'.format(
                epoch, valid_loss.data.cpu().float().item(), transfer_loss_valid.data.cpu().float().item(), classifier_loss_valid.data.cpu().float().item()))
            print('epoch {}: valid total loss={:0.4f}, valid transfer loss={:0.4f}, valid classifier loss={:0.4f}\n'.format(
                epoch, valid_loss.data.cpu().float().item(), transfer_loss_valid.data.cpu().float().item(), classifier_loss_valid.data.cpu().float().item()))
            config['out_file'].flush()
            
            # Logging for tensorboard:
            writer.add_scalar("validation total loss", valid_loss.data.cpu().float().item(), epoch)
            writer.add_scalar("validation classifier loss", classifier_loss_valid.data.cpu().float().item(), epoch)
            writer.add_scalar("validation transfer loss", transfer_loss_valid.data.cpu().float().item(), epoch)

            plot_loss_epochs = np.append(int(epoch), plot_loss_epochs)
            plot_loss_classifier = np.append(classifier_loss_valid.data.cpu().float().item(),plot_loss_classifier)
            plot_loss_transfer = np.append(transfer_loss_valid.data.cpu().float().item(),plot_loss_transfer)
            # Early stop in case we see overfitting

            if early_stop_engine.is_stop_training(classifier_loss_valid.cpu().float().item()):
                config["out_file"].write("no improvement after {}, stop training at step {}\n".format(
                config["early_stop_patience"], epoch))
                
                sys.exit()  
        if i % config["test_interval"] == 0:
            base_network.train(False)
            
            snapshot_obj = {'epoch': epoch, 
                "base_network": base_network.state_dict(), 
                'valid loss': valid_loss,
                'train loss' : train_loss,
                }
                
            snapshot_obj['center_criterion'] = center_criterion.state_dict()
        
            if (i+1) % config["snapshot_interval"] == 0:
                torch.save(snapshot_obj, osp.join(config["output_path"], "epoch_{}_model.pth.tar".format(i/len(dset_loaders["source"]))))
                
            if best_loss == None:
                best_loss = valid_loss

                print('first iteration', valid_loss)
                # Save best model
                torch.save(snapshot_obj, 
                           osp.join(config["output_path"], "best_model.pth.tar"))

            elif valid_loss < best_loss:
                best_loss = valid_loss
                
                # Save best model
                torch.save(snapshot_obj, 
                           osp.join(config["output_path"], "best_model.pth.tar"))
                print('loss decreased')
                
                #Plot Output - Classifier
                fig = plt.figure()
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
                #name temp data
                sfr_pred = source_logits.cpu().detach().numpy()[:, :1]
                sfr_real = labels_valid_source.cpu().detach().numpy()[:, :1]
                mstar_pred = source_logits.cpu().detach().numpy()[:, 1:]
                mstar_real = labels_valid_source.cpu().detach().numpy()[:, 1:]
                #scatter plot
                axs[0,0].plot(sfr_pred, sfr_real, '.', color='k')
                axs[0,1].plot(mstar_pred, mstar_real, '.', color='k')
                #y=x lines
                lims_1 = [0,1]
                axs[0,0].plot(lims_1, lims_1, 'r-')
                lims_2 = [0,1]
                axs[0,1].plot(lims_2, lims_2, 'r-')
                #Losses
                axs[1,0].plot(plot_loss_epochs, plot_loss_classifier, color='k')
                axs[1,1].plot(plot_loss_epochs, plot_loss_transfer, color='k')
                #axis labels
                for ii in range(0,2):
                    axs[0, ii].set_xlabel('Predicted')
                    axs[0, ii].set_ylabel('True')
                    axs[0, ii].set_aspect('equal', adjustable='box')
                    axs[1, ii].set_xlabel('Epoch')
                    axs[1, ii].set_ylabel('Loss')

                title = 'Results at epoch ' + str(epoch)
                axs[0, 0].set_title('SFR Classification')
                axs[0, 1].set_title('MStar Classification')
                axs[1, 0].set_title('Overall Classifier Loss')
                axs[1, 1].set_title('Overall Transfer Loss')
                plt.suptitle(title)
                #save
                loc = '/content/drive/My Drive/Deepskies/DeepMergeDomainAdaptation/python_files/'
                plt.savefig(loc + 'output_plots/mmd_training/' + str(epoch))
                plt.show()
            else:
                print('best loss is still', best_loss)
    ###################
    ###### TEST ######
    ###################
    #Set Up Figure
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
    
    print("start test: ")
    base_network.train(False)
    with torch.no_grad():
        loss_s = 0
        loss_t = 0
        for i in range(0, len(dset_loaders["source_test"])):
            inputs_test_source, labels_test_source = iter(dset_loaders["source_test"]).next()
            inputs_test_target, labels_test_target = iter(dset_loaders["target_test"]).next()
    
            inputs_test_source, inputs_test_target, labels_test_source, labels_test_target = \
                                Variable(inputs_test_source).cuda(), \
                                Variable(inputs_test_target).cuda(), \
                                Variable(labels_test_source).cuda(), \
                                Variable(labels_test_target).cuda()

            #Source -------------------------------------------------
            #Get Data
            test_source_batch_size = inputs_test_source.size(0)
            features, logits = base_network(inputs_test_source)
            source_logits = logits.narrow(0, 0, test_source_batch_size)
            #name temp data
            sfr_pred_s,sfr_real_s = source_logits.cpu().detach().numpy()[:, 0], labels_test_source.cpu().detach().numpy()[:, 0]
            mstar_pred_s, mstar_real_s = source_logits.cpu().detach().numpy()[:, 1], labels_test_source.cpu().detach().numpy()[:, 1]

            #scatter plot
            axs[0, 0].plot(unnormalize(sfr_pred_s), unnormalize(sfr_real_s), '.')
            axs[0, 1].plot(unnormalize(mstar_pred_s), unnormalize(mstar_real_s), '.')
            axs[0, 2].plot(unnormalize(sfr_pred_s), unnormalize(sfr_real_s), '.')
            axs[0, 3].plot(unnormalize(mstar_pred_s), unnormalize(mstar_real_s), '.')
            
            #Target Plots -------------------------------------------------
            #Get Data
            test_target_batch_size = inputs_test_target.size(0)
            features, logits = base_network(inputs_test_target)
            target_logits = logits.narrow(0, 0, test_target_batch_size)
            #name temp data
            sfr_pred_t, sfr_real_t = target_logits.cpu().detach().numpy()[:, 0], labels_test_target.cpu().detach().numpy()[:, 0]
            mstar_pred_t, mstar_real_t = target_logits.cpu().detach().numpy()[:, 1], labels_test_target.cpu().detach().numpy()[:, 1]

            #scatter plot
            axs[1, 0].plot(unnormalize(sfr_pred_t), unnormalize(sfr_real_t), '.')
            axs[1, 1].plot(unnormalize(mstar_pred_t), unnormalize(mstar_real_t), '.')
            axs[1, 2].plot(unnormalize(sfr_pred_t), unnormalize(sfr_real_t), '.')
            axs[1, 3].plot(unnormalize(mstar_pred_t), unnormalize(mstar_real_t), '.')
            
            #Calulate Loss
            loss_s += class_criterion(source_logits, labels_test_source.float())
            loss_t += class_criterion(target_logits, labels_test_target.float())
            
    #y=x lines
    zero = unnormalize(0)
    one = unnormalize(1)
    axs[0, 0].plot([zero, one],[zero, one], 'k-')
    axs[0, 1].plot([zero, one],[zero, one], 'k-')
    axs[1, 0].plot([zero, one],[zero, one], 'k-')
    axs[1, 1].plot([zero, one],[zero, one], 'k-')
    
    axs[0, 2].plot([zero, one],[zero, one], 'k-')
    axs[0, 3].plot([zero, one],[zero, one], 'k-')
    axs[1, 2].plot([zero, one],[zero, one], 'k-')
    axs[1, 3].plot([zero, one],[zero, one], 'k-')
    
    #axis labels
    for k in range (0,2):
        for m in range(0, 4):
            axs[k, m].set_xlabel('Predicted')
            axs[k, m].set_ylabel('True')
            #axs[k, m].set_aspect('equal', adjustable='box')
            if m < 2:
                axs[k, m].set_xscale('log')
                axs[k, m].set_yscale('log')


    #Titles
    axs[0, 0].set_title('Source SFR')
    axs[0, 1].set_title('Source MStar')
    axs[1, 0].set_title('Target SFR')
    axs[1, 1].set_title('Target MStar')
    
    axs[0, 2].set_title('Source SFR')
    axs[0, 3].set_title('Source MStar')
    axs[1, 2].set_title('Target SFR')
    axs[1, 3].set_title('Target MStar')

    #save
    loc = '/content/drive/My Drive/Deepskies/DeepMergeDomainAdaptation/python_files/'
    plt.savefig(loc + 'output_plots/mmd_training/eval')
    plt.show()
    
    #Losses Plots
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
    axs[0].plot(plot_loss_epochs, plot_loss_classifier, color = 'k')
    axs[1].plot(plot_loss_epochs, plot_loss_transfer, color = 'k')
    axs[2].plot(plot_loss_epochs, plot_loss_classifier + loss_params["trade_off"] * plot_loss_transfer, color = 'k')
    
    axs[0].set_title('Classifier Loss')
    axs[1].set_title('Transfer Loss')
    axs[2].set_title('Total Loss')
    plt.savefig(loc + 'output_plots/mmd_training/loss')
    
    print('\n Source Loss: ', loss_s)
    print('Target Loss: ', loss_t)
    return best_loss

# Adding all possible arguments and their default values
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature-based Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--loss_type', type=str, default='mmd', choices=["coral", "mmd"], help="type of transfer loss.")
    parser.add_argument('--ly_type', type=str, default="cosine", choices=["cosine", "euclidean"], help="type of classification loss.")
    parser.add_argument('--trade_off', type=float, default=1.0, help="coef of transfer_loss")
    parser.add_argument('--intra_loss_coef', type=float, default=0.0, help="coef of intra_loss.")
    parser.add_argument('--inter_loss_coef', type=float, default=0.0, help="coef of inter_loss.")
    parser.add_argument('--em_loss_coef', type=float, default=0.0, help="coef of entropy minimization loss.")
    parser.add_argument('--fisher_loss_type', type=str, default="tr", 
                        choices=["tr", "td"], 
                        help="type of Fisher loss.")
    parser.add_argument('--inter_type', type=str, default="global", choices=["sample", "global"], help="type of inter_class loss.")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,DeepMerge")
    parser.add_argument('--dset', type=str, default='galaxy', help="The dataset or source dataset used")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model")
    parser.add_argument('--optim_choice', type=str, default='SGD', help='Adam or SGD')
    parser.add_argument('--epochs', type=int, default=200, help='How many epochs do you want to train?')
    parser.add_argument('--grad_vis', type=str, default='no', help='Do you want to visualize your gradients?')
    parser.add_argument('--dset_path', type=str, default=None, help="The dataset directory path")
    parser.add_argument('--source_x_file', type=str, default='SB_version_00_numpy_3_filters_pristine_SB00_augmented_3FILT.npy',
                         help="Source domain x-values filename")
    parser.add_argument('--source_y_file', type=str, default='SB_version_00_numpy_3_filters_pristine_SB00_augmented_y_3FILT.npy',
                         help="Source domain y-values filename")
    parser.add_argument('--target_x_file', type=str, default='SB_version_00_numpy_3_filters_noisy_SB25_augmented_3FILT.npy',
                         help="Target domain x-values filename")
    parser.add_argument('--target_y_file', type=str, default='SB_version_00_numpy_3_filters_noisy_SB25_augmented_y_3FILT.npy',
                         help="Target domain y-values filename")
    parser.add_argument('--one_cycle', type=str, default = None, help='Do you want to turn on one-cycle learning rate?')
    parser.add_argument('--lr_scan', type=str, default = 'no', help='Set to yes for learning rate scan')
    parser.add_argument('--cycle_length', type=int, default = 2, help = 'If using one-cycle learning, how many epochs should one learning rate cycle be?')
    parser.add_argument('--early_stop_patience', type=int, default = 10, help = 'Number of epochs for early stopping.')
    parser.add_argument('--weight_decay', type=float, default = 5e-4, help= 'How much do you want to penalize large weights?')
    parser.add_argument('--blobs', type=str, default=None, help='Plot tSNE plots.')
    parser.add_argument('--fisher_or_no', type=str, default='Fisher', help='Run the code without fisher loss')
    parser.add_argument('--seed', type=int, default=3, help='Set random seed.')
    parser.add_argument('--ckpt_path', type=str, default=None, help="path to load ckpt")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Train config
    config = {}
    config["epochs"] = args.epochs
    config["output_for_test"] = True
    config["output_path"] = args.output_dir
    config["optim_choice"] = args.optim_choice
    config["grad_vis"] = args.grad_vis
    config["lr_scan"] = args.lr_scan
    config["cycle_length"] = args.cycle_length
    config["early_stop_patience"] = args.early_stop_patience
    config["weight_decay"] = args.weight_decay
    config["blobs"] = args.blobs
    config["fisher_or_no"] = args.fisher_or_no
    config["seed"] = args.seed

    # Set log file
    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
        config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if osp.exists(config["output_path"]):
        config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w") 

    # Set loss
    loss_dict = {"coral":loss.CORAL, "mmd":loss.mmd_distance}
    fisher_loss_dict = {"tr": loss.FisherTR, 
                         "td": loss.FisherTD, 
                         }
    config["loss"] = {"name": loss_dict[args.loss_type], 
                      "ly_type": args.ly_type, 
                      "fisher_loss_type": args.fisher_loss_type,
                      "discriminant_loss": fisher_loss_dict[args.fisher_loss_type],
                      "trade_off":args.trade_off, "update_iter":200,
                      "intra_loss_coef": args.intra_loss_coef, "inter_loss_coef": args.inter_loss_coef, "inter_type": args.inter_type, 
                      "em_loss_coef": args.em_loss_coef, }
    
    # Set parameters that depend on the choice of the network
    if "DeepMerge" in args.net:
        config["network"] = {"name":network.DeepMerge, \
            "params":{"class_num":2, "new_cls":True, "use_bottleneck":False} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":False, "new_cls":True} }

    # Set optimizer parameters
    if config["optim_choice"] == "Adam":
        config["optimizer"] = {"type":"Adam", "optim_params":{"lr":1e-6, "betas":(0.7,0.8), "weight_decay": config["weight_decay"], "amsgrad":True, "eps":1e-8}, \
                        "lr_type":"inv", "lr_param":{"init_lr":0.0001, "gamma":0.001, "power":0.75} }
    else:
        config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                               "weight_decay": config["weight_decay"], "nesterov":True}, "lr_type":"inv", \
                               "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }
        
    # Learning rate paramters
    if args.lr is not None:
        config["optimizer"]["optim_params"]["lr"] = args.lr
        config["optimizer"]["lr_param"]["init_lr"] = args.lr
        config["frozen lr"] = args.lr

    # One-cycle parameters
    if args.one_cycle is not None:
        config["optimizer"]["lr_type"] = "one-cycle"

    # Set up loading of the pretrained model if we want to do TL
    if args.ckpt_path is not None:
        config["ckpt_path"] = args.ckpt_path

    # Set paramaters needed for lr_scan
    if args.lr_scan == 'yes':
        config["optimizer"]["lr_type"] = "linear"
        config["optimizer"]["optim_params"]["lr"] = 1e-6
        config["optimizer"]["lr_param"]["init_lr"] = 1e-6
        config["frozen lr"] = 1e-6
        config["epochs"] = 5
        
    config["dataset"] = args.dset
    config["path"] = args.dset_path

    if config["dataset"] == 'galaxy':
        pristine_x = array_to_tensor(osp.join(config['path'], args.source_x_file))
        pristine_y = array_to_tensor(osp.join(config['path'], args.source_y_file))

        noisy_x = array_to_tensor(osp.join(config['path'], args.target_x_file))
        noisy_y = array_to_tensor(osp.join(config['path'], args.target_y_file))

        update(pristine_x, noisy_x)

        config["network"]["params"]["class_num"] = 2

    train(config)

    config["out_file"].write("finish training! \n")
    config["out_file"].close()
