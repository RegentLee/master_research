"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from data.test_dataset import create_test_dataset
from models import create_model
from util.visualizer import Visualizer

import numpy as np
import pandas as pd
import pickle

from util import my_util
from data.MyFunction import my_transforms


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    trainset = create_test_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    trainset_size = len(trainset)    # get the number of images in the dataset.
    print('The number of training images = %d' % trainset_size)
    my_util.val = True
    valset = create_test_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    valset_size = len(valset)    # get the number of images in the dataset.
    print('The number of validation images = %d' % valset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    score = my_util.MAE
    result_train = []
    result_val = []

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        
        my_util.x = np.random.randint(0, 10)
        my_util.y = np.random.randint(0, 10)
        my_util.val = False
        for name in model.model_names:
            if isinstance(name, str):
                net = getattr(model, 'net' + name)
                net.train()
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            #for name, param in model.net.named_parameters():
            #    print('name : ', name)

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # my_util.train(model, data)
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # model.compute_visuals()
            # visualizer.display_current_results(model.get_current_visuals(), epoch, False)
            # time.sleep(1)

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)

            iter_data_time = time.time()
        # if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        '''model.eval()
        temp = np.zeros(3)
        for i, data in enumerate(dataset):
            temp += my_util.test(model, data)
        print(temp/dataset_size)

        temp = np.zeros(3)
        for i, data in enumerate(valset):
            temp += my_util.test(model, data)
        print(temp/valset_size)'''

        if not opt.diff:
            model.eval()
            my_util.val = True
            temp = [] # np.zeros(3)
            tempB = np.zeros(3)
            dt = []
            for i, data in enumerate(trainset):
                model.set_input(data)
                model.test()
                answer = model.fake_B.to('cpu').detach().numpy().copy()
                data_A = data['A'].numpy()[0][0]
                data_A = (data_A + 1)*my_util.distance[-1]
                data_B = model.real_B.to('cpu').detach().numpy().copy()[0][0]
                m_max = my_util.distance[-1]
                m_min = 0
                data_A = np.where(data_A > m_max, m_max, data_A)
                data_A = (data_A - m_min)/(m_max - m_min)*2 - 1
                org = score(data_A, data_B)
                last = score(data_A, answer[0][0])
                first = score(answer[0][0], data_B)
                dt.append([data_A, answer[0][0], data_B])
                # print(np.array([org, last, first]))
                # if data['domain'] == 0:
                temp.append([org, last, first])
                tempB += np.array([org, last, first])
                # else:
                #     tempB += np.array([org, last, first])
                # print(org, last, first)
                # model.compute_visuals()
                # visualizer.display_current_results(model.get_current_visuals(), epoch, False)
                # time.sleep(1)
            print(tempB/(trainset_size))
            # print(tempB/(dataset_size//2))
            # model.compute_visuals()
            # visualizer.display_current_results(model.get_current_visuals(), epoch, False)
            result_train.append(temp)
            with open('result_' + opt.name + '/result_train', 'wb') as f:
                pickle.dump(result_train, f)
            # result = pd.DataFrame(result_train)
            # result.to_csv('result_' + opt.name + '/result_train.csv')

            temp = []
            tempB = np.zeros(3)
            dtv = []
            for i, data in enumerate(valset):
                model.set_input(data)
                model.test()
                answer = model.fake_B.to('cpu').detach().numpy().copy()
                data_A = data['A'].numpy()[0][0]
                data_A = (data_A + 1)*my_util.distance[-1]
                data_B = model.real_B.to('cpu').detach().numpy().copy()[0][0]
                m_max = my_util.distance[-1]
                m_min = 0
                data_A = np.where(data_A > m_max, m_max, data_A)
                data_A = (data_A - m_min)/(m_max - m_min)*2 - 1
                org = score(data_A, data_B)
                last = score(data_A, answer[0][0])
                first = score(answer[0][0], data_B)
                dtv.append([data_A, answer[0][0], data_B])
                temp.append([org, last, first])
                tempB += np.array([org, last, first])
                # model.compute_visuals()
                # visualizer.display_current_results(model.get_current_visuals(), epoch, False)
                # time.sleep(1)
                # print(np.array([org, last, first]))
                # rmsd = np.sqrt(np.sum(((answer - data['B'].numpy())**2).flatten())/len(answer.flatten()))
            print(tempB/valset_size)
            # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, False)
            result_val.append(temp)
            with open('result_' + opt.name + '/result_val', 'wb') as f:
                pickle.dump(result_val, f)
            # result = pd.DataFrame(result_val)
            # result.to_csv('result_' + opt.name + '/result_val.csv')
        '''
        else:
            model.eval()
            temp = np.zeros(3)
            for i, data in enumerate(dataset):
                model.set_input(data)
                model.test()
                answer = model.image.to('cpu').detach().numpy().copy()
                org = score(0, data['B'].numpy()[0][0])
                last = score(0, answer[0][0])
                first = score(answer[0][0], data['B'].numpy()[0][0])
                # print(np.array([org, last, first]))
                temp += np.array([org, last, first])
                # print(org, last, first)
            print(temp/dataset_size)
            result_train.append(list(temp/dataset_size))

            temp = np.zeros(3)
            for i, data in enumerate(valset):
                model.set_input(data)
                model.test()
                answer = model.image.to('cpu').detach().numpy().copy()
                # print(answer[0][0])
                org = score(0, data['B'].numpy()[0][0])
                last = score(0, answer[0][0])
                first = score(answer[0][0], data['B'].numpy()[0][0])
                temp += np.array([org, last, first])
                # print(np.array([org, last, first]))
                # rmsd = np.sqrt(np.sum(((answer - data['B'].numpy())**2).flatten())/len(answer.flatten()))
            print(temp/valset_size)
            result_val.append(list(temp/valset_size))
        '''

    # result_train = pd.DataFrame(result_train)
    # result_train.to_csv('result/result_train.csv')

    with open('result_' + opt.name + '/train_' + str(epoch), 'wb') as f:
        pickle.dump(dt, f)

    # result_val = pd.DataFrame(result_val)
    # result_val.to_csv('result/result_val.csv')

    with open('result_' + opt.name + '/val_' + str(epoch), 'wb') as f:
        pickle.dump(dtv, f)
    '''
    for i, data in enumerate(valset):
        model.set_input(data)
        model.test()
        answer = model.fake_B.to('cpu').detach().numpy().copy()
        answer = pd.DataFrame(answer[0][0])
        answer.to_csv('result/'+str(i)+'.csv')'''