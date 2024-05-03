import sys
import os         
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import deepxde as dde # version 0.11 or higher
import MS_utils
import pinn
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from generate_plots import plot_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-vf', '--v-file-name', dest='v_file_name', required = True, type = str, help='igb File name for input voltage data')
    parser.add_argument('-hf', '--h-file-name', dest='h_file_name', required = True, type = str, help='igb File name for h input data')
    parser.add_argument('-ptf', '--pt-file-name', dest='pt_file_name', required = True, type = str, help='pt File name for coordinates of input nodes, but please do not put .pts at the end')
    parser.add_argument('-m', '--model-folder-name', dest='model_folder_name', required = False, type = str, help='Folder name to save model (prefix /)')
    parser.add_argument('-d', '--dimension', dest='dim', required = False, type = int, help='Model dimension. Needs to match the input data')
    parser.add_argument('-n', '--noise', dest='noise', action='store_true', help='Add noise to the data')
    parser.add_argument('-w', dest='w_input', required = False, action='store_true', help='Add w to the model input data')
    parser.add_argument('-v', '--inverse', dest='inverse', required = False, type = str, help='Solve the inverse problem, specify variables to predict (e.g. a / ad / abd')
    parser.add_argument('-ht', '--heter', dest='heter', required = False, action='store_true', help='Predict heterogeneity - only in 2D')    
    parser.add_argument('-p', '--plot', dest='plot', required = False, action='store_true', help='Create and save plots')
    parser.add_argument('-a', '--animation', dest='animation', required = False, action='store_true', help='Create and save 2D Animation')
    args = parser.parse_args()

## General Params
noise = 0.1 # noise factor
test_size = 0.9 # precentage of testing data

def main(args):
    
    ## Get Dynamics Class
    dynamics = MS_utils.system_dynamics()
    
    ## Parameters to inverse (if needed)
    params = dynamics.params_to_inverse(args.inverse)
    
    ## Generate Data 
    v_file_name = args.v_file_name
    h_file_name = args.h_file_name 
    pt_file_name = args.pt_file_name 
    observe_x, V, h, Vsav, len_t = dynamics.generate_data(v_file_name, h_file_name, pt_file_name)

    ## Split data to train and test
    observe_train, observe_test, v_train, v_test, h_train, h_test = train_test_split(observe_x,V,h,test_size=test_size)
    
    ## Add noise to training data if needed
    if args.noise:
        v_train = v_train + noise*np.random.randn(v_train.shape[0], v_train.shape[1])

    ## Geometry and Time domains
    geomtime = dynamics.geometry_time()
    ## Define Boundary Conditions
    bc = dynamics.BC_func(geomtime)
    ## Define Initial Conditions
    ic = dynamics.IC_func(observe_train, v_train)
    
    ## Model observed data
    observe_v = dde.PointSetBC(observe_train, v_train, component=0)
    input_data = [bc, ic, observe_v]
    if args.w_input: ## If h required as an input
        observe_h = dde.PointSetBC(observe_train, h_train, component=1)
        input_data = [bc, ic, observe_v, observe_h]
    
    ## Select relevant PDE (Heterogeneity) and define the Network
    model_pinn = pinn.PINN(dynamics, args.heter, args.inverse)
    model_pinn.define_pinn(geomtime, input_data, observe_train)
            
    ## Train Network
    out_path = dir_path + args.model_folder_name
    model, losshistory, train_state = model_pinn.train(out_path, params)
    
    ## Compute rMSE
    pred = model.predict(observe_test)   
    v_pred, h_pred = pred[:,0:1], pred[:,1:2]
    rmse_v = np.sqrt(np.square(v_pred - v_test).mean())
    print('--------------------------')
    print("V rMSE for test data:", rmse_v)
    print('--------------------------')
    print("Arguments: ", args)
    
    ## Save predictions, data
    np.savetxt("train_data.dat", np.hstack((observe_train, v_train, h_train)),header="observe_train,v_train, h_train")
    np.savetxt("test_pred_data.dat", np.hstack((observe_test, v_test,h_pred, h_test, h_pred)),header="observe_test,v_test, v_pred, h_test, h_pred")
    
    plot_results(Vsav, observe_train, v_train, observe_test, v_pred, len_t, args.model_folder_name, args.animation)

## Run main code
model = main(args)
