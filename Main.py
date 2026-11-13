import numpy as np
import os
import pandas as pd
from numpy import matlib
from pykalman import KalmanFilter
from sklearn.preprocessing import LabelEncoder
from BCO import BCO
from BOA import BOA
from EHO import EHO
from GTO import GTO
from Global_Vars import Global_Vars
from Model_BiLSTM import Model_BiLSTM
from Model_FHO_Modified_BiLSTM import Model_FHO_Modified_BiLSTM
from Model_GP_DCNN_SPE_ACGAN import Model_GP_DCNN_SPE_ACGAN
from Model_XSCA_DENet import Model_XSCA_DENet
from Objective_Function import objfun
from Plot_Results import *
from Proposed import Proposed

# Read Dataset 1
an = 0
if an == 1:
    file = './Dataset/Dataset 1/heart_disease_uci.csv'
    File1 = pd.read_csv(file)
    Dataset1 = File1.values
    Data = Dataset1
    # Create a copy
    encoded_data = Data.copy()

    # Store encoders if needed later
    encoders = {}

    for col in range(encoded_data.shape[1]):
        try:
            # Check if column is numeric
            encoded_data[:, col].astype(float)
        except:
            # Encode categorical column
            le = LabelEncoder()
            encoded_data[:, col] = le.fit_transform(encoded_data[:, col].astype(str))
            encoders[col] = le

    # Convert entire array to float
    encoded_data = encoded_data.astype(float)


    Target = Dataset1[:, 14]
    tar = File1.thal
    tar.replace(np.nan, 'normal', inplace=True)
    ta1 = tar.values
    uniq = np.unique(ta1)
    Tar = np.zeros((len(tar), len(uniq)))
    for j in range(len(uniq)):
        Index = np.where(tar == uniq[j])
        Tar[Index[0], j] = 1
    np.save('Data_1.npy', encoded_data)
    # np.save('Target_1.npy', Tar)

# Read Dataset 2
an = 0
if an == 1:
    file = './Dataset/Dataset 2/heart.csv'
    File1 = pd.read_csv(file)
    Dataset1 = File1.values
    Data = Dataset1
    Target = Dataset1[:, 13]
    np.save('Data_2.npy', Data)
    np.save('Target_2.npy', Target.reshape(-1, 1))

no_of_dataset = 2

# preprocessing
an = 0
if an == 1:
    for n in range(no_of_dataset):
        # Load .npy file
        data = np.load('Data_'+str(n+1)+'.npy', allow_pickle=True)
        # Apply Kalman Filter column-wise
        filtered_data = np.zeros_like(data, dtype=np.float64)

        for col in range(data.shape[1]):
            kf = KalmanFilter(
                initial_state_mean=data[0, col],
                n_dim_obs=1
            )

            state_means, _ = kf.filter(data[:, col].reshape(-1, 1))
            filtered_data[:, col] = state_means.flatten()

        # Min-Max Normalization
        min_val = filtered_data.min(axis=0)
        max_val = filtered_data.max(axis=0)

        normalized_data = (filtered_data - min_val) / (max_val - min_val + 1e-8)

        # Save preprocessed data
        np.save('Preprocessed_data_'+str(n+1)+'.npy', normalized_data)

# Optimization for Classification:
an = 0
if an == 1:
    Fitness = []
    Bestsol = []
    for n in range(no_of_dataset):
        Data = np.load('Preprocessed_data_'+str(n+1)+'.npy', allow_pickle=True)
        Target = np.load('Target_'+str(n+1)+'.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # hidden Neuron Count, Epochs, batch Size
        xmin = matlib.repmat([5, 5, 0.01], Npop, 1)
        xmax = matlib.repmat([255, 50, 0.99], Npop, 1)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("GTO...")
        [bestfit1, fitness1, bestsol1, time1] = GTO(initsol, fname, xmin, xmax, Max_iter)

        print("BCO...")
        [bestfit2, fitness2, bestsol2, time2] = BCO(initsol, fname, xmin, xmax, Max_iter)

        print("EHO...")
        [bestfit3, fitness3, bestsol3, time3] = EHO(initsol, fname, xmin, xmax, Max_iter)

        print("BOA...")
        [bestfit4, fitness4, bestsol4, time4] = BOA(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

        sol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])

        fitness = ([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])
        Bestsol.append(sol)
        Fitness.append(fitness)
    np.save('BestSol.npy', Bestsol)
    np.save('Fitness.npy', Fitness)

# Classification
an = 0
if an == 1:
    Evaluate = []
    for n in range(no_of_dataset):
        Data = np.load('Preprocessed_data_'+str(n+1)+'.npy', allow_pickle=True)
        Target = np.load('Target_'+str(n+1)+'.npy', allow_pickle=True)
        BestSol = np.load('BestSol.npy', allow_pickle=True)
        EVAL = []
        batch = ['Linear', 'ReLU', 'TanH', 'Sigmoid', 'Softmax']
        for learn in range(len(batch)):
            learnperc = round(Data.shape[0] * 0.75)
            Train_Data = Data[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Data[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 15))
            for j in range(BestSol.shape[0]):
                print(learn, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :] = Model_XSCA_DENet(Train_Data, Train_Target, Test_Data, Test_Target, batch[learn], sol)
            Eval[5, :] = Model_GP_DCNN_SPE_ACGAN(Train_Data, Train_Target, Test_Data, Test_Target, batch[learn])
            Eval[6, :] = Model_FHO_Modified_BiLSTM(Train_Data, Train_Target, Test_Data, Test_Target, batch[learn])
            Eval[7, :] = Model_BiLSTM(Train_Data, Train_Target, Test_Data, Test_Target, batch[learn])
            Eval[8, :] = Model_XSCA_DENet(Train_Data, Train_Target, Test_Data, Test_Target, batch[learn])
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Evaluate.append(EVAL)
    np.save('Eval_all.npy', Evaluate)  # Save the Eval_all

plotConvResults()
Plot_ROC_Curve()
Plots_Results()
Line_PlotTesults()
Proposed_PlotResults()
Table()
