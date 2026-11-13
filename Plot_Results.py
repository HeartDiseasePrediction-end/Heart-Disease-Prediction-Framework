import numpy as np
import warnings
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pylab
warnings.filterwarnings("ignore")



def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    for n in range(2):
        Fitness = np.load('Fitness.npy', allow_pickle=True)[n]
        Algorithm = ['TERMS', 'GTO-ESCA-DECA', ' BCO-ESCA-DECA', 'EHO-ESCA-DECA', 'BOA-ESCA-DECA', 'SHBOA-ESCA-DECA']
        Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[1])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Convergence Curve')
        plt.plot(length, Fitness[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Fitness[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Fitness[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Fitness[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Fitness[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
        plt.savefig("./Results/Conv-%s.png" % (n+1))
        plt.show()


def Plot_ROC_Curve():
    lw = 3
    cls = ['DRNN', 'SPE-ACGAN', 'Bi-LSTM', 'ESCA-DECA', 'SHBOA-ESCA-DECA']
    for a in range(2):
        Actual = np.load('Target_'+str(a+1)+'.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.25)
        Actual = Actual[lenper:, :]
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('ROC Curve')
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score_'+str(a+1)+'.npy', allow_pickle=True)[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel("True Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
        plt.title("ROC Curve")
        plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 12})
        path = "./Results/ROC-%s.png" % str(a+1)
        plt.savefig(path)
        plt.show()


def Table():
    for n in range(2):
        eval = np.load('Evaluate_all.npy', allow_pickle=True)[n]
        Algorithm = ['Batch Size', 'GTO-ESCA-DECA', ' BCO-ESCA-DECA', 'EHO-ESCA-DECA', 'BOA-ESCA-DECA', 'SHBOA-ESCA-DECA']
        Classifier = ['Activation Function', 'DRNN', 'SPE-ACGAN', 'Bi-LSTM', 'ESCA-DECA', 'SHBOA-ESCA-DECA']
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
                 'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
        Graph_Terms = np.array([0, 4, 6, 9, 12, 15]).astype(int)
        Table_Terms = [0, 4, 6, 9, 12, 15]
        table_terms = [Terms[i] for i in Table_Terms]
        Activation_Fun = ['4', '16', '32', '64', '80']
        for k in range(len(Table_Terms)):
            value = eval[:, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Activation_Fun)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('---------------------------------- Dataset - ', n+1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Activation_Fun)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('-------------------------------- Dataset - ', n+1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Proposed_PlotResults():
    for n in range(2):
        eval = np.load('Evaluate.npy', allow_pickle=True)[n]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC',
                 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'Lrplus', 'Lrminus', 'DOR', 'Prevalence']
        Graph_Term = [0, 5, 7, 9, 12]
        Batch_Size = ['4', '8', '16', '32', '48', '64', '128', '256']
        color = ['#4f5d75', '#ef8354', '#004e98', '#8e7dbe', '#2d3142', '#219ebc', 'm', '#f07167']
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[0], 1))
            for k in range(eval.shape[0]):
                Graph[k, 0] = eval[k, 4, Graph_Term[j] + 4]

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Batch Size Vs Proposed')
            ax.set_axisbelow(True)
            X = np.arange(Graph.shape[0])
            barWidth = 0.7

            bars = []
            for v in range(Graph.shape[0]):
                bar = ax.bar(X[v], Graph[v, 0], color=color[v], width=barWidth)
                bars.append(bar)

            plt.xticks(X, Batch_Size, fontname="Arial", fontsize=14, fontweight='bold', color='#003566')
            plt.xlabel('Batch Size', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#003566')
            for s in ['top', 'right']:
                plt.gca().spines[s].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            path1 = "./Results/Dataset-%s-Prop_%s_bar.png" % (n+1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def Plots_Results():
    for n in range(2):
        eval = np.load('Eval_all.npy', allow_pickle=True)[n]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
                 'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
        Graph_Terms = [0, 4, 5, 10, 12, 13]
        colors = ['#8f2d56', '#b388eb', '#219ebc', '#f77f00', '#8ac926']
        Classifier = ['DRNN', 'SPE-ACGAN', 'Bi-LSTM', 'ESCA-DECA', 'SHBOA-ESCA-DECA']
        Kfold = [1, 2, 3, 4, 5]
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[0:2])
            for k in range(eval.shape[0]):
                for l in range(eval.shape[1]):
                    Graph[k, l] = eval[k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of Kfold')
            X = np.arange(len(Kfold))
            plt.bar(X + 0.00, Graph[:, 5], color='#8f2d56', width=0.15, label=Classifier[0])
            plt.bar(X + 0.15, Graph[:, 6], color='#b388eb', width=0.15, label=Classifier[1])
            plt.bar(X + 0.30, Graph[:, 7], color='#219ebc', width=0.15, label=Classifier[2])
            plt.bar(X + 0.45, Graph[:, 8], color='#f77f00', width=0.15, label=Classifier[3])
            plt.bar(X + 0.60, Graph[:, 4], color='#8ac926', width=0.15, label=Classifier[4])
            plt.xticks(X + 0.30, ['1', '2', '3', '4', '5'], fontname="Arial", fontsize=14,
                       fontweight='bold', color='k')
            plt.xlabel('KFold', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in colors]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=12,
                       frameon=False, ncol=len(Classifier), prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            path = "./Results/Dataset-%s-%s_mod_bar.png" % (n+1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Line_PlotTesults():
    for n in range(2):
        eval = np.load('Eval_all.npy', allow_pickle=True)[n]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
                 'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
        Graph_Terms = [0, 4, 5, 10, 12, 13]
        Algorithm = ['GTO-ESCA-DECA', ' BCO-ESCA-DECA', 'EHO-ESCA-DECA', 'BOA-ESCA-DECA', 'SHBOA-ESCA-DECA']
        Kfold = [1, 2, 3, 4, 5]
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[0:2])
            for k in range(eval.shape[0]):
                for l in range(eval.shape[1]):
                    Graph[k, l] = eval[k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title(Terms[j] + ' Algorithm Comparison of Kfold')
            X = np.arange(len(Kfold))
            plt.bar(X + 0.00, Graph[:, 0], color='#6a994e', width=0.15, label=Algorithm[0])

            plt.bar(X + 0.15, Graph[:, 1], color='#00a8e8', width=0.15, label=Algorithm[1])

            plt.bar(X + 0.30, Graph[:, 2], color='violet', width=0.15, label=Algorithm[2])
            plt.bar(X + 0.45, Graph[:, 3], color='crimson', width=0.15, label=Algorithm[3])
            plt.bar(X + 0.60, Graph[:, 4], color='k', width=0.15, label=Algorithm[4])
            plt.xticks(X + 0.30, ['1', '2', '3', '4', '5'], fontsize=14, fontname="Arial",
                       fontweight='bold', color='k')
            plt.xlabel('KFold', fontsize=14, fontname="Arial", fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontname="Arial", fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            colors = ['#6a994e', '#00a8e8', 'violet', 'crimson', 'k']
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in colors]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=12,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            path = "./Results/Dataset-%s-%s_Alg_bar.png" % (n+1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plot_ROC_Curve()
    Plots_Results()
    Line_PlotTesults()
    Proposed_PlotResults()
    Table()
