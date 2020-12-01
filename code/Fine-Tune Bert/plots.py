import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def refine_hyper_str(hyper):
    parts = hyper.split('_')
    new = parts[0]
    # i=0
    # while i<len(parts):
    #     if parts[i] == 'l':
    #         new += parts[i] + ' ' + parts[i+1] + ' '
    #         i += 2
    #     if parts[i] == 'f':
    #         new += parts[i] + ' ' + parts[i+1] + ' '
    #         i += 2
    #     i += 1
    #
    # i = 0
    # while i<len(parts):
    #     if parts[i] == 'u':
    #         new += parts[i] + ' ' + parts[i+1] + ' '
    #         i += 2
    #
    #     if parts[i] == 'b':
    #         new += parts[i] + ' ' + parts[i+1] + ' '
    #         i += 2
    #     i+=1

    return new

def plot_hist(loss_hist, train_acc_hist, val_acc_hist, plot_name):
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    loss_hist_ = loss_hist[1::50]  # sparse the curve a bit
    plt.plot(loss_hist_, '-o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    val_acc_hist_ = val_acc_hist[::2]
    train_acc_hist_ = train_acc_hist[::2]
    plt.title('Accuracy')
    plt.plot(train_acc_hist_, '-o', label='Training')
    plt.plot(val_acc_hist_, '-o', label='Validation')
    plt.plot([0.5] * len(val_acc_hist_), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.gcf().set_size_inches(15, 12)
    # plt.show()
    plt.savefig('./experiments/plots/lr_plots/{}.png'.format(plot_name))

def multi_plot_hist(train_loss, val_loss, train_acc_hist, val_acc_hist, plot_name):
    colors = ["red", "gold", "limegreen", 'purple']
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    train_loss_ = train_loss[1::50]
    plt.plot(train_loss_, '-o', label='train_loss', alpha = 0.6)
    if val_loss:
        plt.plot(val_loss, '-o', label='val_loss', alpha=0.6)
    plt.xlabel('Iteration')
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')

    plt.plot(train_acc_hist, '-o', label='Training_acc', alpha = 0.6)
    plt.plot(val_acc_hist, '-x', label='Validation_acc', alpha = 0.6)

    plt.plot([0.5] * len(val_acc_hist), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='center right')

    plt.gcf().set_size_inches(15, 12)
    plt.savefig('./{}.png'.format(plot_name))


def pca_plot(x, labels):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    df_labels = pd.DataFrame(data=labels, columns=["labels"])
    finalDf = pd.concat([principalDf, df_labels], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1]
    colors = ['b', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['labels'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50
                   , alpha=0.3)
    targets = ['Huffington Post US', 'Breitbart']
    # ax.legend(targets)
    ax.grid()
    # plt.title('layer 2')
    plt.savefig('./pca_finetune_bert_h6.png')
    plt.show()

def tsne_plot(x, labels):
    tsne = TSNE(n_components=2, verbose=1, perplexity=43, n_iter=1000)
    tsne_results = tsne.fit_transform(x)
    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    df['label'] = labels
    df['label'] = df['label'].map({0: 'Huffington Post US', 1: 'Breitbart'})
    plt.figure(figsize=(8, 8))
    g = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=["b", "r"],
        data=df,
        legend="full",
        alpha=0.5
    )
    # ax.grid()

    plt.savefig('./tsne_finetune_bert.png')
    plt.show()
    print('done')

if __name__ == '__main__':
    with open('new_splits/test_labels.csv') as f:
        labels = f.read().splitlines()

    labels = [int(x) for x in labels]

    x = np.loadtxt('fine_tuned_embeddings_e10_b6_h6.csv', delimiter=',')
    print(x.shape)
    print(len(labels))
    pca_plot(x, labels)
    # tsne_plot(x, labels)