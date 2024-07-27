#pip install pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
sns.set()
 
 
def get_data():
    """生成聚类数据"""
    from sklearn.datasets import make_blobs
    x_value, y_value = make_blobs(n_samples=1000, n_features=40, centers=3, )
    return x_value, y_value
 
 
def plot_xy(x_values, label, title,i):
    """绘图"""
    df = pd.DataFrame(x_values, columns=['x', 'y'])
    #print(label)
    df['label'] = label
    plt.figure(facecolor='white')
    custom_palette = {1: 'red', 3: 'blue', 2: 'green'}
    category_labels = {'1': 'RV', '2': 'Myo', '3': 'LV'}
    sns.scatterplot(x="x", y="y", hue="label",palette=custom_palette,hue_order=[1, 2, 3], data=df)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    #plt.title(title)
  # 去掉网格线
    sns.set_style("whitegrid")
    plt.grid(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    print(handles, labels)
    plt.gca().legend(handles, [category_labels[label] for label in labels])
    # 去掉坐标轴数据
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig('./tsne_our/t-sne_{}.png'.format(i))
    plt.show()
 
 
def main_s(x_value,y_value,i):
    
    # # PCA 降维
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # x_pca = pca.fit_transform(x_value)
    # plot_xy(x_pca, y_value, "PCA")
    # t-sne 降维
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_value)
    plot_xy(x_tsne, y_value, "t-sne_{}".format(i),i)
 
 
