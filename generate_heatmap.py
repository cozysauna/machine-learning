def heatmap(df):
    import seaborn as sns
    corr_mat = df.corr(method='pearson')
    sns.heatmap(corr_mat, 
                vmin = -1.0,
                vmax = 1.0,
                center = 0,
                annot = True,
                fmt = '.1f',
                xticklabels = corr_mat.columns.values,
                yticklabels = corr_mat.columns.values
                )
    plt.show()
    return corr_mat