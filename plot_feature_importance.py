def plot_feature_importance(df):
    n_features = len(df)
    df_plot = df.sort_values('importance')
    f_importance_plot = df_plot['importance'].values
    plt.barh(range(n_features), f_importance_plot, align='center') 
    cols_plot = df_plot['feature'].values             
    plt.yticks(np.arange(n_features), cols_plot)      
    plt.xlabel('Feature importance')                 
    plt.ylabel('Feature')  