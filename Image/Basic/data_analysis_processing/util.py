from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import os
import shutil

# a function to group connected components
def get_clusters(df, sort_by='count', min_count=2, ascending=False):
    # columns to aggregate
    agg_dict = {'filename': list, 'mean_distance': max, 'count': len}

    if 'label' in df.columns:
        agg_dict['label'] = list

    # filter by count
    df = df[df['count'] >= min_count]

    # group and aggregate columns
    grouped_df = df.groupby('component_id').agg(agg_dict)

    # sort
    grouped_df = grouped_df.sort_values(by=[sort_by], ascending=ascending)
    return grouped_df


def vis_cluster_tsne(X, y, filenames, output):
    # Instantiate and fit TSNE model
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(X)

    # Create a pandas dataframe with the embeddings and labels
    df = pd.DataFrame({'tsne_1': tsne_result[:, 0],
                       'tsne_2': tsne_result[:, 1],
                       'tsne_3': tsne_result[:, 2],
                       'component': y,
                       'filename': filenames})

    # Create a Plotly 3D scatter plot with colored points
    fig = px.scatter_3d(df, x='tsne_1', y='tsne_2', z='tsne_3',
                        color='component',
                        opacity=0.5,
                        hover_data=['component', 'filename'])
    # Save the figure to an image file (e.g., 'plot.png')
    fig.write_image(os.path.join(output, 'vis', 'cluster.png'))


