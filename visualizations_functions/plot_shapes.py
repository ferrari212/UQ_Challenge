import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_static_shapes(X: np.array, Y: np.array, Z: np.array, plot_dir: str, data_name: str, feature_index: int):

    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='plasma')

    ax.set_xlabel('Sample Value', fontsize=40, labelpad=20)
    ax.set_ylabel('Time', fontsize=40, labelpad=20)
    ax.set_zlabel('Result', fontsize=40, labelpad=20)
    plt.title(f'3D Shape Plot: feature {feature_index} for {data_name} sample', fontsize=60, pad=40)

    # Change the view direction
    ax.view_init(elev=30, azim=225)

    # Change the axis numbers size
    ax.tick_params(axis='both', which='major', labelsize=30)

    # Create folder if it doesn't exist
    save_path = f"{plot_dir}/feature_{feature_index}_plot_{data_name}_three_dimensional.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Static plot saved at: {save_path}")

def plot_dynamic_shapes(X: np.array, Y: np.array, Z: np.array, feature_index: int):

    # Create a 3D surface plot with Plotly
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Plasma')])

        fig.update_layout(
            title=f"3D Surface Plot for Feature {feature_index}",
            scene=dict(
                xaxis_title='Sample Value',
                yaxis_title='Time',
                zaxis_title='Result'
            ),
            width=1500,
            height=1000,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        fig.show()

def plot_3d_shapes(Z_out: np.ndarray, plot_dir: str, data_name: str, plot_static: bool = True, plot_interactive: bool = True):

    for j in range(Z_out.shape[1]): # for each feature
        
        # Create a meshgrid for the X and Y axes
        X, Y = np.meshgrid(np.linspace(0, 1, Z_out.shape[2]), np.linspace(1, 60, Z_out.shape[0]))
        Z = np.reshape(Z_out[:, j, :], X.shape) # Reshape the data to match the meshgrid 

        if plot_static:
            plot_static_shapes(X, Y, Z, plot_dir, data_name, j)
        
        if plot_interactive:
            plot_dynamic_shapes(X, Y, Z, j)