import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import nibabel as nib

def plot_3d_slices(img_path, seg_path=None, n_slices=10, figsize=(10,8)):
    cmap_red = ListedColormap([(1, 1, 1, 0), (1, 0, 0, 1)])
    # load image
    img = nib.load(img_path)
    data = img.get_fdata()
    x_dim, y_dim, z_dim = data.shape
    if seg_path:
        seg = nib.load(seg_path)
        seg_data = seg.get_fdata()
        
        seg_x_dim, seg_y_dim, seg_z_dim = seg_data.shape
        assert x_dim == seg_x_dim and y_dim == seg_y_dim and z_dim == seg_z_dim, "Image and segmentation dimensions do not match."
    
    slice_indices = np.linspace(0, z_dim - 1, n_slices).astype(int)
    
    # create a 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # set plot limits and erase label
    ax.set_xlim(0, x_dim)
    ax.set_ylim(0, y_dim)
    ax.set_zlim(0, z_dim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    for slice_index in slice_indices:
        
        slice_data = data[:, :, slice_index]
        
        # create a meshgrid for the x and y dimensions
        x, y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
        z = np.full_like(x, slice_index)
        
        if seg_path:
            seg_slice_data = seg_data[:, :, slice_index]
            ax.contourf(x, y, seg_slice_data, zdir='z', offset=slice_index, cmap=cmap_red, alpha=0.6)
        
        ax.contourf(x, y, slice_data, zdir='z', offset=slice_index, cmap='gray', alpha=0.2)    
    # plt.show()