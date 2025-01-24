import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colors import Normalize
import mne
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyvista as pv
import nibabel.freesurfer.io as fsio
from pyvistaqt import BackgroundPlotter


# m: Measurement as string, ex 'M1' 
# s: subject as string, ex '0900'
# data: the data that should be projected, list of 200 parcels in order of the Schaefer labels
# title: titel of the figure as string
# cmin/cmax: the limits for the data value to be used as color limits
# fwd_fixed_path: the path to the forward solution
# mri_dir: the path to mri directory
# brain_kwargs: additional arguments as a dictionary for the visualisation, ex dict(alpha=0.3, background='white', cortex='high_contrast')
# cmap: color map to use
def vis_data_on_brain(s, data, title, cmin, cmax, fwd_fixed_path, mri_dir, 
                      brain_kwargs=dict(alpha=0.3, background='white', cortex='high_contrast'), cmap='Reds', 
                      surface_name='pial', parcs='parc2018yeo7_200'):
    
    #colors = cm.get_cmap(cmap, 256)
    #newcolors = colors(np.linspace(0, 1, 256))
    #grey = np.array([1, 1, 1, 1])
    #newcolors[0:96] = grey  #32
    #cmap = ListedColormap(newcolors)
            
    fwd_fixed = mne.read_forward_solution(fwd_fixed_path)
    src = fwd_fixed['src']
    src_lh = src[0]
    src_rh = src[1]
    
    data_lh = data[:100]
    data_rh = data[100:]
    
    labels_lh = mne.read_labels_from_annot('MRI_example_subject', parc=parcs, hemi='lh', subjects_dir=mri_dir,
                                           surf_name=surface_name)
    labels_lh = labels_lh[:100]
    labels_rh = mne.read_labels_from_annot('MRI_example_subject'.format(s), parc=parcs, hemi='rh', subjects_dir=mri_dir, 
                                           surf_name=surface_name)
    labels_rh = labels_rh[:100]
    
    labels_both = np.concatenate((labels_lh, labels_rh), axis=0)

    surf_verts_lh = src_lh['vertno']
    surf_verts_rh = src_rh['vertno']
    
    data_verts_lh = get_data_verts(labels_lh, surf_verts_lh, data_lh)
    data_verts_rh = get_data_verts(labels_rh, surf_verts_rh, data_rh)
    
    brain_lh1 = mne.viz.Brain('MRI_example_subject', **brain_kwargs, hemi='lh', size=(1200, 600))
    brain_lh2 = mne.viz.Brain('MRI_example_subject', **brain_kwargs, hemi='lh', size=(1200, 600))
    brain_rh1 = mne.viz.Brain('MRI_example_subject', **brain_kwargs, hemi='rh', size=(1200, 600))
    brain_rh2 = mne.viz.Brain('MRI_example_subject', **brain_kwargs, hemi='rh', size=(1200, 600))
    
    c_kwargs = dict(label_font_size=14, n_labels=5)

    brain_lh1.add_data(data_verts_lh, vertices=surf_verts_lh, hemi='lh', fmin=cmin, fmax=cmax, colormap=cmap,
                       colorbar_kwargs=c_kwargs, time_label=None, smoothing_steps='nearest',colorbar=False)
    brain_lh2.add_data(data_verts_lh, vertices=surf_verts_lh, hemi='lh', fmin=cmin, fmax=cmax, colormap=cmap,
                       colorbar_kwargs=c_kwargs, time_label=None, smoothing_steps='nearest',colorbar=False)
    brain_rh1.add_data(data_verts_rh, vertices=surf_verts_rh, hemi='rh', fmin=cmin, fmax=cmax, colormap=cmap, 
                       colorbar_kwargs=c_kwargs, time_label=None, smoothing_steps='nearest',colorbar=False)
    brain_rh2.add_data(data_verts_rh, vertices=surf_verts_rh, hemi='rh', fmin=cmin, fmax=cmax, colormap=cmap,
                       colorbar_kwargs=c_kwargs, time_label=None, smoothing_steps='nearest',colorbar=False)

    brain_lh1.show_view(distance=350)
    brain_lh2.show_view(view='medial', distance=350)
    brain_rh1.show_view(distance=350)
    brain_rh2.show_view(view='medial', distance=350)
    
    img_lh1 = brain_lh1.screenshot()
    img_lh2 = brain_lh2.screenshot()
    img_rh1 = brain_rh1.screenshot()
    img_rh2 = brain_rh2.screenshot()

    img_list = [[img_lh1, img_rh1],[img_lh2, img_rh2]]
    
    fig, axes = plt.subplots(2,2, figsize=(9, 6))
#     fig.patch.set_facecolor('lightgrey')
    for imgs, axs in zip(img_list, axes):
        for img, ax in zip(imgs,axs):
            img = img[100:-65, 300:-300, :]
            imc = ax.imshow(img)
            ax.axis('off')
    
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=cmin, vmax=cmax), cmap=cmap), ax=axes.ravel().tolist())
    # cbar.set_label(label='AUPDC', size='x-large')
    cbar.ax.tick_params(labelsize='large')
    
    cbar_ticks = cbar.ax.get_yticks()
    cbar_ticklabels = []
    c_id = 0
    for i, tick in enumerate(cbar_ticks):
         cbar_ticklabels.append(str(round(tick,3)))
    
    cbar_ticks = cbar.ax.get_yticklabels()
    
    #fig.suptitle('Subject {}\n{}'.format(s, title), size='x-large', color='black')
    plt.show()
    
    
def get_data_verts(labels, surf_verts, data):
    data_verts = np.zeros_like(surf_verts, dtype=float)
    
    for i, label in enumerate(labels):
        label_verts = label.get_vertices_used(surf_verts)
                
        data_value = data[i]

        for i, vert in enumerate(surf_verts):
            if vert in label_verts:
                data_verts[i] = data_value
    
    return data_verts
