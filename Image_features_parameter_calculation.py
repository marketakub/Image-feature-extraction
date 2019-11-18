# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:12:27 2019

@author: mkuban
"""

# -*- coding: utf-8 -*-
#%% 
import numpy as np
import matplotlib.pylab as plt
import dclab
import pandas as pd
from matplotlib import cm
from scipy.stats import gaussian_kde
import pickle
import mahotas
import mahotas.features

######## colormap chan be changed here
cmap_vir = cm.get_cmap('viridis')



####################################################################################################
######################################### scatter plot #############################################
####################################################################################################

def density_scatter( x , y, bins, ax, sort = True, **kwargs )   :

    np.nan_to_num(y, copy=False)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    ax.scatter( x, y, c=z, cmap = cmap_vir, marker = ".", s = 4, picker = True, **kwargs )
    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
    plt.minorticks_on()
    #plt.grid(b=True, which='both', color='0.65', linestyle='-')
    plt.grid(b=True, which='minor', color='0.85', linestyle='--')
    plt.grid(b=True, which='major', color='0.85', linestyle='-')
    plt.xticks()
    plt.yticks()
    plt.rcParams["font.size"] = 12
    plt.gcf().set_tight_layout(False)

    return ax

def onpick(event):
    ind = event.ind
    #print('onpick scatter event number:', ind)
    #print('Shown index', ind[0])
    #print('length of index', len(ind))
    #print('area of event', ds_child["area_um"][ind[0]])

    #plt.figure(figsize=(10,5))
    samples = ds.config["fluorescence"]["samples per event"]
    sample_rate = ds.config["fluorescence"]["sample rate"]
    t = np.arange(samples) / sample_rate * 1e6

    figure, axes = plt.subplots(nrows=5, sharex=False, sharey=False)
    axes[0] = plt.subplot2grid((5, 3), (0, 0), colspan=5)
    axes[1] = plt.subplot2grid((5, 3), (1, 0), colspan=5)
    axes[2] = plt.subplot2grid((5, 3), (2, 1))
    axes[3] = plt.subplot2grid((5, 3), (3, 1))
    axes[4] = plt.subplot2grid((5, 3), (4, 1))
    axes[0].imshow(ds_child["image"][ind[0]], cmap="gray")
    axes[1].imshow(ds_child["mask"][ind[0]])
    axes[2].plot(t, ds_child["trace"]["fl1_median"][ind[0]], color="#16A422",
         label=ds.config["fluorescence"]["channel 1 name"])
    axes[3].plot(t, ds_child["trace"]["fl2_median"][ind[0]], color="#CE9720",
         label=ds.config["fluorescence"]["channel 2 name"])
    axes[4].plot(t, ds_child["trace"]["fl3_median"][ind[0]], color="#CE2026",
         label=ds.config["fluorescence"]["channel 3 name"])
    
    axes[2].set_xlim(0, 570) #(200, 350)
    axes[2].grid()
    axes[3].set_xlim(0, 570) #(200, 350)
    axes[3].grid()
    axes[4].set_xlim(0, 570) #(200, 350)
    axes[4].grid()

    axes[2].axvline(ds_child["fl1_pos"][ind[0]] + ds_child["fl1_width"][ind[0]]/2, color="gray")
    axes[2].axvline(ds_child["fl1_pos"][ind[0]] - ds_child["fl1_width"][ind[0]]/2, color="gray")
    #axes[2].axvline(350, color="black")
    #axes[2].axvline(200, color="black")
    axes[3].axvline(ds_child["fl2_pos"][ind[0]] + ds_child["fl2_width"][ind[0]]/2, color="gray")
    axes[3].axvline(ds_child["fl2_pos"][ind[0]] - ds_child["fl2_width"][ind[0]]/2, color="gray")
    #axes[3].axvline(350, color="black")
    #axes[3].axvline(200, color="black")
    axes[4].axvline(ds_child["fl3_pos"][ind[0]] + ds_child["fl3_width"][ind[0]]/2, color="gray")
    axes[4].axvline(ds_child["fl3_pos"][ind[0]] - ds_child["fl3_width"][ind[0]]/2, color="gray")
    #axes[4].axvline(350, color="black")
    #axes[4].axvline(200, color="black")
    
    plt.show()
    print(ds_child["trace"][ind[0]])

####################################################################################################
########################################### load video #############################################
####################################################################################################
#%%  
def framecapture_notflat(ds):
       
    pix = 0.34
    Images = []
    Images_i_m = []
    #IMS = []
       
    for idx in range(len(ds_child)):
        
        pos_x = round(float((ds_child["pos_x"][idx] / pix)))
        pos_y = round(float((ds_child["pos_y"][idx] / pix)))
        cellimg = (ds_child["image"][idx])
        cellmask = (ds_child["mask"][idx])
        
        cell_i_m = cellimg * cellmask
        
        #cellimg_cropped = cellimg[pos_y-18:pos_y+18,pos_x-24:pos_x+24] #used for spiked blood
        #cellimg_cropped = cellimg[pos_y-11:pos_y+11,pos_x-16:pos_x+14] #used for lymphocytes
        #cellimg_cropped = cellimg[pos_y-12:pos_y+12,pos_x-17:pos_x+17] #used for original embedding Feb 2019
        #cellimg_cropped = cellimg[pos_y-13:pos_y+13,pos_x-20:pos_x+20] # full image with background
        #cellimg_cropped = cellimg[pos_y-20:pos_y+20,pos_x-30:pos_x+30] # full image with background
        cellimg_cropped = cellimg[pos_y-25:pos_y+25,pos_x-35:pos_x+35] # used for lemon cells
        cellimg_cropped_i_m = cell_i_m[pos_y-25:pos_y+25,pos_x-35:pos_x+35] # multiplied by mask
        
        cellimg_cropped = cellimg_cropped #/255
        cellimg_cropped_i_m = cellimg_cropped_i_m #/255
        
        Images.append(cellimg_cropped)
        Images_i_m.append(cellimg_cropped_i_m)
    

    #plt.imshow(cellimg_cropped)
    Images=pd.DataFrame(Images)
    Images_i_m=pd.DataFrame(Images_i_m)

    return[Images, Images_i_m]  


#%%

###################################################################################################################
############################################## LOAD DATA ##########################################################
###################################################################################################################

# CHANGE THESE VARIABLES WHEN LOADING A NEW SAMPLE:
filepath = r"D:\UK ERLANGEN DATA\20191018_Marketa_Alex_fibrotic-lung_3week\0.08 Triple negative -ery lysed- from FACS\M003_data" #filepath of the file to be analysed
sample1 = '20191018-FACSneg1-M003' #here specify experiment details
FL1_threshold = 250
FL2_threshold = 200
FL3_threshold = 100


##########################
ds = dclab.new_dataset(filepath + ".rtdc")
exp_date = ds.config["experiment"]["date"]
exp_time = ds.config["experiment"]["time"]
patient = exp_date + '_' + sample1
identifier = patient

############################## BASIC FILTERS SET TO PREFILTER LEMON CELL DATASET #####################################
ds.config["calculation"]["emodulus temperature"] = ds["temp"]
ds.config["calculation"]["emodulus viscosity"] = 54.7 # 0.8 MC
ds.config["filtering"]["area_um min"] = 15
ds.config["filtering"]["area_um max"] = 600
ds.config["filtering"]["area_ratio min"] = 1
ds.config["filtering"]["area_ratio max"] = 1.2
ds.config["filtering"]["aspect min"] = 1
ds.config["filtering"]["aspect max"] = 3
ds.config["filtering"]["deform min"] = 0
ds.config["filtering"]["deform max"] = 2
ds.config["filtering"]["bright_avg min"] = 70   # to exclude too dark / too bright events
ds.config["filtering"]["bright_avg max"] = 150
ds.config["filtering"]["y_pos min"] = 0   # the y_pos limits have to be changed if the channel was off-center
ds.config["filtering"]["y_pos max"] = 25
ds.apply_filter()  # this step is important!
ds_child = dclab.new_dataset(ds) #dataset filtered according to the set filter
ds_child.apply_filter() ### will change ds_child when the filters for ds are changed
counts = (len(ds), len(ds_child))

images_from_rtdc_dataset, images_from_rtdc_dataset_masks = framecapture_notflat(ds_child)
images_from_rtdc_dataset = images_from_rtdc_dataset[0]
images_from_rtdc_dataset_masks = images_from_rtdc_dataset_masks[0]
images_from_rtdc_dataset_segmented = pd.DataFrame({'images':(images_from_rtdc_dataset * images_from_rtdc_dataset_masks)})


################################################ PLOTTING #######################################################
figure = plt.figure(figsize=(20,5))
ax = plt.subplot(1,5,1, xlabel = 'Area [$\mu$m$^2$]', xlim = (10,300), ylabel = 'Deformation [a.u.]', ylim = (0, 0.4))
density_scatter(ds_child["area_um"], ds_child["deform"], bins = [1000,100], ax = ax)
ax = plt.subplot(1,5,2, xlabel = 'Area [$\mu$m$^2$]', xlim = (10,120), ylabel = 'Brightness average [a.u.]', ylim = (80, 150))
density_scatter(ds_child["area_um"], ds_child["bright_avg"], bins = [1000,100], ax = ax)
ax = plt.subplot(1,5,3, xlabel = 'Area [$\mu$m$^2$]', xlim = (10,120), ylabel = 'Brightness SD [a.u.]', ylim = (0, 30))
density_scatter(ds_child["area_um"], ds_child["bright_sd"], bins = [1000,100], ax = ax)
ax = plt.subplot(1,5,4, xlabel = 'Brightness average [a.u.]', xlim = (80, 150), ylabel = 'Deformation [a.u.]', ylim = (0, 0.4))
density_scatter(ds_child["bright_avg"], ds_child["deform"], bins = [1000,100], ax = ax)
ax = plt.subplot(1,5,5, xlabel = 'Brightness average [a.u.]', xlim = (80, 150), ylabel = 'Brightness SD [a.u.]', ylim = (0, 30))
density_scatter(ds_child["bright_avg"], ds_child["bright_sd"], bins = [1000,100], ax = ax)
figure.canvas.mpl_connect('pick_event', onpick)


#%%

###################################################################################################################
######################################### FEATURES COMPUTATION ####################################################
###################################################################################################################


########## gets rid of empty cells

for idx in range(len(images_from_rtdc_dataset_segmented)):
    if images_from_rtdc_dataset_segmented[idx].shape[1] == 0:
        images_from_rtdc_dataset_segmented = images_from_rtdc_dataset_segmented.drop([idx])


########## calculates haralick features

features_haralick = []
for idx in range(len(images_from_rtdc_dataset_segmented)):
    features_har_1im = mahotas.features.haralick(images_from_rtdc_dataset_segmented['images'].iloc[idx]).mean(0)
    features_haralick.append(features_har_1im)


########## calculates Zernike features






plt.imshow(images_from_rtdc_dataset_segmented['images'][1400], cmap='gray')





















