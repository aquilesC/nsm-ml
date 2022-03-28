import matplotlib.ticker as plticker
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
from utils.YOLO import manYOLO, ConvertYOLOLabelsToCoord, ConvertCoordToYOLOLabels
from utils.funcs import remove_stuck_particle_2,load_all_models,reload_resnet, predict_function

#Load models
unet_path = "Network-weights/U-Net- I0.01-25_loss_0.009943331.h5" #Normal segmentation model
resnet_intensity_path="Network-weights/resnet-D0.01-1 I0.01-30 512x128_loss_5.4842668.h5" #Normal int model
resnet_diffusion_path='Network-weights/resnet-diffusion-09062021-0734012048x128_combinedLoss.h5' #Normal diffusion model

resnet_intensity_base=reload_resnet()
resnet_diffusion_base=reload_resnet()

resnet_intensity_base.load_weights(resnet_intensity_path)
resnet_diffusion_base.load_weights(resnet_diffusion_path)
unet = tf.keras.models.load_model(unet_path,compile=False)

resnet_ensemble = load_all_models()


#%% Enter user variables

#Path to data to be analyzed
mainDataPath='Data/Preprocessed Sample Data/'
#Plot data?
plotImages=False
#Save images?
saveImages = False
#%%    
plt.close('all')

try:
    del intensityArray
    del diffusionArray
    del intensityEnsembleArray
    del diffusionEnsembleArray
    del countarray
except:
    pass

measurements = os.listdir(mainDataPath)
for meas in measurements:
    print("Running measurement: "+meas)
    dataPath = mainDataPath+"/"+meas
    files = os.listdir(dataPath + "/intensity/")

    for fileName in files:
        print("Analyzing file: "+fileName)
        try:
            del YOLOLabels
            del YOLOCoords
        except:
            pass

        file = np.load(dataPath+ "/intensity/"+fileName)
        diff_file = np.load(dataPath+ "/diffusion/"+fileName)
        
        file = np.expand_dims(file,0)
        file = np.expand_dims(file,-1)
        diff_file = np.expand_dims(diff_file,0)
        diff_file = np.expand_dims(diff_file,-1)
        
        
        length = file.shape[2] 
        times = file.shape[1]

        if length == 150:
            lengthCutOff = int((150-128)/2)
        elif length == 640:
            lengthCutOff = int((640-512)/2)
        
        orig_img = file[:,904:904+8192,lengthCutOff:-lengthCutOff,:]
        orig_img_diff = diff_file[:,904:904+8192,lengthCutOff:-lengthCutOff,:]

        pred = unet.predict(orig_img)
        pred_diff = unet.predict(orig_img_diff)

        length = orig_img.shape[2]
        times = orig_img.shape[1]

        if plotImages:
            fig,axs=plt.subplots(1,2,figsize=(16,16))
            ax2 = axs[1]
            ax=axs[0]
            im=ax.imshow(pred_diff[0,:,:,0],aspect='auto')
            plt.colorbar(im,ax=ax)
            ax.set_ylabel('t')
            ax.set_title("Segmented Image")
            ax.set_xlabel('x')
            
            loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
            locY = plticker.MultipleLocator(base=100.0)
            axs[1].xaxis.set_major_locator(loc)
            axs[1].yaxis.set_major_locator(locY)
                   
            file = np.squeeze(file,0)
            file = np.squeeze(file,-1)

            im=ax2.imshow(orig_img[0,:,:,0],aspect='auto')
            plt.colorbar(im,ax=ax2)
            ax2.set_ylabel('t')
            ax2.set_title(fileName)
            ax2.set_xlabel('x')
            
            if saveImages:
                try:
                    plt.savefig("output/"+fileName+".png")
                    #np.save("output/"+fileName,pred_diff[0,:,:,0])
                except:
                    os.mkdir("output")
                    plt.savefig("output/"+fileName+".png")
                    #np.save("output/"+fileName,pred_diff[0,:,:,0])
                    
        YOLOLabels = manYOLO(pred_diff)
    
        if not None in YOLOLabels:
            boxes = ConvertYOLOLabelsToCoord(YOLOLabels[0,:,:],pred_diff.shape[1],pred_diff.shape[2])

            for i in range(0,len(boxes)):
                box=boxes[i,:]
                _,x1,y1,x2,y2 = box
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                box_l = x2-x1
                box_t = y2-y1
                img = pred_diff[:,y1:y2,x1:x2,:]
                labels=manYOLO(np.transpose(img,(0,2,1,3)))

                labels[:,:,1],labels[:,:,2],labels[:,:,3],labels[:,:,4] = np.copy(labels[:,:,2]), np.copy(labels[:,:,1]), np.copy(labels[:,:,4]), np.copy(labels[:,:,3])

                if not None in labels:
                    coords = ConvertYOLOLabelsToCoord(labels[0,:,:],box_t,box_l)
                    coords[:,1] = coords[:,1]+x1
                    coords[:,2] = coords[:,2]+y1
                    coords[:,3] = coords[:,3]+x1
                    coords[:,4] = coords[:,4]+y1
                    labels = ConvertCoordToYOLOLabels(coords,pred_diff.shape[1],pred_diff.shape[2]).reshape(1,-1,5)
                    try:
                        YOLOCoords = np.append(YOLOCoords,coords,0)
                        YOLOLabels = np.append(YOLOLabels,labels,1)
                    except:
                        YOLOLabels = np.copy(labels)
                        YOLOCoords = np.copy(coords)

        YOLOLabels=YOLOLabels[0,:,:]

        if not None in YOLOLabels:

            detections = ConvertYOLOLabelsToCoord(YOLOLabels,pred_diff.shape[1],pred_diff.shape[2])
            detections = detections[:,1:]
            
            try:
                detections
            except:
                detections = np.zeros((0,4))
        
            #Pick out each separate trajectory
            for x1, y1, x2, y2 in detections:
                x1 = np.max([x1,0])
                y1 = np.max([y1,0])
                x2 = np.min([x2,orig_img.shape[2]])
                y2 = np.min([y2,orig_img.shape[1]])
                box_w = x2 - x1
                box_h = y2 - y1    
                yolo_img = orig_img[:,int(y1):int(y2),int(x1):int(x2),:]
                yolo_img_diff = pred_diff[:,int(y1):int(y2),int(x1):int(x2),:]
                
                
                #If the detected trajectory is longer than minTrajLength, predict its intensity & diffusion with the base model and ensemble model.
                if yolo_img.shape[1] > 34 and yolo_img.shape[2] > 34:
                        
                    intensity = resnet_intensity_base(yolo_img,training=False)[0][0][0]
                    diffusion = resnet_diffusion_base(yolo_img_diff,training=False)[0][0][0]**2*57
                                   
                    intensity_ensemble,diffusion_ensemble = predict_function(resnet_ensemble,yolo_img,yolo_img_diff,intensity,diffusion)
                    
                    meanInt = np.mean([intensity,intensity_ensemble])
                    stdInt = np.std([intensity,intensity_ensemble])
                    meanDiff = np.mean([diffusion,diffusion_ensemble])
                    stdDiff = np.std([diffusion,diffusion_ensemble])
                    if stdInt>meanInt or stdDiff>meanDiff:#stdInt>intensity or stdDiff>diffusion:#stdInt > meanInt/2 or stdDiff > meanDiff:
                        print("Ignoring prediction, stdInt = "+str(stdInt)+", meanInt = "+str(meanInt)+", stdDiff = "+str(stdDiff)+", meanDiff = "+str(meanDiff))
                        intensity,diffusion,intensity_ensemble,diffusion_ensemble = 0,0,0,0
                        box_h=0
                else:
                    intensity,diffusion,intensity_ensemble,diffusion_ensemble = 0,0,0,0
                    box_h=0

                if plotImages:
                    # Create a Bounding Box
                    bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, zorder=2,edgecolor="black", facecolor="none")           
                    ax2.text(x1,y1,"I = "+str(np.round(intensity_ensemble,2))+", D = "+str(np.round(diffusion_ensemble,1)),color = "black")
                    ax2.add_patch(bbox)

                try:
                    intensityArray = np.append(intensityArray,intensity)  
                    diffusionArray = np.append(diffusionArray,diffusion)  
                    intensityEnsembleArray = np.append(intensityEnsembleArray,intensity_ensemble)  
                    diffusionEnsembleArray = np.append(diffusionEnsembleArray,diffusion_ensemble)  
                    meanIntArray = np.append(meanIntArray,meanInt)  
                    meanDiffArray = np.append(meanDiffArray,meanDiff)  
                    countArray = np.append(countArray,box_h)

                except:
                    intensityArray = np.array([intensity])
                    diffusionArray = np.array([diffusion])
                    intensityEnsembleArray = np.array(intensity_ensemble)  
                    diffusionEnsembleArray = np.array(diffusion_ensemble)  
                    meanIntArray = np.array(meanInt)  
                    meanDiffArray = np.array(meanDiff)  
                    countArray = np.array([box_h])

#%%
plt.close('all')
intensityArray=intensityArray[intensityArray>0]
diffusionArray=diffusionArray[diffusionArray>0]
intensityEnsembleArray= intensityEnsembleArray[intensityEnsembleArray>0]
diffusionEnsembleArray= diffusionEnsembleArray[diffusionEnsembleArray>0]
countArray=countArray[countArray>0]

from scipy.optimize import curve_fit
from pylab import hist, exp, sqrt, diag, plot, legend
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

data = [[i]*int(j) for i,j in zip(intensityEnsembleArray,countArray)]
data = np.concatenate(data).ravel()
nbr_bins = int(np.max(data)*20)

plt.figure(figsize=(12,6))
y,x,_=hist(data,nbr_bins,alpha=.3,label='data');
x=(x[1:]+x[:-1])/2 # make len(x)==len(y)

iOCRange=[0.75,1,2,5,10,20]
for i in range(0,len(iOCRange)):
    expected_height = len(data)
    expected_peak = iOCRange[i]
    expected_width = expected_peak/100
    
    expected=(expected_peak,expected_width,expected_height)
    print('expected peak = {:.2f}'.format(expected_peak))
    print('expected width = {:.2f}'.format(expected_width))
    params,cov=curve_fit(gauss,x,y,expected)
    sigma=sqrt(diag(cov))
    plot(x,gauss(x,*params),lw=2,label='Gaussian fit, iOC='+str(iOCRange[i]) + "$\cdot 10^{-4}~nm$")
    plt.xlabel("iOC ($\cdot 10^{-4} nm$)")
    plt.ylabel("#Frames")
    peak = params[0]
    sigma = abs(params[1]) # sometimes the width is outputted as a negative number...
    
    legend(fontsize=14)
    
    print('peak = {:.2f}'.format(peak))
    print('width = {:.2f}'.format(sigma))
    
#data = np.copy(diffusionEnsembleArray)
data = [[i]*int(j) for i,j in zip(diffusionEnsembleArray,countArray)]
data = np.concatenate(data).ravel()
nbr_bins = int(np.max(data)*20)

plt.figure(figsize=(12,6))
y,x,_=hist(data,nbr_bins,alpha=.3,label='data');
x=(x[1:]+x[:-1])/2 # make len(x)==len(y)

iOCRange=[10,20,50]
for i in range(0,len(iOCRange)):
    expected_height = len(data)
    expected_peak = iOCRange[i]
    expected_width = expected_peak/10
    
    expected=(expected_peak,expected_width,expected_height)
    print('expected peak = {:.2f}'.format(expected_peak))
    print('expected width = {:.2f}'.format(expected_width))
    params,cov=curve_fit(gauss,x,y,expected)
    sigma=sqrt(diag(cov))
    plot(x,gauss(x,*params),lw=2,label='Gaussian fit, D='+str(iOCRange[i])+"$\mu m^2/s$")
    plt.xlabel("D ($\mu m^2/s$)")
    plt.ylabel("#Frames")
    peak = params[0]
    sigma = abs(params[1]) # sometimes the width is outputted as a negative number...
    
    legend(fontsize=14)
    plt.xlim(0,75)
    
    print('peak = {:.2f}'.format(peak))
    print('width = {:.2f}'.format(sigma))


points=list(set(zip(intensityEnsembleArray,diffusionEnsembleArray))) 
plot_x=[i[0] for i in points]
plot_y=[i[1] for i in points]
count=np.array(countArray)
plt.figure(figsize=(12,12))
plt.scatter(plot_x,plot_y,color="blue",alpha=count/np.max(count))
plt.xlabel("iOC ($\cdot 10^{-4} nm$)")
plt.ylabel("D ($\mu m^2/s$)")
plt.yscale('log')
plt.ylim(0,100)
plt.gca().invert_yaxis()



# np.save(meas+"-intensity.npy",intensityArrayFull)
# np.save(meas+"-diffusion.npy",diffusionArrayFull)

#%%delete this
plt.close('all')
import statistics
from scipy.optimize import curve_fit
from pylab import hist, exp, sqrt, diag, plot, legend
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

intensityArray=intensityArray[intensityArray>0]
diffusionArray=diffusionArray[diffusionArray>0]
intensityEnsembleArray= intensityEnsembleArray[intensityEnsembleArray>0]
diffusionEnsembleArray= diffusionEnsembleArray[diffusionEnsembleArray>0]
results=[intensityArray,diffusionArray,intensityEnsembleArray,diffusionEnsembleArray]

for i in range(0,4):

    data = np.copy(results[i])
    nbr_sigma = 3
    
    expected_peak = statistics.median(data)
    expected_width = expected_peak/10

    nbr_bins = int(np.max(data)*10)
    expected_height = len(data)
    
    plt.figure(figsize=(12,6))
    y,x,_=hist(data,nbr_bins,alpha=.3,label='data');
    
    x=(x[1:]+x[:-1])/2 # make len(x)==len(y)
    
    expected=(expected_peak,expected_width,expected_height)
    print('expected peak = {:.2f}'.format(expected_peak))
    print('expected width = {:.2f}'.format(expected_width))
    params,cov=curve_fit(gauss,x,y,expected)
    sigma=sqrt(diag(cov))
    plot(x,gauss(x,*params),color='red',lw=2,label='model')
    peak = params[0]
    sigma = abs(params[1]) # sometimes the width is outputted as a negative number...
    
    plt.axvline(peak-nbr_sigma*sigma,label='peak +- {}$\sigma$'.format(nbr_sigma),color='black')
    plt.axvline(peak+nbr_sigma*sigma,color='black')
    legend(fontsize=14)
    
    
    print('peak = {:.2f}'.format(peak))
    print('width = {:.2f}'.format(sigma))


    if i % 2:
        plt.figure()
        uniqueIntensities = np.unique(results[i-1],return_index=True)
        uniqueDiffusivities = results[i][uniqueIntensities[1]]
        uniqueIntensities= results[i-1][uniqueIntensities[1]]
        plt.scatter(uniqueIntensities,uniqueDiffusivities)
        #plt.ylim(10,20)
        plt.gca().invert_yaxis()

# np.save(meas+"-intensity.npy",intensityArrayFull)
# np.save(meas+"-diffusion.npy",diffusionArrayFull)
#%%
## import scipy as sp
import scipy.ndimage.morphology
from scipy.signal import convolve2d



def manYOLO(im,treshold=0.05,trajTreshold=256):
    nump = 1
    batchSize = 1
    YOLOLabels =np.reshape([None]*batchSize*1*5,(batchSize,1,5))
    times = im.shape[1]
    length = im.shape[2]
    for j in range(0,batchSize):
        for k in range(0,nump):
            particle_img = im[j,:,:,k]
            #particle_img = particle_img/np.max(particle_img)

            particleOccurence = np.where(particle_img>treshold)
            if np.sum(particleOccurence) <= 0:
                continue
            else:
                trajectoryOccurence = np.diff(particleOccurence[0])
                trajectories = particleOccurence[0][np.where(trajectoryOccurence>trajTreshold)]
                trajectories = np.append(0,trajectories)
                trajectories = np.append(trajectories,times)

                for traj in range(0,len(trajectories)-1): 


                    particleOccurence = np.where(particle_img[trajectories[traj]:trajectories[traj+1],:]>treshold)
                    if np.sum(particleOccurence[1]) <=0 or np.sum(particleOccurence[0]) <=0:
                        continue
                    constant = trajectories[traj]
                    if traj != 0:
                        particleOccurence = np.where(particle_img[trajectories[traj]+trajTreshold:trajectories[traj+1],:]>treshold)
                        constant = trajectories[traj]+trajTreshold

                    x1,x2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])  
                    y1,y2 = np.min(particleOccurence[0])+constant,np.max(particleOccurence[0])+constant

                    A = (x2-x1)*(y2-y1)
                    if A > 100:

                        if YOLOLabels[0,0,0] is None:
                            YOLOLabels = np.reshape([0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)],(1,1,5))   
                        else:
                            YOLOLabels =np.append(YOLOLabels,np.reshape([0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)],(1,1,5)),1)
 
        return YOLOLabels
        
def NMS_test(prediction):
    conf_thres=0
    nms_thres=0
    #prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > 0#nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output[0][:,:4]

def mergeByTrajTreshold(detections,trajTreshold=16,nrIter=1):
    output = [None for _ in range(len(detections))]
    keep_boxes = []
    trajTresholdTime = trajTreshold
    for i in range(0,nrIter):

        while len(detections) >0:
            #x_range = np.array([(detections[0, 2]-detections[0,0])]).reshape(1,1)- (detections[:, 2]-detections[:, 0])
            #y_range = np.array([(detections[0, 3]-detections[0,1])]).reshape(1,1)- (detections[:, 3]-detections[:, 1])

            x_range_1 = np.abs(detections[0, 0]-detections[:, 0])
            x_range_2 = np.abs(detections[0, 2]-detections[:, 2])


            y_range_1 = np.abs(detections[0, 1]-detections[:, 1])
            y_range_2 = np.abs(detections[0, 3]-detections[:, 3])

            box_nearby_x =  np.logical_or(x_range_1<trajTreshold, x_range_2<trajTreshold).reshape(-1)
            box_nearby_y =  np.logical_or(y_range_1<trajTresholdTime, y_range_2<trajTresholdTime).reshape(-1)

            box_nearby = np.logical_and(box_nearby_x,box_nearby_y).reshape(-1)
            #box_nearby = np.logical_and(x_range<trajTreshold, y_range<trajTreshold).reshape(-1)
            newBoxes = detections[box_nearby,:]
            newBox = np.min(newBoxes[:,0]),np.min(newBoxes[:,1]),np.max(newBoxes[:,2]),np.max(newBoxes[:,3])

            keep_boxes += [newBox]
            detections = detections[~box_nearby]
        if keep_boxes and i != nrIter-1:
            detections = np.copy(keep_boxes)
            keep_boxes = []
            trajTreshold=trajTreshold/2 
            trajTresholdTime = trajTresholdTime/2
            
    if keep_boxes:
        output[0] = np.array(torch.Tensor(keep_boxes)) #torch.stack(keep_boxes)
            
    return output[0]

def remove_stuck_particle_2(original_img,M,nbr_its):
    original_img = original_img[0,:,:,0]
    #plt.figure()
    #plt.imshow(original_img,aspect='auto')
    for i in range(0,nbr_its):

        conv_img = original_img - convolve2d(original_img,np.ones((M,1))/M,mode='same',boundary='symm',fillvalue=1)
        
        img = original_img * (1-conv_img)
        try:
            img /= np.max(img)
        except:
            return original_img
        
        img[img<0.99] = 0
        img[img>0] = 1

        identifiedStuckTraj = np.sum(img,0)
        img[:,identifiedStuckTraj<2] = 0
        
        binary_img = scipy.ndimage.morphology.binary_dilation(img,structure=np.ones((M,1)))

        idcs = np.sum(binary_img,axis=0)==0
        cut_img = original_img[:,idcs]
        original_img = np.copy(cut_img)
    #plt.figure()
    #plt.imshow(original_img,aspect='auto')
    return np.expand_dims(original_img,(0,-1))


experimental_data=False
astra = False
exosomes=False
multiYOLO = False
YOLO = False
trajTreshold = 256
treshold=0.05#0.05
if not YOLO:
    YOLOLabels = [1]
plot=1


color="black"
    
def ConvertYOLOLabelsToCoord(YOLOLabels,xdim,ydim):
        #image,x1,y1,x2,y2
        coordinates = np.zeros(YOLOLabels.shape)
        coordinates[:,0] = YOLOLabels[:,0]
        coordinates[:,1] = (xdim-1)*YOLOLabels[:,1]-(xdim-1)/2*YOLOLabels[:,3]
        coordinates[:,2] = (ydim-1)*YOLOLabels[:,2]-(ydim-1)/2*YOLOLabels[:,4]
        coordinates[:,3] = (xdim-1)*YOLOLabels[:,1]+(xdim-1)/2*YOLOLabels[:,3]
        coordinates[:,4] = (ydim-1)*YOLOLabels[:,2]+(ydim-1)/2*YOLOLabels[:,4]

        return coordinates



    

resnet_intensity=reload_resnet()
resnet_diffusion=reload_resnet()

resnet_path_075_1 = "Network-weights/resnet-D0.01-1 I0.0-1.5 512x128_loss_0.1030596.h5"
resnet_path_075 = "Network-weights/resnet-D0.05-1.15 I0.01-0.99 2048x128_loss_0.002396633.h5"
resnet_path_1 = "Network-weights/resnet-D0.05-1.15 I0.8-1.9 2048x128_loss_0.011120575.h5"
resnet_path_2 = "Network-weights/resnet-D0.05-1.15 I1.5-2.5 512x128_loss_0.0037260056.h5"
resnet_path_5 = "Network-weights/resnet-D0.01-1 I2.5-7.5 512x128_loss_0.03088841.h5"
resnet_path_10 = "Network-weights/resnet-D0.05-1.15 I5.1-15 512x128_loss_0.025085581.h5"
resnet_path_20 = "Network-weights/resnet-D0.01-1 I15-25 512x128_loss_0.13881019.h5"

resnet_diff_path_10 = "Network-weights/resnet-diffusion-D0.57-14.25 I0.01-30 2048x128_loss_0.03129236.h5"
resnet_diff_path_20 = "Network-weights/resnet-diffusion-D11.54-36.48 I0-50 512x128_loss_0.13375391.h5"
resnet_diff_path_50 = "Network-weights/resnet-diffusion-D27.93-75.38 I0.01-30 2048x128_loss_0.14572684.h5"

resnet_intensity_075_1 = reload_resnet()
resnet_intensity_075 = reload_resnet()
resnet_intensity_1 = reload_resnet()
resnet_intensity_2 = reload_resnet()
resnet_intensity_5 = reload_resnet()
resnet_intensity_10 = reload_resnet()
resnet_intensity_20 = reload_resnet()

resnet_diffusion_10= reload_resnet()
resnet_diffusion_20 = reload_resnet()
resnet_diffusion_50 = reload_resnet()

resnet_intensity_075_1.load_weights(resnet_path_075_1)
resnet_intensity_075.load_weights(resnet_path_075)
resnet_intensity_1.load_weights(resnet_path_1)
resnet_intensity_2.load_weights(resnet_path_2)
resnet_intensity_5.load_weights(resnet_path_5)
resnet_intensity_10.load_weights(resnet_path_10)
resnet_intensity_20.load_weights(resnet_path_20)

resnet_diffusion_10.load_weights(resnet_diff_path_10)
resnet_diffusion_20.load_weights(resnet_diff_path_20)
resnet_diffusion_50.load_weights(resnet_diff_path_50)
        
pred_factor=1
pred_diff_factor=1
#pred_factor=1/0.75

img_size = 512

unet = tf.keras.models.load_model(unet_path,compile=False)

resnet_intensity.load_weights(resnet_intensity_path)
resnet_diffusion.load_weights(resnet_diffusion_path)
resnet_intensity_base.load_weights("Network-weights/resnet-D0.01-1 I0.01-30 512x128_loss_5.4842668.h5")
resnet_diffusion_base.load_weights("Network-weights/resnet-diffusion-09062021-0734012048x128_combinedLoss.h5")

#resnet_intensity_075_1.load_weights(resnet_path_075_1)
#resnet_intensity_2.load_weights(resnet_path_2)
#resnet_intensity_5.load_weights(resnet_path_5)
#resnet_intensity_10.load_weights(resnet_path_10)
#resnet_intensity_20.load_weights(resnet_path_20)

resnet_diffusion_10.load_weights(resnet_diff_path_10)
resnet_diffusion_20.load_weights(resnet_diff_path_20) 
resnet_diffusion_50.load_weights(resnet_diff_path_50) 

timesInitial = 904
times = timesInitial+ 8192

if experimental_data:
    timesInitial = 976
    times = timesInitial+ 2048
dataSave={}
try:
    del intensityArray
    del diffusionArray
    del intensityArrayFull
    del diffusionArrayFull

except:
    pass
#files = os.listdir(mainDataPath)
#dataPath = mainDataPath+"/"+files[0]

if astra:
    measurements = os.listdir("../../input/astrasample") + os.listdir("../../input/astrasample2")
elif exosomes:
    measurements = os.listdir(mainDataPath)
else:
    measurements = os.listdir(mainDataPath)
    print(measurements)
    #measurements = [measurements[0]]
    


measurements = np.flip(measurements)
#measurements=[measurements[0]]  
print(measurements)
counter=0
counter2=0
for meas in measurements:

    if meas =="0.00075":
        resnet_intensity.load_weights(resnet_path_075)
        measNr = 0
    if meas =="0.001":
        measNr = 4
        resnet_intensity.load_weights(resnet_path_10)
    if meas =="0.002":
        measNr = 5
        resnet_intensity.load_weights(resnet_path_20)

    if counter > 1 or True:
        YOLO=True
        multiYOLO =True# True
        trajTreshold = 32
        treshold=0.05

    else:
        YOLO = False
        multiYOLO = False
        trajTresholdMerge = 256
        YOLOLabels = [1]
        treshold=0.05
        
    tresholdMerge=0.05
    trajTresholdMerge = 256
    counter+=1
    
    minImgArea = 100

    dataPath = mainDataPath+"/"+meas
    print(dataPath)
    files = os.listdir(dataPath + "/intensity/")
    
    measSave = np.copy(meas)
    for fileName in files:
        if "0.0001-0.0005" in measSave:
            if "iOC0.0001" in fileName:
                meas = "0.0001"
                measNr = 1
                resnet_intensity.load_weights(resnet_path_1)
            if "iOC0.0002" in fileName:
                meas = "0.0002"
                measNr = 2
                resnet_intensity.load_weights(resnet_path_2)
            if "iOC0.0005" in fileName:
                meas = "0.0005"
                measNr = 3
                resnet_intensity.load_weights(resnet_path_5) 
                
        if "iOC7.5e-05" in fileName:
            measNr = 0

        if "iOC0.001" in fileName:
            measNr = 4

        if "iOC0.002" in fileName:
            measNr = 5
                
                
        if "D50" in fileName:
            D = "D50"
            DNr = 2
        elif "D20" in fileName:
            D = "D20"
            DNr = 1
        elif "D10" in fileName:
            D = "D10"
            DNr = 0
        
        #if not "D50" in fileName:# or not "iOC0.0001" in fileName:
        #    pred_diff_factor=1
       # else:
       #     pred_diff_factor=50/57


        file = np.load(dataPath+ "/intensity/"+fileName)
        diff_file = np.load(dataPath+ "/diffusion/"+fileName)

        file = np.expand_dims(file,0)
        file = np.expand_dims(file,-1)
        diff_file = np.expand_dims(diff_file,0)
        diff_file = np.expand_dims(diff_file,-1)
    

        

        
        timesLimit = file.shape[1] % img_size
        times = file.shape[1]
        print(timesLimit)

        if not experimental_data:
            orig_img = file[:,904:904+8192,11:-11,:]
            orig_img_diff = diff_file[:,904:904+8192,11:-11,:]
        elif timesLimit > 0:
            if file.shape[1] >= 8192+904:
                orig_img = file[:,904:904+8192,11:-11,:]
                orig_img_diff = diff_file[:,904:904+8192,11:-11,:]
            else:
                orig_img = file[:,0:-int(timesLimit),11:-11,:]
                orig_img_diff = diff_file[:,0:-int(timesLimit),11:-11,:]
        else:
            orig_img = file[:,:,11:-11,:]
            orig_img_diff = diff_file[:,:,11:-11,:]


        #pred = np.copy(unet(orig_img,training=False)) 
        #pred_diff = np.copy(unet(orig_img_diff,training=False))
        pred = unet.predict(orig_img)
        pred_diff = unet.predict(orig_img_diff)

        #pred_diff_2=unet2.predict(orig_img_diff)
        length = orig_img.shape[2]
        times = orig_img.shape[1]


        
        if plot:
            #fig,axs=plt.subplots(1,2) 
            fig,axs=plt.subplots(1,2,figsize=(16,16))
            ax2 = axs[1]
            ax=axs[0]
            #plt.figure(figsize=(16,16))
            im=ax.imshow(pred_diff[0,:,:,0],aspect='auto')
            plt.colorbar(im,ax=ax)
            #axs[1].imshow(pred[0,:,:,0],aspect='auto')
            #axs[1].imshow(pred_diff_2[0,:,:,0],aspect='auto')
            ax.set_ylabel('t')
            ax.set_title("Segmented Image")
            ax.set_xlabel('x')
            

           
            
            file = np.squeeze(file,0)
            file = np.squeeze(file,-1)

            #fig,ax2=plt.subplots(1,figsize=(16,16))
            im=ax2.imshow(orig_img[0,:,:,0],aspect='auto')
            plt.colorbar(im,ax=ax2)
            ax2.set_ylabel('t')
            ax2.set_title(fileName)
            ax2.set_xlabel('x')
            
            

  
        #pred_diff[:,:,0:8,:] = 0
        #pred_diff[:,:,-8:,:] = 0

        if YOLO:
            YOLOLabels = manYOLO(pred_diff,trajTreshold=trajTreshold,treshold=treshold)

    
    
        #YOLOLabels = ConvertTrajToMultiBoundingBoxes(im,length=length,times=times,treshold=0.5,trackMultiParticle=False)
        if multiYOLO and not None in YOLOLabels:
            boxes = ConvertYOLOLabelsToCoord(YOLOLabels[0,:,:],pred_diff.shape[1],pred_diff.shape[2])

            #YOLOLabels = np.zeros(1,len(boxes),5)
            for i in range(0,len(boxes)):
                box=boxes[i,:]
                _,x1,y1,x2,y2 = box
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                box_l = x2-x1
                box_t = y2-y1
                img = pred_diff[:,y1:y2,x1:x2,:]
                labels=manYOLO(np.transpose(img,(0,2,1,3)),trajTreshold=trajTreshold,treshold=treshold)

                labels[:,:,1],labels[:,:,2],labels[:,:,3],labels[:,:,4] = np.copy(labels[:,:,2]), np.copy(labels[:,:,1]), np.copy(labels[:,:,4]), np.copy(labels[:,:,3])

                #labels[:,:,1],labels[:,:,2],labels[:,:,3],labels[:,:,4] = labels[:,:,2], labels[:,:,1], labels[:,:,4], labels[:,:,3]




                if not None in labels:
                    coords = ConvertYOLOLabelsToCoord(labels[0,:,:],box_t,box_l)
                    coords[:,1] = coords[:,1]+x1
                    coords[:,2] = coords[:,2]+y1
                    coords[:,3] = coords[:,3]+x1
                    coords[:,4] = coords[:,4]+y1
                    labels = ConvertCoordToYOLOLabels(coords,pred_diff.shape[1],pred_diff.shape[2]).reshape(1,-1,5)
                    if i ==0:
                        YOLOLabels = np.copy(labels)
                        YOLOCoords = np.copy(coords)
                    else:
                        YOLOLabels = np.append(YOLOLabels,labels,1)
                        YOLOCoords = np.append(YOLOCoords,coords,0)
                   ## try:
                  #     YOLOLabels = np.append(YOLOLabels,labels,1)
                   #    YOLOCoords = np.append(YOLOCoords,coords,0)
                   # except:
                   #     YOLOLabels = np.copy(labels).reshape(1,-1,5)
                   #     YOLOCoords = np.copy(coords)
       # else:
       #      YOLOLabels=YOLOLabels[0,:,:]
        if YOLO:    
            YOLOLabels=YOLOLabels[0,:,:]
        
        

        if not None in YOLOLabels:
            if YOLO and (YOLOCoords[:,4]-YOLOCoords[:,2] < 9000).all():
                detections = ConvertYOLOLabelsToCoord(YOLOLabels,pred_diff.shape[1],pred_diff.shape[2])
                detections = detections[:,1:]
            else:
                try:
                    del detections
                except:
                    pass
                
                if True:
                    yolo_img_diff = np.copy(pred_diff[0,:,:,0])
                    ono=np.ones((50,1))
                    ono=ono/np.sum(ono)
                    yolo_img_diff-=convolve2d(yolo_img_diff,ono,mode="same")
                    yolo_img_diff-=convolve2d(yolo_img_diff,np.transpose(ono),mode="same")

                    yolo_img_diff[yolo_img_diff<tresholdMerge] = 0
                    #yolo_img_diff[yolo_img_diff!=0] = 1
                    objects, num_objects = sp.ndimage.label(yolo_img_diff)
                    object_slices =  sp.ndimage.find_objects(objects)
                    #detections = np.zeros((len(object_slices),4))              
                    for objSlice in object_slices:
                         x1 = objSlice[1].start
                         y1 = objSlice[0].start
                         x2 = objSlice[1].stop
                         y2 = objSlice[0].stop
                         #detections[i,...] = x1,y1,x2,y2

                         if (x2-x1)*(y2-y1)>0:
                             try:
                                 detections = np.append(detections,np.array([x1,y1,x2,y2]).reshape(1,4),0)
                             except:
                                 detections = np.array([x1,y1,x2,y2])
                                 detections = detections.reshape(1,4)
                                
                else:
                    for i in range(0,32):
                        for j in range(0,2):
                            x1 = 256*j
                            x2 = 256*(j+1)
                            y1 = 256*i
                            y2 = 256*(i+1)
                            try:
                                 detections = np.append(detections,np.array([x1,y1,x2,y2]).reshape(1,4),0)
                            except:
                                 detections = np.array([x1,y1,x2,y2])
                                 detections = detections.reshape(1,4)
           
            try:
                detections
            except:
                detections = np.zeros((0,4))
        
            if len(detections) > 200 and False:
                detections = np.array(NMS_test(torch.Tensor(np.expand_dims(np.append(detections,np.ones((len(detections),2)),1),0))))
                detections = mergeByTrajTreshold(detections,trajTreshold= trajTresholdMerge,nrIter=5)

                
            for x1, y1, x2, y2 in detections:
                conf = 1
                cls_conf = 1
                cls_pred = 0
                x1 = np.max([x1,0])
                y1 = np.max([y1,0])
                x2 = np.min([x2,orig_img.shape[2]])
                y2 = np.min([y2,orig_img.shape[1]])
                #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                # x1 = 0
                # x2 = 127
                #y1 = 0
                # y2 = 8191
                box_w = x2 - x1
                box_h = y2 - y1    

                #Resnet-intensity predictions
                yolo_img = orig_img[:,int(y1):int(y2),int(x1):int(x2),:]
                yolo_img_diff = pred_diff[:,int(y1):int(y2),int(x1):int(x2),:]
                


                
                print(yolo_img.shape)
                if yolo_img.shape[1] >= 32 and yolo_img.shape[2] >= 32 and cls_conf > 0.5:
                    
                    #try:
                    M = 10
                    nbr_its = 5
                    #pred = tf.convert_to_tensor(remove_stuck_particle_2(pred.cpu().numpy(),M,nbr_its))
                    #pred_diff = tf.convert_to_tensor(remove_stuck_particle_2(pred_diff.cpu().numpy(),M,nbr_its))
                    
                    if False: #remember to turn this on again
                        yolo_img_diff = remove_stuck_particle_2(yolo_img_diff,M,nbr_its)
                        
                    #except:
                   #     pass
                    #intensity = resnet_intensity.predict(yolo_img)[0][0][0]*pred_factor
                    intensity_base=resnet_intensity_base(yolo_img,training=False)[0][0][0]
                    diffusion_base=resnet_diffusion_base(yolo_img_diff,training=False)[0][0][0]**2*57
                    intensity = resnet_intensity(yolo_img,training=False)[0][0][0]*pred_factor
                    diffusion = resnet_diffusion(yolo_img_diff,training=False)[0][0][0]**2*57*pred_diff_factor
                    
                    if "D50" in fileName:
                        D = "D50"
                        diffusion = resnet_diffusion_50(yolo_img_diff,training=False)[0][0][0]**2*57*pred_diff_factor
                    elif "D20" in fileName:
                        D = "D20"
                        diffusion = resnet_diffusion_20(yolo_img_diff,training=False)[0][0][0]**2*57*pred_diff_factor
                    elif "D10" in fileName:
                        D = "D10"
                        diffusion = resnet_diffusion_10(yolo_img_diff,training=False)[0][0][0]**2*57*pred_diff_factor
                    
                    meanInt = np.mean([intensity_base,intensity])
                    stdInt = np.std([intensity_base,intensity])
                    meanDiff = np.mean([diffusion_base,diffusion])
                    stdDiff = np.std([diffusion_base,diffusion])
                    
                    if stdInt>intensity or stdDiff>diffusion:#stdInt > meanInt/2 or stdDiff > meanDiff:
                        print("ignoring prediction, stdInt = "+str(stdInt)+", meanInt = "+str(meanInt)+", stdDiff = "+str(stdDiff)+", meanDiff = "+str(meanDiff))
                        intensity=0
                        diffusion=0
                        box_h=1

                else:
                    intensity = 0
                    diffusion = 0
                    box_h=1
                
                #diffusion = diffusion/3
                #if intensity > 100:
                   # diffusion = diffusion/(intensity*2/100)
                if plot:
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, zorder=2,edgecolor="black", facecolor="none")
                    #bboxReal = patches.Rectangle((x1, y1), int(x1)+closest_power2(int(x2)-int(x1)), int(y1)+closest_power2(int(y2)-int(y1)), linewidth=2, zorder=2,edgecolor="white", facecolor="none")
                                        # Add the bbox to the plot
                   # if int(cls_pred) == 0:
                    ax2.text(x1,y1,"I = "+str(np.round(intensity,2))+", D = "+str(np.round(diffusion,1))+", P = "+str(np.round(conf,2))+", $P_c$ = "+str(np.round(cls_conf,2)),color = color)
                    ax2.add_patch(bbox)



                propConstant = int(box_h)
                intensityFull = [intensity]*propConstant
                diffusionFull = [diffusion]*propConstant

                #intensityFull = [intensity]*trajectoryPixels
                #diffusionFull = [diffusion]*trajectoryPixels
                
                #savePath = "output/"
                #with open(savePath+meas+"-"+D+"-intensity.npy","ab+") as f:
                    #np.save(f, np.array(intensityFull))
                #with open(savePath+meas+"-"+D+"-diffusion.npy","ab+") as f:    
                    #np.save(f, np.array(diffusionFull))

                    
                try:
                    intensityArray = np.append(intensityArray,intensity)  
                    diffusionArray = np.append(diffusionArray,diffusion)  
                    
                    intensityArrayFull = np.append(intensityArrayFull,intensityFull)  
                    diffusionArrayFull = np.append(diffusionArrayFull,diffusionFull)  
                except:
                    intensityArray = np.array([intensity])
                    diffusionArray = np.array([diffusion])
                    intensityArrayFull = np.array(intensityFull)  
                    diffusionArrayFull = np.array(diffusionFull)

    
#%% delete this
intensityArrayFull=intensityArrayFull[intensityArrayFull>0]
diffusionArrayFull=diffusionArrayFull[diffusionArrayFull>0]
import statistics
from scipy.optimize import curve_fit
from pylab import hist, exp, sqrt, diag, plot, legend
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

nbr_sigma = 3
#diffusionArrayFull
#intensityArrayFull
# Expected peak and width parameters can be changed (and automatized more effectively), but the current settings seem to work well.
expected_peak = statistics.median(intensityArrayFull)
expected_width = expected_peak/10

data = np.copy(intensityArrayFull)
nbr_bins = 200 # nbr of bins can be automatized
expected_height = len(data) # not a very sensitive parameter.

plt.figure(figsize=(12,6))
y,x,_=hist(data,nbr_bins,alpha=.3,label='data');

x=(x[1:]+x[:-1])/2 # make len(x)==len(y)

expected=(expected_peak,expected_width,expected_height)
print('expected peak = {:.2f}'.format(expected_peak))
print('expected width = {:.2f}'.format(expected_width))
params,cov=curve_fit(gauss,x,y,expected)
sigma=sqrt(diag(cov))
plot(x,gauss(x,*params),color='red',lw=2,label='model')
peak = params[0]
sigma = abs(params[1]) # sometimes the width is outputted as a negative number...

plt.axvline(peak-nbr_sigma*sigma,label='peak +- {}$\sigma$'.format(nbr_sigma),color='black')
plt.axvline(peak+nbr_sigma*sigma,color='black')
legend(fontsize=14)


print('peak = {:.2f}'.format(peak))
print('width = {:.2f}'.format(sigma))

nbr_sigma = 3
#diffusionArrayFull
intensityArrayFull
# Expected peak and width parameters can be changed (and automatized more effectively), but the current settings seem to work well.
expected_peak = statistics.median(diffusionArrayFull)
expected_width = expected_peak/10

data = np.copy(diffusionArrayFull)
nbr_bins = 200 # nbr of bins can be automatized
expected_height = len(data) # not a very sensitive parameter.

plt.figure(figsize=(12,6))
y,x,_=hist(data,nbr_bins,alpha=.3,label='data');

x=(x[1:]+x[:-1])/2 # make len(x)==len(y)

expected=(expected_peak,expected_width,expected_height)
print('expected peak = {:.2f}'.format(expected_peak))
print('expected width = {:.2f}'.format(expected_width))
params,cov=curve_fit(gauss,x,y,expected)
sigma=sqrt(diag(cov))
#plot(x,gauss(x,*params),color='red',lw=2,label='model')
peak = params[0]
sigma = abs(params[1]) # sometimes the width is outputted as a negative number...

#plt.axvline(peak-nbr_sigma*sigma,label='peak +- {}$\sigma$'.format(nbr_sigma),color='black')
#plt.axvline(peak+nbr_sigma*sigma,color='black')
legend(fontsize=14)


print('peak = {:.2f}'.format(peak))
print('width = {:.2f}'.format(sigma))

plt.figure()
uniqueIntensities = np.unique(intensityArrayFull,return_index=True)
uniqueDiffusivities = diffusionArrayFull[uniqueIntensities[1]]
uniqueIntensities= intensityArrayFull[uniqueIntensities[1]]
plt.scatter(uniqueIntensities,uniqueDiffusivities)
#plt.ylim(10,20)
plt.gca().invert_yaxis()

np.save(meas+"-intensity.npy",intensityArrayFull)
np.save(meas+"-diffusion.npy",diffusionArrayFull)

