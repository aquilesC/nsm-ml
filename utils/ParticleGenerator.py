#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import scipy.io as IO
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow_addons.layers import InstanceNormalization
import os
K=keras.backend
import sys
sys.path.append("../../../../deeptrack/")
# sys.path.append("..")
# sys.path.insert(0,'DeepTrack-2.0/')
import deeptrack as dt
import deeptrack.models
# Font parameters
sz = 14
plt.rc('font', size=sz)          # controls default text sizes
plt.rc('axes', titlesize=sz)     # fontsize of the axes title
plt.rc('axes', labelsize=sz)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=sz)    # fontsize of the tick labels
plt.rc('ytick', labelsize=sz)    # fontsize of the tick labels
plt.rc('legend', fontsize=sz)    # legend fontsize
plt.rc('figure', titlesize=sz)  # fontsize of the figure title

times = 1024
length = 2048


from deeptrack.features import Feature
import skimage.measure
import numpy as np
    


class init_particle_counter(Feature):
    
    def __init__(self, vel=0, D=0, I=0, s=0, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

    def get(self, image, vel, D, I,s, **kwargs):
        
        # Init particle counter
        nbr_particles = 0
        image.append({"nbr_particles":nbr_particles})
        
        return image

    
class get_trajectory(Feature):

    
    def __init__(self, vel=0, D=0.1, I=[0.01, 5],s=0.05, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, nbr_particles=0, **kwargs
        )
        self.I=I

    def get(self, image, vel, D, I, s, **kwargs):
        I = self.I[0]+self.I[1]*np.random.rand()
        I = I/10000
        # Particle counter
        nbr_particles = image.properties[1]['nbr_particles']
        particle_index = nbr_particles + 1
        image.properties[1]['nbr_particles'] += 1
        
        
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        x0=0
        x0+=np.cumsum(vel+D*np.random.randn(times))
        v1=np.transpose(I*f2(1,x0,s,0,Y))
        image[...,0]*=(1-v1)
        
        particle_trajectory = np.transpose(f2(1,x0,0.05,0,Y))
        
        # Add trajectory to image
        image[...,1] += particle_trajectory 
        
        # Save single trajectory as additional image
        image[...,-particle_index] = particle_trajectory      
        
        
        try:
            image.properties["D"]+=10*D#*np.sum(np.transpose(f2(1,x0,.1,0,Y)))
            image.properties["I"]+=s*I*np.sqrt(2*np.pi)*256*.03 ##256 should be length!
        except:
            image.append({"D":10*D,"I":s*I*np.sqrt(2*np.pi)*256*.03})
            
        return image
    
class get_exosome_trajectory(Feature):

    
    def __init__(self, vel=0, D=0.1, I=[0.01, 5],s=0.05, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, nbr_particles=0, **kwargs
        )
        self.I=I

    def get(self, image, vel, D, I, s, **kwargs):
        I = self.I[0]+self.I[1]*np.random.rand()
        I = I/10000
        
        
        # Particle counter
        nbr_particles = image.properties[1]['nbr_particles']
        particle_index = nbr_particles + 1
        image.properties[1]['nbr_particles'] += 1
        
        
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        x0=0
        x0+=np.cumsum(vel+D*np.random.randn(times))
        v1=np.transpose(I*f2(1,x0,s,0,Y))
        image[...,0]*=(1-v1)
        
        particle_trajectory = np.transpose(f2(1,x0,0.05,0,Y))
        
        # Add trajectory to image
        image[...,1] += particle_trajectory 
        
        # Save single trajectory as additional image
        image[...,-particle_index] = particle_trajectory      
        
        
        try:
            image.properties["D"]+=10*D#*np.sum(np.transpose(f2(1,x0,.1,0,Y)))
            image.properties["I"]+=s*I*np.sqrt(2*np.pi)*256*.03 ##256 should be length!
        except:
            image.append({"D":10*D,"I":s*I*np.sqrt(2*np.pi)*256*.03})
            
        return image
    
    
class gen_noise(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0,biglam=0,bgnoiseCval=0,bgnoise=0,bigx0=0,sinus_noise_amplitude=0,freq=0,**kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA,biglam=biglam,bgnoiseCval=bgnoiseCval,bgnoise=bgnoise,bigx0=bigx0,sinus_noise_amplitude=sinus_noise_amplitude,freq=0, **kwargs
        )

    def get(self, image, noise_lev, dX, dA,biglam,bgnoiseCval,bgnoise,bigx0,sinus_noise_amplitude,freq, **kwargs):
        from scipy.signal import convolve
        
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        bgnoise*=np.random.randn(length)

        tempcorr=3*np.random.rand()
        dAmp=dA#*np.random.rand()
        shiftval=dX*np.random.randn()
        dx=0
        dx2=0
        dAmp0=0
        
        bg0=f2(1,bigx0,biglam,0,x)
        image.append({"bg0":bg0})
        
        ll=(np.pi-.05)
        for j in range(times):
            dx=(sinus_noise_amplitude*np.random.randn()+np.sin(freq*j))*dX
            
            bgnoiseC=f2(1,0,bgnoiseCval,dx,x)
            bgnoiseC/=np.sum(bgnoiseC)
            bg=f2(1,bigx0+dx,biglam,0,x)*(1+convolve(bgnoise,bgnoiseC,mode="same"))
            dAmp0=dA*np.random.randn()
            bg*=(1+dAmp0)
            image[j,:,0]=bg*(1+noise_lev*np.random.randn(length))+.4*noise_lev*np.random.randn(length)
        
        return image
    
class post_process(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0, **kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA, **kwargs
        )

    def get(self, image, **kwargs):
        from scipy.signal import convolve2d
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        
        
        #Perform same preprocessing as done on experimental images
        image[:,:,0]=(image[...,0]-np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0))/np.mean(image[...,0],axis=0)
        ono=np.ones((200,1))
        ono=ono/np.sum(ono)
        image[:,:,0]-=convolve2d(image[:,:,0],ono,mode="same")
        image[:,:,0]-=convolve2d(image[:,:,0],np.transpose(ono),mode="same")
        
        image[:,:,0]-=np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0)
        a=np.std(image[...,0],axis=0)
        image[:,:,0]/=a
        try:
            image.properties["I"]/=a
        except:
            pass        
        return image

class post_process_basic(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0, **kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA, **kwargs
        )

    def get(self, image, **kwargs):              
        #Perform same preprocessing as done on experimental images
        image[:,:,0]=(image[...,0]-np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0))/np.mean(image[...,0],axis=0)
        return image

    
class input_array(Feature):
    __distributed__ = False
    def get(self,image, **kwargs):
        image=np.zeros((times,length,10))
        return image
    
def heaviside(a):
    a[a>0] = 1
    a[a!=1] = 0
    return a

class get_diffusion(Feature):
    
    def __init__(self, vel=0, D=0, I=0, s=0, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

    def get(self, image, vel, D, I,s, **kwargs):
        LOW = 0.1
        HIGH = 1.15
        D = (LOW + HIGH*np.random.rand())
        image.append({"D":D})
        
        return image
    
class get_long_DNA_trajectory(Feature):

    
    def __init__(self, vel=0, D=0.1, I=0.1,I_2=0.5,particle_width=0.1,s=0.01, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,I_2=I_2,particle_width=particle_width,s=s, nbr_particles=0, **kwargs
        )

    def get(self, image, vel, D, I,I_2, particle_width, s, **kwargs):
        
        # vel= np.random.choice([(500+500*np.random.rand())*10**-6,(1500+1000*np.random.rand())*10**-6])
        # D=  0.0005+0.0005*np.random.rand()
        # I=5
        # I_2=20
        # particle_width =  0.08 + 0.04*np.random.rand()
        
        from scipy.signal import convolve
        I = I/10000
        # Particle counter
        nbr_particles = image.properties[1]['nbr_particles']
        particle_index = nbr_particles + 1
        image.properties[1]['nbr_particles'] += 1
        
        f=lambda a,b,x0,x: a*(heaviside((x-x0)-b/2)-heaviside((x-x0)+b/2))
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        x0=-1+2*np.random.rand()
        x0+=np.cumsum(vel+D*np.random.randn(times))

        x1= x0-particle_width/2+np.random.rand()*particle_width
        
        vv=np.transpose(convolve(I*f(1,particle_width,x0,Y),f2(1,0,s,0,Y[:,0:1]),mode="same"))
        v1=np.array(vv)-np.transpose(I*I_2*f2(1,x1,s,0,Y))         

        
        image[...,0]*=(1+v1)
        image[...,2]*=(1+v1)
        
        particle_trajectory =np.transpose(convolve(f(1,particle_width,x0,Y),f2(1,0,s,0,Y[:,0:1]),mode="same"))#v1

        # Add trajectory to image
        image[...,1] += particle_trajectory 
        image[...,1]=np.clip(np.abs(image[...,1]),0,1)
        # Save single trajectory as additional image
        image[...,-particle_index] = particle_trajectory             
        
        try:
            image.properties["D"]+=10*D#*np.sum(np.transpose(f2(1,x0,.1,0,Y)))
            image.properties["I"]+=s*I*np.sqrt(2*np.pi)*256*.03
        except:
            image.append({"D":10*D,"hash_key":list(np.zeros(4))})
            image.append({"I":s*I*np.sqrt(2*np.pi)*256*.03,"hash_key":list(np.zeros(4))})
            
        return image

#%% Plot  lambda/DNA
plt.close('all')
vel=lambda: np.random.choice([(500+500*np.random.rand())*10**-6,(1500+1000*np.random.rand())*10**-6])
#vel=lambda: (50000*np.random.rand())*10**-6
D= lambda: 0.0005+0.0005*np.random.rand()

I=lambda: 4+2*np.random.rand()
I_2=0
s = 0.01 #lambda: 0.01 + 0.01*np.random.rand()
particle_width = 0.1

longDNAImage = dt.FlipLR(dt.FlipUD(input_array() + init_particle_counter() 
                        + gen_noise(dX=.00001+.00003*np.random.rand(),
                                                dA=0,
                                                noise_lev=.0001,
                                                biglam=0.6+.4*np.random.rand(),
                                                bgnoiseCval=0.03+.02*np.random.rand(),
                                                bgnoise=.08+.04*np.random.rand(),
                                                bigx0=lambda: .1*np.random.randn())
                                    + get_long_DNA_trajectory(vel=vel,D=D,I=I,I_2=I_2,s=s,particle_width=particle_width)
                                    + post_process_basic()))

from scipy.ndimage.interpolation import rotate

for i in range(0,3):
    plotDNAImage = longDNAImage.resolve()
    plt.figure()
    plt.imshow(plotDNAImage[:,:,0],aspect='auto')
    plt.colorbar()
    plt.figure()
    plt.imshow(plotDNAImage[:,:,1],aspect='auto')
    plt.colorbar()
    plt.clim()
    
    
    DNALength = np.copy(plotDNAImage[...,1])
    indicesOfDNA = np.where(DNALength>np.max(DNALength)*0.99)
    x,y = np.max(indicesOfDNA[1])-np.min(indicesOfDNA[1]),np.max(indicesOfDNA[0])-np.min(indicesOfDNA[0])
    
    angle = np.rad2deg(np.arctan(y/x))
    
    DNALength = rotate(DNALength,angle=angle)
    
    indicesOfDNA = np.where(DNALength>0.5*np.max(DNALength))
    x2,x1,y2,y1 = np.max(indicesOfDNA[1]),np.min(indicesOfDNA[1]),np.max(indicesOfDNA[0]),np.min(indicesOfDNA[0])
    DNALength = DNALength[int(y1-(y2-y1)*0.5):int(y2+(y2-y1)*0.5),int(x1+(x2-x1)*0.2):int(x2-(x2-x1)*0.2)]
    plt.figure()
    plt.imshow(DNALength,aspect='auto')
    
    
    intensityProfile = np.mean(DNALength,1)
    
    plt.figure()
    plt.plot(intensityProfile)

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
simPath="E:/NSM/Data/Preprocessed Simulated DNA-lambda/intensity/"
trajPath = "E:/NSM/Data/Preprocessed Simulated DNA-lambda Ground Truth/"
DNAFiles = os.listdir(simPath)[:5]
trajFiles = os.listdir(trajPath)
for file in DNAFiles:
    fig,axs=plt.subplots(1,2,figsize=(16,16))
    DNA = np.load(simPath+file)
    axs[0].imshow(DNA,aspect='auto')
    DNA = np.load(trajPath+file)
    axs[1].imshow(DNA,aspect='auto')
#%% Plot exosomes

import deeptrack as dt
from deeptrack.features import Feature
import numpy as np
import skimage.measure
from scipy.signal import convolve
from scipy.signal import convolve2d
K=keras.backend
class get_diffusion(Feature):
    
    def __init__(self, vel=0, D=0, I=0, s=0, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

    def get(self, image, vel, D, I,s, **kwargs):
        LOW = 0.1
        HIGH = 1.15
        D = (LOW + HIGH*np.random.rand())
        image.append({"D":D})
        
        return image

    
class gen_noise(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0,biglam=0,bgnoiseCval=0,bgnoise=0,bigx0=0, **kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA,biglam=biglam,bgnoiseCval=bgnoiseCval,bgnoise=bgnoise,bigx0=bigx0, **kwargs
        )

    def get(self, image, noise_lev, dX, dA,biglam,bgnoiseCval,bgnoise,bigx0, **kwargs):
        
        
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        G= lambda a,b,x0,s,x: a*np.exp(-(x-x0)**2/s**2)+b
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        bgnoise*=np.random.randn(length)

        tempcorr=3*np.random.rand()
        dAmp=dA
        shiftval=dX*np.random.randn()
        dx=0
        dx2=0
        dAmp0=0
        bg0=f2(1,bigx0,biglam,0,x)
        ll=(np.pi-.05)
        for j in range(times):
            dx=(2*np.random.randn()+np.sin(ll*j))*dX
            
            bgnoiseC=f2(1,0,bgnoiseCval,dx,x)
            bgnoiseC/=np.sum(bgnoiseC)
            bg=f2(1,bigx0+dx,biglam,0,x)*(1+convolve(bgnoise,bgnoiseC,mode="same"))
            dAmp0=dA*np.random.randn()
            bg*=(1+dAmp0)
            image[j,:,0]=bg*(1+noise_lev*np.random.randn(length))+.4*noise_lev*np.random.randn(length)
        return image
    
class post_process(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0, **kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA, **kwargs
        )

    def get(self, image, **kwargs):
        from scipy.signal import convolve2d
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        
        # New preprocess 7 Mars
        image[:,:,0]=(image[...,0]-np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0))/np.mean(image[...,0],axis=0)
        
        #Perform same preprocessing as done on experimental images
        ono=np.ones((200,1))
        ono[0:80]=1
        ono[120:]=1
        ono=ono/np.sum(ono)
        image[:,:,0]-=convolve2d(image[:,:,0],ono,mode="same")
        image[:,:,0]-=convolve2d(image[:,:,0],np.transpose(ono),mode="same")
        
        image[:,:,0]-=np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0)
        a=np.std(image[...,0],axis=0)
        image[:,:,0]/=a
        try:
            image.properties["I"]/=a
        except:
            pass      
        
        return image    
    
class input_array(Feature):
    __distributed__ = False
    def get(self,image, **kwargs):
        image=np.zeros((times,length,2))
        return image


class get_trajectory(Feature):

    
    def __init__(self, vel=0, D=0.1, I=[0.1,50],s=0.05, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

    def get(self, image, vel, D, I,s, **kwargs):
        #I = 0.01+(np.random.rand()*25)
        s = 0.04 + 0.01*np.random.rand()
        D = image.get_property("D")/10
        I=I[0]+I[1]*np.random.rand()
        I = s*I*np.sqrt(2*np.pi)*256*0.03/10000
        v1 = 0
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(0,0.5,times)
        X, Y=np.meshgrid(t,x)
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        x0=-20+22*np.random.rand()
        x0+=np.cumsum(vel+D*np.random.randn(times))
        v1=np.transpose(I*f2(1,x0,s,0,Y))
        temp_img = np.copy(image[...,0]*(1-v1))
            
        image[...,0]*=(1-v1)
        image[...,1]+=np.transpose(f2(1,x0,0.05,0,Y))
        
        return image

vel=lambda: (15000+100000*np.random.rand())*10**-6
#vel=lambda:0
#vel = lambda:1
D = lambda: 0.10*np.sqrt((0.05 + 2*np.random.rand()))
I=[0.1,50]
s = 0.01 #lambda: 0.01 + 0.01*np.random.rand()
nbr_particles=lambda:10
times=512
length=512


image=dt.FlipLR(dt.FlipUD(input_array() + get_diffusion() 
                        + gen_noise(dX=.00001+.00003*np.random.rand(),
                                                dA=0,
                                                noise_lev=.0001,
                                                biglam=0.6+.4*np.random.rand(),
                                                bgnoiseCval=0.03+.02*np.random.rand(),
                                                bgnoise=.08+.04*np.random.rand(),
                                                bigx0=lambda: .1*np.random.randn())
                                    +get_trajectory(vel=vel,D=D,I=I)**nbr_particles
                                    + post_process()))

from scipy.ndimage.interpolation import rotate

plotDNAImage = image.resolve()
plt.figure()
plt.imshow(plotDNAImage[:,:,0],aspect='auto')
#plt.colorbar()
plt.figure()
plt.imshow(plotDNAImage[:,:,1],aspect='auto')
#plt.colorbar()



# ### Function to plot simulated data during training
#%%

Int = [0.01,5]
Ds = lambda: 0.10*np.sqrt((0.05 + 2*np.random.rand()))
st = lambda: 0.04 + 0.01*np.random.rand()  
nump= lambda: 1

cmap = 'plasma'

Traj1 = get_trajectory(I=[1,1.1],D=0.1,s=st)
Traj2 = get_trajectory(I=[2,2.1],D=0.05,s=st)
Traj3 = get_trajectory(I=[5,5.1],D=0.08,s=st)


image=dt.FlipLR(dt.FlipUD(input_array() + init_particle_counter() 
                        + gen_noise(dX=.00001+.00003*np.random.rand(),
                                                dA=0,
                                                noise_lev=.0001,
                                                biglam=0.6+.4*np.random.rand(),
                                                bgnoiseCval=0.03+.02*np.random.rand(),
                                                bgnoise=.08+.04*np.random.rand(),
                                                bigx0=lambda: .1*np.random.randn())
                                    + Traj3
                                    + post_process()))

plotTrajImage = image.resolve()[...,1]


plt.imsave("abstractTrajectoriesPlasma.svg",plotTrajImage,cmap=cmap,dpi=300,format='svg')

plt.figure()
plotTrajImage[plotTrajImage<10**-60] = np.nan
plt.imshow(plotTrajImage,aspect='auto',cmap=cmap)

plt.xticks([])
plt.yticks([])



plt.savefig("abstractTrajectoriesPlasmaTransparent.svg",transparent=True)

#%%
plotTrajImage = image.resolve()[...,1]

plt.figure()
plt.imshow(plotTrajImage,aspect='auto',cmap=cmap)

plt.figure()
plotTrajImage[plotTrajImage<10**-20] = np.nan
plt.imshow(plotTrajImage,aspect='auto',cmap=cmap)