#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:12:16 2020

@author: pierre-louis
"""



import rosbag
from cv_bridge import CvBridge
import cv2
import numpy as np
from numpy.linalg import inv
import time



################### FUNCTIONS ###################################


def init(testInputBagFile):
    """ Initialisation function
    
    param : testInputBagFile = name of the inuput bag file (example :'2020-05-20-15-39-47.bag')
    
    return : minV = minimum value for thresholding
             maxV = maximum value for thresholding
             readTopic = name of the topic we would work on
             bag = information about the bag file
    
    """
    
    minV=90 #90
    maxV=255
    readTopic = '/image_data'
    bag = rosbag.Bag(testInputBagFile)
    return(minV,maxV,readTopic,bag) 
    

    
def kal_init():
    """ Kalman algorithm initialisation 
        
    return : sig_q,sig_px,sig_py,Te,xp,yp,x0,G_0,u,G_a,G_b,A,C,y_kal : constant initialisation for the kalman equations
    
    """
    sig_q=0.01 #0.1
    sig_px=0.01 #some constants 
    sig_py=0.01 
    Te=0.04 # period time 
    xp=0 # initial dx/dt
    yp=0 # initial dy/dt
    x0=np.array([[125],[xp],[300],[yp]]) #initial state vector (x,dx/dt, y,dy/dt)
    G_0=0.01*np.eye(4) # covariance matrix
    u=0 #control command
    G_a=(sig_q)**2 * np.array([[Te**3/3,Te**2/2,0,0],[Te**2/2,Te,0,0],[0,0,Te**3/3, Te**2/2],[0,0,Te**2/2,Te]]) # gaussian white noise in time for the estimation
    
    G_b=np.array([ [sig_px**2,0],[0,sig_py**2]]) # gaussian white noise in time for the observation 
    A=np.array([[1,Te,0,0],[0,1,0,0],[0,0,1, Te],[0,0,0,1]]) # equation matrix 1
    C=np.array([ [1,0,0,0],[0,0,1,0]]) # equation matrix 2
    y_kal=0 # first observation
    
    return(sig_q,sig_px,sig_py,Te,xp,yp,x0,G_0,u,G_a,G_b,A,C,y_kal)
    
    

def kalman_predict(xup,Gup,u,G_a,A):
    """ kalman prediction algorithm
    
    param : xup = corrected estimation 
            Gup = corrected covariance
            u = control 
            G_a = gaussian noise
            A = matrix of the kalman equation
            
    return : x1 = predicted estimation
             G_1 = predicted covariance
    
    """
    
    G_1 = A.dot(Gup).dot(A.T) + G_a
    x1 = A.dot(xup) + u    
    return(x1,G_1)    

def kalman_correc(x0,G_0,y,G_b,C):
    """ kalman correcion algorithm
    
    param : x0 = estimation
            G_0 = covariance matrix
            y = observation
            G_b = gaussian noise
            C = matrix of the kalman equation
            
    return : xup = corrected estimation
             Gup = corrected covariance
    """
    
    S = C.dot(G_0).dot(C.T) + G_b        
    K = G_0.dot(C.T).dot(inv(S)) 
    ytilde = y - C.dot(x0)   
    Gup = ( np.eye(len(x0)) - K.dot(C) ).dot(G_0) 
    xup = x0 + K.dot(ytilde)
    return(xup,Gup) 
    
    
    
def kalman(x0,G_0,u,y,G_a,G_b,A,C):
    """ Kalman implementation algorithm :kalman correction + kalman prediction
    
    param : all of kalman_correc and kalman_predict
    
    return : x1 = predicted estimation
             G_1 = predicted covariance
    
    """
    
    xup,Gup = kalman_correc(x0,G_0,y,G_b,C)
    x1,G_1=kalman_predict(xup,Gup,u,G_a,A)
    return(x1,G_1)  



def black_filter(image):
    """ filter that hides groups of light points (e.g. the central light beam of the sonar or noise added at the end). 
        Setting to 0 the pixels in a row or column whose average pixel size is above a certain threshold.
    
    param : image = image before filtering
    
    return : image = image after filtering
    
    """
    threshold1=31 #threshold for the columns 31
    for i in range(image.shape[1]): # columns loop
        if int(np.mean(image[:,i])) >=threshold1: #if the mean of the column > threshold
            for j in range(image.shape[0]):
                image[j,i]=0 # set to 0 for all pixels in the concerned column
                
    threshold2=28 #threshold for the line 28
    for i in range(image.shape[0]-50): #lines loops
        if int(np.mean(image[i,:])) >=threshold2: #if the mean of the line > threshold
            for j in range(image.shape[1]):
                image[i,j]=0 # set to 0 for all pixels in the concerned line
    return(image)
    
def troncate(x,y,image):
    """
    function that truncates each detection by setting the image dimensions as maximum values
    
    param : x,y = coordonates of the given detection
            image = the current image
            
    return : x,y = the truncate coordonates of the given detection
    """
    
    # truncate the x coordonates
    if x<0:
        x=0
    if x>image.shape[1]: 
        x=image.shape[1] 
    # truncate the y coordonates
    if y<0:
        y=0
    if y>image.shape[0]:
        y=image.shape[0]
    return(x,y)
    
def delete_double(list_contours,contours):
    """
    Function for deleting all contours that appear twice. 
    It is assumed that if the same point is detected twice in several image sequences, 
    then it is very likely that the point corresponds to sonar noise.
    
    param : list_contours = list of all contours detected from start. 
            contours = contours of the current image.
            
    return = contours_finals = duplicate-free contours.
    """
    contours_final=[]
    for i in range(len(contours)): # for all elements in the current contours
        boolean=False
        for j in range(len(list_contours)): # for all elements in the list of all the previous contours
            if len(list_contours[j])!=0:
                if np.all(contours[i]==list_contours[j][0]) == True: # if all elements of the first list is egal to all elements of the second list
                    boolean=True
                
        if boolean==False: # if contours[i] has no duplicate
            contours_final.append(contours[i])
        
    return(contours_final)
    
    
def compare(contours,contours_before):
    """
    Function that selects the closest contour between all detected contours 
    of the current frame and those of the previous frame
    This makes it possible to detect only one contour at the end based on the detection of the previous contours
    because between 2 frames, the cable has moved very little.. 
    
    param : contours = list of the current contours
            contours_before = list of the previous contours
            
    return = contours = list containing the closest contour.
    """

    if len(contours)>1 and len(contours_before)>0:
        minimum=np.inf #minimum value for the research algorithm of minimum
        for i in range(len(contours)): # for all the current contours
            for j in range(len(contours_before)): # for all the previous contours
                a=np.sqrt( (np.abs(contours[i][0][0][0]-contours_before[j][0][0][0])) +( np.abs(contours[i][0][0][1]-contours_before[j][0][0][1]) )) # compute the euclidian distance between the contours
                if a<minimum:
                    minimum=a
                    index=i
        
        contours_final=[contours[index]] # list of the closest contour
        contours=contours_final

    return(contours)

def transfo(image,x0,G_0,u,G_a,G_b,A,C,y_kal,boolean,k,contours_init,list_contours):
    """ Transformation function 
        Function performing thresholding, cropping, detects the contours of the light point and performs a kalman filter.
        Function that displays the image as well as drawings such as the light point, an approximation rectangle and the coordinates of the point on the image.
    
    
    param : image = openCV matrix of the current image
            x0 = previous (of the previous image) state vector 
            G_0 = previous (of the previous image) covariance matrix 
            u = control command
            G_a, G_b = gaussian noise white in time
            A,C = matrix equation
            y_kal = previous observation
    
    return : x0 = current state vector
             G_0 = current covariance matrix
             y_kal = current observation
             
    """
    
    
    image_init=image.copy()                
    image=black_filter(image) # performing a black filtering    
    ret,seuil = cv2.threshold(image,minV,maxV,cv2.THRESH_BINARY) #thresholding 
    image=seuil
    p=30 # value for cropping
    p2=10 # value for cropping
    image_recadrage=image[p:image_init.shape[0]-50,p2:image_init.shape[1]-p2] # cropping in order to do not detect the white extremity of the image
    edged = cv2.Canny(image_recadrage, 30, 200) 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#find the contour of the light dot

    
    
    if k!=0: # if it is not the first frame
        # This the part of the code that try to reduce as much noise as possible
        contours=contours_final(contours_init, contours) 
        list_contours.append(contours) # list containing all the contours since the beginning
        contours=delete_double(list_contours[:-1],contours) 
        if len(list_contours)>=2: # if there are more than 2 frames to be compared
            contours=compare(contours,list_contours[-2])
    
    if len(contours) !=0: # if the algorithme has detect the light dot 
        x_list,y_list=[contours[0][i][0][0] for i in range(contours[0].shape[0])],[contours[0][i][0][1] for i in range(contours[0].shape[0])]
        x_obs,y_obs=int(np.floor(np.mean(x_list))),int(np.floor(np.mean(y_list))) # performing the mean of all contours detected
        y=np.array([[x_obs],[y_obs]]) #observation vector 
        x0,G_0=kalman(x0,G_0,u,y,G_a,G_b,A,C) #performing the kalman filtering
        y_kal=y # the current observation becomes the last observation for the next frame
        x_pred,y_pred=int(x0[0][0])+p2,int(x0[2][0])+p #adjustment due to cropping
        x_pred,y_pred=troncate(x_pred,y_pred,image_init) #truncation
        cv2.drawContours(image_init, np.array([[[x_pred,y_pred]]]), -1, (2000, 2000, 2000), 3) #draw the contours of the light dot
        cv2.rectangle(image_init, (x_pred-15, y_pred-15), (x_pred+15, y_pred+15), (255, 0, 0), 2) #draw a rectangle around the dot 
        cv2.putText(image_init,"x:{} and y:{}".format(x_pred,y_pred),(10,500),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) #coordinates 
        cv2.imshow('Target tracking', image_init) #showing the image with drawing
        boolean=True
        


    elif boolean==True and len(contours) ==0: # if the algorithm don't detect any light dot
        y=y_kal # the current observation = the last observation
        x0,G_0=kalman(x0,G_0,u,y,G_a,G_b,A,C) #kalman filtering 
        y_kal=np.array([[x0[0][0]],[x0[2][0]]])
        x_pred,y_pred=int(x0[0][0])+p2,int(x0[2][0])+p #adjustment due to cropping
        x_pred,y_pred=troncate(x_pred,y_pred,image_init) #truncation
        cv2.drawContours(image_init, np.array([[[x_pred,y_pred]]]), -1, (2000, 2000, 2000), 3) #draw the contours of the light dot
        cv2.rectangle(image_init, (x_pred-15, y_pred-15), (x_pred+15, y_pred+15), (255, 0, 0), 2) #draw a rectangle around the dot 
        cv2.putText(image_init,"x:{} and y:{}".format(x_pred,y_pred),(10,500),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) #coordinates 
        cv2.imshow('Target tracking', image_init) #showing the image with drawing
    
    else : # if the algorithm does not detect the cable in the frist frames
        cv2.imshow('Target tracking', image_init) #juste show the image without processing
        return(np.array([[125],[0],[300],[0]]), 0.01*np.eye(4), 0, 140,418,boolean,list_contours )
    
    
    return(x0,G_0,y_kal,x_pred,y_pred,boolean,list_contours)
    
    
    
def contours_init_compute(image):   
    """
    Function that processes the first image of the sequence by detecting contours.
    param : image = array representign the image of the first frame
    return : contours = contours detected of the first frame
    """            
    image=black_filter(image) # performing a black filtering
    ret,seuil = cv2.threshold(image,minV,maxV,cv2.THRESH_BINARY) #thresholding 
    image=seuil
    p=30
    p2=10
    image_recadrage=image[p:image.shape[0]-50,p2:image.shape[1]-p2] # cropping in order to do not detect the white extremity of the image
    edged = cv2.Canny(image_recadrage, 30, 200) 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#find the contour of the light dot
    return(contours)

def contours_final(contours_init, contours):
    """  
    Function that compares the points detected at each images with the points detected on the first image. 
    Enable to delete the points that are detected at each images and do not represent the cable. 
    """
    final=[]
    for q in range(len(contours)): # for all the current contours
        boolean=True
        for i in range(len(contours_init)): # for all the previous contours
    
            if compute_contours(contours[q],contours_init[i])==True: # the 2 elements are close from each other
                boolean=False
                break
    
        if boolean !=False:
            final.append(contours[q])
    return(final)


def compute_contours(list1,list2): 
    """
    Function that returns "true" if there is an element whose absolute value of the difference
    with another element in the other list is less than a threshold. 
    This function allows to detect if a point belongs to cluster of a previous region.
    
    param : list1 = list of contours
            list2 = another list of contours
            
    return : Boolean = False or True
    """
    thresh=7 # treshold to compare the coordinates of the 2 points
    a,b=list1[0][0][0],list1[0][0][1] # the coordinates of the contours that is to to be compared
    
    for i in range(len(list2)): # for all elements in the current frame
        x,y=list2[i][0][0],list2[i][0][1] 
        if abs(a-x)<thresh and abs(b-y)<thresh:
            return(True)
            
    return(False)
        

def main(readTopic):
    """ Read the bag file and convert images into OpenCV matrix for transformation
    
    param : readTopic = name of the bag file topic 
    
    return : None
    
    """
    sig_q,sig_px,sig_py,Te,xp,yp,x0,G_0,u,G_a,G_b,A,C,y_kal=kal_init() # initialisation of the constants
    genBag = bag.read_messages(readTopic) #read the given topic
    boolean=False
    list_contours=[] # list containing all the contours since the first frame
    frequency=[] # list of the frequency of processing each image
    for k,b in enumerate(genBag): # for each frame
        cb = CvBridge()
        image = cb.imgmsg_to_cv2( b.message, b.message.encoding ) # convert bag file image into an OpenCv matrix
        if k==0: # if it is the first image
            contours_init=contours_init_compute(image) 
        a1=time.time()
        x0,G_0,y_kal,x_pred,y_pred,boolean,list_contours=transfo(image,x0,G_0,u,G_a,G_b,A,C,y_kal,boolean,k,contours_init,list_contours) # transformation of the current image
        b1=time.time()
        frequency.append(1/(b1-a1))
        key = cv2.waitKey(40) # wait time (ms)

        
        if 113 == key: # touch "q" to interrupt
            print("q pressed. Abort.")
            break
    
    print(' The processing frequency of this file is : ',np.mean(frequency))
    cv2.destroyAllWindows() 
    bag.close()
    



######### MAIN PROGRAM #######################

if __name__ == '__main__':
    """ Main programm
    
    We have to choose between the proposed bag files (comment one of the two below)
    
    """
    
    ########  comment one of the two below #################
    testInputBagFile = '2020-05-20-15-37-56.bag' ## unoised file
    #testInputBagFile = '2020-05-20-15-39-47.bag' ## unoised file
    #testInputBagFile='2020-05-20-15-44-21.bag'  ## unoised file
    #testInputBagFile = '2020-07-06-17-26-24.bag' ## noisy file
       
    
    
    minV,maxV,readTopic,bag=init(testInputBagFile) # initialisation
    main(readTopic) # main fucntion

    
    


