from distutils import dir_util
from multiprocessing.reduction import DupFd
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
import pandas as pd
import os

# sources:
# OCR:
# https://pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
# Finding frames:
# https://stackoverflow.com/questions/57791203/python-take-screenshot-from-video


def getFrames(inputFile, steps, initialSecs):
    
    # initialization
    currentFrame = 0
    framesCaptured = 1
    
    # adding suffix to file to create folder for screenshots
    outputFolder = os.path.splitext(inputFile)[0] + "-data"

    # creating/finding folder
    try:
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
            print("Creating directory: ", outputFolder)
    except OSError:
        print("ERROR: Could not create directory!")
        
    # reading video file
    cam = cv2.VideoCapture(inputFile)
    fps = cam.get(cv2.CAP_PROP_FPS)

    # setting step based on dictionary inputted
    for key in steps.keys():
        if key in inputFile:
            step = steps[key]
            print('key:', key)
            print('step:', step)
            break
            
    os.chdir(outputFolder)  

    # iterating through frames of video and taking screenshots at every step
    while(True):
        ret,frame = cam.read()
        if ret:
            # safety feature: to prevent taking too many screenshots
            if framesCaptured > 1000:
                print("ERROR: 1000 frames captured. Stopping for safety!")
                return None
            # takes a screenshot every second for initialSecs (to tell when to start recording data)
            if framesCaptured <= initialSecs and currentFrame > fps:
                currentFrame = 0
                name = 'frame'+str(framesCaptured)+'.jpg'
                print('Creating: ' + name)
                cv2.imwrite(os.path.join(outputFolder,name),frame)
                if not(cv2.imwrite(name,frame)):
                    print('ERROR: Could not write image file')
                framesCaptured += 1
            # after initial 5 seconds, begins taking screenshots every step seconds
            elif currentFrame > (step*fps):
                currentFrame = 0
                name = 'frame'+str(framesCaptured)+'.jpg'
                print('Creating: ' + name)
                cv2.imwrite(os.path.join(outputFolder,name),frame)
                if not(cv2.imwrite(name,frame)):
                    print('ERROR: Could not write image file')
                framesCaptured += 1    
            currentFrame += 1
        if ret == False:
            break
    cam.release()

    # outputs step for allNums to know how to index dataframe
    # outputs framesCaptured for allNums to know how many entries into dataframe
    return (step, framesCaptured - 1)


def locateDisplay(image):

    # filters to make contours/edges more visible
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    edged = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)

    cv2.imshow("edged",edged)
    cv2.waitKey(0)

    # find contours (locates continous points with same color/intensity)
    cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # sort contours by largest area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # contours of display
    displayCnt = None

    # finding rectangle by approximating contour shape and finding one with 4 vertices
    for c in cnts:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.05*peri,True)
        if len(approx) == 4:
            displayCnt = approx
            break

    # warped is the filtered image to use to find digits
    # output is to use to check what function has identified as the display
    warped = four_point_transform(gray, displayCnt.reshape(4,2))
    output = four_point_transform(image, displayCnt.reshape(4,2))
    edged = output = four_point_transform(edged, displayCnt.reshape(4,2))

    return warped, output, edged

def locateDigitstwo(output,warped,edged):

    kernel = np.ones((5,5),np.uint8)
    edged = cv2.erode(edged,kernel,iterations=1)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(edged, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 100:   #keep
            result[labels == i + 1] = 255

    kernel = np.ones((9,9),np.uint8)
    result = cv2.dilate(result,kernel,iterations=1)

    cv2.imshow("Binary", edged)
    cv2.imshow("Result", result)
    cv2.waitKey(0)

    cnts = cv2.findContours(result,cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # stores the contours of the digits
    digitCnts = []

    # find the boxes for each digit
    for c in cnts:
        # x and y are the top left coordinates of the rectangle
        # w and h are the width and height of rectangle
        (x,y,w,h) = cv2.boundingRect(c)


        # the ranges for width and height needed to be satisfied for bounding box
        # to be accepted as a contour for a digit
        # these will have to be adjusted depending on camera angle and distance
        # look at issue 1 1 on README.md
        if w >= 18 and (h >= 50 and h <= 200):
            digitCnts.append(c)
            cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),3)

    # sort the contours from left to right (the same way we read numbers)
    if len(digitCnts) > 0:
        digitCnts = contours.sort_contours(digitCnts, method='left-to-right')[0]
    else:
        print("ERROR: No contours found!")

    return output, result, digitCnts



def locateDigits(output, warped, edged):

    # filter to remove shadow effect
    # source: https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv

    rgb_planes = cv2.split(warped)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    # results (result is not normalized while warped is)
    result = cv2.merge(result_planes)
    warped = cv2.merge(result_norm_planes)

    # eroding and dilating image to remove noise
    # since using cv2.THRESH_BINARY_INV (inversed), cv2.dilate has 
    # effect of erosion and cv2.erode has effect of dilation
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.dilate(warped,kernel,iterations=1)

    kernel = np.ones((7,7),np.uint8)
    dilation = cv2.erode(erosion,kernel,iterations=1)

    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.erode(dilation,kernel,iterations=1)

    thresh = cv2.threshold(dilation,0,225,
                     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    kernel = np.ones((7,7),np.uint8)
    # thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)

    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)

    # CHAIN_APPROX_SIMPLE only stores the necessary points for the contours
    # i.e. if there is a rectangle, it will only store vertices instead of 
    # every dot on the line
    cnts = cv2.findContours(thresh,cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # stores the contours of the digits
    digitCnts = []

    # find the boxes for each digit
    for c in cnts:
        # x and y are the top left coordinates of the rectangle
        # w and h are the width and height of rectangle
        (x,y,w,h) = cv2.boundingRect(c)


        # the ranges for width and height needed to be satisfied for bounding box
        # to be accepted as a contour for a digit
        # these will have to be adjusted depending on camera angle and distance
        # look at issue 1 1 on README.md
        if w >= 18 and (h >= 50 and h <= 200):
            digitCnts.append(c)
            cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),3)

    # sort the contours from left to right (the same way we read numbers)
    if len(digitCnts) > 0:
        digitCnts = contours.sort_contours(digitCnts, method='left-to-right')[0]
    else:
        print("ERROR: No contours found!")

    return output, thresh, digitCnts


def identifyDigits(output, thresh, digitCnts):

    # define dictionary
    # references to the sections of LCD which need to be on for a number
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9  
    }

    # there is a specific dictionary for number one since the bounding box for 1
    # is different (much smaller)
    # therefore different sections used to identify 1
    ONE_LOOKUP = {
        (1,1): 1
    }

    # list of digits
    digits = []

    # iterate through each box (to identify each individual digit)
    for c in digitCnts:
        # dimensions of bounding box for digit
        # roi : region of interest
        (x,y,w,h) = cv2.boundingRect(c)
        roi = thresh[y:y+h,x:x+w]
        
        # values used to split bounding box into segments
        (roiH,roiW) = roi.shape
        (dH,dW) = (int(roiH*0.15),int(roiW*0.3))
        dHC = int(roiH * 0.1)

        # splits the roi into seven areas where the LCD can be ON or NOT
        # for all numbers except 1
        segments = [
            ((0, 0), (w, int(dH*0.8))), # top
            ((0, 0), (int(dW*1.2), h // 2)), # top-left
            ((w - dW, 0), (w, h // 2)), # top-right
            ((dW, (h // 2) - dHC) , (w-dW, (h // 2) + dHC)), # center
            ((0, h // 2), (dW, h)), # bottom-left
            ((w - dW, h // 2), (w, h)), # bottom-right
            ((0, h - dH), (w-dH, h)) # bottom
        ]
        on = [0] * len(segments)

        # splits roi into two areas where LCD can be ON or NOT for 1 specifically
        one_segments = [
            ((int(w*0.4),int(h*0.1)), (int(w*0.7),int(h*0.5))), # top of 1
            ((0,int(h*0.5)),(int(w*0.6),h)) # bottom of 1
        ]
        one_on = [0] * len(one_segments)

        # identifies if specific region of LCD is ON or NOT by counting 
        # the nonzero pixels. If they account for more than 45% of region,
        # LCD region is ON

        # checking if segments for 1 are ON or Not
        for (i, ((xA, yA), (xB, yB))) in enumerate(one_segments):
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            if total / float(area) > 0.45:
                one_on[i]= 1
        
        try:
            digit = ONE_LOOKUP[tuple(one_on)]
            digits.append(digit)

        # if not 1, check for other digits
        except KeyError:
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                if total / float(area) > 0.45:
                    on[i]= 1
            try:
                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(digit)
            except KeyError:
                continue

        # output number found by computer next to bounding box of number on real image
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(output, str(digit), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # return list of individual digits on LCD
    return digits


def getNum(frameNumber):

    # get image
    image = cv2.imread("frame"+str(frameNumber)+".jpg")

    # process image
    image = imutils.resize(image,height=500)

    # locate roi for display
    warped, output, edged = locateDisplay(image)

    # locate roi for digits on display
    output, thresh, digitCnts = locateDigits(output, warped, edged)

    # list of individual digits in frame
    digits = identifyDigits(output, thresh, digitCnts)

    # forming whole number on display using list of digits
    num = 0
    for x in range(len(digits)):
        num = num * 10 + digits[x]
    
    # output number found
    return num


def allNums(step, initialSecs, framesCaptured):

    # initialization
    temps = []
    times = []

    # adds num and time to lists for initial screenshots at START
    previous = 9999999999
    startTime = 0
    for x in range(initialSecs):
        num = getNum(x+1)
        if num > previous:
            startTime = x-1
            diff = initialSecs - startTime
            for y in range(diff):
                times.append(y)
                temps.append(getNum(y+startTime+1))
            break
        previous = num
    diff = initialSecs - startTime

    # adds num and time to lists after initial screenshots
    for x in range(framesCaptured-initialSecs):
        temps.append(getNum(x+1+diff))
        times.append((x*step)+diff)
    df = pd.DataFrame({'time':times,
                       'temps':temps})

    # finds current directory to title table in Excel file
    path = os.getcwd()
    dir = os.path.basename(path)

    # removes last 5 chars because suffix of directory is '-data'
    # if you change the suffix in getFrames(), need to change this here too
    dir = dir[:-5]

    # output df and title (for excel sheet)
    return df, dir


def fileList(path):

    # list of all files in folder
    os.chdir(path)
    listOfFiles = os.listdir(path)

    # remove folders, excel files and .DS_Store files (cannot/don't want to read digits from them)
    remove = []
    for file in listOfFiles:
        if 'data' in file:
            remove.append(file)
            continue
        if '.xlsx' in file:
            remove.append(file)
            continue
        if '.DS_Store' in file:
            remove.append(file)

    for file in remove:
        listOfFiles.remove(file)

    # sort list in alphabetical order
    # e.g. 25C-40A-1, 25C-40A-2, 25C-40A-3 are all next to each other
    listOfFiles = sorted(listOfFiles)

    return listOfFiles


def folderToData(path, fileName, spaces, steps, initialSecs):

    # list of videos in alphabetical order
    listOfFiles = fileList(path)

    # creating excel file with fileName
    writer = pd.ExcelWriter(fileName,engine = 'xlsxwriter')

    # iterate through all files in folder
    for x, file in enumerate(listOfFiles):
        os.chdir(path)

        # get frames from file
        frameInfo = getFrames(file,steps,initialSecs)

        # get dataframe and directory name from file
        df,dir = allNums(frameInfo[0],initialSecs,frameInfo[1])
        print(df)

        # input dataframe into excel file
        df.to_excel(writer,startrow=2,startcol=x*spaces)

        # title dataframe with directory name in excel file
        worksheet = writer.sheets['Sheet1']
        worksheet.write_string(1, (x*spaces)+1, dir)

    writer.save()


# path is the location of the folder with the videos you want to be processed
path = '/Users/lukasgrunzke/Desktop/MCBData-Heating'
path = '/Users/lukasgrunzke/Desktop/NEWVID'


# fileName is the name of the excel file that will be outputted in folder reference in path
fileName = 'TestingData.xlsx'

# This is the dictionary where you can input the tags in file names which refer to step time
# For example: if file contains the string "40A", step will = 1. If it contains "20A", step = 6.
# You can add/remove the strings as you like but they must be in the format:
# steps = {
#   "exampleString1" : exampleStepTime1,
#   "exampleString2" : exampleStepTime2,
#   "exampleString3" : exampleStepTime3 <-- last entry does not have a comma! (,)
# }
steps = {
    "40A":1,
    "30A":1,
    "20A":6,
    "10A":10,
}

# spaces is the distance you want between each individual table in your excel spread sheet
spaces = 1
spaces = spaces + 3

# initialSecs is how many seconds at START of video take screenshots
# this is useful if you start video and then turn on power for example
# if your step is normally 10 but you need lots of frames in the beginning to identify when power is 
# turned on, initialSecs useful
initialSecs = 5

folderToData(path,fileName,spaces,steps,initialSecs)