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

# getting frames from video
# can choose step (how many seconds between frames)
def getFrames(inputFile):
    currentFrame = 0
    framesCaptured = 1
    
    outputFolder = os.path.splitext(inputFile)[0] + "-data"

    try:
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
            print("Creating directory: ", outputFolder)
    except OSError:
        print("ERROR: Could not create directory!")
        
    cam = cv2.VideoCapture(inputFile)
    fps = cam.get(cv2.CAP_PROP_FPS)

    if "50A" in inputFile:
        step = 1
    elif "40A" in inputFile:
        step = 2
    elif "30A" in inputFile:
        step = 4
    elif "20A" in inputFile:
        step = 6
    else:
        step = 5

    os.chdir(outputFolder)  

    while(True):
        ret,frame = cam.read()
        if ret:
            if currentFrame > (step*fps):
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

    # used for iterations for getNum
    return (step, framesCaptured - 1)




def getNum(frameNumber):

    # get image
    image = cv2.imread("frame"+str(frameNumber)+".jpg")

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

    ONE_LOOKUP = {
        (1,1): 1
    }

    # process image
    image = imutils.resize(image,height=500)

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
    
    # sort by largest area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # contours for display
    displayCnt = None

    # finding rectangle (approx shape and find one with 4  vertices)
    for c in cnts:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.05*peri,True)
        if len(approx) == 4:
            displayCnt = approx
            break

    # if you give it four points, it will output rectangle as if you are looking at it straigth on
    warped = four_point_transform(gray, displayCnt.reshape(4,2))
    output = four_point_transform(image, displayCnt.reshape(4,2))


    # remove shadow
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

    result = cv2.merge(result_planes)
    warped = cv2.merge(result_norm_planes)

    cv2.imshow("shadow",warped)
    cv2.waitKey(0)

    # define kernel for erosion
    # erosion actually reduces the size of the white but since we use 
    # BINARY_THRESH_INV (inversed) it does the same effect as cv2.dilate
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.dilate(warped,kernel,iterations=1)

    kernel = np.ones((7,7),np.uint8)
    dilation = cv2.erode(erosion,kernel,iterations=1)

    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.erode(dilation,kernel,iterations=1)

    thresh = cv2.threshold(dilation,0,225,
                     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)

    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)

    # CHAIN_APPROX_SIMPLE only stores the necessary points for the contours
    # i.e. if there is a rectangle, it will only store vertices instead of 
    # every dot on the line
    cnts = cv2.findContours(thresh,cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # stores the contours of the DIGITS
    digitCnts = []

    # find the boxes for each digit
    for c in cnts:
        # x and y are the top left coordinates of the rectangle
        # w and h are the width and height of rectangle
        (x,y,w,h) = cv2.boundingRect(c)

        # these need to be tuned for the specific image since
        # the width and height will be different depending on the 
        # camera angle and distance from LCD
        # POSSIBLE IMPROVEMENT: write algorithm to automatically find
        # the correct w and h
        if w >= 15 and (h >= 50 and h <= 200):
            digitCnts.append(c)
            # cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),3)
            # cv2.imshow("output boxes",output)
            # cv2.waitKey(0)

    # sort the contours from left to right (the same way we read numbers)
    if len(digitCnts) > 0:
        digitCnts = contours.sort_contours(digitCnts, method='left-to-right')[0]
    else:
        print("ERROR: No contours found!")

    # list of digits
    digits = []

    # iterate through each box (representing the space of each digit)
    for c in digitCnts:
        (x,y,w,h) = cv2.boundingRect(c)
        # roi: region of interest
        roi = thresh[y:y+h,x:x+w]
        
        (roiH,roiW) = roi.shape
        (dH,dW) = (int(roiH*0.15),int(roiW*0.3))
        dHC = int(roiH * 0.1)

        # splits the roi into 7 areas where the LCD can be on or not
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

        one_segments = [
            ((int(w*0.4),int(h*0.1)), (int(w*0.7),int(h*0.5))), # top of 1
            ((0,int(h*0.5)),(int(w*0.6),h)) # bottom of 1
        ]
        one_on = [0] * len(one_segments)

        # identifies if specific region of LCD is on or not by counting 
        # the nonzero pixels. If they account for more than 50% of region,
        # LCD region is on
        for (i, ((xA, yA), (xB, yB))) in enumerate(one_segments):
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            # print("ONE: percentage covered for area ",i,": ",total/float(area))
            if total / float(area) > 0.45:
                one_on[i]= 1
        
        try:
            digit = ONE_LOOKUP[tuple(one_on)]
            digits.append(digit)
        except KeyError:
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                # print("percentage covered for area ",i,": ",total/float(area))
                if total / float(area) > 0.45:
                    on[i]= 1
            try:
                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(digit)
            except KeyError:
                print("ERROR: Could not find digit in dictionary!")
                continue
        # print(on)

        # output number on screen
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(output, str(digit), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("output again",output)
        cv2.waitKey(0)

    num = 0
    for x in range(len(digits)):
        num = num * 10 + digits[x]
    # print(num)
    return num




def allNums(step, framesCaptured):
    temps = []
    times = []
    for x in range(framesCaptured):
        temps.append(getNum(x+1))
        times.append(x*step)
    df = pd.DataFrame({'time':times,
                       'temps':temps})
    print(df)
    return df
    



def multipleDfs(dfList, outputFolder, sheet, file_name, spaces):
    os.chdir(outputFolder)
    writer = pd.ExcelWriter(file_name,engine = 'xlsxwriter')
    row = 0
    for dataframe in dfList:
        dataframe.to_excel(writer,sheet_name=sheet,startrow=row,startcol=0)
        row = row + len(dataframe.index) + spaces + 1
    writer.save()




def folderToData(path):
    listOfFiles = os.listdir(path)
    listOfFiles = listOfFiles[1:]
    print(listOfFiles)

    dfs = []

    for file in listOfFiles:
        frameInfo = getFrames(file)
        df = allNums(frameInfo[0],frameInfo[1])
        print('Dataframe: ')
        print(df)
        print('\n')
        dfs.append(df)








folderToData('/Users/lukasgrunzke/Desktop/MCBData')

# testframe = getFrames('/Users/lukasgrunzke/Desktop/testvid5.mov')
# step = testframe[0]
# framesCaptured = testframe[1]
# testdf = allNums(step,framesCaptured)

# frames1 = getFrames('/Users/lukasgrunzke/Desktop/MCBData/25C-40A-1.mov')

# step = frames1[0]
# framesCaptured = frames1[1]
# df1 = allNums(step,framesCaptured)

# frames2 = getFrames('/Users/lukasgrunzke/Desktop/MCBData/25C-20A-2.mov')

# step = frames2[0]
# framesCaptured = frames2[1]
# df2 = allNums(step,framesCaptured)

# outputFolder = '/Users/lukasgrunzke/Desktop/MCBData'
# dfList = [df1,df2]
# multipleDfs(dfList,outputFolder,'Testing','test1.xlsx',5)


