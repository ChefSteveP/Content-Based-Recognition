import cv2
import numpy as np
import matplotlib.pyplot as plt

#Helper Functions########################################################
##https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d
#Source of calcHist Documentation, tun with 'bins' the number of bins for the r,g,b color
#channels. I took the advice of professor and made blue have the least bins, green the most, 
# and red a little in the middle. I found 6 to made good distictions between the reds. 
def gen_hist(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = (img // 128) * 128
    # Define the number of bins for each channel
    bins = [6,8,2]

    # Compute the histogram with a different number of bins for each channel
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    return hist

#eq from class notes
def L1Dist(h1,h2):
    dist = np.sum(np.abs(h1-h2))
    normalized_dist = dist/(2 * 60 * 89)
    return normalized_dist

#Creates a list of all (L1 Distance, img.jpg) pairs associated with a query image.
def QSimList (q, hist_list, mode):
    L1Dist_list = []
    j = 1
    for hist in hist_list:
        if j <= 9:
            num_img = ('0'+ str(j))
        else:
            num_img = str(j)

        if q is hist:
            L1Dist_list.append([0,'self'])
        else:
            if mode == 'color':
               L1Dist_list.append([L1Dist(q,hist),'i'+ num_img +'.jpg'])
            elif mode == 'texture':
                L1Dist_list.append([L1Dist(q,hist),'i'+ num_img +'.jpg'])
            elif mode == 'shape':
                L1Dist_list.append([overlap(q,hist),'i'+ num_img +'.jpg'])
            elif mode == 'symmetry':
                L1Dist_list.append([symmetry(q,hist),'i'+ num_img +'.jpg'])

        j+=1
    return L1Dist_list

def topThree(q, hist_list, mode):
    if mode == 'color':
        L1Dist_list = QSimList(img_hist_arr[q-1], hist_list, 'color')
    elif mode == 'texture':
        L1Dist_list = QSimList(laplacianImgs[q-1], hist_list, 'texture')
    elif mode == 'shape':
        L1Dist_list = QSimList(binaryImgs[q-1], hist_list, 'shape')
    elif mode == 'symmetry':
        L1Dist_list = QSimList(binaryImgs2[q-1], hist_list, 'symmetry')

    top_three = []
    top_three_idx = []
    totalScore = []
    temp_list = L1Dist_list.copy()

    #Remove Self Element
    idx = 0
    for i in range(len(temp_list)):
        if temp_list[i][1]=='self':
            idx = i
    temp_list.pop(idx)

    #First Target
    for i in range(0,3):
        #grab list of top three closest (L1dist, 'iXX.jpg') pairs
        top_three.append(min(temp_list))
        #Grab the index used for score calculation
        top_three_idx.append(L1Dist_list.index(min(temp_list)))
        totalScore.append(crowdData[q-1][L1Dist_list.index(min(temp_list))])
        temp_list.remove(min(temp_list))
    return top_three, totalScore

def topThreeSimplex(q,simplex):
    #P = a*C + b*T + c*S + d*Y
    #a+b=c+d=1 "Standard Simplex"

    a = simplex[0]
    b = simplex[1]
    c = simplex[2]
    d = simplex[3]

    L1Dist1 = QSimList(img_hist_arr[q-1], img_hist_arr, 'color')
    L1Dist2 = QSimList(laplacianImgs[q-1], laplacianImgs, 'texture')
    overlapDist = QSimList(binaryImgs[q-1], binaryImgs, 'shape')
    SymmetryDist = QSimList(binaryImgs2[q-1], binaryImgs2, 'symmetry')
    newWeighting = []


    for i in range(0,40):
        num_img = leadingZeroStr(i+1)
        newWeighting.append([a*L1Dist1[i][0] + b*L1Dist2[i][0]+c*overlapDist[i][0]+d*SymmetryDist[i][0],'i'+ num_img +'.jpg'])


    top_three = []
    top_three_idx = []
    totalScore = []
    temp_list = newWeighting.copy()

    #Remove Self Element
    idx = 0
    for i in range(len(temp_list)):
        if temp_list[i][1]=='self':
            idx = i
    temp_list.pop(idx)

    #First Target
    for i in range(0,3):
        #grab list of top three closest (L1dist, 'iXX.jpg') pairs
        top_three.append(min(temp_list))
        #Grab the index used for score calculation
        top_three_idx.append(newWeighting.index(min(temp_list)))
        totalScore.append(crowdData[q-1][newWeighting.index(min(temp_list))])
        temp_list.remove(min(temp_list))
    return top_three, totalScore

def leadingZeroStr(num):
    if num < 10:
        return ('0'+ str(num))
    else:
        return str(num)
    

#counts the number of pixels between 
def overlap(bin1, bin2):
    diff_count = 0
    for y in range(len(bin1)):
        for x in range(len(bin1[0])):
            if bin1[y,x] != bin2[y,x]:
                diff_count += 1
    diff_count_norm = diff_count/(len(bin1[0])*len(bin1))
    return diff_count_norm


##Returns the Count of how many binary pixels are symetrical along a vertical axis
def symmetry(bin1, bin2):
    #iterate col 1-44 
    tot_mirror_sum1 = 0
    tot_mirror_sum2 = 0
    
    for col in range(0, len(bin1[0])//2):
        mirror_sum1 = 0
        mirror_sum2 = 0
        query1 = bin1[:,col]
        query2 = bin2[:,col]

        target1 = bin1[:,len(bin1[0]) - col - 1]
        target2 = bin2[:,len(bin2[0]) - col - 1]

        for row in range(len(query1)):
            if not(int(query1[row]) ^ int(target1[row])):
                mirror_sum1 += 1
            if not(int(query2[row]) ^ int(target2[row])):
                mirror_sum2 += 1
        
        tot_mirror_sum1 += mirror_sum1
        tot_mirror_sum2 += mirror_sum2

    diff_tot_mirror_sum = np.abs(tot_mirror_sum1-tot_mirror_sum2)
    symmetry_norm = diff_tot_mirror_sum / (len(bin1) * len(bin1[0])/2)
    return symmetry_norm

##############################################################

#Project######################################################

img_hist_arr = []
##############################################################
#read Crowd.txt into 2D array
crowdData = []
file1 = open('Crowd.txt', 'r')
Lines1 = file1.readlines()
# Strips the newline character and adds list of row elements to crowdData
for line in Lines1:
    line = line.strip().split()
    ##map to a integer list of the values
    crowdData.append(list(map(int, line)))

#read MyPreferences.txt into a 2D array
preferences = []
file2 = open('MyPreferences.txt', 'r')
Lines2 = file2.readlines()
for line in Lines2:
    line = line.strip().split()
    ##map to a integer list of the values
    preferences.append(line[1:])


###########PART ONE################################################

#Step 1.2.1:  Generate each of the 40 image's three color histogram
for i in range(1,41):
    
    num_img = leadingZeroStr(i)
    hist = gen_hist('images/i'+ num_img +'.jpg')
    #add histogram to list of all images histograms
    img_hist_arr.append(hist)

'''
#Step 1.2.2 For Each For each of the 40 query images, find the three target images that have the 
smallest normalized L1 distance to the query image. Then, to get a score for each query image q, 
add up the counts given in Crowd.txt for each of the three target images that your algorithm selects
##Did this with topThree method##
'''
#Step 1.3: Visualization and performance evaluation
output = ''
totScore = 0
similarity = 0
#Formate data in an HTML table
for i in range(1,41):
    closest, score = topThree(i,img_hist_arr,'color')

    t_list = [(closest[0][1][1:3]),(closest[1][1][1:3]),(closest[2][1][1:3])]
    similarity += len(set(t_list) & set(preferences[i-1])) 
    output += '<tr>'

    num_img = leadingZeroStr(i)
    output += '<td><img src = "images/i'+ num_img + '.jpg"><br>query = '+ num_img +'</td>'
    
    output += '<td><img src = "images/i'+ t_list[0] + '.jpg"><br>t1 = '+ t_list[0] + ', count = '+ str(score[0]) +'</td>'

    output += '<td><img src = "images/i'+ t_list[1] + '.jpg"><br>t2 = '+ t_list[1] + ', count = '+ str(score[1]) +'</td>'

    output += '<td><img src = "images/i'+ t_list[2] + '.jpg"><br>t3 = '+ t_list[2] + ', count = '+ str(score[2]) +'</td>'

    output += '<td>row score = '+ str(sum(score)) +'</td>'
    totScore += sum(score)
    output += '</tr>'

output += '<tr><td>Total Score = '+ str(totScore) +'</td></tr>'

output += '<tr><td>Predictions intersecting with Preferences = '+ str(similarity) +'/120</td></tr>'

##Geeks for geeks syntax for creating and writing to an html file.
Func = open("output.html","w")

# Adding input data to the HTML file
Func.write("<html>\n<head>\n<title> \nOutput Data for art 1.3 \
           </title>\n</head> <body><h1>Visualization Results Given Color</h1><table><tbody>"+output+"</tbody></table>\
            \n</body></html>")
              
# Saving the data into the HTML file
Func.close()

############PART TWO######################################################
#Texture#
greyScaleImgs = []
laplacianImgs = []

for i in range(1,41):
    num_img = leadingZeroStr(i)
    with open('images/i'+ num_img +'.ppm', 'rb') as f:
        #will be p6
        magic_number = f.readline().decode().strip()
        #Will be 
        trademark = f.readline().decode().strip()
        #In all cases 89, 60
        width, height = [int(x) for x in f.readline().decode().split()]
        #In all cases is 255
        maxColor = int(f.readline().decode())

        #read the color byte data from the file.
        data = f.read()
        #convert color byte data into a 3D numpy array. Can be indexed by (x,y(r,b,g))
        pixels = np.frombuffer(data, dtype=np.uint8).reshape((height, width,3))

        #Convert 3D RGB pixel Matrix into a 2D Intensity matrix
        greyscale = (pixels[:,:,0] + pixels[:,:,1] + pixels[:,:,2]) // 3

        greyScaleImgs.append(greyscale)
        laplacian_img = cv2.Laplacian(greyscale, cv2.CV_64F, ksize=3, scale=1)

        # Convert the result to 8-bit image
        laplacian_img = cv2.convertScaleAbs(laplacian_img)

        #Absolute Game Changer. My results were really whacky before I normalized the images
        #everything used to have the onion in the top three and now it is barely present.
        #This also doubled my total score and preference matches. 
        laplacian_img_norm = cv2.normalize(laplacian_img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        laplacianImgs.append(laplacian_img_norm)



#2.3 Texture Visualization
output = ''
totScore = 0
similarity = 0
#Formate data in an HTML table
for i in range(1,41):
    closest, score = topThree(i,laplacianImgs,'texture')

    t_list = [(closest[0][1][1:3]),(closest[1][1][1:3]),(closest[2][1][1:3])]
    similarity += len(set(t_list) & set(preferences[i-1])) 
    output += '<tr>'

    num_img = leadingZeroStr(i)
    output += '<td><img src = "images/i'+ num_img + '.jpg"><br>query = '+ num_img +'</td>'
    
    output += '<td><img src = "images/i'+ t_list[0] + '.jpg"><br>t1 = '+ t_list[0] + ', count = '+ str(score[0]) +'</td>'

    output += '<td><img src = "images/i'+ t_list[1] + '.jpg"><br>t2 = '+ t_list[1] + ', count = '+ str(score[1]) +'</td>'

    output += '<td><img src = "images/i'+ t_list[2] + '.jpg"><br>t3 = '+ t_list[2] + ', count = '+ str(score[2]) +'</td>'

    output += '<td>row score = '+ str(sum(score)) +'</td>'
    totScore += sum(score)
    output += '</tr>'

output += '<tr><td>Total Score = '+ str(totScore) +'</td></tr>'

output += '<tr><td>Predictions intersecting with Preferences = '+ str(similarity) +'/120</td></tr>'

##Geeks for geeks syntax for creating and writing to an html file.
Func2 = open("output2.html","w")

# Adding input data to the HTML file
Func2.write("<html>\n<head>\n<title> \nOutput Data for art 2.3 \
           </title>\n</head> <body><h1>Visualization Results Given Texture</h1><table><tbody>"+output+"</tbody></table>\
            \n</body></html>")
              
# Saving the data into the HTML file
Func2.close()


#print(L1Dist(laplacianImgs[18], laplacianImgs[16]))
#print(QSimList(laplacian_img[8], laplacian_img))
# Display the result
#cv2.imshow('Laplacian Image', laplacianImgs[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###############PART THREE##############################################################
#Shape#

#Convert GreyScale images from part 2 into Binary Images

##Used m = 20 as my threshold values in which to count as foreground. Any value below 20 a began
##to create noise surrounding my object from small amounts of light reflecting off the object on the black background.
##This means a strong outline, even though there are some gaps within the sillouhette.




binaryImgs = []

for i in range(0,40):

    # apply thresholding to convert grayscale to binary image
    
    ret,binary = cv2.threshold(greyScaleImgs[i],20,255,0)
    #This did not change my performance as much as it did in the texture step but 
    binary_img_norm = cv2.normalize(binary, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    binaryImgs.append(binary_img_norm)


#3.3 Shape Overlap Visualization
output = ''
totScore = 0
similarity = 0
#Format data in an HTML table
for i in range(1,41):
    closest, score = topThree(i,binaryImgs,'shape')

    t_list = [(closest[0][1][1:3]),(closest[1][1][1:3]),(closest[2][1][1:3])]
    similarity += len(set(t_list) & set(preferences[i-1])) 
    output += '<tr>'

    num_img = leadingZeroStr(i)
    output += '<td><img src = "images/i'+ num_img + '.jpg"><br>query = '+ num_img +'</td>'
    
    output += '<td><img src = "images/i'+ t_list[0] + '.jpg"><br>t1 = '+ t_list[0] + ', count = '+ str(score[0]) +'</td>'

    output += '<td><img src = "images/i'+ t_list[1] + '.jpg"><br>t2 = '+ t_list[1] + ', count = '+ str(score[1]) +'</td>'

    output += '<td><img src = "images/i'+ t_list[2] + '.jpg"><br>t3 = '+ t_list[2] + ', count = '+ str(score[2]) +'</td>'

    output += '<td>row score = '+ str(sum(score)) +'</td>'
    totScore += sum(score)
    output += '</tr>'

output += '<tr><td>Total Score = '+ str(totScore) +'</td></tr>'

output += '<tr><td>Predictions intersecting with Preferences = '+ str(similarity) +'/120</td></tr>'

##Geeks for geeks syntax for creating and writing to an html file.
Func3 = open("output3.html","w")

# Adding input data to the HTML file
Func3.write("<html>\n<head>\n<title> \nOutput Data for Part 3.3 \
           </title>\n</head> <body><h1>Visualization Results Given Shape: Foreground v. Background</h1><table><tbody>"+output+"</tbody></table>\
            \n</body></html>")
              
# Saving the data into the HTML file
Func3.close()


#cv2.imshow("Binary Image", binaryImgs[22])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###########PART FOUR#########################################################################
#Shape:Symmetry#

binaryImgs2 = []

for i in range(0,40):

    # apply thresholding to convert grayscale to binary image
    ################Justifacation#########################
    ##I chose to have a lower threshold of m = 10 to fill in the gaps within the object. 
    ## This decision allows for higher internal symmetry which is far more signifigant than the static
    ## that results. The static also in symetrical images is less likely to mirror. It only minorly buffs the results
    ## for the large gain that it provides me.
    ret,binary2 = cv2.threshold(greyScaleImgs[i],10,255,0)
    #This did not change my performance as much as it did in the texture step but 
    binary_img_norm2 = cv2.normalize(binary2, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    binaryImgs2.append(binary_img_norm2)



#3.3 Shape Overlap Visualization
output = ''
totScore = 0
similarity = 0
#Format data in an HTML table
for i in range(1,41):
    closest, score = topThree(i,binaryImgs2,'symmetry')

    t_list = [(closest[0][1][1:3]),(closest[1][1][1:3]),(closest[2][1][1:3])]
    similarity += len(set(t_list) & set(preferences[i-1])) 
    output += '<tr>'

    num_img = leadingZeroStr(i)
    output += '<td><img src = "images/i'+ num_img + '.jpg"><br>query = '+ num_img +'</td>'
    
    output += '<td><img src = "images/i'+ t_list[0] + '.jpg"><br>t1 = '+ t_list[0] + ', count = '+ str(score[0]) +'</td>'

    output += '<td><img src = "images/i'+ t_list[1] + '.jpg"><br>t2 = '+ t_list[1] + ', count = '+ str(score[1]) +'</td>'

    output += '<td><img src = "images/i'+ t_list[2] + '.jpg"><br>t3 = '+ t_list[2] + ', count = '+ str(score[2]) +'</td>'

    output += '<td>row score = '+ str(sum(score)) +'</td>'
    totScore += sum(score)
    output += '</tr>'

output += '<tr><td>Total Score = '+ str(totScore) +'</td></tr>'

output += '<tr><td>Predictions intersecting with Preferences = '+ str(similarity) +'/120</td></tr>'

##Geeks for geeks syntax for creating and writing to an html file.
Func4 = open("output4.html","w")

# Adding input data to the HTML file
Func4.write("<html>\n<head>\n<title> \nOutput Data for Part 4.3 \
           </title>\n</head> <body><h1>Visualization Results Given Shape: Symmetry</h1><table><tbody>"+output+"</tbody></table>\
            \n</body></html>")
              
# Saving the data into the HTML file
Func4.close()

#print(topThree(25,binaryImgs2,'symmetry'))
#cv2.imshow("Binary Image", binaryImgs2[33])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###############PART FIVE#######################################################
#Gestalt Perception#

simplex = [0.5,0.25,0.15,0.1]

for q in range(1,41):
    closest, score = topThreeSimplex(q,simplex)

    t_list = [(closest[0][1][1:3]),(closest[1][1][1:3]),(closest[2][1][1:3])]
    similarity += len(set(t_list) & set(preferences[i-1])) 
    output += '<tr>'

    num_img = leadingZeroStr(i)
    output += '<td><img src = "images/i'+ num_img + '.jpg"><br>query = '+ num_img +'</td>'
    
    output += '<td><img src = "images/i'+ t_list[0] + '.jpg"><br>t1 = '+ t_list[0] + ', count = '+ str(score[0]) +'</td>'

    output += '<td><img src = "images/i'+ t_list[1] + '.jpg"><br>t2 = '+ t_list[1] + ', count = '+ str(score[1]) +'</td>'

    output += '<td><img src = "images/i'+ t_list[2] + '.jpg"><br>t3 = '+ t_list[2] + ', count = '+ str(score[2]) +'</td>'

    output += '<td>row score = '+ str(sum(score)) +'</td>'
    totScore += sum(score)
    output += '</tr>'

output += '<tr><td>Total Score = '+ str(totScore) +'</td></tr>'

output += '<tr><td>Predictions intersecting with Preferences = '+ str(similarity) +'/120</td></tr>'

##Geeks for geeks syntax for creating and writing to an html file.
Func5 = open("output5.html","w")

# Adding input data to the HTML file
Func5.write("<html>\n<head>\n<title> \nOutput Data for Part 5.3 \
           </title>\n</head> <body><h1>Visualization Results Given Simple Simplex</h1><br><h2>Simplex(a,b,c,d) = "+str(simplex)+"</h2><table><tbody>"+output+"</tbody></table>\
            \n</body></html>")
              
# Saving the data into the HTML file
Func5.close()








