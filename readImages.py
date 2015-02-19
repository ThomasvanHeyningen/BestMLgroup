
class ImageReader:

    def __init__(self, directory_names):
        self.directory_names=directory_names

    def readimages(self):
        #Rescale the images and create the combined metrics and training labels
        #get the total training images
        numberofImages = 0
        for folder in self.directory_names:
            for fileNameDir in os.walk(folder):
                for fileName in fileNameDir[2]:
                     # Only read in the images
                    if fileName[-4:] != ".jpg":
                      continue
                    numberofImages += 1

        # We'll rescale the images to be 25x25
        maxPixel = 25
        imageSize = maxPixel * maxPixel
        num_rows = numberofImages # one row for each image in the training dataset
        num_features = imageSize + 3 # for our ratio

        # X is the feature vector with one row of features per image
        # consisting of the pixel values and our metric
        X = np.zeros((num_rows, num_features), dtype=float)
        # y is the numeric class label
        y = np.zeros((num_rows))

        files = []
        # Generate training data
        i = 0
        label = 0
        # List of string of class names
        namesClasses = list()

        print "Reading images"
        # Navigate through the list of directories
        for folder in directory_names:
            # Append the string class name for each class
            currentClass = folder.split(os.pathsep)[-1]
            namesClasses.append(currentClass)
            for fileNameDir in os.walk(folder):
                for fileName in fileNameDir[2]:
                    # Only read in the images
                    if fileName[-4:] != ".jpg":
                      continue

                    # Read in the images and create the features
                    nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                    image = imread(nameFileImage, as_grey=True)
                    files.append(nameFileImage)
                    (axisratio, width, height) = getMinorMajorRatio(image)
                    image = resize(image, (maxPixel, maxPixel))

                    # Store the rescaled image pixels and the axis ratio
                    X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
                    X[i, imageSize+0] = axisratio
                    X[i, imageSize+1] = height # this might not be good
                    X[i, imageSize+2] = width# this might not be good

                    # Store the classlabel
                    y[i] = label
                    i += 1
                    # report progress for each 5% done
                    report = [int((j+1)*num_rows/20.) for j in range(20)]
                    if i in report: print np.ceil(i *100.0 / num_rows), "% done"
            label += 1
        return (X,y,namesClasses, num_features)
