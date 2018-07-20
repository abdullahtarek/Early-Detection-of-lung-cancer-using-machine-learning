import SimpleITK as sitk
import numpy as np
def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
    lines = []
    with open(filename, "rb") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def getcrops(numpyImage):
    frame=64
    crops = []
    for d in range(int(numpyImage.shape[0]/64)):
        for r in range(int(numpyImage.shape[1]/64)):
            for c in range(int(numpyImage.shape[2]/64)):
                dd = d *frame
                rr= r*frame
                cc= c*frame
                crop = numpyImage[dd:dd+frame,rr:rr+frame,cc:cc+frame]
                crops.append(crop)

    for r in range(int(numpyImage.shape[1]/64)):
        for c in range(int(numpyImage.shape[2]/64)):
            dd = d *frame
            rr= r*frame
            cc= c*frame
            crop = numpyImage[-64:,rr:rr+frame,cc:cc+frame]
            crops.append(crop)          
    return crops