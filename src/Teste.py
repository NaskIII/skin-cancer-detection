from MPP import Pre_Processing_Module
from MEC import Features_extractor
import cv2

mpp = Pre_Processing_Module.PreProcessing_Module()
mec = Features_extractor.Features_Extractor()

imgBGR = mpp.openImg(r'C:\Users\Raphael Nascimento\Pictures\Melanoma\1.jpg')
drawIMG = mpp.openImg(r'C:\Users\Raphael Nascimento\Pictures\Melanoma\1.jpg')
img = mpp.removeHair(imgBGR, (11, 11))
grayImg = mpp.bgr2Gray(img)
gaussianBlurImg = mpp.gaussianBlur(grayImg, (11, 11))
# grayImg = mpp.equalizeImg(grayImg)
binaryImg = mpp.binaryImg(grayImg, 0, 255)
binaryImg = mpp.morphologyClose(binaryImg, (11, 11), 0)
mpp.removeSmallAreas(binaryImg)
# borderedImg = mpp.canny(binaryImg)
contours, hierarchy = mec.findContours(binaryImg)
moments = mec.findMoments(binaryImg)
(cX, cY) = mec.findCenterOfMass(moments)
(rows, cols) = binaryImg.shape
binaryPixels = mec.findBinaryPixels(binaryImg, rows, cols)
symmetryX = mec.findSymmetryX(binaryImg, rows, cols, cX)
symmetryY = mec.findSymmetryY(binaryImg, rows, cols, cY)
(x, y), diameter = mec.findMinEnclosingCircle(contours[0])
(blueMean, greenMean, redMean) = mec.findMeanBGR(imgBGR, binaryImg,
                                                         rows,
                                                         cols,
                                                         binaryPixels)
(blueVariance, greenVariance, redVariance) = mec.findVarianceBGR(
            imgBGR, binaryImg, rows, cols, binaryPixels, blueMean,
            greenMean, redMean)

descriptors = []
descriptors.append(symmetryX)
descriptors.append(symmetryY)
descriptors.append(diameter)
descriptors.append(redMean)
descriptors.append(redVariance)
descriptors.append(greenMean)
descriptors.append(greenVariance)
descriptors.append(blueMean)
descriptors.append(blueVariance)

drawIMG = mec.drawContours(drawIMG, contours, -1)
drawIMG = mec.drawCircle(drawIMG, (int(cX), int(cY)), 3)
drawIMG= mec.drawCircle(drawIMG, (int(x), int(y)), int(diameter))

print(descriptors)

mec.show2Img(imgBGR, img, "Imagem Original e Remocao de Cabelos")
mec.show2Img(grayImg, binaryImg, "Imagem Pre-Processada")
mec.showSingleImg(drawIMG, "Resultado")
