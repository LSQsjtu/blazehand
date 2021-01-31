import numpy as np
import torch
import cv2
import math

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches

palmUpId = 2
palmDownId = 0
palm_shift_y = 0.5
palm_box_scale = 2.6

fo = open("center.txt", "w")

def plot_detections(img, detections):
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d hands" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        cv2.imshow("frame",img)
        xscale = xmax - xmin 
        yscale = ymax - ymin
         
        handUpx = detections[i,4+palmUpId * 2]
        handUpy=detections[i,4+palmUpId * 2+1]
        handDownx = detections[i,4 + palmDownId * 2]
        handDowny=detections[i,4 + palmDownId * 2+1]
        angleRad = math.atan2(handDownx - handUpx, handDowny - handUpy)                        
        angleDeg = angleRad * 180 / math.pi
        x_center = xmin + xscale * (0.5 - palm_shift_y * math.sin(angleRad))
        y_center = ymin + yscale * (0.5 - palm_shift_y * math.cos(angleRad));
        xscale = yscale = max(xscale, yscale)
        yrescale =xrescale = xscale * palm_box_scale
        
        h, w = img.shape[:2]

        dst_w = dst_h = int(w * math.fabs(math.sin(angleRad)) + h * math.fabs(math.cos(angleRad)))
         
        matrix = cv2.getRotationMatrix2D((128,128), -angleDeg, 1)
        matrix[0, 2] += dst_w // 2 - 128
        matrix[1, 2] += dst_h // 2 - 128
        rotFrame = cv2.warpAffine(img, matrix, (dst_w, dst_h))

        # cv2.imshow("rotate",rotFrame)
        # fo.write("x_center_1=")
        # fo.write(str(x_center))
        # fo.write(", y_center_1=")
        # fo.write(str(y_center))
        # fo.write("\n")
        # pointMat = np.array([[x_center-128],[y_center-128],[0]])
        # mat = np.dot(matrix,pointMat)*(dst_h/256)
        # x_center = mat[0]+dst_h/2.0
        # y_center = mat[1]+dst_h/2.0
        poinMat = np.array([[x_center],[y_center],[1.0]])
        mat= np.dot(matrix,poinMat)
        x_center = mat[0]
        y_center = mat[1]
        # fo.write("x_center=")
        # fo.write(str(x_center))
        # fo.write(", y_center=")
        # fo.write(str(y_center))
        # fo.write("\n")
        xrescale_2 = xrescale / 2
        yrescale_2 = yrescale / 2
        xDwHalf = min(x_center, xrescale_2)
        yDwHalf = min(y_center, yrescale_2)
        xUpHalf = dst_w - x_center if x_center + xrescale_2 > dst_w else xrescale_2
        yUpHalf = dst_h - y_center if y_center + yrescale_2 > dst_h else yrescale_2
        cropHand = rotFrame[int(x_center - xDwHalf):int(x_center+xUpHalf), int(y_center - yDwHalf): int(y_center+ yUpHalf)]
        cropHand = cv2.copyMakeBorder(cropHand, int(yrescale_2 - yDwHalf), int(yrescale_2 - yUpHalf), int(xrescale_2 - xDwHalf), int(xrescale_2 - xUpHalf), borderType=cv2.BORDER_CONSTANT)
        cropImage_Affine = cv2.resize(cropHand,(256,256),0,0,cv2.INTER_LINEAR)
        cv2.imshow('crop', cropImage_Affine)

    
from blazepalm import BlazePalm

net = BlazePalm().to(gpu)
net.load_weights("blazepalm.pth")
net.load_anchors("anchors.npy")

# Optionally change the thresholds:
net.min_score_thresh = 0.75
net.min_suppression_threshold = 0.3

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret, img = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_ori = cv2.resize(img, (256, 256))
    img = img_ori

    detections = net.predict_on_image(img_ori)
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)
    if (detections.shape[0]==0):
       cv2.imshow('frame', img)
    else:
        img = torch.from_numpy(img).permute((2, 0, 1)).to(gpu)
        img = img.unsqueeze(0)

        x = net._preprocess(img)

        with torch.no_grad():
            out = net(x)

        outputs = net._tensors_to_detections(out[0], out[1], net.anchors)[0]

        plot_detections(img_ori, outputs[-1].cpu())

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
fo.close()