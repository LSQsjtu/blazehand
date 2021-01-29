import numpy as np
import torch
import cv2

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_detections(img, detections):
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d hands" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        ymin = int(detections[i, 0] * img.shape[0])
        xmin = int(detections[i, 1] * img.shape[1])
        ymax = int(detections[i, 2] * img.shape[0])+1
        xmax = int(detections[i, 3] * img.shape[1])+1
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        
    cv2.imshow('frame', img)

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