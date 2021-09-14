# kaggle-wheat-detection

Model development for Wheat detection kaggle competition.
Prediction task description: https://www.kaggle.com/c/global-wheat-detection
The best model I submitted is developed with PyTorch. It uses a Faster R-CNN based on a resnet50 with FPN pretrained on COCO.
Evaluated on the mean average precision at different intersection over union, it got 0.6046.

![image](https://user-images.githubusercontent.com/24422668/133229551-b16aa54a-3272-45f3-bf87-18ef3659b61d.png)
