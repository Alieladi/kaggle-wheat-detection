from torchvision.transforms import functional as F
import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import train

def output(model, root = "global-wheat-detection/test"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    submissions = {'image_id':[], 'PredictionString':[]}
    for _, _, filenames in os.walk(root):
        for filename in filenames:
            img_path = os.path.join(root, filename)
            img = Image.open(img_path).convert("RGB")
            pred = model([F.to_tensor(img).to(device)])[0]
            image_id = filename[:-4]
            boxes = pred['boxes'].tolist()
            scores = pred['scores'].tolist()

            PredictionString = ""
            for box, score in zip(boxes, scores):
                if score > .5:
                    x0, y0, x1, y1 = box
                    width = x1 - x0
                    height = y1 - y0
                    PredictionString += f"{round(score,4)} {int(x0)} {int(y0)} {int(width)} {int(height)} "
                #' '.join([str(score)]+list(map(box,str)))

            submissions['image_id'].append(image_id)
            submissions['PredictionString'].append(PredictionString[:-1])
        df = pd.DataFrame(data=submissions)
    return df

    def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               font = ImageFont.load_default(),
                               thickness=10):
    # Adds a bounding box to an image.
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
             width=thickness,
             )

if __name__ = "__main__"
    # initialize model and optimizer
    num_classes = 2 # 1 class (wheat head) + background
    model = get_model(num_classes)

    # load existing checkpoint (comment if none existing)
    epoch = 10
    PATH = f"model_weights_v1_/model_weights_v1_{epoch}.tar"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    df = output(model)
    df.to_csv("submission.csv", index=False)

    # display a case:
    img_filename = "51b3e36ab.jpg"
    img_path = os.path.join(root, "test", img_filename)
    im = Image.open(img_path).convert("RGB")

    # predict
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = model([F.to_tensor(img).to(device)])
    boxes = predictions[0]['boxes']#.tolist()
    scores = predictions[0]['scores']
    max_boxes = 50

    # draw bounding boxes
    for i in range(min(len(boxes), max_boxes)):
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        if scores[i] >= 5:
            draw_bounding_box_on_image(im, ymin, xmin, ymax, xmax, thickness=4)
            _len += 1
    print("number of boxes", len(boxes))
    im.save("51b3e36ab_boxes.jpg")
