import sys
import os
import pydbow
import pyvlad
import cv2
from tqdm import tqdm
from imagetoolbox.ORB import detect_orb, visualize_orb
import slamtoolbox.sthereo_dataset as sthereo


###############################################

# Create a new Vocabulary
def create_vocabulary():
    dataloader = sthereo.Dataloader("/home/hj/Dropbox/Dataset/STHEREO-raw/01", 
                                    "image/rgb_left").downsample(25)
    voc = pydbow.Vocabulary(4,3)
    training_features = []
    for data in tqdm(dataloader, ncols=100):
        image = cv2.imread(data.path, 0)
        keypoints, descriptors = detect_orb(image)
        training_features.append(descriptors)

    voc.create(training_features)
    print(f'voc size: {voc.size()}')
    voc.save("./config/sthereo_01_rgb_4_3.yaml")

###############################################


# load vocabulary from file
voc_load = pyvlad.Vocabulary("./config/sthereo_01_rgb_4_3.yaml")
print(f'voc_load: {voc_load}')

# Create a new Database
db = pyvlad.Database("./config/sthereo_01_rgb_4_3.yaml")
print(f'db: {db}')

dataloader = sthereo.Dataloader("/home/hj/Dropbox/Dataset/STHEREO-raw/01", 
                                "image/rgb_left").downsample(5)

images = []
keypoints_list = []
for i, data in enumerate(dataloader):
    image = cv2.imread(data.path, 0)
    keypoints, descriptors = detect_orb(image)
    # print(descriptors.shape)
    db.add(descriptors)
    images.append(image)
    keypoints_list.append(keypoints)

    print(f'db size: {db.size()} \r', end='')

    results = db.query(descriptors, 1, max(db.size()-100, 0))
    if len(results) == 0:
        continue
    best_idx = results[0][0]
    best_score = results[0][1]

    query_image = visualize_orb(image, keypoints)
    best_image = visualize_orb(images[best_idx], keypoints_list[best_idx])
    concat = cv2.hconcat([query_image, best_image])
    concat = cv2.resize(concat, (0, 0), fx=0.5, fy=0.5)

    near = data.pose.distance(dataloader[best_idx].pose) < 25
    color = (0, 255, 0) if near else (0, 0, 255)
    cv2.putText(concat, f'best query results for {i}th image: '
                        f'id: {best_idx}, score: {best_score:.4f}',
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    
    cv2.imshow('query and best', concat)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pscore = db.compute_pairwise_score()
print(f'pscore: {pscore.shape}')
pscore = (pscore * 255).astype('uint8')
cv2.imshow('pscore', pscore)
cv2.imwrite('pscore_01_vlad.png', pscore)
cv2.waitKey(0)