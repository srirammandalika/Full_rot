import numpy as np
from tqdm import tqdm
from pathlib import Path
from imageio import imsave

# STL10
'''reference: https://www.kaggle.com/code/pratt3000/generate-stl10/notebook'''
HomePath = './dataset/stl10_binary'  # Corrected path to stl10_binary directory.

MetaNames = ['train', 'test', 'unlabeled']  # Correctly named 'unlabeled'
for MetaName in MetaNames:
    if MetaName == 'unlabeled':
        DestPath = './dataset/STL10_unlabelled/train'  # Unlabeled data
    else:
        DestPath = f'./dataset/STL10/{MetaName}'  # Train/test data

    # Read images
    with open(f'{HomePath}/{MetaName}_X.bin', 'rb') as f:
        Metadata = np.fromfile(f, dtype=np.uint8)
        Images = np.reshape(Metadata, (-1, 3, 96, 96))  # Reshape to (Num_Images, Channels, Height, Width)
        Images = np.transpose(Images, (0, 3, 2, 1))  # Transpose to (Num_Images, Height, Width, Channels)

    # Read labels for train/test, skip for unlabeled
    if MetaName != 'unlabeled':
        with open(f'{HomePath}/{MetaName}_y.bin', 'rb') as f:
            Labels = np.fromfile(f, dtype=np.uint8)
    else:
        Labels = np.array([0] * len(Images))  # Placeholder for unlabeled data

    # Determine the number of classes (only for labelled data)
    if MetaName != 'unlabeled':
        NumClasses = len(set(Labels))
    else:
        NumClasses = 1  # For unlabelled data, treat it as 1 class

    # Prepare class names for train/test data
    ClassNames = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

    # Counter to keep track of image counts for each class
    Counter = np.zeros((NumClasses,), dtype=np.int32)

    # Save images
    for i, Img in enumerate(tqdm(Images, colour='green', ncols=60)):
        if MetaName != 'unlabeled':
            Label = Labels[i] - 1  # STL-10 labels are 1-based, adjust to 0-based index.
            ClassName = ClassNames[Label]
            Counter[Label] += 1
            SaveStr = f'{ClassName}_{Counter[Label]}'
        else:
            # For unlabelled, use a sequential image name
            SaveStr = f'unlabelled_img_{i + 1}'

        # Create directories if they don't exist
        Path(DestPath).mkdir(parents=True, exist_ok=True)

        # Save image as PNG
        imsave(f'{DestPath}/{SaveStr}.png', Img, format="png")

    print(f"Images for {MetaName} saved to {DestPath}.")
