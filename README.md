# Brain Tumor MRI Images Classification and Segmentation

Imperial College Winter School Honor Group Project, 2025.

We adopted a **VGG16 model** for brain tumor classification task and a **U-Net3D model** for brain tumor segmentation task.
Finally achieve a Dice score of 0.8194 and a 95 Hausdorff Distance of 3 on our test set.
## 1 Introduction

Non-invasive imaging techniques, particularly Magnetic Resonance Imaging (MRI), play a pivotal role in diagnosing brain tumors by providing detailed anatomical and functional insights. In the classification task, we employed the VGG16 neural network architecture and achieved an accuracy of 96% on the test set. For the segmentation task, we referred to the U-Net architecture, designed our own loss function, and incorporated extensive data augmentation techniques. Ultimately, we developed a U-Net3D network architecture that directly processes 3D files. The Dice coefficient reached 83.0% on the training set and 83.2% on the test set.


## 2 Classification

### 2.1 Design of a Transfer Learning Classification Model Based on VGG16

#### 2.1.1 DataGenerator

Use keras.utils.Sequence to implement a custom data generator (DataGenerator) for batch loading of .npy format image data, performing data preprocessing, augmentation, and binary classification based on segmentation data.

#### 2.1.2 VGG16 Transfer Learning

Using VGG16 shown in Figure 1 as a feature extractor by loading ImageNet pre-trained weights, removing the original fully connected layers while retaining only the convolutional part. The last 12 convolutional layers are unfrozen, allowing the model to adapt to a new classification task while reducing computational cost.

<center><img src=.github/pic1.png width=60%>
<br>
<div style="font-size: 16px; color: #333;">
    Figure 1: VGG16 network architecture.
</div></center>

#### 2.1.3 Training Process and Early Stopping Strategy

A well-designed early stopping strategy can effectively prevent the model from overfitting. If the validation accuracy (val_accuracy) does not improve within 15 epochs, training will be halted, and the model will revert to the best weights. The learning rate is dynamically adjusted if the validation loss (val_loss) does not show improvement within 6 epochs; the learning rate is reduced by 20%, with a lower bound of 1e-9 to prevent the learning rate from becoming too small. In the model training phase, the optimal model was obtained by adjusting the early stopping condition, modifying the number of trainable layers, and modifying the dynamic adjustment of the learning rate parameter.

<center>
<img src=.github/pic2.png width=60%>
<br>
<div style="font-size: 16px; color: #333;">
    Figure 2: Test Accuracy Improvement Over Adjustments.
</div>
</center>

## 3 Segmentation

### 3.1 Data Visualization

The provided dataset consists of 210 3D brain MRI images with a resolution of 240×240×155. Each entry includes two files: xxx_fla.nii.gz and xxx_seg.nii.gz, which represent the 3D brain MRI image and the tumor mask data, respectively. .nii is a medical imaging data format that stores 3D MRI image data of the brain. Each data entry comprises 155 axial slices. There are two ways to visualize the data.

The most straightforward method is to utilize the ITK-SNAP software. By importing the two files as visualization data and mask data respectively, one can observe the brain MRI images and the corresponding tumor locations from various sectional views, as depicted in Figure 3.

<center>
<img src=.github/pic3.png width=60%>
<br>
<div style="font-size: 16px; color: #333;">
    Figure 3: Visualization using ITK-SNAP software. The tumor region is marked in red in Figure 3b.
</div>
</center>

Alternatively, the Python library Nibabel can be employed to read and load the data as three-dimensional arrays. We have also conducted some visualization demonstrations, as shown in Figure 4.

<center>
<img src=.github/pic4.png width=60%>
<br>
<div style="font-size: 16px; color: #333;">
    Figure 4: Visualization using Nibabel library in Python.The brain slices are visualized by overlaying multiple axial
views, with the tumor regions marked in red.
</div>
</center>

### 3.2 Data Argumentation

In the initial training phase, we did not incorporate data augmentation. However, this led to overfitting, as evidenced by the high accuracy on the training set and the significantly lower accuracy on the validation set. By introducing data augmentation, we were able to reduce the model’s sensitivity to specific image features, thereby enhancing its generalization capability.

<center>
<img src=.github/pic5.png width=60%>
<br>
<div style="font-size: 16px; color: #333;">
    Figure 5: Several data augmentation methods
</div>
</center>

### 3.3 Methodology

#### 3.3.1 Loss Function

For medical image segmentation, the IoU loss and Dice loss are commonly used as loss functions. They are defined as below.

$$
\text{IoU loss} = 1 - \frac{Area\quad of\quad Overlap}{Area\quad of\quad Union}
$$

$$
\text{Dice loss} = 1 - \frac{2 \times Area\quad of\quad Overlap}{Total\quad Area}
$$

Here we combine the binary cross-entropy loss with the dice loss [15]. The overall loss function is calculated as below:

$$
    L_{total} = L_{BCE} - D
$$

$$
    L_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$


#### 3.3.2 Optimizer

We chose Adam optimizer as our optimizer. It is widely used in deep learning due to its efficiency and adaptability. Adam combines the advantages of both RMSprop and Momentum methods, allowing for adaptive learning rates and faster convergence.

#### 3.3.3 Network Architecture

We have replicated the 3D U-Net architecture. The network consists of an encoder path and a decoder path, each with four resolution steps.

<center>
<img src=.github/pic6.png width=60%>
<br>
<div style="font-size: 16px; color: #333;">
    Figure 6: 3D U-Net architecture
</div>
</center>

### 3.4 Experiment Design

All experiments were conducted on a NVIDIA GeForce RTX 4090 with 24GB of memory.

**Data Preprocessing**: Dataset is randomly divided into an 80% training set and a 20% validation set. Before training, the non-zero regions of the 3D images in the entire dataset are normalized. Subsequently, data augmentation is performed on the training set, including random flipping, random rotation, random scaling, adding noise, and random cropping.

**Training**: Due to limitations of memory, we set the batch size of the model at 1 and the learning rate at 3 × 10−4. We used StepLR module to dynamically decrease the learning rate in the training process to achieve higher accuracy.

**Results**: During training, we initially refrained from using extensive data augmentation techniques and only applied simple normalization. The results are shown in Figure 7a. The model was trained for a total of 30 epochs, achieving a Dice coefficient of 80% on the training set and 83% on the validation set. After employing data augmentation techniques, the results are shown in Figure 7b. The model was trained for 30 epochs, achieving a Dice coefficient of 83.0% on the training set and 83.2% on the validation set. It can be observed that the accuracy on both the training and validation sets is more balanced, which reduces the overfitting of the model and enhances its robustness.

<center>
<img src=.github/pic7.png width=60%>
<br>
<div style="font-size: 16px; color: #333;">
    Figure 7: Training Results
</div>
</center>

Finally, a series of data was randomly selected, and the prediction results along with the ground truth are shown in Figure 8. It can be observed that most regions of the prediction are consistent with the ground truth, with only a few areas of significant color change failing to be accurately predicted.

<center>
<img src=.github/pic8.png width=60%>
<br>
<div style="font-size: 16px; color: #333;">
    Figure 8: Training Results shown with ITK-SNAP
</div>
</center>
