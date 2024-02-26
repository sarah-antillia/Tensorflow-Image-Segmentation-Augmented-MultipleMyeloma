<h2>Tensorflow-Image-Segmentation-Augmented-MultipleMyeloma(2024/02/27)</h2>

This is the second experimental Image Segmentation project for MultipleMyeloma based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1QiGah4_0yY-5B7s2kIZ2AjbEVu2ejB3G/view?usp=sharing">MultipleMyeloma-ImageMask-Dataset_V2_X.zip</a>
<br>
<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/segmentation_samples.png" width="720" height="auto">
<br>
<br>

In order to improve segmentation accuracy, we will use an online dataset augmentation strategy based on Python script <a href="./src/ImageMaskAugmentor.py">
ImageMaskAugmentor.py</a> to train a MultipleMyeloma Segmentation Model.<br><br>
Please see also our first experiment 
<a href="https://github.com/atlan-antillia/Image-Segmentation-Multiple-Myeloma">Image-Segmentation-Multiple-Myeloma</a>
<br>
<br>
As a first trial, we use the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this MultipleMyelomaSegmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
The original dataset used here has been take from the following  web site:<br><br>
<b>SegPC-2021-dataset</b><br>
SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images<br>
<pre>
https://www.kaggle.com/datasets/sbilab/segpc2021dataset
</pre>

<b>Citation:</b><br>

<pre>
Anubha Gupta, Ritu Gupta, Shiv Gehlot, Shubham Goswami, April 29, 2021, "SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells 
in Microscopic Images", IEEE Dataport, doi: https://dx.doi.org/10.21227/7np1-2q42.

BibTex
@data{segpc2021,
doi = {10.21227/7np1-2q42},
url = {https://dx.doi.org/10.21227/7np1-2q42},
author = {Anubha Gupta; Ritu Gupta; Shiv Gehlot; Shubham Goswami },
publisher = {IEEE Dataport},
title = {SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images},
year = {2021} }

IMPORTANT:
If you use this dataset, please cite below publications-
1. Anubha Gupta, Rahul Duggal, Shiv Gehlot, Ritu Gupta, Anvit Mangal, Lalit Kumar, Nisarg Thakkar, and Devprakash Satpathy, 
 "GCTI-SN: Geometry-Inspired Chemical and Tissue Invariant Stain Normalization of Microscopic Medical Images," 
 Medical Image Analysis, vol. 65, Oct 2020. DOI: 
 (2020 IF: 11.148)
2. Shiv Gehlot, Anubha Gupta and Ritu Gupta, 
 "EDNFC-Net: Convolutional Neural Network with Nested Feature Concatenation for Nuclei-Instance Segmentation,"
 ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
 Barcelona, Spain, 2020, pp. 1389-1393.
3. Anubha Gupta, Pramit Mallick, Ojaswa Sharma, Ritu Gupta, and Rahul Duggal, 
 "PCSeg: Color model driven probabilistic multiphase level set based tool for plasma cell segmentation in multiple myeloma," 
 PLoS ONE 13(12): e0207908, Dec 2018. DOI: 10.1371/journal.pone.0207908

License
CC BY-NC-SA 4.0
</pre>


<h3>
<a id="2">
2 MultipleMyeloma ImageMask Dataset
</a>
</h3>
 If you would like to train this MultipleMyelomaSegmentation model by yourself,
 please download this dataset from the google drive 
<a href="https://drive.google.com/file/d/1QiGah4_0yY-5B7s2kIZ2AjbEVu2ejB3G/view?usp=sharing">MultipleMyeloma-ImageMask-Dataset_V2_X.zip</a>
<br>
On that dataset, please see also <a href="https://github.com/sarah-antillia/MultipleMyeloma-ImageMask-Dataset">MultipleMyeloma-ImageMask-Dataset</a>
<br>
Please expand the downloaded ImageMaskDataset and place them under <b>./dataset</b> folder to be

<pre>
./dataset
└─MultipleMyeloma
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
 
 
<b>MultipleMyelomaDataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/MultipleMyeloma_Statistics.png" width="512" height="auto"><br>

As shown above, the number of images of train and valid dataset is not necessarily large. Therefore the online dataset augmentation strategy may 
be effective to improve segmentation accuracy.

<br>

<h3>
<a id="3">
3 TensorflowSlightlyFlexibleUNet
</a>
</h3>
This <a href="./src/TensorflowUNet.py">TensorflowUNet</a> model is slightly flexibly customizable by a configuration file.<br>
For example, <b>TensorflowSlightlyFlexibleUNet/MultipleMyeloma</b> model can be customizable
by using <a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/train_eval_infer.config">train_eval_infer.config</a>
<pre>
; train_eval_infer.config
; Pancreas, GENERATOR_MODE=True
; 2024/02/26 (C) antillia.com
; 2024/02/26 Modified to use 
; loss           = "bce_dice_loss"

[model]
generator     = True
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 1
base_filters   = 16
base_kernels   = (5,5)
num_layers     = 7
dropout_rate   = 0.08
learning_rate  = 0.0001
clipvalue      = 0.5
dilation       = (2,2)
loss           = "bce_dice_loss"
metrics        = ["binary_accuracy"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 4
steps_per_epoch  = 200
validation_steps = 100
patience      = 10
metrics       = ["binary_accuracy", "val_binary_accuracy"]
model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "../../../dataset/MultipleMyeloma/train/images/"
mask_datapath  = "../../../dataset/MultipleMyeloma/train/masks/"
create_backup  = False
learning_rate_reducer = False
save_weights_only = True

[eval]
image_datapath = "../../../dataset/MultipleMyeloma/valid/images/"
mask_datapath  = "../../../dataset/MultipleMyeloma/valid/masks/"

[test] 
image_datapath = "../../../dataset/MultipleMyeloma/test/images/"
mask_datapath  = "../../../dataset/MultipleMyeloma/test/masks/"

[infer] 
images_dir    = "./mini_test"
output_dir    = "./mini_test_output"
merged_dir    = "./mini_test_output_merged"

[segmentation]
colorize      = False
black         = "black"
white         = "green"
blursize      = None

[mask]
blur      = True
blur_size = (3,3)
binarize  = True
#threshold = 128
threshold = 74

[generator]
debug     = True
augmentation   = True

[augmentor]
vflip    = True
hflip    = True
rotation = True
angles   = [5, 10]
shrinks  = [0.8]
shears   = [0.2]
transformer = True
alpah       = 1300
sigmoid     = 8
</pre>

Please note that the online augementor 
<a href="./src/ImageMaskAugmentor.py">
ImageMaskAugmentor.py</a> reads the parameters in [generator] and [augmentor] sections, and yields some images and mask depending on the batch_size,
 which are used for each epoch of the training and evaluation process of this UNet Model. 
<pre>
[augmentor]
vflip    = True
hflip    = True
rotation = True
angles   = [5, 10,]
shrinks  = [0.8]
shears   = [0.2]
transformer = True
alpah       = 1300
sigmoid     = 8
</pre>
Depending on these parameters in [augmentor] section, it will generate vflipped, hflipped, rotated, shrinked,
sheared, elastic-transformed images and masks
from the original images and masks in the folders specified by image_datapath and mask_datapath in 
[train] and [eval] sections.<br>
<pre>
[train]
image_datapath = "../../../dataset/MultipleMyeloma/train/images/"
mask_datapath  = "../../../dataset/MultipleMyeloma/train/masks/"
[eval]
image_datapath = "../../../dataset/MultipleMyeloma/valid/images/"
mask_datapath  = "../../../dataset/MultipleMyeloma/valid/masks/"
</pre>

For more detail on ImageMaskAugmentor.py, please refer to
<a href="https://github.com/sarah-antillia/Image-Segmentation-ImageMaskDataGenerator">
Image-Segmentation-ImageMaskDataGenerator.</a>.
    
<br>

<h3>
3.1 Training
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma</b> folder,<br>
and run the following bat file to train TensorflowUNet model for MultipleMyeloma.<br>
<pre>
./1.train_generator.bat
</pre>
, which simply runs <a href="./src/TensorflowUNetGeneratorTrainer.py">TensorflowUNetGeneratorTrainer.py </a>
in the following way.

<pre>
python ../../../src/TensorflowUNetGeneratorTrainer.py ./train_eval_infer.config
</pre>
Train console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/train_console_output_at_epoch_32.png" width="720" height="auto"><br>
<br>
Train metrics:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/train_metrics_at_epoch_32.png" width="720" height="auto"><br>
<br>
Train losses:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/train_losses_at_epoch_32.png" width="720" height="auto"><br>
<br>
The following debug setting is helpful whether your parameters in [augmentor] section are good or not good.
<pre>
[generator]
debug     = True
</pre>
You can check the yielded images and mask files used in the actual train-eval process in the following folders under
<b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/</b>.<br> 
<pre>
generated_images_dir
generated_masks_dir
</pre>

Sample images in generated_images_dir<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/sample_images_in_generated_images_dir.png"
 width="1024" height="auto"><br>
Sample masks in generated_masks_dir<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/sample_masks_in_generated_masks_dir.png"
 width="1024" height="auto"><br>

<h3>
3.2 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for MultipleMyeloma.<br>
<pre>
./2.evaluate.bat
</pre>
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer.config
</pre>
This evalutes loss and accuray for test dataset specified [test] section.<br>
<pre>
[test] 
image_datapath = "../../../dataset/MultipleMyeloma/test/images/"
mask_datapath  = "../../../dataset/MultipleMyeloma/test/masks/"
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/evaluate_console_output_at_epoch_32.png" width="720" height="auto">
<br>

<pre>
Test loss    :0.1865
Test accuracy:0.9589999914169312
</pre>
 
<h2>
3.3 Inference
</h2>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for MultipleMyeloma.<br>
<pre>
./3.infer.bat
</pre>
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
mini test images<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/mini_test.png" width="1024" height="auto"><br>
<!--
Sample test mask (ground_truth)<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/sample_test_masks.png" width="1024" height="auto"><br>
-->
<br>
Inferred mini_test masks<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
Merged test images and inferred masks<br> 
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/asset/mini_test_output_merged.png" width="1024" height="auto"><br> 


Enlarged samples<br>
<table>

<tr>
<td>
mini_test/605.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/mini_test/605.jpg" width="512" height="auto">
</td>
<td>
Inferred merged/605.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/mini_test_output_merged/605.jpg" width="512" height="auto">
</td> 
</tr>

<tr>
<td>
mini_test/1735.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/mini_test/1735.jpg" width="512" height="auto">
</td>
<td>
Inferred merged/1735.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/mini_test_output_merged/1735.jpg" width="512" height="auto">
</td> 
</tr>

<tr>
<td>
mini_test/1923.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/mini_test/1923.jpg" width="512" height="auto">
</td>
<td>
Inferred merged/1923.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/mini_test_output_merged/1923.jpg" width="512" height="auto">
</td> 
</tr>

<tr>
<td>
mini_test/2123.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/mini_test/2123.jpg" width="512" height="auto">
</td>
<td>
Inferred merged/2123.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/mini_test_output_merged/2123.jpg" width="512" height="auto">
</td> 
</tr>

<tr>
<td>
mini_test/2405.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/mini_test/2405.jpg" width="512" height="auto">
</td>
<td>
Inferred merged/2123.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-MultipleMyeloma/mini_test_output_merged/2405.jpg" width="512" height="auto">
</td> 
</tr>

</table>

<h3>
References
</h3>
<b>1. SegPC-2021-dataset</b><br>
SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images<br>
<pre>
https://www.kaggle.com/datasets/sbilab/segpc2021dataset
</pre>
Citation:<br>
<pre>
Anubha Gupta, Ritu Gupta, Shiv Gehlot, Shubham Goswami, April 29, 2021, "SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells 
in Microscopic Images", IEEE Dataport, doi: https://dx.doi.org/10.21227/7np1-2q42.
BibTex
@data{segpc2021,
doi = {10.21227/7np1-2q42},
url = {https://dx.doi.org/10.21227/7np1-2q42},
author = {Anubha Gupta; Ritu Gupta; Shiv Gehlot; Shubham Goswami },
publisher = {IEEE Dataport},
title = {SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images},
year = {2021} }
IMPORTANT:
If you use this dataset, please cite below publications-
1. Anubha Gupta, Rahul Duggal, Shiv Gehlot, Ritu Gupta, Anvit Mangal, Lalit Kumar, Nisarg Thakkar, and Devprakash Satpathy, 
 "GCTI-SN: Geometry-Inspired Chemical and Tissue Invariant Stain Normalization of Microscopic Medical Images," 
 Medical Image Analysis, vol. 65, Oct 2020. DOI: 
 (2020 IF: 11.148)
2. Shiv Gehlot, Anubha Gupta and Ritu Gupta, 
 "EDNFC-Net: Convolutional Neural Network with Nested Feature Concatenation for Nuclei-Instance Segmentation,"
 ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
 Barcelona, Spain, 2020, pp. 1389-1393.
3. Anubha Gupta, Pramit Mallick, Ojaswa Sharma, Ritu Gupta, and Rahul Duggal, 
 "PCSeg: Color model driven probabilistic multiphase level set based tool for plasma cell segmentation in multiple myeloma," 
 PLoS ONE 13(12): e0207908, Dec 2018. DOI: 10.1371/journal.pone.0207908
License
CC BY-NC-SA 4.0
</pre>

<b>2. Deep Learning Based Approach For MultipleMyeloma Detection</b><br>
Vyshnav M T, Sowmya V, Gopalakrishnan E A, Sajith Variyar V V, Vijay Krishna Menon, Soman K P<br>
<pre>
https://www.researchgate.net/publication/346238471_Deep_Learning_Based_Approach_for_Multiple_Myeloma_Detection
</pre>
<br>

<b>3. Image-Segmentation-Multiple-Myeloma</b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/atlan-antillia/Image-Segmentation-Multiple-Myeloma
</pre>
<br>


