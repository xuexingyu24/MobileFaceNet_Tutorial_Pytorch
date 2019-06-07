# MobileFaceNet_Tutorial_Pytorch
This repo illustrates how to implement MobileFaceNet and Arcface for face recognition task.  

## Demo

<img src="images/ipy_pic/output.gif"  width="700" style="float: left;">

## Test
1. Test Picture
  ```
  python MTCNN_MobileFaceNet.py -img {image_path}
  ```
2. Take Picture for Face Database
* From cam
  ```
  python take_picture.py -n {name}
  ```   
* From photo
  ```
  python take_ID.py -i {image_path} -n {name}
  ```
3. Test Video
* From cam
  ```
  python cam_demo.py
  ```
* From video file
  ```
  python video_demo.py
  ```
4. Instruction 
  ```
  MobileFaceNet_Step_by_Step.ipynb
  ```
## Train
Download training and evaluation data from [Model Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo).
Put images and annotation files into "data_set" folder. The structure should be arranged as follows:
  ```
  data_set/
              ---> AgeDB-30
              ---> CASIA_Webface_Image
              ---> CFP-FP
              ---> faces_emore_images
              ---> LFW
  ```
1. Convert .bin and .rec file to images 
  ```
  python data_set/load_images_from_bin.py
  ```
2. Generate annotation file 
  ```
  python data_set/anno_generation.py
  ```
3. Train MobileFaceNet
  ```
  python Train.py
  ```
4. Instruction 
  ```
  MobileFaceNet_Training_Step_by_Step.ipynb
  ```
The training results are shown below:
<table><tr>
<td> <img src="images/ipy_pic/loss_train.png"  width="500" style="float: left;"> </td>
<td> <img src="images/ipy_pic/accuracy_train.png"  width="500" > </td>
</tr></table>

## Evaluation 
  ```
  python Evaluation.py
  ``` 
Here is the evaluation result. 'Flip' the image can be applied to encode the embedding feature vector with ~ 0.07% higer accuracy. L2 distance score slightly outperforms cos similarity (not necessarily the same trend for other cases, but it is what we conclude in this work) 

|  Eval Type     |   Score   |   LFW   | AgeDB-30 | CFP-FP 
|:--------------:|:---------:|:-------:|:--------:|:-------
|Flip            |  L2       |  99.52  |   96.30  |  92.93    
|Flip            |  Cos      |  99.50  |   96.18  |  92.84   
|UnFlip          |  L2       |  99.45  |   95.63  |  93.10   
|UnFlip          |  Cos      |  99.45  |   95.65  |  93.10 

Don't forget to star the repo if it is helpful for your research 
