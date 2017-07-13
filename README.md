# Tensorflow-VGG16
Implement VGG16 by Tensorflow using the pre-trained model from [MatConvNet](http://www.vlfeat.org/matconvnet/).

## Pre-requisites
* Python 2.7
* Scipy
* Tensorflow
* Pre-trained model [vgg-face.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat) (MD5: f72d927587ca4c97fbd165ec9cb9997f)

## Test on the pre-trained model

```
$ python test_vgg16.py
```

## Result

```
Classification Result:
        Category Name: Aamir_Khan
        Propbability: 51.60%
        
        Category Name: Adam_Driver
        Propbability: 6.78%
        
        Category Name: Manish_Dayal
        Propbability: 1.95%
```
