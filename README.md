# Tensorflow-VGG16
Implement VGG16 by Tensorflow using the pre-trained model from [MatConvNet](http://www.vlfeat.org/matconvnet/).

## Pre-requisites
* Python 2.7
* Scipy
* Tensorflow
* Pre-trained model [imagenet-vgg-verydeep-16.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat) (MD5: f72d927587ca4c97fbd165ec9cb9997f)

## Test on the pre-trained model

```
$ python vgg16.py
```

## Result

```
Classification Result:
        Category Name: [u'weasel']
        Probability: 82.63%

        Category Name: [u'polecat, fitch, foulmart, foumart, Mustela putorius']
        Probability: 10.86%

        Category Name: [u'mink']
        Probability: 5.95%

        Category Name: [u'black-footed ferret, ferret, Mustela nigripes']
        Probability: 0.55%

        Category Name: [u'mongoose']
        Probability: 0.01%
```
