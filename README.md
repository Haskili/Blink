<h1 align="center">Blink</h1> 
  <p align="center">
  <img src="https://i.imgur.com/zB2YVNx.png"  alt="blink-icon"  width="100"  height="70"><br>
    An OpenCV approach to CCTV
    <br/><br/><br/>
    [<a href="https://docs.opencv.org/4.0.1/index.html">OpenCV</a>]
    [<a href="https://github.com/Haskili/Blink#acknowledgements">Acknowledgements</a>]
    [<a href="https://github.com/Haskili/Blink/issues">Issues</a>]
  </p>
</p>

## Overview

Blink records frames from a device like normal recording software, but only keeps those that change significantly from the frame in the previously recorded event to minimize space requirements.

It utilizes [SSIM](https://en.wikipedia.org/wiki/Structural_similarity), Flat Difference, and [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) to detect how different two images are and only records the frame if that difference exceeds a threshold set through the arguments. 

Each time that happens, the program then identifies objects in the image and their location. All identified objects are labeled with a number, weight of identification, and a outline depicting the region of interest. For any human detections using the [Single Shot Detector](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2), it will try to predict if any two people are too close for social-distancing by estimating the relative distance of every detected person from all other people it found in the image.

<p align="center">
	<img src="https://imgur.com/EBVqhyc.gif" alt="output2" border="0">
</p>

Finally, it will alert the user as to the time of the event, event number, and then saves this image to a separate file labeled with that event number.
<br></br>

## Requirements

**OpenCV (4.3.0-7)**
```sh
pacman -S opencv (optional dependencies)
```
Please see dependencies listed [here](https://www.archlinux.org/packages/extra/x86_64/opencv/).
<br></br>

**Classification & Detection Files**
<br></br>
To use the Haar-Cascade Classification functionality it is required to have a cascade file. In my own testing, I used the cascade file created by Rainer Lienhart that comes with `opencv-samples` from the [AUR](https://www.archlinux.org/packages/extra/x86_64/opencv-samples/) for front-facing human faces. These files are also available [here](https://github.com/opencv/opencv/blob/master/data/haarcascades/), in the OpenCV repository.

Alternatively, to use the Single-Shot-Detector functionality it is required to have the class labels as well as both the model and network configuration in whatever file format you choose. Please see the [documentation](https://docs.opencv.org/master/d6/d0f/group__dnn.html#ga3b34fe7a29494a6a4295c169a7d32422) for information on supported backends in OpenCV. A good place to look for these is the [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and the [Tensor Flow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) repositories. Depending on what you use, you will need to set the scaling for your images and mean value used in the SSD. In my own testing, I've used MobileNet, which has performed well in general application.
<br></br>

## Usage

```sh
./driver -<OPTION>=<VALUE>
```
```sh
./driver -<MULTI-VALUE-OPTION>=<VALUE>,<VALUE>,...
```
e.g. Capture an image every 1000000ns interval on devices 0 & 2 with a percent difference threshold of 15%
```sh
./driver -i=1000000 -t=15 -d=0,2
```

### Arguments
|Option                 |Full Name |Description                                                                                              |
|-----------------------|----------|---------------------------------------------------------------------------------------------------------|
|`d`			        |device    | Sets the device used to read frames from		      		                                             |
|`i`			        |interval  | Sets the interval between capturing frames in units of nanoseconds	                                     |
|`m`			        |method    | Defines which method of calculating image difference is used (0: FTPD -- 1: SSIM)		      		     |
|`f`					|fthresh   | Set the pixel-difference percentage threshold for FTPD		      		                                 |
|`t`					|threshold | Specify the percent difference threshold to use for what qualifies as 'different images'              |
|`n`					|type      | Set the detection method (-1: NONE -- 0: SSD -- 1: HCC)                                                 |
|`c`					|classifier| Set the path to the file(s) used during classification                                                 |
|`p` (req: SSD)	        |humanLID  | Specify the index of 'human' in class labels given for SSD			  		                             |
|`a` (req: HCC)	        |neighbours| Set the minumum amount of neighbours for HCC detection			      		                             |
|`s` (req: HCC)	        |scale     | Set the image scaling to use for HCC detection		      		                                         |
|`b` (req: HCC)	        |blur  	   | Specify whether to blur identified objects for HCC		      		                                     |
|`r` (req: HCC)	        |rotation  | Specify whether to try rotation for HCC			      		                                         |

<br></br>

## Acknowledgements
*'Blink'* was originally meant as a educational resource for OpenCV that turned into my Senior Project. As such, it is heavily commented and there may be certain sections with verbose documentation.

A good portion of the comments for `SSIM()` & `PSNR()` refer to the formulas, most if not all of which can be found in the two Wikipedia pages below:
* [SSIM Wikipedia page](https://en.wikipedia.org/wiki/Structural_similarity)
* [PSNR Wikipedia page](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)

To learn how to calculate SSIM in a timely and efficient manner with the matrices in OpenCV as well as use the cascade classifiers provided, I referenced the pages below: 
* [OpenCV Documentation on SSIM & PSNR](https://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html)
* [OpenCV Documentation on forEach() and matrix iteration](https://docs.opencv.org/4.0.1/d3/d63/classcv_1_1Mat.html#a952ef1a85d70a510240cb645a90efc0d)
* [OpenCV Documentation on cascade classifiers and functional usage](https://docs.opencv.org/4.0.1/db/d28/tutorial_cascade_classifier.html)