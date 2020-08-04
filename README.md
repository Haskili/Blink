
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

It utilizes [SSIM](https://en.wikipedia.org/wiki/Structural_similarity), Flat Difference, and [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) to detect how different two images are and only records the frame if that difference exceeds a threshold set through the arguments. Each time that happens, the program then uses a cascade classifier that can be set through arguments to identify objects in the image. Any and all identified objects are labeled with a number, weight of identification, and a outline depicting the region of interest. 

Finally, it will alert the user through the standard output as to the time of the event, event number, and then saves this image to a separate file with the same label as that event number.

## Getting Started

### Prerequisites
OpenCV (4.3.0-7)
```sh
pacman -S opencv (optional dependencies)
```
Please see dependencies listed [here](https://www.archlinux.org/packages/extra/x86_64/opencv/).

To use the classification functionality, it is required to have a cascade file. In my own testing, I used the cascade file created by Rainer Lienhart that comes with `opencv-samples` from the [AUR](https://www.archlinux.org/packages/extra/x86_64/opencv-samples/) for front-facing human faces. These files are also available [here](https://github.com/opencv/opencv/blob/master/data/haarcascades/), in the OpenCV repository.

## Usage

```sh
./blink (OPTIONS)
```

### Arguments
* `-i` sets the interval between captures in units of nanoseconds
* `-m` defines which method of calculating image difference is used (SSIM VS FTPD)
* `-t` specifies the threshold to use for what percent difference qualifies as 'different images'
* `-d` sets the device used to read frames from
* `-c` sets the path to the cascade file used during classification
* `-r` specifies whether or not to try rotating the image during classification
* `-s` changes the scale of the image used in classification
* `-b` specifies whether or not to blur any identifications (people) during classification

### Example
```sh
// Capture every 1000000ns interval on device 0 with a threshold of 15%
./blink -i=1000000 -t=15 -d=0
```

## Acknowledgements
*'Blink'* is meant as a educational resource for OpenCV usage and is heavily commented.</br>
I apologize for any lengthy sections with overly verbose documentation.

A good portion of the comments for `SSIM()` & `PSNR()` refer to the formulas, most if not all of which can be found in the two Wikipedia pages below:
* [SSIM Wikipedia page](https://en.wikipedia.org/wiki/Structural_similarity)
* [PSNR Wikipedia page](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)

To learn how to calculate SSIM in a timely and efficient manner with the matrices in OpenCV as well as use the cascade classifiers provided, I referenced the pages below: 
* [OpenCV Documentation on SSIM & PSNR](https://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html)
* [OpenCV Documentation on forEach() and matrix iteration](https://docs.opencv.org/4.0.1/d3/d63/classcv_1_1Mat.html#a952ef1a85d70a510240cb645a90efc0d)
* [OpenCV Documentation on cascade classifiers and functional usage](https://docs.opencv.org/4.0.1/db/d28/tutorial_cascade_classifier.html)
