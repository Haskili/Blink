// C1 = [((2^8)-1)*0.01]^2 = 6.5025
// C2 = [((2^8)-1)*0.03]^2 = 58.5225 
#define C1 6.5025
#define C2 58.5225

// Detector Types
#define DT_SSD   0
#define DT_HCC   1
#define DT_NONE -1

// Difference measurement types
#define DIFF_SSIM 0
#define DIFF_FTPD 1

// OCV Main dependencies
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// OCV Video dependencies
#include <opencv2/videoio.hpp>

// OCV HR dependencies
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>

// Basic dependencies
#include <ctime>
#include <thread>
#include <sstream>
#include <fstream>

using namespace cv;
using std::thread;
using std::string;
using std::stringstream;
using std::vector;

// Define functions
int fdWait(int seconds, int microseconds);
int detectObjHCC(Mat& inIMG, Mat& outIMG, CascadeClassifier& cascade,
				double scale, int rotOpt, int minN, int blurOpt);

int detectObjSSD(vector<string> classNames, int CLSize, int hID, 
				Mat& inIMG, Mat& outIMG, dnn::Net& net);

double FTPD(Mat& A, Mat& B, float thrsh);
double PSNR(Mat& A, Mat& B);
double SSIM(Mat& A, Mat& B);

float thrshCalibrate(VideoCapture &cap, int iter, double tolerance);
float thrshCalibrate(Mat& A, Mat& B, int iter, double tolerance);

void detectCalibrate(VideoCapture cap, CascadeClassifier cascade);
void deviceProcNA(int devID, int mode, double threshold, 
					double ftpdThresh, struct timespec ts);

void deviceProcHCC(int devID, int blur, int mode, int rotOpt, int minN,
					double scale, double threshold, double ftpdThresh,
					struct timespec ts, CascadeClassifier cascade);

void deviceProcSSD(int devID, int mode,  double threshold, 
					double ftpdThresh, int hID, struct timespec ts, 
					dnn::Net net, vector<string> classesSSD);

static void calibrationTrackbar(int e, void* data);

// Define valid arguments
static const string args = "{help h || Prints this help message}"
	"{device d 		|0| Get video input from a specific device}"
	"{interval i 	|500000000| Set interval (ns) between capturing frames}"
	"{method m 		|1| Set the image comparison method (0: FTPD -- 1: SSIM)}"
	"{threshold t 	|15| Set the percent difference threshold for 'events')}"
	"{type n		|0| Set the detection method (-1: NONE -- 0: SSD -- 1: HCC)}"
	"{humanLID p 	|0| Specify the index of 'human' in classes given for SSD}"
	"{classifier c 	|INVALID| Set file(s) used in detection}"
	"{blur b 		|0| Specify whether to blur identified objects}"
	"{rotate r 		|0| Specify whether to try rotation during detection}"
	"{scale s 		|1.1| Set the image scaling to use}"
	"{fthresh f 	|-1| Set the threshold for FTPD}"
	"{neighbours a 	|2| Set the minumum amount of neighbours for HCC detection}";