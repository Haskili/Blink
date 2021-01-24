// C1 = [((2^8)-1)*0.01]^2 = 6.5025
// C2 = [((2^8)-1)*0.03]^2 = 58.5225 
#define C1 6.5025
#define C2 58.5225

// OCV Main dependencies
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// OCV Video dependencies
#include <opencv2/videoio.hpp>

// OCV HR dependencies
#include <opencv2/objdetect.hpp>

// Basic dependencies
#include <ctime>

using namespace cv;
using std::string;
using std::vector;

// Define functions
int startup(int argc, char* argv[]);
int fdWait(int seconds, int microseconds);
int detectObj(Mat& img, CascadeClassifier& cascade,
			double scale, int rotOpt, int blurOpt);

double FTPD(Mat& A, Mat& B, float thrsh);
double PSNR(Mat& A, Mat& B);
double SSIM(Mat& A, Mat& B);
float thrshCalibrate(VideoCapture &cap, int iter, double tolerance);
float thrshCalibrate(Mat& A, Mat& B, int iter, double tolerance);

// Define valid arguments
static const string args = "{help h || Prints this help message}"
	"{scale s |0.5| Set the image scaling to use}"
	"{device d |0| Get video input from a specific device}"
	"{blur b |0| Specify whether to blur identified objects}"
	"{threshold t |15| Set the percent difference threshold for 'events')}"
	"{ftpd |-1| Set the percent difference threshold for FTPD)}"
	"{mode m |0| Set the comparison method (0: FTPD -- 1: SSIM)}"
	"{classifier c |./hc_ff.xml| Set classifier used in detection}"
	"{rotate r |0| Specify whether to try rotation during detection}"
	"{interval i |500000000| Set interval (ns) between capturing frames}";