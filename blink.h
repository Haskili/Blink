// C1 = [((2^8)-1)*0.01]^2 = 6.5025
// C2 = [((2^8)-1)*0.03]^2 = 58.5225 
#define C1 6.5025
#define C2 58.5225

// Detector Types
#define DT_SSD   0
#define DT_HCC   1
#define DT_YOLO  2
#define DT_NONE -1

// Define alias for OpenCV error code
#define STSE Error::StsError

// Define capture image compression factor
#define IM_COMPRESSION_FACTOR 1

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
#include <fstream>
#include <atomic>
#include <csignal>
#include <mutex>

using namespace cv;
using std::thread;
using std::string;
using std::stringstream;
using std::vector;

// CaptureEvent Definition
//
//     Event Number: Chronological number of the event
//            Input: The device number or path to the device
//       Event Time: Local time at occurance of event
//   Inference Time: Amount of time in milliseconds required for HCC/SSD
// Image Difference: The dissimilarity of the two frames measured
//
class CaptureEvent {

    public:
        CaptureEvent(int EN, string IN, char* ET, double IT, double DIFF):
                            	eventNumber(EN), input(IN), eventTime(ET), 
                            	inferenceTime(IT), imageDifference(DIFF) {}

        int eventNumber;
        string input;
        char* eventTime;
        double inferenceTime;
        double imageDifference;
};

// Function prototypes
int fdWait(int seconds, int microseconds);
int HCC_RAW(Mat& inIMG, Mat& outIMG, CascadeClassifier& cascade,
                double scale, int rotOpt, int minN, int blurOpt);

int SSD_RAW(vector<string> labels, int CLSize, int hID,
                Mat& inIMG, Mat& outIMG, dnn::Net& net);

int SSD_NMS(vector<string> labels, int CLSize, int hID,
                Mat& inIMG, Mat& outIMG, dnn::Net& net);

int YOLO_RAW(vector<string> labels, int CLSize, int hID,
                 Mat& inIMG, Mat& outIMG, dnn::Net& net, 
                             vector<string> outputNames);

int YOLO_NMS(vector<string> labels, int CLSize, int hID,
                 Mat& inIMG, Mat& outIMG, dnn::Net& net, 
                             vector<string> outputNames);

int NMSProcessing(vector<string> labels, int CLSize, int hID, 
                      Mat& inIMG, Mat& outIMG, dnn::Net& net,
                                     vector<int> &indicesNMS, 
                                     vector<int> &classLabel, 
                                      vector<Rect> &classROI, 
                              vector<float> &classConfidence);

double FTPD(Mat& A, Mat& B, float thrsh);
double PSNR(Mat& A, Mat& B);
double SSIM(Mat& A, Mat& B);

float thrshCalibrate(VideoCapture &cap, int iter, double tolerance);
float thrshCalibrate(Mat& A, Mat& B, int iter, double tolerance);

void deviceThread(int devNum, int blur, int mode, int rotOpt, int minN, 
                    int detectionType, int hID, int compressionFactor,
                    double ftpdThresh, double scale, double threshold, 
                    struct timespec ts, CascadeClassifier cascade, 
                    dnn::Net net, vector<string> labels, string devID);

void detectCalibrate(VideoCapture cap, CascadeClassifier cascade);

static void calibrationTrackbar(int e, void* data);
static void operator<<(FileStorage& lf, const CaptureEvent evt);

// Definition of valid arguments for program
static const string args = "{help h || Prints this help message}"
    "{device d      |0| Get video input from a specific device}"
    "{interval i    |500000000| Set interval (ns) between capturing frames}"
    "{method m      |1| Set the image comparison method (0: FTPD -- 1: SSIM)}"
    "{threshold t   |15| Set the percent difference threshold for 'events')}"
    "{type n        |0| Set the detection method (-1: NONE -- 0: SSD -- 1: HCC)}"
    "{humanLID p    |0| Specify the index of 'human' in classes given for SSD}"
    "{classifier c  |INVALID| Set file(s) used in detection}"
    "{blur b        |0| Specify whether to blur identified objects}"
    "{rotate r      |0| Specify whether to try rotation during detection}"
    "{scale s       |1.1| Set the image scaling to use}"
    "{fthresh f     |-1| Set the threshold for FTPD}"
    "{compression k |1| Set the compression level of captured PNG images}"
    "{neighbours a  |2| Set the minumum amount of neighbours for HCC detection}";

// Interupt flag for threads to abort
static std::atomic_bool signalFlag;

// Mutex lock for the non-thread-safe
// section of detection functions
static std::mutex m;