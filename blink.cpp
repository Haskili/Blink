#include "blink.h"

/*
    deviceProc() is the function called as a detached thread
    where each device is handled as it's own "Blink" process;
    Meaning that devices and deviceProc() threads are 1:1

    The general idea is that each thread will be able to write
    to it's own set of uniquely named images & etc. using it's 
    device number as part of the name to differentiate itself

    This allows it to avoid certain messy I/O bounded wait times
    it would have with other implementations
*/
void deviceProc(int devID, int blur, int mode, int rotOpt, int minN, 
                int detectionType, int hID, struct timespec ts, 
                double scale, double threshold, double ftpdThresh,
                CascadeClassifier cascade, dnn::Net net, 
                vector<string> classesSSD) {

    // Set the SIGINT signal handler so the thread
    // can finish properly when interrupted
    signal(SIGINT, [](int signalNumber) {
        signalFlag = true;
        printf("\nINTERRUPT SIGNAL CAUGHT\n");
    });

    // Open the device specified, check for issues,
    // and then capture a single image from the device
    Mat idIMG, cIMG, pIMG;
    VideoCapture cap;
    if (!cap.open(devID, CAP_V4L2))
        CV_Error_(STSE, ("ERR open() failed on device '%i'\n", devID));

    if (!cap.set(CAP_PROP_BUFFERSIZE, 1))
        CV_Error_(STSE, ("ERR set() op not supported by '%s' API\n",
                        cap.getBackendName()));

    cap >> pIMG;

    // If the FTPD is being used and the threshold wasn't
    // specified then calculate it now with thrshCalibrate()
    if (mode == 0 && ftpdThresh < (float)0) {

        // Get threshold and check for bad return
        ftpdThresh = thrshCalibrate(cap, 300, 0.0000);
        if (ftpdThresh == EXIT_FAILURE)
            CV_Error_(STSE, ("ERR thrshCalibrate() empty capture\n"));
    }

    // Create the logfile before entering the primary loop 
    // for capturing from the device
    char buffer[50];
    sprintf(buffer, "logfile_%i.json", devID);
    FileStorage lf(buffer, FileStorage::WRITE
                         | FileStorage::APPEND 
                         | FileStorage::FORMAT_JSON);

    if (!lf.isOpened())
        CV_Error_(STSE, ("ERR isOpened() failed on '%s'\n", buffer));

    // Enter primary loop for capturing and reading from the device
    int event = 1, detections = 0, lblSz = (int)classesSSD.size();
    while (!signalFlag) {

        // Wait for a 'ts' time period and then capture an 
        // image from the device
        nanosleep(&ts, NULL);
        cap >> cIMG;
        if (cIMG.empty())
            CV_Error_(STSE, ("ERR empty() capture inside loop\n"));

        // Check if the PSNR value is significant enough
        // to warrant a thorough difference calculation
        if (PSNR(cIMG, pIMG) > 45.0)
        	continue;

        // Calculate the difference using specified method
        double diff = mode? SSIM(cIMG, pIMG):FTPD(cIMG, pIMG, ftpdThresh);

        // Check if the difference percentage between the captures 
        // excedes the threshold for 'different images'
        if (diff >= threshold) {

            // Get the current time and put it in 'buffer'
            time_t ctime = time(nullptr);
            strftime(buffer, sizeof(buffer), "%F_%T", 
                        std::localtime(&ctime));

            // If the detector type is set to Single Shot Detector,
            // use the SSD to detect objects in the image
            if (detectionType == DT_SSD)
                detections = detectObjSSD(classesSSD, lblSz, hID, 
                                                cIMG, idIMG, net);

            // Else if the detector type is set to Haar-Cascade Classifier,
            // use the HCC to detect objects in the image
            else if (detectionType == DT_HCC)
                detections = detectObjHCC(cIMG, idIMG, cascade, scale,
                                                  rotOpt, minN, blur);

            // Log the event information
            lf << ("EVENT_" + std::to_string(event)) 
               << "{"
               << "TIME"       << buffer
               << "DEVICE"     << devID
               << "DETECTIONS" << detections
               << "DIFFERENCE" << std::to_string(diff*100.0) 
               << "}";

            // Write the captured frame to a seperately saved image and
            // reset 'pIMG' to 'cIMG' for the next comparison
            sprintf(buffer, "captures/DEV_%i-CAP_%i.png", devID, event++);
            imwrite(buffer, idIMG);
            pIMG = cIMG.clone();
        }
    }

    // Close the logfile and exit the thread
    lf.release();
    pthread_exit(NULL);
}

/*
    FTPD() takes in two images A & B, and returns the flat 
    difference % for all pixels using a given threshold for 
    what qualifies as enough BRG distance to be considered 
    different at any given pixel

    It works by looking through the absolute difference of  of A & B, 
    checking all pixels in the result for a BRG value meeting
    the 'thrsh' threshold to say that there's different pixels at 
    this point, and finally returning the percentage marked

    For more information on the forEach() usage, please see the 
    documentation at:
    
    "https://docs.opencv.org/4.0.1/d3/d63/classcv_1_1Mat.html"
*/
double FTPD(Mat& A, Mat& B, float thrsh) {

    // Setup the absolute difference of the two images, 'diffIMG'
    // diffIMG = |A - B|
    Mat diffIMG(A.rows, A.cols, CV_8UC3, Scalar(0,0,0));
    absdiff(A, B, diffIMG);

    // Create an empty matrix that will hold values at points where
    // the two images differ significantly enough
    Mat maskIMG = Mat::zeros(diffIMG.rows, diffIMG.cols, CV_8UC1);

    // Iterate through every pixel in 'diffIMG' for all pixels 'p' 
    // and see if the %diff at 'p' exceedes the threshold 'thrsh' to
    // toggle the pixel in the mask image
    typedef cv::Point3_<uint8_t> Pixel;
    diffIMG.forEach<Pixel>(
        [&](Pixel& p, const int pos[]) -> void {
            if (((float)p.x + (float)p.y + (float)p.z)/(float)765 > thrsh)
                maskIMG.at<unsigned char>(pos[0], pos[1]) = 1;
    });

    // Return the percentage of marked pixels in the mask
    return (double)countNonZero(maskIMG)/(double)(maskIMG.rows*maskIMG.cols);
}

/*
    PSNR() calculates the PSNR(dB) between two images A & B

    It will return lower values (~10) for less similar images 
    and higher values (~50) otherwise. The PSNR value is a
    relatively quick calculation compared to the SSIM() where
    a significantly higher amount of resources are required

    A function for calculating the ratio exists in OpenCV as
    cv::PSNR(), but it is something worth going over regardless

    References:
    "https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio"
    "double cv::PSNR(...)"
    "https://docs.opencv.org/2.4/doc/tutorials/highgui/
            video-input-psnr-ssim/video-input-psnr-ssim.html"
*/
double PSNR(Mat& A, Mat& B) { 
    
    // Get 'diffSQD', the abs. difference of A & B squared
    // diffSQD = (|A(x,y) - B(x,y)|)^2, for all pixels in the 
    // difference image 'diffSQD'
    Mat diffSQD;
    absdiff(A, B, diffSQD);
    diffSQD.convertTo(diffSQD, CV_32F);
    diffSQD = diffSQD.mul(diffSQD);

    // Get the sum of all channels to calculate the SSE of the two images
    Scalar total = sum(diffSQD);
    double SSE;
    for (int i = 0; i < A.channels(); SSE += total.val[i], i++);

    // Calculate MSE and then return the PSNR as,
    // 20*log_10(MAX_I) - 10*log_10(MSE)
    double MSE = SSE/(double)(A.channels() * A.total());
    return (20.0*log10(255) - 10.0*log10(MSE));
}

/*
    SSIM() returns the (mean) structural similarity (SSIM) of two images

    This function gives the mean SSIM calculation across all channels
    for two images A & B. It's exceptionally time consuming compared to
    something like PSNR(), and should only be used after a call to
    PSNR() returns a value exceeding the threshold for 'different images'

    Any references to 'z' variables are often the product of that section's
    x & y, so please keep that in mind when considering the formulas for SSIM

    References:
    "https://en.wikipedia.org/wiki/Structural_similarity"
    "https://docs.opencv.org/2.4/doc/tutorials/highgui/
            video-input-psnr-ssim/video-input-psnr-ssim.html"
*/
double SSIM(Mat& A, Mat& B) {

    // Get converted matricies to use for calculations
    Mat x, y;
    A.convertTo(x, CV_32F);
    B.convertTo(y, CV_32F);

    // Get squared of X, Y, and then calculate Z matrix
    Mat xSQD = x.mul(x), ySQD = y.mul(y), z = x.mul(y);

    // Get averages of X and Y matricies
    Mat xAvg, yAvg;
    GaussianBlur(x, xAvg, Size(), 1.5);
    GaussianBlur(y, yAvg, Size(), 1.5);

    // Get the average-squared matricies
    Mat XAvgSQD = xAvg.mul(xAvg), YAvgSQD = yAvg.mul(yAvg);
    Mat ZAvgSQD = xAvg.mul(yAvg);

    // Calculate the sigma matricies by first taking the
    // average of the squared matrix and then subtracting
    // the respective average-squared matrix
    Mat sigX, sigY, sigZ;
    GaussianBlur(xSQD, sigX, Size(), 1.5);
    GaussianBlur(ySQD, sigY, Size(), 1.5);
    GaussianBlur(z, sigZ, Size(), 1.5);
    sigX -= XAvgSQD, sigY -= YAvgSQD, sigZ -= ZAvgSQD;

    // Calculate the numerator and denominator
    // (See definitions at top of header file for C1 & C2)
    Mat num, den;
    num = ZAvgSQD.mul(2) + Scalar(C1, C1, C1);
    den = XAvgSQD + YAvgSQD + Scalar(C1, C1, C1);

    num = num.mul(sigZ.mul(2) + Scalar(C2, C2, C2));
    den = den.mul(sigX + sigY + Scalar(C2, C2, C2));
    
    // Divide the two and find the Scalar mean() of that matrix
    Mat ret;
    divide(num, den, ret);
    Scalar s = mean(ret);

    // Return the mean SSIM for all channels
    return 1.0-((s[0]+s[1]+s[2])/3.0);
}

/*
    fdWait() is an alternative sleep method made using select()

    It uses select() to wait on a non-existant file-descriptor
    for a specified amount of time; it never sees activity on a
    file-descriptor that doesn't exist, which means we get to 
    wait the full amount of time specified in the arguments
*/
int fdWait(int seconds, int microseconds) {
    
    // Create and clear the fdset
    int s = -1;
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(s, &fds);

    // Create and set the time-values
    struct timeval tv = {tv.tv_sec = seconds, tv.tv_usec = microseconds};

    // Return (-1: Activity -- 0: No activity) on 's'
    return ((select(s+1, &fds, NULL, NULL, &tv)) > 0) ? -1 : 0;
}

/*
    detectObjHCC() finds objects within an image using a HCC; 
    It is based off the OpenCV example(s) for classification
    using Haar-like features

    It will try to find indentifiable objects using a cascade
    classifier, draw bounding boxes onto ROI's, label each ROI 
    with it's weight value, and then finally return how many 
    identifications it made for that image

    The biggest thing(s) that distinguish it from all of 
    the examples I've seen for similar tasks is that it 
    tries multiple rotations, labels each ROI uniquely,
    and it returns the identification amount
*/
int detectObjHCC(Mat& inIMG, Mat& outIMG, CascadeClassifier& cascade,
                    double scale, int rotOpt, int minN, int blurOpt) {

    // Perform pre-processing steps on 'procIMG'
    Mat procIMG;
    outIMG = inIMG.clone();
    cvtColor(outIMG, procIMG, COLOR_BGR2GRAY);
    resize(procIMG, procIMG, Size(), 
        (double)(1/scale), (double)(1/scale), 5);

    equalizeHist(procIMG, procIMG);

    // Try to detect any objects in the image and put
    // anything we found into 'regions'
    vector<Rect> regions;
    vector<double> weights;
    vector<int> levels;
    cascade.detectMultiScale(procIMG, regions, levels, weights, scale, minN,
                            CASCADE_DO_CANNY_PRUNING, Size(), Size(), true);

    // Try rotating image different ways and searching 
    // for objects in each different orientation if specified
    if (rotOpt) {
        vector<Rect> rx;
        vector<double> wx;

        // 90 rotation
        transpose(procIMG, procIMG);
        cascade.detectMultiScale(procIMG, regions, levels, weights, scale, minN,
                                CASCADE_DO_CANNY_PRUNING, Size(), Size(), true);

        for (size_t i = 0; i < rx.size(); i++)
            regions.push_back(rx[i]), weights.push_back(wx[i]);

        // 270 rotation
        flip(procIMG, procIMG, 1);
        cascade.detectMultiScale(procIMG, regions, levels, weights, scale, minN,
                                CASCADE_DO_CANNY_PRUNING, Size(), Size(), true);

        for (size_t i = 0; i < rx.size(); i++)
            regions.push_back(rx[i]), weights.push_back(wx[i]);
    }

    // For every ROI (identified object)
    for (size_t i = 0; i < regions.size(); i++) {
        Rect ROI = regions[i];

        // Get dimensions for current ROI rectangle
        int LX, LY, RX, RY;
        LX = ROI.x*scale, RX = (ROI.x + ROI.width)*scale;
        LY = ROI.y*scale, RY = (ROI.y + ROI.height)*scale;

        // Blur if specified to do so
        if (blurOpt) {
            Mat maskIMG = outIMG(Range(LY, RY), Range(LX, RX));
            blur(maskIMG, maskIMG, Size(20, 20), Point(-1,-1));
        }
        
        // Draw bounding rectangle to show ROI
        rectangle(outIMG, Point(RX, RY), Point(LX, LY),
                Scalar(0,255,0), 2, 8, 0);

        // Draw text to indicate identification number
        // and associated weight-value onto the ROI
        char buf[50];
        sprintf(buf, "ID #%i: %4.2f", i+1, weights[i]);
        putText(outIMG, buf, Point(LX + 5, LY - 5), 3, 0.5, 
            Scalar(0,255,0), 0.15*scale, 8, false);
    }
    return (int)regions.size();
}

/*
    thrshCalibrate() returns the calculated threshold value for FTPD()
    that gets it close to SSIM()'s returned value for the input given

    It does this by slowly changing the threshold for a certain number of
    iterations or until it reaches the tolerance for inaccuracy;
    each iteration it compares what the FTPD value is to the SSIM value
    for that iteration's threshold estimate, and then modifying the
    estimate accordingly
*/
float thrshCalibrate(VideoCapture &cap, int iter, double tolerance) {

    // Setup the structures involved and then initialize the estimated
    // threshold at 0.025 (2.5%) as a rough guess to start with
    Mat cIMG, pIMG;
    double curFTPD = 0.0, curSSIM = 0.0;
    float estThrsh = 0.025f;

    // Begin iteration loop
    cap >> pIMG;
    for (int i = 0; i < iter; i++) {

        // Get an image from the VideoCapture object and use
        // that for calculating new SSIM & FTPD values
        cap >> cIMG;
        if (cIMG.empty())
            return EXIT_FAILURE;
        
        curSSIM = SSIM(cIMG, pIMG);
        curFTPD = FTPD(cIMG, pIMG, estThrsh);

        // If we reached the tolerated level of inaccuracy,
        // return value for estimated threshold
        if (abs((curSSIM-curFTPD)/curSSIM) <= tolerance)
            return estThrsh;

        // Else we need to change the threshold to be either more or
        // less sensitive using the accuracy of our FTPD measurement
        // and the current threshold estimate
        float altValue = estThrsh * (float)(abs((curSSIM-curFTPD)/curSSIM));

        // If the threshold value made it not sensitive enough...
        if (curFTPD < curSSIM)
            estThrsh -= altValue;

        // Else if the threshold value made it too sensitive...
        else if (curFTPD > curSSIM)
            estThrsh += altValue;

        pIMG = cIMG.clone();

    }

    return estThrsh;
}

/*
    thrshCalibrate() returns the calculated threshold value for FTPD()
    that gets it close to SSIM()'s returned value for the input given
    
    This function differs from the previous only in that it takes in 
    two images A & B rather than a VideoCapture stream to continually 
    take images from; meaning that while it may be quicker, it is 
    significantly less accurate over the long run
*/
float thrshCalibrate(Mat& A, Mat& B, int iter, double tolerance) {

    // Setup the structures involved and then initialize the estimated
    // threshold at 0.025 (2.5%) as a rough guess to start with
    double curFTPD = 0.0;
    double curSSIM = SSIM(A, B);
    float estThrsh = 0.025f;

    // Begin capture and value-setting loop
    // for [0, iter] iterations
    for (int i = 0; i < iter; i++) {
        
        // Get the current FTPD with the updated threshold
        curFTPD = FTPD(A, B, estThrsh);

        // If we reached the tolerated level of inaccuracy,
        // return value for estimated threshold
        if (abs((curSSIM-curFTPD)/curSSIM) <= tolerance)
            return estThrsh;

        // Else we need to change the threshold to be either more or
        // less sensitive using the accuracy of our FTPD measurement
        // and the current threshold estimate
        float altValue = estThrsh * (float)(abs((curSSIM-curFTPD)/curSSIM));

        // If the threshold value made it not sensitive enough...
        if (curFTPD < curSSIM)
            estThrsh -= altValue;

        // Else if the threshold value made it too sensitive...
        else if (curFTPD > curSSIM)
            estThrsh += altValue;
    }

    return estThrsh;
}

/*
    The detectCalibrate() function is a basic GUI implementation
    for allowing the user to calibrate the variables used in the
    detection process of detectObjHCC()

    At current state, it will not give back the value(s) in any way,
    but will allow one to calibrate the values manually and record
    the values used themselves
*/
static void calibrationTrackbar(int e, void* data) {
    // Empty function for detection calibration function 
    // to use; left open for future developement
}

void detectCalibrate(VideoCapture cap, CascadeClassifier cascade) {
    
    // Create an image and a window for the trackbars to lie on
    Mat img;
    namedWindow("Calibrate", WINDOW_AUTOSIZE);
    int scale = 1, minN = 1;

    // Create two trackbars, one for minimum neighbours and
    // one for how scaled the image is during detection
    createTrackbar("Minimum Neighbours", "Calibrate", 
                    &minN, 10, calibrationTrackbar, &img);
    
    createTrackbar("Image Scaling", "Calibrate", 
                    &scale, 10, calibrationTrackbar, &img);

    // While the user is selecting the values desired, continually
    // show the video from stream and await input
    while(1) {
        cap >> img;
        detectObjHCC(img, img, cascade, 1.0 +((double)scale/10.0), minN, 0, 0);
        imshow("Result", img);
        char key = (char)waitKey(1);
        if (key == 'q' || key == 27)
            break;
    }
}

/*
    detectObjSSD() is used to determine what type of things are 
    in an image given and where those things might be using a 
    Single-Shot-Detector (SSD). It also performs a small but 
    powerful algorithim to determine and infer distances of
    detected people to see if they are far enough apart.

    The SSD is significantly more efficient and accurate in 
    certain scenarios than the Haar-Cascade Classifier used 
    in the other detection function. However, there are 
    multiple pitfalls, and it does not fit all use cases.

    It is advised that the user try both methods to see what 
    works best for their particular use case(s).
*/
int detectObjSSD(vector<string> classNames, int CLSize, int hID, 
                        Mat& inIMG, Mat& outIMG, dnn::Net& net) {

    // Resize the input image, 'inIMG'
    Mat scaledIMG;
    outIMG = inIMG.clone();
    resize(inIMG, scaledIMG, Size(300, 300), 0, 0, INTER_CUBIC);

    // Get a blob from the resized input and set
    // the network input to the returned blob
    Mat blob = dnn::blobFromImage(scaledIMG, 0.007843, Size(300, 300), 
                                  Scalar(127.5, 127.5, 127.5), false);

    net.setInput(blob);

    // Get prediction(s) of the network
    //
    // NOTE: The 'detections' is a 32FC1 with 4 total dimensions,
    //       requiring another matrix 'detTF' ('detections 3D/4D')
    //       to access [0, 0, i, 1..6] in a semi-reasonable manner
    //
    //       There's a mutex lock on dnn::forward() because 
    //       it isn't thread-safe
    //
    m.lock();
    Mat detections = net.forward();
    Mat detTF(detections.size[2], detections.size[3], 
              CV_32F, detections.ptr<float>());

    m.unlock();

    // For all POSSIBLE detections...
    int detectHits = 0;
    for (int i = 0; i < detections.size[2]; i++) {

        // Get the confidence value and 
        // if it's above the threshold...
        float conf = *detTF.ptr<float>(i,2);
        if (conf >= 0.40f) {

            // Check class index of identification for valid entry
            int classIndex = (int)(*detTF.ptr<float>(i,1));
            if (classIndex >= CLSize)
                continue;

            // Get coordinates of source identification
            // (needs to be rescaled using inIMG size)
            int s_xLB = (int)(*detTF.ptr<float>(i,3) * inIMG.cols);
            int s_xRT = (int)(*detTF.ptr<float>(i,5) * inIMG.cols);
            int s_yLB = (int)(*detTF.ptr<float>(i,4) * inIMG.rows);
            int s_yRT = (int)(*detTF.ptr<float>(i,6) * inIMG.rows);

            // Draw ROI and associated information onto the image
            rectangle(outIMG, Point(s_xRT, s_yRT), Point(s_xLB, s_yLB), 
                        Scalar(0,255,0), 2, 8, 0);

            char buf[50];
            sprintf(buf, "ID#%i %s: %f%%", ++detectHits, 
                        classNames[classIndex].c_str(), conf);

            putText(outIMG, buf, Point(s_xLB+5, s_yLB-5), 
                        3, 0.5, Scalar(0,255,0));

            // Check for distance violations if the current 
            // identification is a human
            if (classIndex != hID)
                continue;

            // For every other ("target") identification...
            for (int j = 0; j < detections.size[2]; j++) {
                
                // Check for target confidence
                if (*detTF.ptr<float>(j,2) < 0.40f)
                    continue;

                // Check that we're going TO a TARGET that's human
                // and not the SOURCE we're looking FROM
                if (j == i || (int)(*detTF.ptr<float>(j,1)) != hID)
                    continue;

                // Get coordinates of identification
                int t_xLB = (int)(*detTF.ptr<float>(j,3) * inIMG.cols);
                int t_xRT = (int)(*detTF.ptr<float>(j,5) * inIMG.cols);
                int t_yLB = (int)(*detTF.ptr<float>(j,4) * inIMG.rows);
                int t_yRT = (int)(*detTF.ptr<float>(j,6) * inIMG.rows);

                // Get the target ROI and source ROI coordinates 
                // as seperate coordinate points
                Point target = Point(((t_xLB+t_xRT)/2), ((t_yLB+t_yRT)/2));
                Point source = Point(((s_xLB+s_xRT)/2), ((s_yLB+s_yRT)/2));

                // Get distance in Y-AXIS & X-AXIS of the two ROI's 
                // from eachother
                int yDistance = abs(((t_yLB+t_yRT)/2) - ((s_yLB+s_yRT)/2));
                int xDistance = abs(((t_xLB+t_xRT)/2) - ((s_xLB+s_xRT)/2));
                
                // Get distance that uses each ROI's relative height 
                // averaged to determine the "minimum safe distance"
                //
                // NOTE:   Height_Relative = (Top Y - Bottom Y)/2
                //       Distance_Required = (HR_Source + HR_Target)/2  
                //
                int distanceReq = ((t_yRT - t_yLB) + (s_yRT - s_yLB))/2;

                // Check if percent difference between height/width 
                // of ROI's is >= 60%, as high values COULD indicate 
                // a large difference in Z-axis
                double PDY = 100.0*((double)abs((t_yRT-t_yLB)-(s_yRT-s_yLB))
                                    /(double)(distanceReq));

                double PDX = 100.0*((double)abs((t_xRT-t_xLB)-(s_xRT-s_xLB))
                                    /(double)(distanceReq));

                if (PDY >= 60.0 || PDX >= 60.0)
                    continue;

                // NOTE: The below condition(s) work well for 
                //       forward facing views and best when 
                //       looking from above at a downward angle
                //
                if (xDistance < distanceReq && yDistance < distanceReq)
                    line(outIMG, target, source, Scalar(0, 0, 255), 2, 8);

                else
                    line(outIMG, target, source, Scalar(255, 0, 0), 2, 8);
            }
        }
    }
    return detectHits;
}