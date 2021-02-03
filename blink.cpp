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
void deviceProc(int devID, int blur, int mode, int tryRotate, 
				double scale, double threshold, double ftpdThresh,
				struct timespec ts, CascadeClassifier cascade) {

	// Open the camera and capture a single image
	Mat idIMG, curr, prev;
	VideoCapture cap;
	if (!cap.open(devID, CAP_V4L2)) {
		fprintf(stderr, "ERR open() failed on device '%i'\n", devID);
		return;
	}
	if (!cap.set(CAP_PROP_BUFFERSIZE, 1)) {
		fprintf(stderr, "ERR set() operation not supported by '%s' API\n",
				cap.getBackendName());

		return;
	}
	cap >> prev;

	// If the FTPD is being used and the threshold wasn't
	// specified, calculate it now with thrshCalibrate()
	if (mode == 0 && ftpdThresh < (float)0) {

		// Get threshold and check for bad return
		ftpdThresh = thrshCalibrate(cap, 300, 0.0000);
		if (ftpdThresh == EXIT_FAILURE) {
			fprintf(stderr, "ERR thrshCalibrate() empty capture\n");
			return;
		}
	}

	// Enter loop for capturing and reading from the device
	for (int cnum = 0;;) {

		// Wait for 'ts' and then capture an image from the device
		nanosleep(&ts, NULL);
		cap >> curr;
		if (curr.empty()) {
			fprintf(stderr, "ERR empty() capture inside loop\n");
			return;
		}

		// Check for non-matching dimensions before comparing
		if (curr.size != prev.size) {
			fprintf(stderr, "ERR deviceProc() bad capture dimensions\n");
			return;
		}

		// Check if the difference is significant enough
		// to warrant a thorough difference calculation
		if (PSNR(curr, prev) > 45.0)
			continue;

		// Calculate the difference using specified method
		double diff = mode? SSIM(curr, prev):FTPD(curr, prev, ftpdThresh);

		// Check if the difference percentage between the captures 
		// excedes the threshold for 'different images'
		if (diff >= threshold) {

			// Log the event and check for identifiable objects
			time_t ctime;
			time(&ctime);
			fprintf(stdout, "\nID: %i - %sEvent #%i -- %4.2f%% difference\n",
					devID, asctime(localtime(&ctime)), cnum, diff*100);

			idIMG = curr.clone();
			fprintf(stdout, "Identified '%i' object(s)...\n\n",
				detectObj(idIMG, cascade, scale, tryRotate, blur));

			// Write the capture to a seperately saved image and
			// reset 'prev' for the next comparison
			char nameBuff[50];
			sprintf(nameBuff, "device_%i-capture_%i.png", devID, cnum++);
			imwrite(nameBuff, idIMG);
			prev = curr.clone();
		}
	}
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
	// (See definitions at top of file for C1 & C2)
	Mat num = (2 * ZAvgSQD + C1), den = (XAvgSQD + YAvgSQD + C1);
	num = num.mul((2 * sigZ + C2));
	den = den.mul((sigX + sigY + C2));

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
	detectObj() finds objects within an image using a HCC; 
	it is based off the OpenCV example(s) for cascade classification

	It will try to find indentifiable objects using a cascade
	classifier, draw bounding boxes onto ROI's, label each ROI 
	with it's weight value, and then finally return how many 
	identifications it made for that image

	The biggest thing(s) that distinguish it from all of 
	the examples I've seen for similar tasks is that it 
	tries multiple rotations, labels each ROI uniquely,
	and it returns the identification amount
*/
int detectObj(Mat& img, CascadeClassifier& cascade,
			double scale, int rotOpt, int blurOpt) {

	// Perform pre-processing steps on 'procIMG'
	Mat procIMG;
	cvtColor(img, procIMG, COLOR_BGR2GRAY);
	resize(procIMG, procIMG, Size(), 
		(double)(1/scale), (double)(1/scale), 5);

	equalizeHist(procIMG, procIMG);

	// Try to detect any objects in the image and put
	// anything we found into 'regions'
	vector<Rect> regions;
	vector<double> weights;
	vector<int> levels;
	cascade.detectMultiScale(procIMG, regions, levels, weights, 
		1.1, 3, 0, Size(), Size(), true);

	// Try rotating image different ways and searching 
	// for objects in each different orientation if specified
	if (rotOpt) {
		vector<Rect> rx;
		vector<double> wx;

		// 90 rotation
		transpose(procIMG, procIMG);
		cascade.detectMultiScale(procIMG, rx, levels, wx, 
			1.1, 3, 0, Size(), Size(), true);

		for (size_t i = 0; i < rx.size(); i++)
			regions.push_back(rx[i]), weights.push_back(wx[i]);

		// 270 rotation
		flip(procIMG, procIMG, 1);
		cascade.detectMultiScale(procIMG, rx, levels, wx, 
			1.1, 3, 0, Size(), Size(), true);

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
			Mat maskIMG = img(Range(LY, RY), Range(LX, RX));
			blur(maskIMG, maskIMG, Size(20, 20), Point(-1,-1));
		}
		
		// Draw bounding rectangle to show ROI
		rectangle(img, Point(RX, RY), Point(LX, LY),
				Scalar(0,255,0), 2, 8, 0);

		// Draw text to indicate identification number
		// and associated weight-value onto the ROI
		char buf[50];
		sprintf(buf, "ID #%i: %4.2f", i+1, weights[i]);
		putText(img, buf, Point(LX + 5, LY - 5), 3, 0.5, 
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

	// Begin iteration loop
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