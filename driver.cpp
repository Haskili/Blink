#include "blink.h"

int main(int argc, char* argv[]) {

	// Retrieve and validate arguments
	CommandLineParser parser(argc, argv, args);
	if (parser.has("help")) {
		parser.printMessage();
		return EXIT_SUCCESS;
	}
	int devID = parser.get<int>("device");
	int blur = parser.get<int>("blur");
	int mode = parser.get<int>("mode");
	int tryRotate = parser.get<int>("rotate");
	double threshold = parser.get<double>("threshold") * 0.01;
	double scale = (parser.get<double>("scale"));
	long interval = stol(parser.get<string>("interval"), 0, 10);
	string cascPath = parser.get<string>("classifier");

	// Setup timer 'ts' for interval between captures
	// e.g Default: 500000000ns -> 500ms -> 0.5s
	struct timespec ts = {ts.tv_sec = 0, ts.tv_nsec = interval};
	while (ts.tv_nsec >= 1000000000L)
		ts.tv_sec++, ts.tv_nsec -= 1000000000L;

	// Open the camera and capture a single image 
	// after performing error checking on the VideoCapture
	Mat idIMG, curr, prev;
	VideoCapture cap;
	if (!cap.open(devID)) {
		fprintf(stderr, "ERR isOpened() failed opening '%i'\n", devID);
		return EXIT_FAILURE;
	}
	if (!cap.set(cv::CAP_PROP_BUFFERSIZE, 1)) {
		fprintf(stderr, "ERR set() operation not supported by backend\n");
		return EXIT_FAILURE;
	}
	cap >> prev;

	// Load the classifier for identifying object(s) in frame
	CascadeClassifier cascade;
	if (!cascade.load(cascPath)) {
		fprintf(stderr, "ERR load() failed loading '%s'\n", cascPath);
		return EXIT_FAILURE;
	}

	// Enter main loop for capturing and reading
	for (int cnum = 0;;) {
		
		// Wait and then capture an image from the device
		nanosleep(&ts, NULL);
		cap >> curr;
		if (curr.empty()) {
			fprintf(stderr, "ERR main() empty capture\n");
			return EXIT_FAILURE;
		}

		// Check for invalid dimensions before comparing
		if (curr.cols != prev.cols || curr.rows != prev.rows) {
			fprintf(stderr, "ERR main() bad capture dims\n");
			return EXIT_FAILURE;
		}

		// Check if the difference is significant enough
		// to warrant a thorough difference calculation
		if (PSNR(curr, prev) > 45.0)
			continue;

		// Calculate the difference using specified method
		double diff = mode? SSIM(curr, prev):FTPD(curr, prev, 0.025f);

		// Check if the difference percentage between the captures 
		// excedes the threshold for 'different images'
		if (diff >= threshold) {

			// Log the event and check for identifiable objects
			time_t ctime;
			time(&ctime);
			fprintf(stdout, "\n%sEvent #%i -- %4.2f%% difference\n",
				asctime(localtime(&ctime)), cnum, diff*100);

			idIMG = curr.clone();
			fprintf(stdout, "Identified '%i' object(s)...\n\n",
				detectObj(idIMG, cascade, scale, tryRotate, blur));

			// Write the capture to a seperately saved image and
			// reset 'prev' for the next comparison
			char nameBuff[50];
			sprintf(nameBuff, "capture-%i.png", cnum++);
			imwrite(nameBuff, idIMG);
			prev = curr.clone();
		}
	}
	return EXIT_SUCCESS;
}