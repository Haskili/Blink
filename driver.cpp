#include "blink.h"

int main(int argc, char* argv[]) {

	// Retrieve and validate arguments
	CommandLineParser parser(argc, argv, args);
	if (parser.has("help")) {
		parser.printMessage();
		return EXIT_SUCCESS;
	}

	// Get value of each argument available in 'parser'
	int blur = parser.get<int>("blur");
	int mode = parser.get<int>("method");
	int minN = parser.get<int>("neighbours");
	int tryRotate = parser.get<int>("rotate");
	int detectionType = parser.get<int>("type");
	int hLID = parser.get<int>("humanLID");
	long int interval = stol(parser.get<string>("interval"), 0, 10);
	double threshold = parser.get<double>("threshold") * 0.01;
	double scale = (parser.get<double>("scale"));
	float ftpdThresh = parser.get<float>("fthresh") * 0.01f;
	string filePath = parser.get<string>("classifier");
	string dlStr = parser.get<string>("device");
	string ifStr = parser.get<string>("type");

	// Setup timer 'ts' for interval between captures
	// e.g Default: 500000000ns -> 500ms -> 0.5s
	struct timespec ts = {ts.tv_sec = 0, ts.tv_nsec = interval};
	while (ts.tv_nsec >= 1000000000L)
		ts.tv_sec++, ts.tv_nsec -= 1000000000L;

	// Parse the list of devices from args into 'devices[]', 
	// continually allocating more space as needed
	stringstream dlStream(dlStr);
	int dlCount = 0;
	int* devices = (int*)malloc(1*sizeof(int));
	for (string tkn; getline(dlStream, tkn, ',');) {
		devices = (int*)realloc(devices, (dlCount+1)*sizeof(int));
		devices[dlCount++] = stoi(tkn);
	}

	// Based on the arguments, load in the files for a HCC 
	// or a SSD or output an error for bad args
	dnn::Net net;
	CascadeClassifier cascade;
	vector<string> SSDLabels;
	if (detectionType == DT_SSD) {

		// Seperate out the input into two distinct file paths
		stringstream ifStream(filePath);
		string filePrimary, fileSecondary, fileTeritary;
		getline(ifStream, filePrimary, ',');
		getline(ifStream, fileSecondary, ',');
		getline(ifStream, fileTeritary, ',');

		// Read in the network-related files
		// NOTE:
		//			e.g.    .prototxt    .caffemodel
		net = dnn::readNet(filePrimary, fileSecondary);

		// Take in labels from given file
		std::ifstream lfile(fileTeritary);
		for (string tkn; getline(lfile, tkn); SSDLabels.push_back(tkn));
	}

	else if (detectionType == DT_HCC) {
		if (!cascade.load(filePath)) {
			fprintf(stderr, "ERR load() failed loading '%s'\n", filePath);
			return EXIT_FAILURE;
		}
	}

	// Launch detached threads for each [1..n] cameras based 
	// on what type of detector the user specified in the
	// argument by calling the threads with deviceProc(...)
	for (int i = 1; i < dlCount; i++) {
		thread thd;

		if (detectionType == DT_SSD)
			thd = thread(deviceProcSSD, devices[i], mode, threshold, 
								ftpdThresh, hLID, ts, net, SSDLabels);

		else
			thd = thread(deviceProcHCC, devices[i], blur, mode, tryRotate,
						minN, scale, threshold, ftpdThresh, ts, cascade);

		thd.detach();
	}

	// Launch the primary camera from the parent thread
	// using the correct deviceProc(...)
	if (detectionType == DT_SSD)
		deviceProcSSD(devices[0], mode, threshold, ftpdThresh, 
						hLID, ts, net, SSDLabels);

	else
		deviceProcHCC(devices[0], blur, mode, tryRotate, minN,
						scale, threshold, ftpdThresh, ts, cascade);

	// In the case that the primary device has returned,
	// return from main() with bad status
	free(devices);
	return EXIT_FAILURE;
}