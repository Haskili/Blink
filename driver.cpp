#include "blink.h"

int main(int argc, char* argv[]) {

	// Retrieve and validate arguments
	CommandLineParser parser(argc, argv, args);
	if (parser.has("help")) {
		parser.printMessage();
		return EXIT_SUCCESS;
	}
	int blur = parser.get<int>("blur");
	int mode = parser.get<int>("mode");
	int minN = parser.get<int>("neighbours");
	int tryRotate = parser.get<int>("rotate");
	long int interval = stol(parser.get<string>("interval"), 0, 10);
	double threshold = parser.get<double>("threshold") * 0.01;
	double scale = (parser.get<double>("scale"));
	float ftpdThresh = parser.get<float>("ftpd") * 0.01f;
	string cascPath = parser.get<string>("classifier");
	string dlStr = parser.get<string>("device");

	// Setup timer 'ts' for interval between captures
	// e.g Default: 500000000ns -> 500ms -> 0.5s
	struct timespec ts = {ts.tv_sec = 0, ts.tv_nsec = interval};
	while (ts.tv_nsec >= 1000000000L)
		ts.tv_sec++, ts.tv_nsec -= 1000000000L;

	// Load the classifier for identifying object(s) in frame
	CascadeClassifier cascade;
	if (!cascade.load(cascPath)) {
		fprintf(stderr, "ERR load() failed loading '%s'\n", cascPath);
		return EXIT_FAILURE;
	}

	// Parse the list of devices from args into 'devices[]', 
	// continually allocating more space as needed
	stringstream dlStream(dlStr);
	int dlCount = 0;
	int* devices = (int*)malloc(1*sizeof(int));
	for (string tkn; getline(dlStream, tkn, ',');) {
		devices = (int*)realloc(devices, (dlCount+1)*sizeof(int));
		devices[dlCount++] = stoi(tkn);
	}

	// Create detached threads for all devices except for 
	// the first, which is called from parent thread as deviceProc(...)
	for (int i = 1; i < dlCount; i++) {
		thread thd(deviceProc, devices[i], blur, mode, tryRotate, minN, 
							scale, threshold, ftpdThresh, ts, cascade);

		thd.detach();
	}
	deviceProc(devices[0], blur, mode, tryRotate, minN,
				scale, threshold, ftpdThresh, ts, cascade);

	// In the case that the primary device has returned,
	// return from main() with bad status
	free(devices);
	return EXIT_FAILURE;
}