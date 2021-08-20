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
    int comp = parser.get<int>("compression");
    long int interval = stol(parser.get<string>("interval"), 0, 10);
    double threshold = parser.get<double>("threshold") * 0.01;
    double scale = (parser.get<double>("scale"));
    float ftpdThresh = parser.get<float>("fthresh") * 0.01f;
    string filePath = parser.get<string>("classifier");
    string dlStr = parser.get<string>("device");

    // Setup timer 'ts' for interval between captures
    // e.g Default: 500000000ns -> 500ms -> 0.5s
    struct timespec ts = {ts.tv_sec = 0, ts.tv_nsec = interval};
    while (ts.tv_nsec >= 1000000000L)
        ts.tv_sec++, ts.tv_nsec -= 1000000000L;

    // Parse the list of devices from args into device list 
    stringstream dlStream(dlStr);
    vector<string> devices;
    for (string tkn; getline(dlStream, tkn, ',');)
        devices.push_back(tkn);

    // Based on the arguments, check if we need to load in 
    // the files for a Single Shot Detector
    dnn::Net net;
    CascadeClassifier cascade;
    vector<string> SSDLabels;
    if (detectionType == DT_SSD || detectionType == DT_YOLO) {

        // Seperate out the input into two distinct file paths
        stringstream ifStream(filePath);
        string filePrimary, fileSecondary, fileTeritary;
        getline(ifStream, filePrimary, ',');
        getline(ifStream, fileSecondary, ',');
        getline(ifStream, fileTeritary, ',');

        // Read in the network-related files
        // NOTE:
        //          e.g.    .prototxt    .caffemodel
        net = dnn::readNet(filePrimary, fileSecondary);

        // Take in labels from given file
        std::ifstream lfile(fileTeritary);
        for (string tkn; getline(lfile, tkn); SSDLabels.push_back(tkn));
    }

    // Else if it's specifed to HCC, we need to load the cascade file
    else if (detectionType == DT_HCC) {
        if (!cascade.load(filePath)) {
            CV_Error_(STSE, ("ERR load() failed for '%s'\n", filePath));
            return EXIT_FAILURE;
        }
    }

    // Launch threads for each [0, n] devices the user specified
    thread threads[(int)devices.size()];
    for (int i = 0; i < (int)devices.size(); i++) {
        threads[i] = thread(deviceThread, i, blur, mode, tryRotate, minN, 
                            detectionType, hLID, comp, ftpdThresh, scale, 
                            threshold, ts, cascade, net, SSDLabels, devices[i]);
    }

    // Join each device thread and then return
    for (int i = 0; i < (int)devices.size(); i++)
        threads[i].join();

    return EXIT_SUCCESS;
}