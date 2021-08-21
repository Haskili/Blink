CXX = g++
FLAGS = `pkg-config --cflags $(LIBS)`
LINK = `pkg-config --libs $(LIBS)`
LIBS = opencv4

SSD_FILES = -c=<Configuration>,<Model>,<Labels>
YOLO_FILES = -c=<Configuration>,<Model>,<Labels>
HCC_FILES = -c=<Classifier>

DET_NONE = -n=-1
DET_SSD = -n=0 $(SSD_FILES)
DET_HCC = -n=1 $(HCC_FILES)
DET_YOLO = -n=2 $(YOLO_FILES)

DEVICE = ...
MISC = ...

driver: driver.o blink.o
	@echo "Linking 'Blink'..."
	$(CXX) $(FLAGS) -o driver driver.o blink.o $(LINK) -pthread
	@mkdir -p captures

driver.o: driver.cpp blink.h
	@echo "Compiling driver..."
	$(CXX) $(FLAGS) -c -g driver.cpp

blink.o: blink.cpp blink.h
	@echo "Compiling library files..."
	$(CXX) $(FLAGS) -c -g blink.cpp

recording:
	@echo "Starting video-generation script"
	@echo -e "--------------------------------\n"
	@./makeRecording.sh

perftest:
	@echo "Initiating performance test..."
	@rm -f resources.log
	@(./driver $(DEVICE) $(DET_SSD) $(MISC)) &
	@sleep 1
	@for i in {1..60}; do (ps --no-headers -C driver -o %cpu,%mem >> resources.log; sleep 1); done
	@pkill driver
	@python graph_usage.py
	@echo "Finished, please check 'resource_usage.png'"

clean:
	@echo "Removing object files, executables, and capture.png files from the directory..."
	rm -rf ./*.o ./driver ./captures/*.png