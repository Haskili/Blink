CXX = g++
FLAGS = `pkg-config --cflags $(LIBS)`
LINK = `pkg-config --libs $(LIBS)`
LIBS = opencv4

DEVICE = -d=/dev/video0
MISC = -m=0 -f=15.0 -p=15 -t=0.0

SSD_FILES = -c=<Configuration>,<Model>,<Labels>
YOLO_FILES = -c=<Configuration>,<Model>,<Labels>
HCC_FILES = -c=<Configuration>,<Model>,<Labels>

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

perftest-NONE:
	@echo "Initiating performance test of no detector"
	@rm -f resources-NONE.log
	@(./driver $(DEVICE) $(MISC) -n=-1) &
	@sleep 1
	@for i in {1..60}; do (ps --no-headers -C driver -o %cpu,%mem >> resources-NONE.log; sleep 1); done
	@pkill driver
	@python graph_usage.py NONE

perftest-SSD:
	@echo "Initiating performance test of Single-Shot-Detector"
	@rm -f resources-SSD.log
	@(./driver $(DEVICE) $(SSD_FILES) $(MISC) -n=0) &
	@sleep 1
	@for i in {1..60}; do (ps --no-headers -C driver -o %cpu,%mem >> resources-SSD.log; sleep 1); done
	@pkill driver
	@python graph_usage.py SSD

perftest-YOLO:
	@echo "Initiating performance test of You-Only-Look-Once detector"
	@rm -f resources-YOLO.log
	@(./driver $(DEVICE) $(YOLO_FILES) $(MISC) -n=2) &
	@sleep 1
	@for i in {1..60}; do (ps --no-headers -C driver -o %cpu,%mem >> resources-YOLO.log; sleep 1); done
	@pkill driver
	@python graph_usage.py YOLO

perftest-HCC:
	@echo "Initiating performance test of no Haar-Cascade Classifier"
	@rm -f resources-HCC.log
	@(./driver $(DEVICE) $(HCC_FILES) $(MISC) -n=1 -s=1.1 -a=1) &
	@sleep 1
	@for i in {1..60}; do (ps --no-headers -C driver -o %cpu,%mem >> resources-HCC.log; sleep 1); done
	@pkill driver
	@python graph_usage.py HCC

perftest-ALL: perftest-NONE perftest-SSD perftest-YOLO perftest-HCC

clean:
	@echo "Removing object files, executables, and capture.png files from the directory..."
	rm -rf ./*.o ./driver ./captures/*.png