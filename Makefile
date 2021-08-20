CXX = g++
FLAGS = `pkg-config --cflags $(LIBS)`
LINK = `pkg-config --libs $(LIBS)`
LIBS = opencv4

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

perftest-SSD:
	@(./driver -d=/dev/video0 -n=0 -c=<Configuration>,<Model>,<Labels> -m=0 -f=15.0 -p=15) &
	@sleep 1
	@for i in {1..60}; do (ps --no-headers -C driver -o %cpu,%mem >> resources-SSD.log; sleep 1); done
	@pkill driver
	@python graphRU.py

perftest-HCC:
	@(./driver -d=/dev/video0 -n=1 -c=<Cascade File> -m=0 -f=15.0 -p=15 -s=1.1 -a=1) &
	@sleep 1
	@for i in {1..60}; do (ps --no-headers -C driver -o %cpu,%mem >> resources-HCC.log; sleep 1); done
	@pkill driver
	@python graphRU.py

perftest-NONE:
	@(./driver -d=/dev/video0 -n=-1 -m=0 -f=15.0 -p=15) &
	@sleep 1
	@for i in {1..60}; do (ps --no-headers -C driver -o %cpu,%mem >> resources-NONE.log; sleep 1); done
	@pkill driver
	@python graphRU.py

clean:
	@echo "Removing object files, executables, and capture.png files from the directory..."
	rm -rf ./*.o ./driver ./captures/*.png