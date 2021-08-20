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

clean:
	@echo "Removing object files, executables, and capture.png files from the directory..."
	rm -rf ./*.o ./driver ./captures/*.png