CXX = g++
FLAGS = `pkg-config --cflags $(LIBS)`
LINK = `pkg-config --libs $(LIBS)`
LIBS = opencv4

driver: driver.o blink.o
	@echo -e "Linking 'Blink'..."
	$(CXX) $(FLAGS) -o driver driver.o blink.o $(LINK) -pthread

driver.o: driver.cpp blink.h
	@echo -e "Compiling driver..."
	$(CXX) $(FLAGS) -c -g driver.cpp

blink.o: blink.cpp blink.h
	@echo -e "Compiling library files..."
	$(CXX) $(FLAGS) -c -g blink.cpp

clean:
	@echo "Removing object files, executables, and capture.png files from directory..."
	rm -rf ./*.o ./driver ./*.png
