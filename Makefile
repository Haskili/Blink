CXX = g++
FLAGS = `pkg-config --cflags $(LIBS)`
LINK = `pkg-config --libs $(LIBS)`
LIBS = opencv4

all: blink

blink: blink.cpp blink.h
	@echo "Compiling 'Blink'"
	$(CXX) $(FLAGS) -o blink blink.cpp $(LINK)

clean:
	@echo "Removing executables and capture.png files from directory..."
	rm -rf ./blink ./*.png 