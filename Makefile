all:
	g++ -o 2dto3d src/main.cpp -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgproc
