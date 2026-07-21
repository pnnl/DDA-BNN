shape_file: shape_file.o
	g++ -o shape_file shape_file.o
shape_file.o: shape_file.cpp
	g++ -c -Wno-deprecated shape_file.cpp

clean:
	rm *.o *~ shape_file
