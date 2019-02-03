#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;
Mat Image,Gray,GaussianB,edge,draw,ROI,Image2;




int main(int argc, char* argv[])
{
	Image = imread("lena.jpg");
	cvtColor(Image,Gray,COLOR_BGR2GRAY);
	GaussianBlur(Gray, GaussianB, Size(5,5), 0, 0);
	// Canny Edge
	Canny( GaussianB, edge, 50, 150, 3);
	edge.convertTo(draw, CV_8U);
	// Select ROI
	Gray.setTo(Scalar::all(0)); // fill gray hitam
	Mat roiMuka = Gray( Rect(200,200,200,200) ); // rect(x,y,width,height)
	roiMuka.setTo(Scalar::all(255)); // fill muka putih
	Gray.copyTo(Image2) ; // Duplicate
	Rect WhereRec(200, 200, roiMuka.cols, roiMuka.rows); // Posisi dan ukuran
	roiMuka.copyTo(Gray(WhereRec)); // copy
	
	//Bitwise
    bitwise_and(Gray,edge,ROI); // Region Of Interest
    
    
    
    // Muncul
	namedWindow("hasil",CV_WINDOW_AUTOSIZE);
	imshow("hasil",ROI);
	waitKey(0);
	
}

