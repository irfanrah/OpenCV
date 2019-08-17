#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
using namespace cv;
using namespace std;

//convert double to string
namespace patch
{
    template<typename T> std::string to_string(const T& n)
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

/**
 * Helper function to display text in the center of a contour
 */
void setLabel(Mat& im, const std::string label, std::vector<Point>& contour)
{
	int fontface = FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 0;

	Size text = getTextSize(label, fontface, scale, thickness, &baseline);
	Rect r = boundingRect(contour);

	Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	rectangle(im, pt + Point(0, baseline), pt + Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
	putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

int main(int argc, char** argv)
{
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "CAP BROKEN" << endl;
		return -1;
	}
	namedWindow("Circle Detector", CV_WINDOW_AUTOSIZE);
	while (1)
	{
		Mat Original;
		bool yes = cap.read(Original);
		if (!yes)
		{
			cout << "MAT BROKEN" << endl;
			break;
		}
		
		Mat gray,bw,gray2;
		cvtColor(Original, gray, COLOR_BGR2GRAY);
		threshold( gray, bw,0, 115,3 );
		Canny(bw, gray2, 0, 50, 5);
		vector<vector<Point> > contours;
		findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		vector<Point> approx;
		Mat dst = Original.clone();

	for (int i = 0; i < contours.size(); i++)
	{
		// Approximate contour with accuracy proportional
		// to the contour perimeter
		approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

		// Skip small or non-convex objects 
		if (std::fabs(contourArea(contours[i])) < 700 || !isContourConvex(approx))
			continue;

		if (approx.size() == 3)
		{
			if (std::fabs(contourArea(contours[i])) < 10000 || !isContourConvex(approx))
			{
				continue;
				
			}
			setLabel(dst, "TRI", contours[i]);

		}
		 else if (approx.size() >= 4 && approx.size() <= 6)
		{
			// Number of vertices of polygonal curve
			int vtc = approx.size();

			// Get the cosines of all corners
			std::vector<double> cos;
			for (int j = 2; j < vtc+1; j++)
				cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));

			// Sort ascending the cosine values
			std::sort(cos.begin(), cos.end());

			// Get the lowest and the highest cosine
			double mincos = cos.front();
			double maxcos = cos.back();

			// Use the degrees obtained above and the number of vertices
			// to determine the shape of the contour
			if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
				{
				double area = contourArea(contours[i]);
				double keliling = arcLength( contours[i], true );
				double ganteng = (keliling*keliling)/(16*area);
				
				if (ganteng >= 0.8 and ganteng <= 1.15)
					{	
					setLabel(dst, "SQR", contours[i]);
					}
				else 
				{
					setLabel(dst, "RECT", contours[i]);
				}
				
				
				
				}
			else if (vtc == 5 && mincos >= -0.34 && maxcos <= -0.27)
				setLabel(dst, "PENTA", contours[i]);
			else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45)
				setLabel(dst, "HEXA", contours[i]);
		}
		else
		{
			// Detect and label circles
			double area = contourArea(contours[i]);
			Rect r = boundingRect(contours[i]);
			int radius = r.width / 2;

			if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
			    std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
				setLabel(dst, "CIR", contours[i]);
		}
	}
		
		imshow("Circle Detector", dst);
	
		if (waitKey(1) == 27)
		{
			cout << "EXIT" << endl;
			break;
		}
	}
	
	return 0;
}
