#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "RealSenseManager.hpp"
#include <ctime>

// const parameter
const double UPDATE_RATE = 0.9;
int lowTreshHold = 0;
const int RATIO = 3;
const int KERNEL_SIZE = 3;
const int DRAW_ALL_CONTOURS = -1;
const int FILLED_CONTOURS = -1;
const double AREA2FILTERED = 1700.0;
const double AREA2SMALL = 20000.0;
const int BACKGROUND_HISTORY_TRESHOLD = 500;
const double DIST2TRESHOLD = 400.0;
bool DETECT_SHADOWS = false;
//cv::RNG rng(12345);
int TOTAL_PEOPLE;
// time to delete total people
std::time_t START_TIME;
std::time_t RESULT_TIME;

int main()
{
    // realsense constructor
    //RealSenseManager realSenseManager;
    // Read video
    cv::VideoCapture video("../ManTrap_depth_640p.mp4");
    // Web cam
    // cv::VideoCapture video(1);
    // check video is opened
    if(!video.isOpened())
    {
        std::cout << "Could not read video file!!" << std::endl;
        return -1;
    }
    // background Subtraction statement
    // MOG2
    //cv::Ptr<cv::BackgroundSubtractor> BackgroundSubtraction = cv::createBackgroundSubtractorMOG2();
    // KNN
    cv::Ptr<cv::BackgroundSubtractor> BackgroundSubtraction = cv::createBackgroundSubtractorKNN(BACKGROUND_HISTORY_TRESHOLD, DIST2TRESHOLD, DETECT_SHADOWS);
    // image container
    cv::Mat frame, foreground, canny_filtered, final_output, frame_threshold;
    // get size of image to draw in out line
    double width, height;
    // start processing video
    while( video.read(frame) )
    //while(1)
    {
//        frame = realSenseManager.getDepthStream();
//        frame = realSenseManager.getRGBStream();
        cv::imshow("original", frame);
        //threshold(frame, frame, 10, 255, cv::THRESH_BINARY_INV);
//        cv::imshow("frame", frame);
        // vector Point contours
        std::vector < std::vector < cv::Point > > contours, area_filtered_contours;
        // Detect the object based on HSV Range Values
        cv::inRange(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 180, 0), frame_threshold);
        // Show the frames
        //imshow(window_capture_name, frame);
        // apply background subtractor
        BackgroundSubtraction->apply(frame_threshold, foreground); //,UPDATE_RATE);
        cv::imshow("foreground", foreground);
        // detect edges with Canny algorithm
        //cv::Canny(foreground, canny_filtered, lowTreshHold, lowTreshHold * RATIO, KERNEL_SIZE);
        //cv::imshow("foreground", foreground);
        // remove outlier with erosion
        int erosion_type = 0;
        int erosion_size = 1; // do erosion 3 times
        cv::Mat element = getStructuringElement( erosion_type, cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ), cv::Point( erosion_size, erosion_size ) );
        cv::erode(foreground, foreground, element);
        // apply dilation to normal size
        cv::dilate(foreground, foreground, element);
        // find contours
        //cv::findContours(canny_filtered, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));
        cv::findContours(foreground, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));
        // create empty matrix of canny_filtered size to put drawing contours
        //cv::Mat drawing_contours = cv::Mat::zeros( canny_filtered.size(), CV_8UC3 );
        cv::Mat drawing_contours = cv::Mat::zeros( foreground.size(), CV_8UC3 );
        //remove small area contours
        for( std::vector < std::vector < cv::Point > >::iterator it = contours.begin() ; it != contours.end() ; ++it )
        {
            double contour_area = cv::contourArea(*it);
            if(contour_area > AREA2FILTERED && contour_area < AREA2SMALL)
                area_filtered_contours.push_back(*it);
        }
        // draw filtered contours
        cv::Scalar color( 255, 0, 0 ); // blue
        cv::drawContours( drawing_contours, area_filtered_contours, DRAW_ALL_CONTOURS, color, CV_FILLED, cv::LINE_8 );
        //std::cout << "area filtered : " << area_filtered_contours.size() << std::endl;
        // create hull array for convex hull points
        std::vector< vector< cv::Point > > hull (area_filtered_contours.size());
        for(int i = 0; i < area_filtered_contours.size(); i++)
            cv::convexHull(cv::Mat(area_filtered_contours[i]), hull[i], false);
        //std::cout << "Hull size : " << hull.size() << std::endl;
        for(int i = 0; i < area_filtered_contours.size(); i++)
        {
            // draw ith convex hull
            drawContours(drawing_contours, hull, i, cv::Scalar(255,255,255), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
        }

        // detect number of person
        if(area_filtered_contours.size() > 1)
        {
            START_TIME = std::time(nullptr);
            TOTAL_PEOPLE = area_filtered_contours.size();
        }
        if((RESULT_TIME-START_TIME)>5)
            TOTAL_PEOPLE = 0;
     
        // draw line
        width = drawing_contours.size().width;
        height = drawing_contours.size().height;
        cv::line(drawing_contours, cv::Point(0, (height/5))*2, cv::Point(width, (height/5))*2, cv::Scalar(0, 0, 255) , 2, cv::LINE_8);
        cv::line(drawing_contours, cv::Point(0, (height/5))*3, cv::Point(width, (height/5))*3, cv::Scalar(0, 255, 0) , 2, cv::LINE_8);
        // put text
        std::string output_test_total_People = "Multiple people : " + std::to_string(TOTAL_PEOPLE);
        std::string output_text_Contour = "People detect : " + std::to_string(area_filtered_contours.size());
        cv::putText(drawing_contours, output_test_total_People, cv::Point(0,50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255));
        cv::putText(drawing_contours, output_text_Contour, cv::Point(0,100), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255));
        // render
        //cv::imshow("canny_filtered", canny_filtered);
        //cv::imshow("foreground", foreground);
        cv::imshow("drawing_contours", drawing_contours);
        //cv::imshow("final_output", final_output);
        RESULT_TIME = std::time(nullptr);
        // Exit if ESC pressed
        int k = cv::waitKey(1);
        if(k == 27)
        {
            break;
        }
    }
    return 0 ;
}
