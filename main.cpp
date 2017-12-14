#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

#define COMPOSED_IMAGE_WIDTH 10 // Each compose image's width
#define OUTPUT_IMAGE_WIDTH 500 // Output image's width

void ListAllFiles(const char* dirName, vector<string>& imageNames)
{
    if(nullptr == dirName)
    {
        cout << "dirName is null !" << endl;
        return;
    }

    struct stat s;
    lstat(dirName, &s);
    if(! S_ISDIR(s.st_mode))
    {
        cout << "dirName is not a valid directory!" << endl;
        return;
    }


    DIR* dir;
    dir = opendir(dirName);
    if(nullptr == dir)
    {
        cout << "Can not open dir" << dirName << endl;
        return;
    }

    struct dirent* filename;
    while((filename = readdir(dir)) != nullptr)
    {
        if( strcmp( filename->d_name , "." ) == 0 ||
            strcmp( filename->d_name , "..") == 0    )
            continue;
        string imageName(filename->d_name);
        imageNames.push_back(imageName);
    }

    closedir(dir);
}

void ReadAllImages(const vector<string>& imageNames, vector<Mat>& images)
{
    for (const auto &it : imageNames)
    {
        string imageName = "../pictures/" + it;
        Mat imageMaterial = imread(imageName);
        double ratio = (double)COMPOSED_IMAGE_WIDTH / imageMaterial.rows;
        resize(imageMaterial, imageMaterial, Size(), ratio, ratio);
        images.push_back(imageMaterial);
    }

}

void ComposeImage(const Mat& srcImg, const vector<Mat>& images, Mat& outputImage)
{
    default_random_engine rngEng(time(nullptr));
    uniform_int_distribution<> dis(0, images.size() - 1);

    int curRow = 0, curCol = 0;
    while(curRow <= outputImage.rows - COMPOSED_IMAGE_WIDTH)
    {
        int rngIndex = dis(rngEng);
        while(curCol < outputImage.cols)
        {
            int colsToCopy = curCol < outputImage.cols - images[rngIndex].cols ? images[rngIndex].cols
                                                                               : outputImage.cols - curCol;
            Rect roiRect(curCol, curRow, colsToCopy, images[rngIndex].rows);
            Mat outputROI = outputImage(roiRect);

            Scalar meanScalar = mean(srcImg(roiRect));
            Scalar normalizedScalar;
            normalize(meanScalar, normalizedScalar);

            Mat imageToCopy;
            imageToCopy = images[rngIndex](Rect(0, 0, colsToCopy, images[rngIndex].rows)) +
                    (meanScalar - Scalar(127, 127, 127)).mul(normalizedScalar);
            imageToCopy.copyTo(outputROI);

            curCol += images[rngIndex].cols;
            int tmp = dis(rngEng);
            while (tmp == rngIndex)
                tmp = dis(rngEng);
            rngIndex = tmp; // renew rngIndex
        }

        curCol = 0;
        curRow += COMPOSED_IMAGE_WIDTH;
    }
}

int main(int argc, char* argv[]) {
    if(argc < 3)
    {
        cout << "Please input the srcImg's name and Composed \n";
        cout << "images' dirctory name!";
        return -1;
    }

    char* srcImgName = argv[1];
    char* imgsDirName = argv[2];

    Mat srcImg = imread(srcImgName);
    imshow("srcImg", srcImg);
    resize(srcImg, srcImg, Size(500, 500));
    vector<string> imageNames;
    ListAllFiles(imgsDirName, imageNames);

    vector<Mat> images;
    ReadAllImages(imageNames, images);

    Mat outputImage(srcImg.rows, srcImg.cols, CV_8UC3, Scalar(255,255,255));

    ComposeImage(srcImg, images, outputImage);
    imshow("Composed", outputImage);
    waitKey(0);
    return 0;
}