#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include<opencv2/imgproc.hpp>
#include "gms_matcher.h"

using namespace std;
using namespace cv;

#define ORB_N_FEATURE				500	// 需要提取的特征点数目
#define ORB_N_OCTAVE_LAYERS			8		// 8, 默认值
#define ORB_FAST_THRESHOLD			20		// 20, default value
#define ORB_EDGE_THRESHOLD			31		// 31, default value
#define ORB_PATCH_SIZE				31		// 31, default value
#define ORB_SCALE					1.2		//  default value 1.2 

void ConvertMatches12(const vector<DMatch> &vDMatches, vector<pair<int, int> > &vMatches)
{
    vMatches.resize(vDMatches.size());
    for (size_t i = 0; i < vDMatches.size(); i++)
    {
        vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
    }
}

/**
 * @brief 归一化特征点到同一尺度，作为后续normalize DLT的输入
 *  [x' y' 1]' = T * [x y 1]' 
 *  归一化后x', y'的均值为0，sum(abs(x_i'-0))=1，sum(abs((y_i'-0))=1
 *
 *  为什么要归一化？
 *  在相似变换之后(点在不同的坐标系下),他们的单应性矩阵是不相同的
 *  如果图像存在噪声,使得点的坐标发生了变化,那么它的单应性矩阵也会发生变化
 *  我们采取的方法是将点的坐标放到同一坐标系下,并将缩放尺度也进行统一 
 *  对同一幅图像的坐标进行相同的变换,不同图像进行不同变换
 *  缩放尺度是为了让噪声对于图像的影响在一个数量级上
 * 
 *  Step 1 计算特征点X,Y坐标的均值 
 *  Step 2 计算特征点X,Y坐标离均值的平均偏离程度
 *  Step 3 将x坐标和y坐标分别进行尺度归一化，使得x坐标和y坐标的一阶绝对矩分别为1 
 *  Step 4 计算归一化矩阵：其实就是前面做的操作用矩阵变换来表示而已
 * 
 * @param[in] vKeys                               待归一化的特征点
 * @param[in & out] vNormalizedPoints             特征点归一化后的坐标
 * @param[in & out] T                             归一化特征点的变换矩阵
 */
void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)                           //将特征点归一化的矩阵
{
    // 归一化的是这些点在x方向和在y方向上的一阶绝对矩（随机变量的期望）。

    // Step 1 计算特征点X,Y坐标的均值 meanX, meanY
    float meanX = 0;
    float meanY = 0;

	//获取特征点的数量
    const int N = vKeys.size();

	//设置用来存储归一后特征点的向量大小，和归一化前保持一致
    vNormalizedPoints.resize(N);

	//开始遍历所有的特征点
    for(int i=0; i<N; i++)
    {
		//分别累加特征点的X、Y坐标
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    //计算X、Y坐标的均值
    meanX = meanX/N;
    meanY = meanY/N;

    // Step 2 计算特征点X,Y坐标离均值的平均偏离程度 meanDevX, meanDevY，注意不是标准差
    float meanDevX = 0;
    float meanDevY = 0;

    // 将原始特征点减去均值坐标，使x坐标和y坐标均值分别为0
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

		//累计这些特征点偏离横纵坐标均值的程度
        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    // 求出平均到每个点上，其坐标偏离横纵坐标均值的程度；将其倒数作为一个尺度缩放因子
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;
    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // Step 3 将x坐标和y坐标分别进行尺度归一化，使得x坐标和y坐标的一阶绝对矩分别为1 
    // 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值（期望）
    for(int i=0; i<N; i++)
    {
		//对，就是简单地对特征点的坐标进行进一步的缩放
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // Step 4 计算归一化矩阵：其实就是前面做的操作用矩阵变换来表示而已
    // |sX  0  -meanx*sX|
    // |0   sY -meany*sY|
    // |0   0      1    |
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}



int main( int argc, char** argv )
{
    Mat image1 = imread( "../1.png");
    Mat image2 = imread( "../2.png");
//    Mat image1 = imread( "../3.png");
//    Mat image2 = imread( "../4.png");
    assert(image1.data && image2.data && "can not load images");

    vector<KeyPoint> kp1, kp2;
    Mat desp1, desp2;

    Ptr<ORB> orb = ORB::create(ORB_N_FEATURE);
    orb->setFastThreshold(ORB_FAST_THRESHOLD);
    orb->setEdgeThreshold(ORB_EDGE_THRESHOLD);
    orb->setPatchSize(ORB_PATCH_SIZE);
    orb->setNLevels(ORB_N_OCTAVE_LAYERS);
    orb->setScaleFactor(ORB_SCALE);
    orb->setMaxFeatures(ORB_N_FEATURE);
    orb->setWTA_K(2);
    orb->setScoreType(ORB::HARRIS_SCORE); // HARRIS_SCORE，标准Harris角点响应函数
    orb->detectAndCompute(image1, Mat(), kp1, desp1);
    orb->detectAndCompute(image2, Mat(), kp2, desp2);

    vector< DMatch > matches;

    BFMatcher matcher_bf(NORM_HAMMING, true); //使用汉明距离度量二进制描述子，允许交叉验证
    vector<DMatch> Matches_bf;
    matcher_bf.match(desp1, desp2, matches);

    std::cout<<"Find total "<<matches.size()<<" matches."<<endl;


//GMS筛点
    vector<DMatch> matches_gms;
    vector<bool> vbInliers;

    gms_matcher gms(kp1, image1.size(), kp2, image2.size(), matches);
    int num_inliers = gms.GetInlierMask(vbInliers, false, false);

    cout << "# Refine Matches (after GMS):" << num_inliers  << "/" << matches.size() <<endl;
    // 筛选正确的匹配
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(matches[i]);
        }
    }
    
        // 继续筛选匹配对
    vector< DMatch > goodMatches;

    

    double minDis = 9999.9;
    
    for ( size_t i=0; i<matches_gms.size(); i++ )
    {
        if ( matches_gms[i].distance < minDis )
            minDis = matches_gms[i].distance;
    }
    cout<<"mindistance"<<minDis<<endl;

    for ( size_t i=0; i<matches_gms.size(); i++ )
    {
        if (matches[i].distance <= max(2*minDis,30.0))
            goodMatches.push_back( matches[i] );
    }
    cout<<"good total number: "<<goodMatches.size()<<endl;
    

    Mat img_goodmatch_gms; //放图
    drawMatches(image1,kp1,image2,kp2,goodMatches,img_goodmatch_gms);
    imshow("final matches",img_goodmatch_gms);

    vector< Point2f > pts1, pts2;
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        pts1.push_back(kp1[goodMatches[i].queryIdx].pt);
        pts2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }


    Mat statusF;//得出内外点状态，内点对应位置为1
    Mat statusH;
    //confidencce越高，将导致更多的迭代次数和计算时间，结果更精确
    Mat F21= findFundamentalMat(pts1, pts2,FM_RANSAC,1.0,0.99,statusF);
    Mat H21= findHomography(pts1,pts2,RANSAC,1.0,statusH,2000,0.99);

    Mat H12 = H21.inv();


    cout<<"F_matrix"<<F21<<endl;
    cout<<"H_matrix"<<H21<<endl;
    // 特征点匹配个数
    const int N=goodMatches.size();

    float scoreF=0;
    // 基于卡方检验计算出的阈值（自由度1）
    const float th = 3.841;
    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
    const float thScore = 5.991;
    float sigma=1;
	// 信息矩阵，或 协方差矩阵的逆矩阵
    const float invSigmaSquare = 1.0/(sigma*sigma);

    // 提取基础矩阵中的元素数据
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    // Step 1 获取从参考帧到当前帧的单应矩阵的各个元素
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

	// 获取从当前帧到参考帧的单应矩阵的各个元素
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);
    // 初始化scoreH值
    float scoreH=0;
// ----------- 开始你的代码 --------------//

    std::vector<cv::KeyPoint> mvKeys1 = kp1;
    std::vector<cv::KeyPoint> mvKeys2 = kp2; 
    vector<pair<int, int>> mvMatches12;


//------H矩阵-------//

    ConvertMatches12(goodMatches,mvMatches12);


    // Step 2 通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权重投影误差
    // H21 表示从img1 到 img2的变换矩阵
    // H12 表示从img2 到 img1的变换矩阵 

    // cout << mvMatches12.size() <<endl;
    // cout << N << endl;

    for(int i = 0; i < N; i++)
    {
		// 一开始都默认为Inlier
        bool bIn = true;

		// Step 2.1 提取参考帧和当前帧之间的特征匹配点对
        const cv::KeyPoint &kp11 = mvKeys1[mvMatches12[i].first];  //感觉问题出在这里，但是试了好多种输入都不对，一直得分0
        const cv::KeyPoint &kp22 = mvKeys2[mvMatches12[i].second];
        // const float u1 = kp1[i].pt.x;
        // const float u1 = pts1[i].x;
        const float u1 = kp11.pt.x;
        cout << "u1x = " << u1 << endl;
        // const float v1 = pts1[i].y;
        const float v1 = kp11.pt.y;
        cout << "u1y = " << v1 << endl;
        const float u2 = kp22.pt.x;
        const float v2 = kp22.pt.y;


        // Step 2.2 计算 img2 到 img1 的重投影误差
        // x1 = H12*x2
        // 将图像2中的特征点通过单应变换投影到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|   |u2in1|
        // |v1| = |h21inv h22inv h23inv||v2| = |v2in1| * w2in1inv
        // |1 |   |h31inv h32inv h33inv||1 |   |  1  |
		// 计算投影归一化坐标
        const float w2in1inv = 1.0/(h31inv * u2 + h32inv * v2 + h33inv);
        const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
        const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;
   
        // 计算重投影误差 = ||p1(i) - H12 * p2(i)||2
        const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);
        const float chiSquare1 = squareDist1 * invSigmaSquare;

        cout << "chiSquare1 = "<< chiSquare1 << endl;

        // Step 2.3 用阈值标记离群点，内点的话累加得分
        if(chiSquare1>th){
            bIn = false;    
            cout << "ji le" << endl;
        }
        else{
            // 误差越大，得分越低
            scoreH += th - chiSquare1;
            cout << "scoreH = " << scoreH << endl;
        }

        // 计算从img1 到 img2 的投影变换误差
        // x1in2 = H21*x1
        // 将图像2中的特征点通过单应变换投影到图像1中
        // |u2|   |h11 h12 h13||u1|   |u1in2|
        // |v2| = |h21 h22 h23||v1| = |v1in2| * w1in2inv
        // |1 |   |h31 h32 h33||1 |   |  1  |
		// 计算投影归一化坐标
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        // 计算重投影误差 
        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
        const float chiSquare2 = squareDist2*invSigmaSquare;
 
        // 用阈值标记离群点，内点的话累加得分
        if(chiSquare2>th)
            bIn = false;
        else
            scoreH += th - chiSquare2;   


    }


//------F矩阵-------//


    // Step 2 计算img1 和 img2 在估计 F 时的score值
    for(int i=0; i<N; i++)
    {
		//默认为这对特征点是Inliers
        bool bIn = true;

	    // Step 2.1 提取参考帧和当前帧之间的特征匹配点对
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

		// 提取出特征点的坐标
        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // Step 2.2 计算 img1 上的点在 img2 上投影得到的极线 l2 = F21 * p1 = (a2,b2,c2)
		const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;
    
        // Step 2.3 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2);
        // 带权重误差
        const float chiSquare1 = squareDist1*invSigmaSquare;
		
        // Step 2.4 误差大于阈值就说明这个点是Outlier 
        // ? 为什么判断阈值用的 th（1自由度），计算得分用的thScore（2自由度）
        // ? 可能是为了和CheckHomography 得分统一？
        if(chiSquare1>th)
            bIn = false;
        else
            // 误差越大，得分越低
            scoreF += thScore - chiSquare1;

        // 计算img2上的点在 img1 上投影得到的极线 l1= p2 * F21 = (a1,b1,c1)
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        // 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        // 带权重误差
        const float chiSquare2 = squareDist2*invSigmaSquare;

        // 误差大于阈值就说明这个点是Outlier 
        if(chiSquare2>th)
            bIn = false;
        else
            scoreF += thScore - chiSquare2;
        

    }
    

 // ----------- 结束你的代码 --------------//
    cout<<"F score = "<< scoreF<<endl;
    cout<<"H score = "<< scoreH<<endl;
    float ratio=scoreH/(scoreH+scoreF);

    if(ratio > 0.4)
    cout<<"choose H"<<endl;
    else
    cout<<"choose F"<<endl;
    waitKey(0);
    return 0;
}

