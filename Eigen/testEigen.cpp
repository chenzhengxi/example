#include "Hungarian.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <pangolin/pangolin.h>  //基于openGL的画图工具Pangolin头文件
#include <unistd.h>             // C++中提供对操作系统访问功能的头文件，如fork/pipe/各种I/O（read/write/close等等）
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <boost/format.hpp>   // 具有格式化输出功能

using namespace std;
using namespace Eigen;
//using Eigen::MatrixXd;
int testEigen1()
{

  // [12  7  9  /7  9]
  // [ 8  9  6  6  /6]
  // [ 7 17 12 14  9]
  // [15 14  /6 11 10]
  // [ /4 10  7 10  9]
  // [ 1  /2  3  4  5]

  MatrixXd m(2, 2);            //MatrixXd表示是任意尺寸的矩阵ixj, m(2,2)代表一个2x2的方块矩阵
  m(0, 0) = 3;                 //代表矩阵元素a11
  m(1, 0) = 2.5;               //a21
  m(0, 1) = -1;                //a12
  m(1, 1) = m(1, 0) + m(0, 1); //a22=a21+a12
  cout << m << endl;           //输出矩阵m

  std::vector<std::vector<double>> DistMatrix(6);
  std::vector<int> Assignment;
  DistMatrix[0] = {12, 7, 9, 7, 9};
  DistMatrix[1] = {8, 9, 6, 6, 6};
  DistMatrix[2] = {7, 17, 12, 14, 9};
  DistMatrix[3] = {15, 14, 6, 11, 10};
  DistMatrix[4] = {4, 10, 7, 10, 9};
  DistMatrix[5] = {1, 2, 3, 4, 5};
  HungarianAlgorithm algo;
  algo.Solve(DistMatrix, Assignment);
  for (auto &&value : Assignment)
  {
    std::cout << value << std::endl;
  }
  return 0;
}

int testEigen5()
{
    Matrix3d rotation_matrix = Matrix3d::Identity();
 //旋转向量使用AngleAxis，运算可以当做矩阵
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0,0,1));     //眼Z轴旋转45°
    cout.precision(3);                                         //输出精度为小数点后两位
    cout << "rotation matrix = \n" << rotation_vector.matrix() << endl;
    //用matrix转换成矩阵可以直接赋值
    rotation_matrix = rotation_vector.toRotationMatrix();

    //使用Amgleanxis可以进行坐标变换
    Vector3d v(1, 0, 0);
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;

    //使用旋转矩阵
    v_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose() << endl;

    //欧拉角：可以将矩阵直接转换成欧拉角
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);       //按照ZYX顺序
    cout << "yaw pitch row = "<< euler_angles.transpose() << endl;

    //欧式变换矩阵使用Eigen::Isometry
    Isometry3d T = Isometry3d::Identity();      //实质为4*4的矩阵
    T.rotate(rotation_vector);                  //按照rotation_vector进行转化
    T.pretranslate(Vector3d(1, 3, 4));          //平移向量设为（1， 3， 4）
    cout << "Transform matrix = \n" << T.matrix() <<endl;

    //变换矩阵进行坐标变换
    Vector3d v_transformed = T *v;
    cout << "v transormed =" << v_transformed.transpose() << endl;

    //四元数
    //直接把AngleAxis赋值给四元数，反之亦然
    Quaterniond q = Quaterniond(rotation_vector);
    cout << "quaternion from rotation vector = " << q.coeffs().transpose() << endl;
    q = Quaterniond(rotation_matrix);
    cout << "quaternion from rotation matrix = "<< q.coeffs().transpose() << endl;

    //使用四元数旋转一个向量，使用重载的乘法即可
    v_rotated = q * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
    cout << "should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;
    cout << "......................." << endl;
//   // 1 定义旋转矩阵与平移向量　R t
//   // 沿Z轴转90度的旋转矩阵
//   Eigen::Matrix3f R = Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f(0.707106781, 0, 0.707106781)).toRotationMatrix();
//   //定义平移向量
//   Eigen::Vector3f t(1, 0, 0); // 沿X轴平移1
//   cout << R << endl;
//   cout << t << endl;
//   // SO(3) 旋转矩阵李群与李代数
//   //旋转矩阵李群SO(3)可以由　旋转矩阵/旋转向量/四元素得到,并且都是等效的　
//   //(注意李群的表示形式    Sophus::SO3)
//   Sophus::SO3f SO3_R(R);                             // Sophus::SO(3)可以直接从旋转矩阵构造
//   cout << "SO(3) from VECTOR: " << SO3_R.log().transpose() << endl;
//   Sophus::SO3f SO3_V = Sophus::SO3f::rotZ(M_PI / 2); // 亦可从旋转向量构造(注意此时旋转变量的形式)
//   cout << "SO(3) from vector: " << SO3_V.log().transpose() << endl;
//   Eigen::Quaternionf q(R);                           // 或者四元数
//   Sophus::SO3f SO3_q(q);
//   cout << "SO(3) from quaternion :" << SO3_q.log().transpose() << endl;

//   //旋转矩阵李代数　（李群的对数映射）
//   //(SO(3)李代数表示形式    Eigen::Vector3d
//   Eigen::Vector3f so3 = SO3_R.log();
//   cout << "so3 = " << so3.transpose() << endl;
//   // hat 为向量==>反对称矩阵 (李代数向量　对应的反对称矩阵)
//   cout << "so3 hat=\n"
//        << Sophus::SO3f::hat(so3) << endl;
//   // 相对的，vee为反对称==>向量
//   cout << "so3 hat vee= " << Sophus::SO3f::vee(Sophus::SO3f::hat(so3)).transpose() << endl; // transpose纯粹是为了输出美观一些

//   //旋转矩阵李代数的 增量扰动模型的更新
//   Eigen::Vector3f update_so3(1e-4, 0, 0); //假设更新量为这么多
//   Sophus::SO3f SO3_updated = Sophus::SO3f::exp(update_so3) * SO3_R;
//   cout << "SO3 updated = " << SO3_updated.log() << endl;

//   //SE(3) 变换矩阵李群与李代数
//   //变换矩阵李群SE(3)可以由　旋转矩阵/四元素 + 平移向量得到,并且都是等效的　
//   Sophus::SE3f SE3_Rt(R, t); // 从R,t构造SE(3)
//   Sophus::SE3f SE3_qt(q, t); // 从q,t构造SE(3)
//   cout << "SE3 from R,t= " << endl
//        << SE3_Rt.log() << endl;
//   cout << "SE3 from q,t= " << endl
//        << SE3_qt.log() << endl;

//   //变换矩阵李代数　（李群的对数映射）
//   //(SE(3)李代数表示形式    Eigen::Matrix<double,6,1>   sophus中旋转在前，平移在后
//   typedef Eigen::Matrix<float, 6, 1> Vector6f;
//   Vector6f se3 = SE3_Rt.log();
//   cout << "se3 = " << se3.transpose() << endl;
//   //向量的反对称矩阵表示形式的变换
//   cout << "se3 hat = " << endl
//        << Sophus::SE3f::hat(se3) << endl;
//   cout << "se3 hat vee = " << Sophus::SE3f::vee(Sophus::SE3f::hat(se3)).transpose() << endl;
//   //变换矩阵李代数的 增量扰动模型的更新
//   Vector6f update_se3; //更新量
//   update_se3.setZero();
//   update_se3(0, 0) = 1e-4;
//   cout << "se3_update =  " << update_se3.transpose() << endl;
//   Sophus::SE3f SE3_updated = Sophus::SE3f::exp(update_se3) * SE3_Rt;
//   cout << "SE3 updated = " << endl
//        << SE3_updated.matrix() << endl;
  return 0;
}

int testEigen6()
{
    //读取图像
   cv::Mat img_1 = cv::imread("/home/chenzhengxi/study/slam/aloeL.jpg", cv::IMREAD_COLOR);
   cv::Mat img_2 = cv::imread("/home/chenzhengxi/study/slam/aloeR.jpg", cv::IMREAD_COLOR);
   assert(img_1.data != nullptr && img_2.data != nullptr);
   //在程序运行时cv::assert()计算括号内的表达式，如果表达式为FALSE (或0), 程序将报告错误，并终止执行。
   //如果表达式不为0，则继续执行后面的语句。

   //初始化
   vector<cv::KeyPoint> keypoints_1, keypoints_2;   //关键点/角点
   /**
   opencv中keypoint类的默认构造函数为：
   CV_WRAP KeyPoint() : pt(0,0), size(0), angle(-1), response(0), octave(0), class_id(-1) {}
   pt(x,y):关键点的点坐标； // size():该关键点邻域直径大小； // angle:角度，表示关键点的方向，值为[0,360)，负值表示不使用。
   response:响应强度，选择响应最强的关键点;   octacv:从哪一层金字塔得到的此关键点。
   class_id:当要对图片进行分类时，用class_id对每个关键点进行区分，默认为-1。
   **/
   cv::Mat descriptors_1, descriptors_2;      //描述子
   //创建ORB对象，参数为默认值
   cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
   cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
   cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
   /**
   “Ptr<FeatureDetector> detector = ”等价于 “FeatureDetector * detector =”
   Ptr是OpenCV中使用的智能指针模板类，可以轻松管理各种类型的指针。
   特征检测器FeatureDetetor是虚类，通过定义FeatureDetector的对象可以使用多种特征检测及匹配方法，通过create()函数调用。
   描述子提取器DescriptorExtractor是提取关键点的描述向量类抽象基类。
   描述子匹配器DescriptorMatcher用于特征匹配，"Brute-force-Hamming"表示使用汉明距离进行匹配。
   **/

   //第一步，检测Oriented Fast角点位置
   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   detector->detect(img_1, keypoints_1);     //对参数1图像进行特征的提取，并存放入参数2的数组中
   detector->detect(img_2, keypoints_2);

   //第二步，根据角点计算BREIF描述子
   descriptor->compute(img_1, keypoints_1, descriptors_1);   //computer()计算关键点的描述子向量（注意思考参数设置的合理性）
   descriptor->compute(img_2, keypoints_2, descriptors_2);
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;
   cv::Mat outimg1;
   drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
   cv::imshow("ORB features", outimg1);
   cv::imwrite("feaure1.png", outimg1);

   //第三步， 对两幅图像中的描述子进行匹配，使用hamming距离
   vector<cv::DMatch> matches;    //DMatch是匹配关键点描述子 类, matches用于存放匹配项
   t1 = chrono::steady_clock::now();
   matcher->match(descriptors_1, descriptors_2, matches); //对参数1 2的描述子进行匹配，并将匹配项存放于matches中
   t2 = chrono::steady_clock::now();
   time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "match the ORB cost: " << time_used.count() << "seconds. " << endl;

   //第四步，匹配点对筛选
   //计算最小距离和最大距离
   auto min_max = minmax_element(matches.begin(), matches.end(),
       [](const cv::DMatch &m1, const cv::DMatch &m2){ return m1.distance < m2.distance; });
   // auto 可以在声明变量的时候根据变量初始值的类型自动为此变量选择匹配的类型
   // minmax_element()返回指向范围内最小和最大元素的一对迭代器。参数1 2为起止迭代器范围
   // 参数3是二进制函数，该函数接受范围内的两个元素作为参数，并返回可转换为bool的值。
   // 返回的值指示作为第一个参数传递的元素是否小于第二个。该函数不得修改其任何参数。
   double min_dist = min_max.first->distance;  // min_max存储了一堆迭代器，first指向最小元素
   double max_dist = min_max.second->distance; // second指向最大元素

   printf("-- Max dist : %f \n", max_dist);
   printf("-- Min dist : %f \n", min_dist);

   //当描述子之间的距离大于两倍最小距离时，就认为匹配有误。但有时最小距离会非常小，所以要设置一个经验值30作为下限。
   vector<cv::DMatch> good_matches;  //存放良好的匹配项
   for(int i = 0; i < descriptors_1.rows; ++i){
       if(matches[i].distance <= max(2 * min_dist, 30.0)){
           good_matches.push_back(matches[i]);
       }
   }

   //第五步，绘制匹配结果
   cv::Mat img_match;         //存放所有匹配点
   cv::Mat img_goodmatch;     //存放好的匹配点
   // drawMatches用于绘制两幅图像的匹配关键点。
   // 参数1是第一个源图像，参数2是其关键点数组；参数3是第二张原图像，参数4是其关键点数组
   // 参数5是两张图像的匹配关键点数组,参数6用于存放函数的绘制结果
   drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
   drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
   imshow("all matches", img_match);
   imshow("good matches", img_goodmatch);
   imwrite("match1.png", img_match);
   imwrite("goodmatch1.png", img_goodmatch);

   cv::waitKey(0);
    return 0;
}
// 记录准确文件路径
string left_file = "/home/chenzhengxi/study/slam/left.png";
string right_file = "/home/chenzhengxi/study/slam/right.jpg";
string disparity_file = "/home/chenzhengxi/study/slam/disparity.png";
boost::format fmt_others("/home/chenzhengxi/study/slam/%01d.png");    // other files

// 在pangolin中画图
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int testEigen4()
{
    // 相机内参，一般为已知数据
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 双目相机基线，一般已知
    double b = 0.573;

    // 读取图像
    cv::Mat left = cv::imread(left_file, 0);   //imread()参数2为0时，表示返回灰度图像，默认值为1，代表返回彩色图像
    cv::Mat right = cv::imread(right_file, 0); //从文件路径中读取两幅图像，返回灰度图像
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 调用OpenCv中的SGBM算法，用于计算左右图像的视差
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);   //将视差的计算结果放入disparity_sgbm矩阵中
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f); //将矩阵disparity_sgbm转换为括号中的格式(32位空间的单精度浮点型矩阵)

    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud; //声明一个4维的双精度浮点型可变长动态数组

    // 如果自己的机器慢，可以把++v和++u改成v+=2, u+=2
    for (int v = 0; v < left.rows; ++v)
        for (int u = 0; u < left.cols; ++u) {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;
            //Mat.at<存储类型名称>(行，列)[通道]，用以遍历像素。省略通道部分时，可以看做二维数组简单遍历，例如M.at<uchar>(512-1,512*3-1)；

            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色。第四维数值归一化。

            // 根据双目模型计算 point 的位置
            double x = (u - cx) / fx;      //像素坐标转换为归一化坐标
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));  //计算各像素点深度
            //计算带深度信息的各点坐标
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            pointcloud.push_back(point);   //将各点信息压入点云数组
        }

    cv::imshow("disparity", disparity / 96.0); //输出显示disparuty，显示窗口命名为引号中的内容
    cv::waitKey(0);           //等待关闭显示窗口，括号内参数为零则表示等待输入一个按键才会关闭，为数值则表示等待X毫秒后关闭
    // 画出点云
    showPointCloud(pointcloud);
    return 0;
}
    //定义画出点云的函数,函数参数为四维动态数组的引用
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);   //创建一个Pangolin的画图窗口,声明命名以及显示的分辨率
    glEnable(GL_DEPTH_TEST);    //启用深度缓存。
    glEnable(GL_BLEND);         //启用gl_blend混合。Blend混合是将源色和目标色以某种方式混合生成特效的技术。
    //混合常用来绘制透明或半透明的物体。在混合中起关键作用的α值实际上是将源色和目标色按给定比率进行混合，以达到不同程度的透明。
    //α值为0则完全透明，α值为1则完全不透明。混合操作只能在RGBA模式下进行，颜色索引模式下无法指定α值。
    //物体的绘制顺序会影响到OpenGL的混合处理。
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  //混合函数。参数1是源混合因子，参数2时目标混合因子。本命令选择了最常使用的参数。

    //定义投影和初始模型视图矩阵
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        //对应为gluLookAt,摄像机位置,参考点位置,up vector(上向量)
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );
    //管理OpenGl视口的位置和大小
    pangolin::View &d_cam = pangolin::CreateDisplay()
        //使用混合分数/像素坐标（OpenGl视图坐标）设置视图的边界
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        //指定用于接受键盘或鼠标输入的处理程序
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        //清除屏幕
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //激活要渲染到视图
        d_cam.Activate(s_cam);
        //glClearColor：red、green、blue、alpha分别是红、绿、蓝、不透明度，值域均为[0,1]。
        //即设置颜色，为后面的glClear做准备，默认值为（0,0,0,0）。切记：此函数仅仅设定颜色，并不执行清除工作。
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        //glPointSize 函数指定栅格化点的直径。一定要在要在glBegin前,或者在画东西之前。
        glPointSize(2);
        //glBegin()要和glEnd()组合使用。其参数表示创建图元的类型，GL_POINTS表示把每个顶点作为一个点进行处理
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);  //在OpenGl中设置颜色
            glVertex3d(p[0], p[1], p[2]); //设置顶点坐标
        }
        glEnd();
        pangolin::FinishFrame();    //结束
        usleep(5000);   // sleep 5 ms
    }
    return;
}

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;  // 存放像素点坐标的数组

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline   双目相机基线
double baseline = 0.573;
// paths

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/// class for accumulator jacobians in parallel
// 定义求雅克比的类
class JacobianAccumulator {
public:
    // 构造函数
    JacobianAccumulator(
        const cv::Mat &img1_,             // 图像 1
        const cv::Mat &img2_,             // 图像 2
        const VecVector2d &px_ref_,       //  参考点像素坐标 数组
        const vector<double> depth_ref_,  // 参考点深度 数组
        Sophus::SE3d &T21_) :   // 坐标系1到坐标系2的变换矩阵
        img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }

    /// accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    /// get hessian matrix
    Matrix6d hessian() const { return H; }

    /// get bias
    Vector6d bias() const { return b; }

    /// get total cost
    double cost_func() const { return cost; }

    /// get projected points
    VecVector2d projected_points() const { return projection; }

    /// reset h, b, cost to zero
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;   // 标准互斥类型
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

/**====
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);

/**==
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 **/
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);

// bilinear interpolation  双线性插值
// 双线性内插法利用待求像素四个相邻像素的灰度在两个方向上做线性内插，在光流法求取某像素位置的灰度值时同样用到了二维线性插值。
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check 边界检测
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)]; // data为指针，指向定位的像素位置
    // step()函数，返回像素行的实际宽度
    float xx = x - floor(x);   // floor()函数返回不大于x的最大整数
    float yy = y - floor(y);   // xx 和 yy 就是小数部分
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}


///////////  主函数
int testEigen7()
 {

    cv::Mat left_img = cv::imread(left_file, 0);  // 读取灰度图像
    cv::Mat disparity_img = cv::imread(disparity_file, 0);  // 读取视差图像

    // randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;   // 点的数量
    int boarder = 20;     // 边界的像素数 ，在这里表示留空边上的一部分区域，不在边上取点
    VecVector2d pixels_ref;
    vector<double> depth_ref;
    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        // rng.uniform()函数返回区间内均匀分布的随机数
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);   // 像素的视差
        double depth = fx * baseline / disparity; // 双目视觉中由视差到深度的计算
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }
    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    for (int i = 1; i < 6; i++) {  // 1~10
        std::string filestr = (fmt_others % i).str();
        cout << filestr << endl;
        cv::Mat img = cv::imread(filestr, cv::IMREAD_COLOR);
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
    return 0;
}


///////  求解图像块的雅克比矩阵和增量方程
void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {
    // 求一个图像块内的雅克比矩阵的累积，为了解决单个像素在直接法中不具有代表性的缺点

    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; ++i) {

        // compute the projection in the second image
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        // 计算参考点(第一幅图像)的三维坐标： ((x_i - c_x )/ fx), (y_i - c_y)/ fy, 1) * depth

        // 计算当前目标点(第二幅图像)的三维坐标
        Eigen::Vector3d point_cur = T21 * point_ref;

        if (point_cur[2] < 0)   // depth invalid
            continue;
        // 计算第i个参考点对应的目标点的像素坐标
        float u = fx * point_cur[0] / point_cur[2] + cx;  // u = c_x + fx * (x / z)
        float v = fy * point_cur[1] / point_cur[2] + cy;  // v = c_y + fy * (y / z)

        // 舍弃越界的像素点坐标
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;

        // prijection 存放的是像素点坐标
        projection[i] = Eigen::Vector2d(u, v);
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
            Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;

        // 计数共有多少个良好的目标点
        cnt_good++;

        // compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {

                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Matrix26d J_pixel_xi;   // 像素坐标对相机位姿李代数的一阶变化关系 : \frac{\partial u}{\partial \Delta epslon}
                Eigen::Vector2d J_img_pixel;  // 对应位置的像素梯度 ： \frac{\partial I}{\partial u}

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
                );

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();  // H 矩阵 ： 2*2
                bias += -error * J;            // b 矩阵 ： - e * J (有时也写作 b = - f * J )
                cost_tmp += error * error;     // 误差的二范数 (累加后是所有好的匹配点的误差二范数之和)
            }
    }

    if (cnt_good) {  // 如果好的目标点不为0，也就是J和b都有计算，那么进行以下操作
        // 计算最终的 H矩阵、b矩阵和误差二范数

        unique_lock<mutex> lck(hessian_mutex);
        // unique_lock 是为了避免 mutex 忘记释放锁。在对象创建时自动加锁，对象释放时自动解锁。
        // std::mutex类是一个同步原语，可用于保护共享数据被同时由多个线程访问。std::mutex提供独特的，非递归的所有权语义。
        // std::mutex是C++11中最基本的互斥量，std::mutex对象提供了独占所有权的特性，不支持递归地对std::mutex对象上锁。

        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good; // 本图像块的像素平均误差二范数
    }
}

//////////////// 单层直接法
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        // cv::parallel_for_是opencv封装的一个多线程接口，利用这个接口可以方便实现多线程，不用考虑底层细节
        // 下面这条语句相当于是此次迭代中的jacobian部分并行计算完了
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
                          // bind()函数起绑定效果，占位符std::placeholders::_1表示第一个参数对应jaco.accu::accumulate_jacobian的第一个参数
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // 求解增量方程
        Vector6d update = H.ldlt().solve(b);;
        T21 = Sophus::SE3d::exp(update) * T21;   // 更新位姿
        cost = jaco_accu.cost_func();
        if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3) {
            // converge
            break;
        }

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n" << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::imshow("222", img2);
    cv::waitKey(0);
    cv::cvtColor(img2, img2_show, cv::COLOR_BGR2GRAY);
    cout << "!!!!!!!!!!! 1" << endl;
    VecVector2d projection = jaco_accu.projected_points();
    cout << "!!!!!!!!!!! 2" << endl;
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}


///////////////// 多层直接法
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {

    // 设置图像金字塔参数
    int pyramids = 4;   // 金字塔共有4层
    double pyramid_scale = 0.5;  // 每一层缩放比率是0.5
    double scales[] = {1.0, 0.5, 0.25, 0.125};
    cout << "......1" << endl;
    // 创建图像金字塔
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            // 第一层，底层是原图像
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            // 上面的层使用resize()函数进行创建
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    // 由粗至精进行求解
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // 存放该层的目标点
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        // 不同的层上面由于进行了缩放，相机内参也相应的进行了改变
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        // 调用单层直接法进行求解
        
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}



#include <iostream>
#include <iomanip>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <pangolin/pangolin.h>

struct RotationMatrix {
  Matrix3d matrix = Matrix3d::Identity();
};

ostream &operator<<(ostream &out, const RotationMatrix &r) {
  out.setf(ios::fixed);
  Matrix3d matrix = r.matrix;
  out << '=';
  out << "[" << setprecision(2) << matrix(0, 0) << "," << matrix(0, 1) << "," << matrix(0, 2) << "],"
      << "[" << matrix(1, 0) << "," << matrix(1, 1) << "," << matrix(1, 2) << "],"
      << "[" << matrix(2, 0) << "," << matrix(2, 1) << "," << matrix(2, 2) << "]";
  return out;
}

istream &operator>>(istream &in, RotationMatrix &r) {
  return in;
}

struct TranslationVector {
  Vector3d trans = Vector3d(0, 0, 0);
};

ostream &operator<<(ostream &out, const TranslationVector &t) {
  out << "=[" << t.trans(0) << ',' << t.trans(1) << ',' << t.trans(2) << "]";
  return out;
}

istream &operator>>(istream &in, TranslationVector &t) {
  return in;
}

struct QuaternionDraw {
  Quaterniond q;
};

ostream &operator<<(ostream &out, const QuaternionDraw quat) {
  auto c = quat.q.coeffs();
  out << "=[" << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << "]";
  return out;
}

istream &operator>>(istream &in, const QuaternionDraw quat) {
  return in;
}

int testEigen() {
  pangolin::CreateWindowAndBind("visualize geometry", 1000, 600);
  glEnable(GL_DEPTH_TEST);
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1000, 600, 420, 420, 500, 300, 0.1, 1000),
    pangolin::ModelViewLookAt(3, 3, 3, 0, 0, 0, pangolin::AxisY)
  );

  const int UI_WIDTH = 500;

  pangolin::View &d_cam = pangolin::CreateDisplay().
    SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -1000.0f / 600.0f).
    SetHandler(new pangolin::Handler3D(s_cam));

  // ui
  pangolin::Var<RotationMatrix> rotation_matrix("ui.R", RotationMatrix());
  pangolin::Var<TranslationVector> translation_vector("ui.t", TranslationVector());
  pangolin::Var<TranslationVector> euler_angles("ui.rpy", TranslationVector());
  pangolin::Var<QuaternionDraw> quaternion("ui.q", QuaternionDraw());
  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);

    pangolin::OpenGlMatrix matrix = s_cam.GetModelViewMatrix();
    Matrix<double, 4, 4> m = matrix;

    RotationMatrix R;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R.matrix(i, j) = m(j, i);
    rotation_matrix = R;

    TranslationVector t;
    t.trans = Vector3d(m(0, 3), m(1, 3), m(2, 3));
    t.trans = -R.matrix * t.trans;
    translation_vector = t;

    TranslationVector euler;
    euler.trans = R.matrix.eulerAngles(2, 1, 0);
    euler_angles = euler;

    QuaternionDraw quat;
    quat.q = Quaterniond(R.matrix);
    quaternion = quat;

    glColor3f(1.0, 1.0, 1.0);

    pangolin::glDrawColouredCube();
    // draw the original axis
    glLineWidth(3);
    glColor3f(0.8f, 0.f, 0.f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(10, 0, 0);
    glColor3f(0.f, 0.8f, 0.f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 10, 0);
    glColor3f(0.2f, 0.2f, 1.f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 10);
    glEnd();

    pangolin::FinishFrame();
  }
}