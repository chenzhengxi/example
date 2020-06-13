#include "rxcpp/rx.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <opencv/cv.hpp>
using json = nlohmann::json;
// create alias' to simplify code
// these are owned by the user so that
// conflicts can be managed by the user.
namespace rx=rxcpp;
namespace rxu=rxcpp::util;
using namespace cv;
#include<string>

int main()
{
    Mat in_image;
    VideoCapture cap;
    cap.open("//home//chenzhengxi//gesturevideo//1.mp4");
    cap >> in_image;
    //读取原始图像
    if (in_image.empty()) {
        //检查是否读取图像
        std::cout << "Error! Input image cannot be read...\n";
        return -1;
    }
    //写入图像
    imwrite("czx.jpg", in_image);
    //imshow(argv[2], out_image);
    return 0;

    json value = "Hello";
    json array_0 = json(0, value);
    json array_1 = json(1, value);
    json array_5 = json(5, value);

    // serialize the JSON arrays
    std::cout << array_0 << '\n';
    std::cout << array_1 << '\n';
    std::cout << array_5 << '\n';
    
    // create a JSON object with different entry types
    json j =
    {
        {"integer", 1},
        {"floating", 42.23},
        {"string", "hello world"},
        {"boolean", true},
        {"object", {{"key1", 1}, {"key2", 2}}},
        {"array", {1, 2, 3}}
    };
    std::string jsonstr = j.dump();
    std::cout << j.dump() << std::endl;
    json j2 = json::parse(jsonstr);
    std::cout << j2.dump() << std::endl;
    std::fstream f;
    f.open("j.json", std::ios::app);
    f.write(jsonstr.c_str(), jsonstr.length());
    f.close();
    // access existing values
    int v_integer = j.value("integer", 0);
    double v_floating = j.value("floating", 47.11);

    // access nonexisting values and rely on default value
    std::string v_string = j.value("nonexisting", "oops");
    bool v_boolean = j.value("nonexisting", false);

    // output values
    std::cout << std::boolalpha << v_integer << " " << v_floating
              << " " << v_string << " " << v_boolean << "\n";
    
    auto get_names = [](){return rx::observable<>::from<std::string>(
        "Matthew",
        "Aaron"
    );};

    std::cout << "===== println stream of std::string =====" << std::endl;
    auto hello_str = [&](){return get_names().map([](std::string n){
        return "Hello, " + n + "!";
    }).as_dynamic();};

    hello_str().subscribe(rxu::println(std::cout));

    std::cout << "===== println stream of std::tuple =====" << std::endl;
    auto hello_tpl = [&](){return get_names().map([](std::string n){
        return std::make_tuple("Hello, ", n, "! (", n.size(), ")");
    }).as_dynamic();};

    hello_tpl().subscribe(rxu::println(std::cout));

    hello_tpl().subscribe(rxu::print_followed_by(std::cout, " and "), rxu::endline(std::cout));
    return 0;
}
