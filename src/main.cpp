/*
MIT License

Copyright (c) 2019 haskell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rxcpp/rx.hpp"

#include <fstream>
#include <opencv2/opencv.hpp>
namespace rx = rxcpp;
namespace rxu = rxcpp::util;
using namespace cv;
#include <string>
#include "client.h"
#include <type_traits>
#include "cvYolo.h"
#include "mqtt_pub.h"
#include "sig.h"

#define COUNT 10
typedef struct tagMaxMatch{
    int edge[COUNT][COUNT];
    bool on_path[COUNT];
    int path[COUNT];
    int max_match;
}GRAPH_MATCH;

void outputRes(int *path){
    for (int i = 0 ; i<COUNT; i++) {
        printf("%d****%d\n",i,*(path+i));   //Yj在前 Xi在后
    }
}

void clearOnPathSign(GRAPH_MATCH *match){
    for (int j = 0 ; j < COUNT ; j++) {
        match->on_path[j] = false;
    }
   
}
//dfs算法
bool FindAugPath(GRAPH_MATCH *match , int xi){
    
    for (int yj = 0 ; yj < COUNT; yj++) {
        if ( match->edge[xi][yj] == 1 && !match->on_path[yj]) { //如果yi和xi相连且yi没有在已经存在的增广路经上
             match->on_path[yj] = true;
            if (match->path[yj] == -1 || FindAugPath(match,match->path[yj])) { // 如果是yi是一个未覆盖点或者和yi相连的xk点能找到增广路经，
                  match->path[yj] = xi; //yj点加入路径;
                  return true;
            }
        }
    }
    return false;
}

void Hungary_match(GRAPH_MATCH *match){
    for (int xi = 0; xi<COUNT ; xi++) {
          FindAugPath(match, xi);
          clearOnPathSign(match);
    }
    outputRes(match->path);
}

// int main() {
    
//     GRAPH_MATCH *graph = (GRAPH_MATCH *)malloc(sizeof(GRAPH_MATCH));
//     for (int i = 0 ; i < COUNT ; i++) {
//         for (int j = 0 ; j < COUNT ; j++) {
//             graph->edge[i][j] = 0;
//         }
//     }
//     graph->edge[0][0] = 1;
//     graph->edge[1][1] = 1;
//     graph->edge[1][3] = 1;
//     graph->edge[2][1] = 1;
//     graph->edge[2][2] = 1;
//     graph->edge[3][3] = 1;
//     graph->edge[3][4] = 1;
//     graph->edge[4][2] = 1;
//     graph->edge[4][3] = 1;
    
//     for (int j = 0 ; j < COUNT ; j++) {
//         graph->path[j] = -1;
//         graph->on_path[j] = false;
//     }
    
//     Hungary_match(graph);
    
    
// }

extern "C" int func(); //注意这里的声明  



int main(int argc, char **argv)
{
    Singleton2 *ins = Singleton2::instance();
    Singleton2 *ins2 = Singleton2::instance();
  // func();
  // auto get_names = []() { return rx::observable<>::from<std::string>(
  //                             "Matthew",
  //                             "Aaron"); };
  // std::cout << "===== println stream of std::string =====" << std::endl;
  // auto hello_str = [&]() { return get_names().map([](std::string n) {
  //                                              return "Hello, " + n + "!";
  //                                            })
  //                              .as_dynamic(); };
  // hello_str().subscribe(rxu::println(std::cout));
  // std::cout << "===== println stream of std::tuple =====" << std::endl;
  // auto hello_tpl = [&]() { return get_names().map([](std::string n) {
  //                                              return std::make_tuple("Hello, ", n, "! (", n.size(), ")");
  //                                            })
  //                              .as_dynamic(); };
  // hello_tpl().subscribe(rxu::println(std::cout));
  // hello_tpl().subscribe(rxu::print_followed_by(std::cout, " and "), rxu::endline(std::cout));


  //testYolo();
  //mqtt_pub();

  return 0;
}
