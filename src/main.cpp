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

#include <fstream>
#include <opencv2/opencv.hpp>
using namespace cv;
#include <string>
#include "client.h"
#include <type_traits>
#include "cvYolo.h"
#include "mqtt_pub.h"
#include "sig.h"
#include "testEigen.h"
#include <queue>
#include <deque>
#include <list>
#include <unordered_map>
#include "test_thrift.h"
#include "Plugin/PluginLoader.h"
#include "Plugin/IPlugin.h"
#include <iostream>
#include "test_ace.h"
#define COUNT 10
typedef struct tagMaxMatch
{
    int edge[COUNT][COUNT];
    bool on_path[COUNT];
    int path[COUNT];
    int max_match;
} GRAPH_MATCH;

void outputRes(int *path)
{
    for (int i = 0; i < COUNT; i++)
    {
        printf("%d****%d\n", i, *(path + i)); //Yj在前 Xi在后
    }
}

void clearOnPathSign(GRAPH_MATCH *match)
{
    for (int j = 0; j < COUNT; j++)
    {
        match->on_path[j] = false;
    }
}
//dfs算法
bool FindAugPath(GRAPH_MATCH *match, int xi)
{

    for (int yj = 0; yj < COUNT; yj++)
    {
        if (match->edge[xi][yj] == 1 && !match->on_path[yj])
        { //如果yi和xi相连且yi没有在已经存在的增广路经上
            match->on_path[yj] = true;
            if (match->path[yj] == -1 || FindAugPath(match, match->path[yj]))
            {                         // 如果是yi是一个未覆盖点或者和yi相连的xk点能找到增广路经，
                match->path[yj] = xi; //yj点加入路径;
                return true;
            }
        }
    }
    return false;
}

void Hungary_match(GRAPH_MATCH *match)
{
    for (int xi = 0; xi < COUNT; xi++)
    {
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

// extern "C" int func(); //注意这里的声明
// extern int mnist(int argc, char **argv);
#include <vector>
#include <sstream>

void trim(string &s) 
{
    if (s.empty()) 
    {
        return ;
    }
    s.erase(0,s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
}

// std::vector<std::string> split(const std::string& s, char delimiter)
// {
//     std::vector<std::string> tokens;
//     std::string token;
//     std::istringstream tokenStream(s);
//     while (std::getline(tokenStream, token, delimiter))
//     {
//         trim(token);
//         tokens.push_back(token);
//     }
//     return tokens;
// }

// int mymax(const std::vector<int> &score, int left, int right)
// {
//     int max_value = score[left-1];
//     for(int i=left; i<right; ++i)
//     {
//         int k = score[i];
//         if(k > max_value)
//             max_value = k;
//     }
//     return max_value;
// }

int main(int argc, char **argv)
{
    test_ace();
    // Path to the plugin we want to load
    std::string pluginPath("./libPluginExample.so");

    // You first need to instantiate a Plugin::PluginLoader.
    // It takes your interface type as a template argument and the plugin path as a constructor argument.
    Plugin::PluginLoader<Plugin::IPlugin> loader(pluginPath);
    Plugin::IPlugin* plugin = NULL;

    // Then, call Plugin::PluginLoader::load() method to actually load in memory your dynamic library.
    bool isLoaded = loader.load();
    // At this time, MyPlugin class is not instantiated yet.
    if (isLoaded)
    {
        // You have to call Plugin::PluginLoader::getPluginInstance() to create the Singleton instance of MyPlugin.
        // It returns a pointer to your plugin interface type.
        plugin = loader.getPluginInstance();
    }

    // Beware that Plugin::PluginLoader::getPluginInterfaceInstance() method does not give you ownership of Plugin::IPlugin pointer.
    // It means that this instance will be destroyed as soon as Plugin::PluginLoader is destroyed.
    // Any call to the Plugin::IPlugin pointer after Plugin::PluginLoader has been destroyed leads to undefined behavior.

    // And that's all. You can now call any method defined in your interface.

    if (plugin)
    {
        std::cout << "Plugin name    = " << plugin->iGetPluginName() << std::endl;
        std::cout << "Plugin name    = " << plugin->iWhat(5, "czx") << std::endl;
        //std::cout << "Plugin version = " << plugin->iGetPluginVersion() << std::endl;
    }
    else
    {
        std::cout << "Failed to load plugin = " << pluginPath << std::endl;
        std::cout << "Reason = " << loader.getErrorMsg() << std::endl;
    }

//]
    //test_thrift_server();
    //test_thrift_client();
    testEigen();
    // mnist(0, nullptr);
    // Singleton2 *ins = Singleton2::instance();
    // Singleton2 *ins2 = Singleton2::instance();
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

int CheckMax(vector<int> & stu, int a, int b){
    int m = stu[a];
    for (int i = a; i <= b; i++) m = max(m, stu[i]);
    return m;
}

std::vector<std::string> split(std::string &s, char d)
{
    std::vector<std::string> vs;
    std::istringstream is(s);
    std::string token;
    while(std::getline(is, token, d))
    {
        vs.push_back(token);
    }
    return vs;
}

string ReverseSentence(string str) {
    auto vs = split(str, ' ');
    for(auto &v: vs)
    {

        std::cout << v << " " << v.size() << std::endl;
    }
    std::string s;
    for(int i=vs.size()-1; i>=0; --i)
    {
        s += vs[i];
        if(i != 0 || vs[i] == "")
            s += " ";
    }
    return s;
}
bool IsContinuous( vector<int> numbers ) 
{
    int n0=0;
    vector<int> nums;
    for(auto &v: numbers)
    {
        if(v==0)
            ++n0;
        else
            nums.push_back(v);
    }
    sort(nums.begin(), nums.end());
    int first = nums[0];
    int diff = 0;
    for(int i=1; i<nums.size(); ++i)
    {
        if(nums[i] == nums[i-1])
            return false;
        diff += (nums[i] - nums[i-1]) - 1;
    }
    if(diff <= n0)
        return true;
    return false;
}


struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};

//{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}
//1 
class Solution {
public:
    int climbStairs(int n) 
    { 
        if (n == 0 || n == 1) { return 1; } 
        const int size = n; 
        int dp[size] = {0}; 
        for (int i = 0; i < n; i++) 
        { 
            if (i==0 || i== 1) 
            { 
                dp[i] = i + 1; 
            } 
            if (i>=2)
            { 
                dp[i] = dp[i - 1] + dp[i - 2]; 
            }
        } 
        return dp[n-1]; 
    } 
};
class Solution2 
{ 
    public: TreeNode* rebuild(vector<int>& pre, int pre_left, int pre_right, vector<int>& vin, int vin_left, int vin_right) 
    { 
        if (pre_left > pre_right) return nullptr; 
        TreeNode* root = new TreeNode(pre[pre_left]); 
        for (int i=vin_left; i<=vin_right; ++i) 
        { 
            if (vin[i] == root->val) 
            {
            root->left = rebuild(pre, pre_left+1, pre_left+i-vin_left, vin, vin_left, i-1); 
            root->right = rebuild(pre, pre_left+i-vin_left+1, pre_right, vin, i+1, vin_right); 
            break; 
            }
        }
        return root; 
    } 
    TreeNode* reConstructBinaryTree(vector<int> pre, vector<int> vin) 
    {
        return rebuild(pre, 0, pre.size()-1, vin, 0, vin.size()-1); 
    } 
};