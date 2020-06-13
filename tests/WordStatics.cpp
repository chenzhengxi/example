#include "WordStatics.h"
#include <map>
#include <vector>
#include <algorithm>

std::string format(const std::pair<std::string, int> &words)
{
    return words.first + ":" + std::to_string(words.second);
}

std::map<std::string, int> split(const std::string &words)
{
    int pos = 0;
    std::map<std::string, int> tmp;
    for (int i = 0; i < words.size(); ++i)
    {
        if (words[i] == ' ')
        {
            tmp[words.substr(pos, i - pos)]++;
            pos = i + 1;
        }
    }
    if (pos < words.size())
    {
        tmp[words.substr(pos, words.size() - pos)]++;
    }

    return tmp;
}

bool cmp_by_value(const std::pair<std::string, int> &lhs, const std::pair<std::string, int> &rhs)
{
    return lhs.second > rhs.second;
}

std::string WordStatics(const std::string &words)
{
    std::map<std::string, int> subword = split(words);
    std::vector<std::pair<std::string, int>> vec;
    for (std::map<std::string, int>::iterator it = subword.begin(); it != subword.end(); it++)
    {
        vec.push_back(std::pair<std::string, int>(it->first, it->second));
    }

    std::sort(vec.begin(), vec.end(), cmp_by_value);
    std::string outstr;
    for (auto &&value : vec)
    {
        outstr += (format(value) + "\r\n");
    }
    return outstr;
}
