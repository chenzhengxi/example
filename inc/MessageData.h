#pragma once
#include "MessageBase.h"
#include <cstdlib>
#include <cstdint>
#include <time.h>
#include <memory>
#include <boost/asio/ip/udp.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <chrono>

//struct TIME_FRAME_ARG final:
//        public MessageBase
//{
//    TIME_FRAME_ARG(int _channel, const timeval &_val,  cv::Mat _ColorFrame):
//        MessageBase(MESSAGE_TYPE::MESSAGE_DETECTFRAME), channel(_channel), val(_val), ColorFrame(_ColorFrame)
//    {
//    }
//    TIME_FRAME_ARG(int _channel, const timeval &_val, const cv::Mat &_ColorFrame, cv::Mat &_DepthFrame):
//        MessageBase(MESSAGE_TYPE::MESSAGE_DETECTFRAME), channel(_channel), val(_val), ColorFrame(_ColorFrame), DepthFrame(_DepthFrame)
//    {
//    }
//    int channel;
//    timeval val;
//    cv::Mat ColorFrame;
//    cv::Mat DepthFrame;
//};

struct MD_1 final:
        public MessageBase
{
    MD_1(int _id):
        MessageBase(typeid(*this).hash_code()), id(_id)
    {
    }
    int id;
};

struct MD_2 final:
        public MessageBase
{
    MD_2(int _id):
        MessageBase(typeid(*this).hash_code()), id(_id)
    {
    }
    int id;
};

struct MD_EXIT final:
        public MessageBase
{
    MD_EXIT(int _err):
        MessageBase(typeid(*this).hash_code()), err(_err)
    {
    }
    int err;
};
