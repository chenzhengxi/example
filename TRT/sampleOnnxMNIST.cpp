/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir] [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "sampleOptions.h"
#include "parserOnnxConfig.h"
#include "sampleEngines.h"
#include "mnist.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

float percentile(float percentage, std::vector<float> &times)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude <= all)
    {
        std::sort(times.begin(), times.end());
        return times[all == exclude ? 0 : all - 1 - exclude];
    }
    return std::numeric_limits<float>::infinity();
}

int anchors[] = {5, 10, 9, 23, 19, 17, 17, 41, 37, 32, 31, 72, 65, 65, 97, 134};
std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BBox &bbox1, BBox &bbox2) -> float {
        float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
        float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
        float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
        float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const BBoxInfo &b1, const BBoxInfo &b2) { return b1.prob > b2.prob; });
    std::vector<BBoxInfo> out;
    for (auto &i : binfo)
    {
        bool keep = true;
        for (auto &j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i.box, j.box);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep)
            out.push_back(i);
    }
    return out;
}
std::vector<BBoxInfo> nmsAllClasses(const float nmsThresh, std::vector<BBoxInfo> &binfo,
                                    const uint numClasses)
{
    std::vector<BBoxInfo> result;
    std::vector<std::vector<BBoxInfo>> splitBoxes(numClasses);
    for (auto &box : binfo)
    {
        splitBoxes.at(box.label).push_back(box);
    }

    for (auto &boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }

    return result;
}
float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

BBox convertBBoxNetRes(const float &bx, const float &by, const float &bw, const float &bh,
                       const uint &stride, const uint &netW, const uint &netH)
{
    BBox b;
    // Restore coordinates to network input resolution
    float x = bx * stride;
    float y = by * stride;

    b.x1 = x - bw / 2;
    b.x2 = x + bw / 2;

    b.y1 = y - bh / 2;
    b.y2 = y + bh / 2;

    b.x1 = clamp(b.x1, 0, netW);
    b.x2 = clamp(b.x2, 0, netW);
    b.y1 = clamp(b.y1, 0, netH);
    b.y2 = clamp(b.y2, 0, netH);

    return b;
}

void convertBBoxImgRes(const float scalingFactor, const float &xOffset, const float &yOffset,
                       BBox &bbox)
{
    // Undo Letterbox
    bbox.x1 -= xOffset;
    bbox.x2 -= xOffset;
    bbox.y1 -= yOffset;
    bbox.y2 -= yOffset;

    // Restore to input resolution
    bbox.x1 /= scalingFactor;
    bbox.x2 /= scalingFactor;
    bbox.y1 /= scalingFactor;
    bbox.y2 /= scalingFactor;
}

void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                     const uint stride, const float scalingFactor, const float xOffset,
                     const float yOffset, const int maxIndex, const float maxProb,
                     std::vector<BBoxInfo> &binfo)
{
    BBoxInfo bbi;
    bbi.box = convertBBoxNetRes(bx, by, bw, bh, stride, 800, 800);
    if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2))
    {
        return;
    }
    convertBBoxImgRes(scalingFactor, xOffset, yOffset, bbi.box);
    bbi.label = maxIndex;
    bbi.prob = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
};

std::vector<BBoxInfo> decodeTensor(const float *hostBuffer, const int imageH, const int imageW, const int gridSize, const int numClasses, const int numBBoxes, int masks[])
{
    sample::BuildOptions option;
    const float *detections = hostBuffer;
    float scalingFactor = std::min(static_cast<float>(800) / imageW, static_cast<float>(800) / imageH);
    float xOffset = (800 - scalingFactor * imageW) / 2;
    float yOffset = (800 - scalingFactor * imageH) / 2;
    std::vector<BBoxInfo> binfo;
    for (uint y = 0; y < gridSize; ++y)
    {
        for (uint x = 0; x < gridSize; ++x)
        {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const float pw = anchors[masks[b] * 2];
                const float ph = anchors[masks[b] * 2 + 1];

                const int numGridCells = gridSize * gridSize;
                const int bbindex = y * gridSize + x;
                const float bx = x + detections[bbindex + numGridCells * (b * (5 + numClasses) + 0)];
                const float by = y + detections[bbindex + numGridCells * (b * (5 + numClasses) + 1)];
                const float bw = pw * detections[bbindex + numGridCells * (b * (5 + numClasses) + 2)];
                const float bh = ph * detections[bbindex + numGridCells * (b * (5 + numClasses) + 3)];
                const float objectness = detections[bbindex + numGridCells * (b * (5 + numClasses) + 4)];

                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint i = 0; i < numClasses; ++i)
                {
                    float prob = (detections[bbindex + numGridCells * (b * (5 + numClasses) + (5 + i))]);

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                if (maxProb > 0.8f && maxProb < 1.0f)
                {
                    addBBoxProposal(bx, by, bw, bh, 800.0 / gridSize, scalingFactor, xOffset, yOffset,
                                    maxIndex, maxProb, binfo);
                }
            }
        }
    }
    return binfo;
}

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build()
{
    cudaSetDevice(0);

    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(sample::loadEngine("data/visdrone.trt", -1, gLogError), samplesCommon::InferDeleter());
    printf("....................3\n");
    if (!mEngine)
    {
        return false;
    }
    int index = mEngine->getBindingIndex(mParams.inputTensorNames[0].c_str());
    mInputDims = mEngine->getBindingDimensions(index);
    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                                       SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                                       SampleUniquePtr<nvonnxparser::IParser> &parser)
{
    auto parsed = parser->parseFromFile(
        locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), 1);
    printf("...........................2\n");

    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1_GiB);
    //config->setFlag(BuilderFlag::kFP16);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    sample::AllOptions options;
    options.inference.batch = 1;
    IExecutionContext *context = mEngine->createExecutionContext();

    // Dump inferencing time per layer basis
    SimpleProfiler profiler("Layer time");
    if (options.reporting.profile)
    {
        context->setProfiler(&profiler);
    }

    for (int b = 0; b < mEngine->getNbBindings(); ++b)
    {
        if (!mEngine->bindingIsInput(b))
        {
            continue;
        }
        auto dims = context->getBindingDimensions(b);
        if (dims.d[0] == -1)
        {
            auto shape = options.inference.shapes.find(mEngine->getBindingName(b));
            if (shape == options.inference.shapes.end())
            {
                gLogError << "Missing dynamic batch size in inference" << std::endl;
                return false;
            }
            dims.d[0] = shape->second.d[0];
            context->setBindingDimensions(b, dims);
        }
    }
    // Use an aliasing shared_ptr since we don't want engine to be deleted when bufferManager goes out of scope.
    std::shared_ptr<ICudaEngine> emptyPtr{};
    std::shared_ptr<ICudaEngine> aliasPtr(emptyPtr, &*mEngine);

    samplesCommon::BufferManager bufferManager(aliasPtr, 1, nullptr);
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(bufferManager))
    {
        return false;
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    unsigned int cudaEventFlags = options.inference.spin ? cudaEventDefault : cudaEventBlockingSync;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventFlags));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventFlags));

    bufferManager.copyInputToDeviceAsync(stream);
    std::vector<void *> buffers = bufferManager.getDeviceBindings();

    std::vector<float> times(options.reporting.avgs);
    for (int j = 0; j < 1; j++)
    {
        float totalGpu{0};  // GPU timer
        float totalHost{0}; // Host timer

        for (int i = 0; i < options.reporting.avgs; i++)
        {
            auto tStart = std::chrono::high_resolution_clock::now();
            cudaEventRecord(start, stream);
            context->enqueueV2(&buffers[0], stream, nullptr);
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);

            auto tEnd = std::chrono::high_resolution_clock::now();
            totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[i] = ms;
            totalGpu += ms;
        }

        totalGpu /= options.reporting.avgs;
        totalHost /= options.reporting.avgs;
        gLogInfo << "Average over " << options.reporting.avgs << " runs is " << totalGpu << " ms (host walltime is "
                 << totalHost << " ms, " << static_cast<int>(options.reporting.percentile) << "\% percentile time is "
                 << percentile(options.reporting.percentile, times) << ")." << std::endl;
    }

    bufferManager.copyOutputToHost();
    int nbBindings = mEngine->getNbBindings();
    std::vector<BBoxInfo> binfo;
    for (int i = 0; i < nbBindings; i++)
    {
        if (!mEngine->bindingIsInput(i))
        {
            const char *tensorName = mEngine->getBindingName(i);
            void *buf = bufferManager.getHostBuffer(tensorName);
            size_t bufSize = bufferManager.size(tensorName);
            nvinfer1::Dims bufDims = mEngine->getBindingDimensions(i);
            nvinfer1::DataType t = mEngine->getBindingDataType(i);
            std::cout << "Dumping output tensor " << tensorName << ":" << bufSize << std::endl;
            std::vector<BBoxInfo> curBInfo;
            if (std::string("output_0") == tensorName)
            {
                int masks[] = {0, 1, 2, 3};
                curBInfo = decodeTensor((const float *)buf, frame.rows, frame.cols, 100, 10, 4, masks);
            }
            else
            {
                int masks[] = {4, 5, 6, 7};
                curBInfo = decodeTensor((const float *)buf, frame.rows, frame.cols, 50, 10, 4, masks);
            }
            binfo.insert(binfo.end(), curBInfo.begin(), curBInfo.end());
            //bufferManager.dumpBuffer(gLogInfo, tensorName);
        }
    }
    cudaStreamSynchronize(stream);
    auto remaining = nmsAllClasses(0.7, binfo, 10);

    for (const auto &b : remaining)
    {
        BBox bb = b.box;
        if (bb.x1 < 0)
            bb.x1 = 0;
        if (bb.x2 > frame.cols - 1)
            bb.x2 = frame.cols - 1;
        if (bb.y1 < 0)
            bb.y1 = 0;
        if (bb.y2 > frame.rows - 1)
            bb.y2 = frame.rows - 1;
        float w = bb.x2 - bb.x1;
        float h = bb.y2 - bb.y1;
        if (w <= 0 || h <= 0)
            continue;
        cv::putText(frame, std::to_string(b.label), cv::Point(bb.x1, bb.y1), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(65, 105, 225));
        cv::rectangle(frame, cv::Rect(bb.x1, bb.y2, w, h), cv::Scalar(255, 140, 0));
    }
    cv::imshow("visdrone", frame);
    cv::waitKey();
    if (options.reporting.profile)
    {
        gLogInfo << profiler;
    }

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    context->destroy();
    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager &buffers)
{
    const int inputH = mInputDims.d[3];
    const int inputW = mInputDims.d[2];
    frame = cv::imread("1.jpg", cv::ImreadModes::IMREAD_COLOR);
    cv::Mat inferImage = cv::dnn::blobFromImage(frame, 1.0, cv::Size(inputW, inputH),
                                                cv::Scalar(0.0, 0.0, 0.0), false, false);
    // inferNet->doInference(inferImage.data, 1);
    // auto binfo = inferNet->decodeDetections(0, frame.rows, frame.cols);
    // auto remaining = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());

    srand(unsigned(time(nullptr)));
    int InputSize = mInputDims.d[1] * mInputDims.d[2] * mInputDims.d[3];
    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    memcpy(hostDataBuffer, inferImage.data, InputSize * sizeof(float));
    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager &buffers)
{
    const int outputSize = mOutputDims.d[0];
    float *output = static_cast<float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
    }
    gLogInfo << std::endl;

    return idx == mNumber && val > 0.9f;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args &args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.batchSize = 1;
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]" << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int mnist(int argc, char **argv)
{
    // samplesCommon::Args args;
    // bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    // if (!argsOK)
    // {
    //     gLogError << "Invalid arguments" << std::endl;
    //     printHelpInfo();
    //     return EXIT_FAILURE;
    // }
    // if (args.help)
    // {
    //     printHelpInfo();
    //     return EXIT_SUCCESS;
    // }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    ///////////////////////////////////////////////
    //SampleOnnxMNIST sample(initializeSampleParams(args));
    samplesCommon::OnnxSampleParams params;
    params.dataDirs.push_back("data");
    params.onnxFileName = "visdrone.onnx";
    params.inputTensorNames.push_back("input_0");
    params.batchSize = 1;
    params.outputTensorNames.push_back("output_0");
    params.outputTensorNames.push_back("output_1");
    //params.fp16 = args.runInFp16;

    SampleOnnxMNIST sample(params);
    ///////////////////////////////////////////////

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
