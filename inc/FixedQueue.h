#pragma once
#include <cstdint>
#include <mutex>
#include <chrono>
#include <memory>
#include <condition_variable>
#include <boost/circular_buffer.hpp>

template <typename T>
class FixedQueue
{
public:
    FixedQueue(std::size_t n) :
        cb(n)
    {
    }

    template <typename K, typename... _Args>
    void write(_Args&&... __args)
    {
        std::unique_lock<std::mutex> lock(mutex);
        auto data = std::make_shared<K>(__args...);
        if(cb.full())
            ++UnReadCnt;
        cb.push_back(data);
        EmptyCond.notify_one();
    }

    void write(const std::shared_ptr<T>& data)
    {
        std::unique_lock<std::mutex> lock(mutex);
        if(cb.full())
            ++UnReadCnt;
        cb.push_back(data);
        EmptyCond.notify_one();
    }
    bool read(std::shared_ptr<T> &data)
    {
        std::unique_lock<std::mutex> lock(mutex);
        if (cb.empty())
            return false;
        data = cb.front();
        cb.pop_front();
        return true;
    }
    bool read(std::shared_ptr<T> &data, int ms)
    {
        std::unique_lock<std::mutex> lock(mutex);
        if (ms)
        {
            if (!EmptyCond.wait_for(lock, std::chrono::milliseconds(ms), [=]() { return !cb.empty(); }))
                return false;
        }
        else
        {
            EmptyCond.wait(lock, [=]() { return !cb.empty(); });
        }
 
        data = cb.front();
        cb.pop_front();
        return true;
    }
    size_t size()
    {
        std::unique_lock<std::mutex> lock(mutex);
        return cb.size();
    }
    size_t GetUnReadCnt()
    {
        std::unique_lock<std::mutex> lock(mutex);
        return UnReadCnt;
    }
private:
    //std::queue<std::shared_ptr<T>> q;
    boost::circular_buffer<std::shared_ptr<T>> cb;
    //size_t MaxCount;
    size_t UnReadCnt{ 0 };
    std::mutex mutex;
    std::condition_variable EmptyCond;
};
