#pragma once
#include "FixedQueue.h"
#include "MessageBase.h"
#include <map>
#include <memory>
#include <chrono>
#include <functional>
#include <exception>
#include <thread>
#include <boost/optional.hpp>

class actor_base
{
public:
    explicit actor_base(std::size_t n) : message(n) {}
    template <typename T>
    bool post(std::shared_ptr<const T> MD);
    
    std::multimap<size_t, std::shared_ptr<actor_base>> output;
    FixedQueue<const MessageBase> message;
    std::map<size_t, std::function<bool(std::shared_ptr<const MessageBase>)>> MessageHandler;
};

class actor : public actor_base
{
    //using MTID = typename MT::MyType;
public:
    /*
     * n: queue size
     * wait: pop timeout, 0 for ever, unit: ms
     */
    explicit actor(int ms = 1000, std::size_t n = 128)
        : actor_base(n), waitms(ms)
    {
    }
    virtual ~actor() = 0;
    void operator()();
    void wait();
    boost::optional<size_t> GetHashID() const
    {
        return hash_id;
    }

    template <typename T, typename... _Args>
    static std::shared_ptr<T> make(_Args &&... __args);

    template <typename T, typename MT2>
    static typename std::enable_if<std::is_base_of<MessageBase, MT2>::value && std::is_base_of<actor_base, T>::value, bool>::type
    connect(actor_base *source, std::shared_ptr<T> sink, bool (T::*handler)(std::shared_ptr<const MT2>));

    template <typename MT2, typename T>
    static typename std::enable_if<std::is_base_of<MessageBase, MT2>::value && std::is_base_of<actor_base, T>::value, bool>::type
    connect(actor_base *source, std::shared_ptr<T> sink);
private:
    virtual void init() = 0;
    virtual void fini() = 0;
    virtual bool timeout() = 0;

private:
    boost::optional<size_t> hash_id;
    std::chrono::system_clock::time_point OldTime;
    const int waitms;
    std::shared_ptr<std::thread> thrActor;
};

template <typename T>
bool actor_base::post(std::shared_ptr<const T> MD)
{
    auto iter = output.find(typeid(T).hash_code());
    if (iter == output.end())
        return false;
    auto ret = output.equal_range(iter->first);
    for (auto p = ret.first; p != ret.second; ++p)
    {
        if (!p->second)
            continue;
        //MD->code = typeid(T).hash_code();
        p->second->message.write(MD);
    }
    return true;
}

// template <typename T, typename... _Args>
// bool actor_base::post_emplace(_Args &&... __args)
// {
//     auto iter = output.find(typeid(T).hash_code());
//     if (iter == output.end())
//         return false;
//     auto ret = output.equal_range(iter->first);
//     for (auto p = ret.first; p != ret.second; ++p)
//     {
//         if (!p->second)
//             continue;
//         auto MD = std::make_shared<const T>(__args...);
//         //MD->code = typeid(T).hash_code();
//         p->second->message.write(MD);
//     }
//     return true;
// }

template <typename T, typename... _Args>
std::shared_ptr<T> actor::make(_Args &&... __args)
{
    auto theActor = std::make_shared<T>(__args...);
    auto thr = std::make_shared<std::thread>(std::ref(*theActor));
    //theActor->setThread(thr);
    theActor->thrActor = thr;
    return theActor;
}

template <typename T, typename MT2>
typename std::enable_if<std::is_base_of<MessageBase, MT2>::value && std::is_base_of<actor_base, T>::value, bool>::type
actor::connect(actor_base *source, std::shared_ptr<T> sink, bool (T::*handler)(std::shared_ptr<const MT2>))
{
    using handlerImpl = bool (T::*)(std::shared_ptr<const MessageBase>);
    if (source)
    {
        source->output.insert(std::pair<size_t, std::shared_ptr<actor_base>>(typeid(MT2).hash_code(), sink));
    }
    if (sink && handler)
    {
        sink->MessageHandler[typeid(MT2).hash_code()] = std::bind(reinterpret_cast<handlerImpl>(handler), sink, std::placeholders::_1);
    }
    return true;
}

template <typename MT2, typename T>
typename std::enable_if<std::is_base_of<MessageBase, MT2>::value && std::is_base_of<actor_base, T>::value, bool>::type
actor::connect(actor_base *source, std::shared_ptr<T> sink)
{
    using handlerImpl = bool (T::*)(std::shared_ptr<const MessageBase>);
    if (source)
    {
        source->output.insert(std::pair<size_t, std::shared_ptr<actor_base>>(typeid(MT2).hash_code(), sink));
    }
    if (sink)
    {
        using myhandler = bool (T::*)(std::shared_ptr<const MT2>);
        sink->MessageHandler[typeid(MT2).hash_code()] = std::bind(reinterpret_cast<handlerImpl>((myhandler)(&T::handle)), sink, std::placeholders::_1);
    }
    return true;
}
