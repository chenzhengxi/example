#include "actor.h"

actor::~actor()
{
}

void actor::operator()()
{
    hash_id = std::hash<std::thread::id>()(std::this_thread::get_id());
    init();
    OldTime = std::chrono::system_clock::now();
    bool flag1{true}, flag2{true};
    while (flag1 && flag2)
    {
        std::shared_ptr<const MessageBase> MD;
        actor_base::message.read(MD, waitms);
        if (MD && MD->code)
        {
            auto iter = actor_base::MessageHandler.find(MD->code.value());
            if (iter != actor_base::MessageHandler.end())
            {
                flag1 = iter->second(MD);
                //try {
                //    iter->second(MD);
                //} catch (const MQException &ex) {
                //    //std::cout << ex.what() << std::endl;
                //    LOG(INFO) << ex;
                //    break;
                //}
            }
        }

        if (waitms)
        {
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - OldTime);
            if (diff >= std::chrono::milliseconds(waitms))
            {
                flag2 = this->timeout();
                OldTime = std::chrono::system_clock::now();
            }
        }
    }
    fini();
}

void actor::wait()
{
    if (thrActor)
        thrActor->join();
}
