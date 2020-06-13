#include <nlohmann/json.hpp>
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <iostream>

bool test_zmq()
{
    using json = nlohmann::json;

    zmq::context_t context(1);
    zmq::socket_t subscriber(context, ZMQ_SUB);
    /*订阅任何内容*/
    //zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);
    subscriber.connect("tcp://localhost:5556");
    char *filter = "timestamp ";
    subscriber.setsockopt(ZMQ_SUBSCRIBE, filter, strlen(filter));

    int ret;
    zmq::pollitem_t item[] = {{subscriber, 0, ZMQ_POLLIN, 0}};
    while (1)
    {
        try
        {
            ret = zmq::poll(item, 1, -1);
            if (ret == -1)
                perror("zmq_poll");
            if (item[0].revents & ZMQ_POLLIN)
            {
                char buff[512];
                zmq::mutable_buffer buffer(buff, sizeof(buff));
                auto ret = subscriber.recv(buffer, zmq::recv_flags::none);
                printf("Timestamp: %d %ld %s \n", ret.value().size, buffer.size(), buffer.data());
                std::string j1((char *)buffer.data() + 10, ret.value().size-10);
                json j2 = json::parse(j1);
                auto s = j2.value("string", "");
                std::cout << s << std::endl;
                break;
            }
        }
        catch (const zmq::error_t &e)
        {
            std::cout << e.what() << '\n';
            return false;
            break;
        }
    }
    //  程序不会运行到这里，以下只是演示我们应该如何结束
    subscriber.close();
    context.close();
    return true;
}
