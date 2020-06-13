#pragma once
#include "MessageData.h"
#include "actor.h"
#include <experimental/filesystem>
#include <vector>
#include <functional>
#include <iostream>

namespace fs = std::experimental::filesystem;

class FunixActor final : public actor
{
public:
    template <typename T>
    bool handle(std::shared_ptr<const T> MD);

private:
    void init() final;
    void fini() final;
    bool timeout() final;

private:
};

template <>
bool FunixActor::handle(std::shared_ptr<const MD_1> MD)
{
    std::cout << __FUNCTION__ << "++++++++++" << MD->id << std::endl;
    auto MD2 = std::make_shared<const MD_2>(89);
    post(MD2);
    return false;
}
