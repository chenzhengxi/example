#pragma once
#include "MessageData.h"
#include "actor.h"
#include <experimental/filesystem>
#include <vector>
#include <functional>
#include <iostream>

namespace fs = std::experimental::filesystem;

class CzxActor final : public actor
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
bool CzxActor::handle(std::shared_ptr<const MD_2> MD)
{
    std::cout << __FUNCTION__ << " " << MD->id << std::endl;
    return false;
}