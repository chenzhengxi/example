#pragma once
#include <boost/optional.hpp>

struct MessageBase
{
    explicit MessageBase(size_t _code):code(_code){}
    const boost::optional<size_t> code;
};

