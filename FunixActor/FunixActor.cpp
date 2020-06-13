#include "FunixActor.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/conversion.hpp>
#include <experimental/functional>
#include <experimental/algorithm>
#include <fstream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

void FunixActor::init()
{
    //LOG(INFO) << "FileStore+ id: " << GetHashID().value_or(0);
}

void FunixActor::fini()
{
    //LOG(INFO) << "FileStore-";
}

bool FunixActor::timeout()
{
    return true;
}
