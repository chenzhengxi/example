#include "CzxActor.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/conversion.hpp>
#include <experimental/functional>
#include <experimental/algorithm>
#include <fstream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

void CzxActor::init()
{
    //LOG(INFO) << "FileStore+ id: " << GetHashID().value_or(0);
}

void CzxActor::fini()
{
    //LOG(INFO) << "FileStore-";
}

bool CzxActor::timeout()
{
    return true;
}
