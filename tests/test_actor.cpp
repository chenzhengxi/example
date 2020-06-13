#include "actor.h"
#include "CzxActor.h"
#include "FunixActor.h"

bool test_actor()
{
    auto theCzxActor = actor::make<CzxActor>();
    std::shared_ptr<FunixActor> theFunixActor = actor::make<FunixActor>();

    //actor::connect(theCzxActor.get(), theFunixActor, &FunixActor::handle<MD_1>);
    actor::connect<MD_1>(theCzxActor.get(), theFunixActor);
    actor::connect<MD_2>(theFunixActor.get(), theCzxActor);
    auto MD1 = std::make_shared<const MD_1>(5);
    theCzxActor->post(MD1);
    if(theCzxActor)
        theCzxActor->wait();
    if(theFunixActor)
        theFunixActor->wait();
    return true;
}
