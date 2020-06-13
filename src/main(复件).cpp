/*
MIT License

Copyright (c) 2019 haskell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rxcpp/rx.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;
// create alias' to simplify code
// these are owned by the user so that
// conflicts can be managed by the user.
namespace rx=rxcpp;
namespace rxu=rxcpp::util;
using namespace cv;
#include<string>

#include "caf/all.hpp"
#include "/usr/local/include/caf/detail/type_traits.hpp"
using namespace std;
using std::string;

using namespace caf;

behavior mirror(event_based_actor* self) {
  // return the (initial) actor behavior
  return {
    // a handler for messages containing a single string
    // that replies with a string
    [=](const string& what) -> string {
      // prints "Hello World!" via aout (thread-safe cout wrapper)
      aout(self) << what << "..." << endl;
      //std::this_thread::sleep_for(std::chrono::seconds(6));
      // reply "!dlroW olleH"
      return string(what.rbegin(), what.rend());
    },
    [=](int value) -> int {
      // prints "Hello World!" via aout (thread-safe cout wrapper)
      aout(self) << "value: " << value << endl;
      // reply "!dlroW olleH"
      return value * value;
    }
  };
}

behavior echo(event_based_actor* self) {
  // return the (initial) actor behavior
  return {
    // a handler for messages containing a single string
    // that replies with a string
    [=](const string& what) -> string {
      // prints "Hello World!" via aout (thread-safe cout wrapper)
      aout(self) << "..." << what << "..." << endl;
      // reply "!dlroW olleH"
      return string(what.begin(), what.end());
    }
  };
}

void hello_world(event_based_actor* self, const actor& buddy) {
  // send "Hello World!" to our buddy ...
  self->request(buddy, std::chrono::seconds(10), "Hello World actor!").then(
    // ... wait up to 10s for a response ...
    [=](const string& what) {
      // ... and print it
      aout(self) << what << "!!!" << endl;
    }
  );
}

behavior foo(event_based_actor* self, const actor& buddy) {
  // send "Hello World!" to our buddy ...
  self->request(buddy, std::chrono::seconds(3), "foo!").then(
    // ... wait up to 10s for a response ...
    [=](const string& what) {
      // ... and print it
      aout(self) << "foo[" << what << "]" << endl;
    }
  );

  self->request(buddy, std::chrono::seconds(12), 25).then(
    // ... wait up to 10s for a response ...
    [=](int value) {
      // ... and print it
      aout(self) << "foo[" << value << "]" << endl;
    }
  );
  return {
    // a handler for messages containing a single string
    // that replies with a string
    [=](const string& what) -> string {
      // prints "Hello World!" via aout (thread-safe cout wrapper)
      aout(self) << "$$$" << what << "$$$" << endl;
      // reply "!dlroW olleH"
      return string(what.begin(), what.end());
    }
  };
}

behavior bar(event_based_actor* self, const actor& buddy) {
  // send "Hello World!" to our buddy ...
  self->request(buddy, std::chrono::seconds(10), "bar!").then(
    // ... wait up to 10s for a response ...
    [=](const string& what) {
      // ... and print it
      aout(self) << "bar[" << what << "]" << endl;
    }
  );
  return {
    // a handler for messages containing a single string
    // that replies with a string
    [=](const string& what) -> string {
      // prints "Hello World!" via aout (thread-safe cout wrapper)
      aout(self) << what << "###" << endl;
      // reply "!dlroW olleH"
      return string(what.begin(), what.end());
    }
  };
}

void caf_main(actor_system& system) {
  // create a new actor that calls 'mirror()'
  auto mirror_actor = system.spawn(mirror);
  // auto echo_actor = system.spawn(echo);
  // create another actor that calls 'hello_world(mirror_actor)';
  // auto hello_world_actor = system.spawn(hello_world, mirror_actor);
  auto foo_actor = system.spawn(foo, mirror_actor);
  //system.spawn(bar, foo_actor);
  // system.spawn(bar, echo_actor);
  // system will wait until both actors are destroyed before leaving main
}

int main(int argc, char **argv)
{
    caf::exec_main(caf_main, argc, argv);
    
    json value = "Hello";
    json array_0 = json(0, value);
    json array_1 = json(1, value);
    json array_5 = json(5, value);

    // serialize the JSON arrays
    std::cout << array_0 << '\n';
    std::cout << array_1 << '\n';
    std::cout << array_5 << '\n';
    
    // create a JSON object with different entry types
    json j =
    {
        {"integer", 1},
        {"floating", 42.23},
        {"string", "hello world"},
        {"boolean", true},
        {"object", {{"key1", 1}, {"key2", 2}}},
        {"array", {1, 2, 3}}
    };
    std::string jsonstr = j.dump();
    std::cout << j.dump() << std::endl;
    json j2 = json::parse(jsonstr);
    std::cout << j2.dump() << std::endl;
    std::fstream f;
    f.open("j.json", std::ios::app);
    f.write(jsonstr.c_str(), jsonstr.length());
    f.close();
    // access existing values
    int v_integer = j.value("integer", 0);
    double v_floating = j.value("floating", 47.11);

    // access nonexisting values and rely on default value
    std::string v_string = j.value("nonexisting", "oops");
    bool v_boolean = j.value("nonexisting", false);

    // output values
    std::cout << std::boolalpha << v_integer << " " << v_floating
              << " " << v_string << " " << v_boolean << "\n";
    
    auto get_names = [](){return rx::observable<>::from<std::string>(
        "Matthew",
        "Aaron"
    );};

    std::cout << "===== println stream of std::string =====" << std::endl;
    auto hello_str = [&](){return get_names().map([](std::string n){
        return "Hello, " + n + "!";
    }).as_dynamic();};

    hello_str().subscribe(rxu::println(std::cout));

    std::cout << "===== println stream of std::tuple =====" << std::endl;
    auto hello_tpl = [&](){return get_names().map([](std::string n){
        return std::make_tuple("Hello, ", n, "! (", n.size(), ")");
    }).as_dynamic();};

    hello_tpl().subscribe(rxu::println(std::cout));

    hello_tpl().subscribe(rxu::print_followed_by(std::cout, " and "), rxu::endline(std::cout));
    return 0;
}
