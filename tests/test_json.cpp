#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

using json = nlohmann::json;

bool test_json()
{
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
          {"array", {1, 2, 3}}};
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
  return true;
}
