#include <string>
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <time.h>

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

using json = nlohmann::json;

void test_json()
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
}

int main(int argc, char *argv[])
{
	zmq::context_t context(1);
	zmq::socket_t publisher(context, ZMQ_PUB);
	publisher.bind("tcp://*:5556");
	json j = {
		{"integer", 1},
		{"floating", 42.23},
		{"string", "hello world"},
		{"boolean", true},
		{"object", {{"key1", 1}, {"key2", 2}}},
		{"array", {1, 2, 3}}};
	std::string jsonstr = j.dump();
	int k{0};
	while (1)
	{
		// Send timestamp to all subscribers
		char timestamp[512] = {0};
		sprintf(timestamp, "timestamp %s", jsonstr.c_str());
		zmq::const_buffer buffer(timestamp, strlen(timestamp));
		publisher.send(buffer, zmq::send_flags::none);
		zmq_sleep(2);
	}
	publisher.close();
	context.close();
	return 0;
}
