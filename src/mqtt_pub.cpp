#include <iostream>
#include <cstdlib>
#include <string>
#include <thread>	// For sleep
#include <atomic>
#include <chrono>
#include <string>
#include "mqtt/async_client.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

using json = nlohmann::json;
using namespace std;

const string DFLT_SERVER_ADDRESS { "tcp://localhost:1883" };

const string TOPIC { "test" };
const int QOS = 1;

const char* PAYLOADS[] = {
	"Hello World!",
	"Hi there!",
	"Is anyone listening?",
	"Someone is always listening.",
	nullptr
};

const auto TIMEOUT = std::chrono::seconds(10);

/////////////////////////////////////////////////////////////////////////////
struct data
{
    int x;
    int y;
};

bool mqtt_pub()
{
	string address = DFLT_SERVER_ADDRESS;

	cout << "Initializing for server '" << address << "'..." << endl;
	mqtt::async_client cli(address, "");

	cout << "  ...OK" << endl;

	try {
		cout << "\nConnecting..." << endl;
		cli.connect()->wait();
		cout << "  ...OK" << endl;

		cout << "\nPublishing messages..." << endl;

		mqtt::topic top(cli, "czx", QOS);
		mqtt::token_ptr tok;

		size_t i = 0;
		// while (PAYLOADS[i]) {
		// 	tok = top.publish(PAYLOADS[i++]);
		// }

  json j =
      {
          {"integer", 1},
          {"floating", 42.23},
          {"string", "hello world"},
          {"boolean", true},
          {"object", {{"key1", 1}, {"key2", 2}}},
          {"array", {1, 2, 3}}};
  std::string jsonstr = j.dump();

        tok = top.publish(jsonstr);
		tok->wait();	// Just wait for the last one to complete.
		
		cout << "OK" << endl;

		// Disconnect
		cout << "\nDisconnecting..." << endl;
		cli.disconnect()->wait();
		cout << "  ...OK" << endl;
	}
	catch (const mqtt::exception& exc) {
		cerr << exc.what() << endl;
		return false;
	}

 	return true;
}
