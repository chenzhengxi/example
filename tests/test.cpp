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

#include <fstream>
#include <opencv2/opencv.hpp>
#include "test_json.h"
#include "test_mqtt.h"
#include "test_zmq.h"
#include "test_actor.h"
#include "WordStatics.h"
#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

TEST_CASE( "测试actor", "[MQ]" ) {
    REQUIRE( test_actor() );
}

TEST_CASE("测试空字符串", "[WORDS]")
{
    REQUIRE( WordStatics("") == "" );
}

TEST_CASE("测试1字符串", "[WORDS]")
{
    REQUIRE( WordStatics("he") == "he:1\r\n" );
}

TEST_CASE("测试2字符串", "[WORDS]")
{
    REQUIRE( WordStatics("he is") == "he:1\r\nis:1\r\n" );
}

TEST_CASE("测试1重复字符串", "[WORDS]")
{
    REQUIRE( WordStatics("he he") == "he:2\r\n" );
}

TEST_CASE("测试排序", "[WORDS]")
{
    REQUIRE( WordStatics("he he is") == "he:2\r\nis:1\r\n" );
    REQUIRE( WordStatics("he is he") == "he:2\r\nis:1\r\n" );
}

TEST_CASE( "测试mqtt", "[MQ]" ) {
    REQUIRE( test_mqtt() );
}

TEST_CASE( "测试ZeroMQ", "[MQ]" ) {
    REQUIRE( test_zmq() );
}

TEST_CASE( "测试json", "[MQ]" ) {
    REQUIRE( test_json() );
}

SCENARIO( "动态数组 can be sized and resized", "[vector]" ) {

    GIVEN( "A vector with some items" ) {
        std::vector<int> v( 5 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 5 );

        WHEN( "the size is increased" ) {
            v.resize( 10 );

            THEN( "the size and capacity change" ) {
                REQUIRE( v.size() == 10 );
                REQUIRE( v.capacity() >= 10 );
            }
        }
        WHEN( "the size is reduced" ) {
            v.resize( 0 );

            THEN( "the size changes but not capacity" ) {
                REQUIRE( v.size() == 0 );
                REQUIRE( v.capacity() >= 5 );
            }
        }
        WHEN( "more capacity is reserved" ) {
            v.reserve( 10 );

            THEN( "the capacity changes but not the size" ) {
                REQUIRE( v.size() == 5 );
                REQUIRE( v.capacity() >= 10 );
            }
        }
        WHEN( "less capacity is reserved" ) {
            v.reserve( 0 );

            THEN( "neither size nor capacity are changed" ) {
                REQUIRE( v.size() == 5 );
                REQUIRE( v.capacity() >= 5 );
            }
        }
    }
}
