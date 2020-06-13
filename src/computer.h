#pragma once
#include <string>
#include <iostream>

using namespace std;

class CComputer
{
public:
    virtual void setBoard(string board) = 0;
    virtual void setOS(string os) = 0;
    virtual void setDisplay(string display) = 0;
    void ShowComputer();

protected:
    CComputer(){};
    string mBoard;
    string mOS;
    string mDisplay;
};

class MacBook : public CComputer
{
public:
    MacBook(){}

public:
    void setBoard(string board)
    {
        std::cout << __FUNCTION__ << std::endl;
    }
    void setOS(string os){std::cout << __FUNCTION__ << std::endl;}
    void setDisplay(string display){std::cout << __FUNCTION__ << std::endl;}
};
