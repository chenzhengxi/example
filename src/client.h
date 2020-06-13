#pragma once
#include <string>
#include "computer.h"

using namespace std;
class CComputer;
class CBuilder
{
public:
	virtual void buildBoard(string board) = 0;
	virtual void buildOS(string os) = 0;
	virtual void buildDisplay(string display) = 0;
	virtual CComputer*CreateComputer()=0;
	virtual ~CBuilder();
protected:
	CBuilder(){}
};

class MacBuilder :public CBuilder
{
public:
	void buildBoard(string board);
	void buildOS(string os);
	void buildDisplay(string display);
	CComputer *CreateComputer();
	MacBuilder();
	CComputer *mMac;
};
