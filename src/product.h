#pragma once
#include <string>
#include <iostream>

using namespace std;

class abstracProduct
{
public:
	virtual ~abstracProduct() {};
	virtual void setPartA() = 0;
	virtual void setPartB() = 0;
	virtual void setPartC() = 0;
};

class Product :public abstracProduct
{
public:
	void setPartA() {};
	void setPartB() {};
	void setPartC() {};
private:
	Product() = default;
	string mPartA;
	string mPartB;
	string mPartC;
	friend class Builder;
};

