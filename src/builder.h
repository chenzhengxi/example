#pragma once
#include "product.h"

class Product;
class abstracProduct;
class abstractBuilder
{
public:
	virtual~abstractBuilder() {};
	virtual abstracProduct*CreateProduct()=0;
};

class Builder :public abstractBuilder
{
public:
	abstracProduct*CreateProduct()
	{
		mProductInstance = new Product;
		return mProductInstance;
	};
	void setPartA() { mProductInstance->setPartA(); };
	void setPartB() { mProductInstance->setPartB(); };
	void setPartC() { mProductInstance->setPartC(); };
private:
	abstracProduct*mProductInstance;
};
