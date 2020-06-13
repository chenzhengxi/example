#pragma once
#include <cstdio>
class czx
{
public:
    czx(int a, int b):a(a), b(b)
    {
        printf("...%d...%d\n", a, b);
    }
private:
    int a;
    int b;
};
