#include <iostream>
using namespace std;

class Singleton
{
  private:
    static Singleton *local_instance;
    Singleton(){
        cout << "构造" << endl;
    };
    ~Singleton(){
        cout << "析构" << endl;
    }
    static int k;
  public:
    static Singleton *getInstance()
    {
        static Singleton locla_s;
        return &locla_s;
    }
};

class Singleton2
{
public:
    static Singleton2 *instance()
    {
        static Singleton2 ins;
        return &ins;
    }
    private:
      Singleton2(){cout << "ggg" << endl;}
      ~Singleton2(){cout << "xxx" << endl;}
      //Singleton2& operator=(const Singleton2&){}
};