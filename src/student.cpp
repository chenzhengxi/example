#include "student.h"

Student::Student()
{
}

Student::~Student()
{
}

string Student::getName()
{
    return this->name;
}
void Student::setName(string name)
{
    this->name = name;
}
int Student::getAge()
{
    return this->age;
}
void Student::setAge(int age)
{
    this->age = age;
}
int Student::getType()
{
    return this->type;
}
void Student::setType(int type)
{
    this->type = type;
}
