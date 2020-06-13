#include "client.h"

CBuilder::~CBuilder()
{

}

MacBuilder::MacBuilder()
{
	mMac = new MacBook;
}
void MacBuilder::buildBoard(string board)
{
	mMac->setBoard(board);
}
void MacBuilder::buildOS(string os)
{
	mMac->setOS(os);
}
void MacBuilder::buildDisplay(string display)
{
	mMac->setDisplay(display);
}
CComputer *MacBuilder::CreateComputer()
{
	return mMac;
}
