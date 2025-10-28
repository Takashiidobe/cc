#include "includes/add.h"

int main(void) {
  int sum = add(4, 5);
  int difference = sub(25, 5);
  int product = mul(3, 4);
  int dividend = div(35, 5);

  return sum + difference + product + dividend;
}
