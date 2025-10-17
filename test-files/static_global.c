static int counter = 3;

static int bump(void) {
  counter += 5;
  return counter;
}

int main(void) { return bump() - 2; }
