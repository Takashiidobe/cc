int main(void) {
  int x = 0;
  int count = 0;
  do {
    x++;
    if (x == 2) {
      continue;
    }
    count++;
  } while (x < 3);
  return count;
}
