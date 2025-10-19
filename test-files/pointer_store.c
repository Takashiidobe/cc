int main(void) {
  int value = 7;
  int *ptr = &value;
  *ptr = *ptr + 5;
  return value;
}
