void *malloc(unsigned long size);

int main(void) {
  void *x = malloc(5);
  return sizeof(x);
}
