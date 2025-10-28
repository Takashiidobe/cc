void *malloc(unsigned long size);
void free(void *ptr);

int main(void) {
  void *x = malloc(5);
  unsigned long size = sizeof(x);
  free(x);
  return size;
}
