int extern_reader(void);
int forward_reader(void);

int main(void) { return (extern_reader() + forward_reader()); }

int extern_value = 5;

int extern_reader(void) {
  extern int extern_value;
  return extern_value;
}

int forward_value = 8;

int forward_reader(void) { return forward_value; }
