int main(void) {
    int data[4];
    data[0] = 3;
    data[1] = 8;
    data[2] = 13;
    data[3] = 21;

    int *ptr = data;
    int *next = ptr + 2;
    return *next;
}
