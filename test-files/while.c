int main(void) {
    int i = 0;
    int sum = 0;
    while (i < 5) {
        i = i + 1;
        if (i == 3) {
            continue;
        }
        sum += i;
        if (i == 4) {
            break;
        }
    }
    return sum + i;
}
