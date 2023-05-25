def find_gen(line):
    near_zero = lambda x, y: abs(x) <= 0.15 and abs(y) <= 0.15
    cnt = 0
    for i in range(0, len(line) - 1, 2):
        x = float(line[i])
        y = float(line[i + 1])
        if near_zero(x, y):
            cnt += 1
    print(1 if 100 <= cnt else 2)


if __name__ == "__main__":
    for _ in range(100):
        line = input().split()
        find_gen(line)
