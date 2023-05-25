import numpy as np
import pandas as pd


def generate1():
    """Генератор 1 из условия задачи"""
    a = np.random.uniform(0, 1)
    b = np.random.uniform(0, 1)
    return (a * np.cos(2 * np.pi * b), a * np.sin(2 * np.pi * b))


def generate2():
    """Генератор 2 из условия задачи"""
    while True:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if x**2 + y**2 > 1:
            continue
        return (x, y)


def create_pointset(n):
    d1 = pd.DataFrame([generate1() for _ in range(n)], columns=["x", "y"])
    d1["gen_number"] = "Генератор 1"
    d2 = pd.DataFrame([generate2() for _ in range(n)], columns=["x", "y"])
    d2["gen_number"] = "Генератор 2"
    pointset = pd.concat([d1, d2]).reset_index().drop(columns=["index"])
    return pointset


def number_nzpoints(pointset):
    near_zero_point = lambda point: abs(point["x"]) <= 0.15 and abs(point["y"]) <= 0.15
    pointset["near_zero_point"] = pointset.apply(near_zero_point, axis=1)
    pointset = pointset.groupby(["gen_number"], as_index=False)["near_zero_point"].agg(
        size="size", near_zero="sum"
    )
    return pointset


def find_threshold(n=100):
    near_zero = lambda x, y: abs(x) <= 0.15 and abs(y) <= 0.15
    gen1, gen2 = [], []
    for _ in range(n):
        cnt1, cnt2 = 0, 0
        for _ in range(1000):
            x1, y1 = generate1()
            if near_zero(x1, y1):
                cnt1 += 1
            x2, y2 = generate2()
            if near_zero(x2, y2):
                cnt2 += 1
        gen1.append(cnt1)
        gen2.append(cnt2)

    return pd.DataFrame({"Генератор 1": gen1, "Генератор 2": gen2}).describe().T


def main():
    pointset = create_pointset(1000)
    count_near_zero_points = number_nzpoints(pointset)
    print(count_near_zero_points)
    check_pattern = find_threshold()
    print(check_pattern)


if __name__ == "__main__":
    main()
