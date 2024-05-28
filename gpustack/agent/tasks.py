import time


def test():
    for i in range(5):
        print(f"test: {i}")
        time.sleep(2)
    print("test: task completed.")


def infinite():
    i = 0
    while True:
        print(f"test: {i}")
        time.sleep(2)
        i += 1
