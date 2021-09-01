from mp import Pool


def func(tid, num, n):
    ans = 1
    x = tid + 1
    while x <= n:
        ans *= x
        x += num
    return ans


if __name__ == '__main__':

    n = 100000
    thread_num = 8
    p = Pool(thread_num)  # 线程数

    ans_list = p.run(func, [n] * thread_num)
    ans = 1
    for x in ans_list:
        ans *= x
    print(ans)
    p.finalize()
