import os
from multiprocessing import Pool, cpu_count
import time as s
import laspy  # 打开las用的库
import numpy as np
import open3d as o3d  # 点云库
import math


def pca(xyz_n, sort=True):  # PCA函数，默认排序
    # 直接从深蓝学院作业照搬，所有参仅供参考
    data_T = xyz_n.T  # 数组转置
    s = np.array(data_T)  # 获取数组的行列数
    n = s.shape[0]  # 获取行数（x,y,z）
    m = s.shape[1]  # 获取列数（点云数）
    mean = [0] * 3  # 定义一个平均值空数组

    for i in range(n):  # 进行行数循环
        mean[i] = np.mean(data_T[i, :])  # 求出每行的平均值
        for j in range(m):  # 进行列数循环
            data_T[i, j] -= mean[i]  # 减去平均值
    dataTT = data_T.T  # 转置修改后的数组
    c = 1 / m * np.matmul(data_T, dataTT)  # 协方差c
    eigenvalues, eigenvectors = np.linalg.eig(c)  # 求出矩阵的特征值和特征向量
    # 判断是否排序
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors  # 返回特征值和特征向量（注意是3*3的）


def curvature_(eigenvalues):  # 计算单个点云的曲率
    c = eigenvalues[-1] / sum(eigenvalues)
    return c


def curvature(xyz, n=15):  # 求每个点云的曲率大小
    xyz_32 = np.column_stack((xyz[:, 0], xyz[:, 1], xyz[:, 2])).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_32)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # id_kntree = np.empty((len(xyz), n), dtype=int)  # 新建一个用来存储下标的新数组
    curvature_all = np.empty(len(pcd.points))
    for i in range(len(pcd.points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], n)  # 求每个点最近的n个点
        # id_kntree[i, :] = idx  # 求出每个点最邻近的n个点的下标
        xyz_n = xyz[idx, :]
        cv, _ = pca(xyz_n)  # 求每个点的特征值和特征向量
        c = curvature_(cv)  # 求出当前点的曲率
        curvature_all[i] = c
    return curvature_all  # 返回所有数组的曲率值组成一个数组


def curvature_singel(xyz, start, end, n=15):
    xyz_32 = np.column_stack((xyz[:, 0], xyz[:, 1], xyz[:, 2])).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_32)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    curvature_all = np.empty(end-start)
    for i in range(start, end):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], n)  # 求每个点最近的n个点
        # id_kntree[i, :] = idx  # 求出每个点最邻近的n个点的下标
        xyz_n = xyz[idx, :]
        cv, _ = pca(xyz_n)  # 求每个点的特征值和特征向量
        c = curvature_(cv)  # 求出当前点的曲率
        # print(c)
        curvature_all[i-start] = c
    return curvature_all


class clas:  # 读取las文件的类
    def __init__(self, path):  # 添加文件路径
        las = laspy.file.File(path, mode="r")  # 打开las文件
        self.xyz = np.array([las.x, las.y, las.z]).T  # 以xyz格式存储
        self.x = las.x  # 存储x
        self.y = las.y  # 存储y
        self.z = las.z  # 存储z
        self.num = len(las.x)

    def block(self, n=cpu_count()):  # 点云索引分块
        global start, end
        if self.num <= n:
            start = 0
            end = self.num
            print('点云数量过少，不能分块')
        else:
            start = []
            end = []
            mid = math.ceil(self.num / n)  # 返回大于它的最小整数
            print('共分了', mid, '块')
            for i in range(0, mid):
                start.append(i*n)
                end.append((i+1)*n-1)
        start = np.array(start)
        end = np.array(end)
        end[-1] = self.num
        return start, end

    def cpu_block(self, mid=cpu_count()):
        global start, end
        if self.num <= mid:
            start = 0
            end = self.num
            print('点云数量过少，不能分块')
        else:
            start = []
            end = []
            n_pool = math.ceil(self.num / mid)  # 每个池处理的最大点云数量
            print(n_pool)
            for i in range(0, mid):
                start.append(i*n_pool)
                end.append((i+1)*n_pool-1)
            start = np.array(start)
            end = np.array(end)
            end[-1] = self.num
        return start, end


if __name__ == '__main__':
    print(cpu_count())
    path1 = 'E:\\PointCloudSourceFile\\las\\ControlTest\\L5_C1_L.las'  # 打开点云路径L5
    path2 = 'E:\\PointCloudSourceFile\\las\\ControlTest\\L6_C1_L.las'  # 打开点云路径L6
    las_file1 = clas(path1)
    las_file2 = clas(path2)
    start1, end1 = clas.cpu_block(las_file1)
    start2, end2 = clas.cpu_block(las_file2)
    xyz1 = las_file1.xyz
    xyz2 = las_file2.xyz
    t = s.time()
    curvature_1 = curvature(xyz1)
    curvature_2 = curvature(xyz2)
    print("单线程求曲率用时", s.time() - t, "秒")
    pool = Pool(processes=cpu_count())
    t = s.time()
    c1 = np.empty(shape=[las_file1.num])
    c2 = np.empty(shape=[las_file2.num])
    # 迭代器，i=0时apply一次，i=1时apply一次等等
    n1 = 0
    n2 = 0
    multi_res1 = [pool.apply_async(curvature_singel, args=(xyz1, start1[i], end1[i])) for i in range(len(start1))]
    # 从迭代器中取出
    for res in multi_res1:
        c1[start1[n1]:end1[n1]] = res.get()
        n1 += 1
    # [c1.append(res.get()) for res in multi_res1]
    multi_res2 = [pool.apply_async(curvature_singel, args=(xyz2, start2[i], end2[i])) for i in range(len(start2))]
    # [c2.append(res.get()) for res in multi_res2]
    for res in multi_res2:
        c2[start2[n2]:end2[n2]] = res.get()
        n2 += 1
    print(c1, c2)
    print(len(c1), len(c2))
    pool.close()
    pool.join()
    print("多线程求曲率用时", s.time() - t, "秒")


