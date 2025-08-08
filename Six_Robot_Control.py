from ctypes import *
import ctypes
import json
import os

# 定义POSE结构体对应的Python类，用于表示机械臂的位姿（位置和旋转）
class POSE(Structure):
    _fields_ = [("px", c_float),  # X轴位置
                ("py", c_float),  # Y轴位置
                ("pz", c_float),  # Z轴位置
                ("Rx", c_float),  # 绕X轴旋转
                ("Ry", c_float),  # 绕Y轴旋转
                ("Rz", c_float)]  # 绕Z轴旋转

# 定义JOINT结构体对应的Python类，用于表示机械臂的关节角度
class JOINT(Structure):
    _fields_ = [("j1", c_float),  # 关节1角度
                ("j2", c_float),  # 关节2角度
                ("j3", c_float),  # 关节3角度
                ("j4", c_float),  # 关节4角度
                ("j5", c_float),  # 关节5角度
                ("j6", c_float)]  # 关节6角度

# 机械臂控制类
class Blinx_Six_Robot_Control:
    def __init__(self):
        # 连接机械臂
        host = b"192.168.8.23"  # 机械臂IP地址
        port = 4196  # 机械臂端口号
        # 使用相对路径加载dll
        dll_path = os.path.join(os.path.dirname(__file__), "dll/py_dll.dll")  # DLL文件路径
        self.robot = CDLL(dll_path)  # 加载DLL
        self.robot.Robot_socket_start(host, port)  # 启动socket连接
        self.robot.start_communication()  # 开始通信
        self.robot.set_robot_cmd_mode(b"SEQ")  # 设置命令模式为SEQ

    # 测试机械臂初始化
    def blinx_home(self):
        ret = self.robot.set_robot_arm_init()  # 调用初始化函数
        return ret

    # 气泵打开
    def blinx_pump_on(self):
        ret = self.robot.set_robot_end_tool(1, 1)  # 参数1表示工具ID，1表示打开
        return ret

    # 气泵关闭
    def blinx_pump_off(self):
        ret = self.robot.set_robot_end_tool(1, 0)  # 参数1表示工具ID，0表示关闭
        return ret

    # 机械臂单个关节角度运动控制
    # id: 关节编号（1-6轴）
    # speed: 运动速度
    # value: 目标角度值
    def blinx_move_angle(self, id, speed, value):
        self.robot.set_joint_degree_by_number.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]  # 设置参数类型
        ret = self.robot.set_joint_degree_by_number(id, speed, value)  # 调用单关节运动函数
        return ret

    # 机械臂所有关节角度运动控制
    # value1-value6: 六个关节的目标角度值
    # speed: 运动速度
    def blinx_move_angle_all(self, value1, value2, value3, value4, value5, value6, speed):
        self.robot.set_all_joint_degree_by_number.argtypes = [JOINT, c_int]  # 设置参数类型
        joint = JOINT(value1, value2, value3, value4, value5, value6)  # 创建JOINT结构体实例
        ret = self.robot.set_all_joint_degree_by_number(joint, speed)  # 调用多关节运动函数
        return ret

    # 机械臂回零
    def blinx_move_home(self):
        ret = self.robot.set_robot_arm_home()  # 调用回零函数
        return ret

    # 机械臂坐标运动控制
    # value1-value3: X/Y/Z轴位置
    # value4-value6: RX/RY/RZ轴旋转角度
    # speed: 运动速度
    def blinx_move_coordinate_all(self, value1, value2, value3, value4, value5, value6, speed):
        self.robot.set_robot_arm_coordinate.argtypes = [POSE, c_int]  # 设置参数类型
        pose = POSE(value1, value2, value3, value4, value5, value6)  # 创建POSE结构体实例
        ret = self.robot.set_robot_arm_coordinate(pose, speed)  # 调用坐标运动函数
        return ret

    # 获取机械臂当前坐标（正解）
    def blinx_positive_solution(self):
        self.robot.get_robot_coordinate.restype = ctypes.c_char_p  # 设置返回类型为字符串指针
        data = self.robot.get_robot_coordinate()  # 获取坐标数据
        data = data.decode('utf-8')  # 解码为UTF-8字符串
        result = json.loads(data)  # 解析JSON格式数据
        return result  # 返回坐标字典

    # 机械臂通讯关闭
    def blinx_close(self):
        ret = self.robot.end_communication()  # 结束通信
        return ret
