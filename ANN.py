
# https://www.youtube.com/@pHicha_Sign_On

import numpy as np 
# numpy การจัดการ Array (หลายมิติ)

import matplotlib.pyplot as plt
# Matplotlib = ไลบรารีสำหรับวาดกราฟ แผนภูมิ และ Visualization

import pandas as pd
# เอาไว้จัดการข้อมูลในรูปแบบตาราง (เหมือน Excel)

from sklearn.model_selection import train_test_split 
# train_test_split ใช้สำหรับ แบ่งข้อมูลชุดใหญ่ (Dataset) ออกเป็น 2 ส่วนหลักๆ คือ
# ชุดข้อมูลสำหรับฝึกสอนโมเดล (Train set)
# ชุดข้อมูลสำหรับทดสอบโมเดล (Test set)


# 1. โหลดข้อมูล
dataset = pd.read_csv(r"/Users/phcsten/Desktop/Training_Course/Neural_Network/dataset_iris_new.csv")
X = dataset.iloc[:, 0:4].values      # ดึงข้อมูล feature (คุณสมบัติ)
Y = dataset.iloc[:, 5].values        # ดึงข้อมูล label (ประเภทดอกไม้)

# 2. แบ่งข้อมูลเป็น Train, Validation และ Test
x_main, x_test, z_main, z_test = train_test_split(X, Y, test_size=25, stratify=Y, random_state=42)
x_train, x_val, z_train, z_val = train_test_split(x_main, z_main, test_size=25, stratify=z_main, random_state=42)

# แปลง label ให้เป็น 1D array (สำหรับ numpy)
z_train = np.ravel(z_train)
z_val   = np.ravel(z_val)
z_test  = np.ravel(z_test)

# 3. ฟังก์ชันช่วยเหลือ
def sigmoid(x):
    """ฟังก์ชัน sigmoid สำหรับแปลงค่าให้อยู่ในช่วง 0 ถึง 1"""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """ฟังก์ชัน softmax สำหรับคำนวณความน่าจะเป็นของแต่ละคลาส"""
    exp_x = np.exp(x.T - x.max(1))
    return (exp_x / exp_x.sum(0)).T

def ha_1hot(z, n):
    """แปลง label เป็น one-hot encoding โดยเปรียบเทียบค่าในแต่ละตำแหน่งกับช่วง 0 ถึง n-1"""
    return (z[:, None] == np.arange(n)).astype(int)

def ha_entropy(z, h):
    """คำนวณ cross-entropy loss โดยเพิ่ม 1e-10 เพื่อป้องกัน log(0)"""
    return -(np.log(h[z == 1] + 1e-10)).mean()

# 4. นิยามคลาส Perceptron สำหรับ Neural Network แบบง่าย
class Perceptron:
    def __init__(self, m, eta):
        self.m = m      # จำนวนโหนดใน hidden layer
        self.eta = eta  # อัตราการเรียนรู้

    def learning(self, x_train, z_train, x_val, z_val, x_test, z_test, loop):
        # กำหนดจำนวนคลาสจากข้อมูล Train (ใช้ max label + 1)
        self.four = int(z_train.max() + 1)
        Z_train = ha_1hot(z_train, self.four)
        Z_val   = ha_1hot(z_val, self.four)
        Z_test  = ha_1hot(z_test, self.four)

        # สุ่มค่า weight และ bias สำหรับแต่ละชั้น
        self.w1 = np.random.normal(0, 1, [x_train.shape[1], self.m])
        self.b1 = np.zeros(self.m)

        self.w2 = np.random.normal(0, 2, [x_train.shape[1], self.m])
        self.b2 = np.zeros(self.m)

        self.w3 = np.random.normal(0, 3, [self.m, self.four])
        self.b3 = np.zeros(self.four)

        # เก็บค่า loss และ accuracy ในแต่ละรอบของการฝึก
        self.entropy_train = []
        self.entropy_val   = []
        self.kernel_train  = []
        self.kernel_val    = []
        self.kernel_test   = []  # เก็บ Test Accuracy

        for i in range(loop):
            # วัดผลการฝึก (Train Set)
            # --- Forward Propagation บน Train Set ---
            a1 = np.dot(x_train, self.w1) + self.b1
            h1 = sigmoid(a1)

            a2 = np.dot(x_train, self.w2) + self.b2
            h2 = sigmoid(a2)

            a3 = np.dot(h1, self.w3) + np.dot(h2, self.w3) + self.b3
            h3 = softmax(a3)

            # คำนวณ Loss และ Accuracy บน Train Set
            J_train = ha_entropy(Z_train, h3)
            acc_train = (h3.argmax(1) == z_train).mean()

            # --- คำนวณ Gradient สำหรับ Backpropagation ---
            ga3 = (h3 - Z_train) / len(Z_train)
            gh2 = np.dot(ga3, self.w3.T)
            ga2 = gh2 * h2 * (1 - h2)

            gh1 = np.dot(ga3, self.w3.T)
            ga1 = gh1 * h1 * (1 - h1)

            # --- อัปเดต Weight และ Bias โดยใช้ Gradient Descent ---
            self.w3 -= self.eta * np.dot(h1.T, ga3)
            self.b3 -= self.eta * ga3.sum(0)

            self.w2 -= self.eta * np.dot(x_train.T, ga2)
            self.b2 -= self.eta * ga2.sum(0)

            self.w1 -= self.eta * np.dot(x_train.T, ga1)
            self.b1 -= self.eta * ga1.sum(0)

            # วัดผลการตรวจสอบ (Validation Set)
            # --- Forward Propagation บน Validation Set ---
            a1_val = np.dot(x_val, self.w1) + self.b1
            h1_val = sigmoid(a1_val)

            a2_val = np.dot(x_val, self.w2) + self.b2
            h2_val = sigmoid(a2_val)

            a3_val = np.dot(h1_val, self.w3) + np.dot(h2_val, self.w3) + self.b3
            h3_val = softmax(a3_val)

            # คำนวณ Loss และ Accuracy บน Validation Set
            J_val = ha_entropy(Z_val, h3_val)
            acc_val = (h3_val.argmax(1) == z_val).mean()

            # วัดผลการทดสอบ (Test Set)
            # --- Forward Propagation บน Test Set ---
            a1_test = np.dot(x_test, self.w1) + self.b1
            h1_test = sigmoid(a1_test)

            a2_test = np.dot(x_test, self.w2) + self.b2
            h2_test = sigmoid(a2_test)

            a3_test = np.dot(h1_test, self.w3) + np.dot(h2_test, self.w3) + self.b3
            h3_test = softmax(a3_test)

            # คำนวณ Accuracy บน Test Set
            acc_test = (h3_test.argmax(1) == z_test).mean()

            # บันทึกผลการฝึก
            self.entropy_train.append(J_train)
            self.kernel_train.append(acc_train)
            self.entropy_val.append(J_val)
            self.kernel_val.append(acc_val)
            self.kernel_test.append(acc_test)

            # แสดงผลทุก 100 รอบ
            if i % 100 == 99:
                print(f'รอบที่ {i+1} | Train Acc: {acc_train:.3f} | Val Acc: {acc_val:.3f} | Test Acc: {acc_test:.3f}')

            # หยุดการฝึกเมื่อ Train และ Validation Accuracy เกิน 95%
            if acc_train > 0.95 and acc_val > 0.95:
                print(f'หยุดการฝึกที่รอบที่ {i+1} เพราะ Train และ Validation Accuracy เกิน 95%')
                break

        # แสดงกราฟการฝึก (Loss และ Accuracy)
        self.plot_results()

    def plot_results(self):
        """แสดงกราฟ Loss และ Accuracy สำหรับ Train, Validation และ Test Set"""
        plt.figure(figsize=(14, 6))

        # กราฟ Loss (Train และ Validation)
        plt.subplot(1, 2, 1)
        plt.plot(self.entropy_train, 'g', label='Train Loss')
        plt.plot(self.entropy_val, 'r', label='Validation Loss')
        plt.xlabel('Training cycle')
        plt.ylabel('Loss')
        plt.legend()

        # กราฟ Accuracy (Train, Validation และ Test)
        plt.subplot(1, 2, 2)
        plt.plot(self.kernel_train, 'g', label='Train Accuracy')
        plt.plot(self.kernel_val, 'r', label='Validation Accuracy')
        plt.plot(self.kernel_test, 'b', label='Test Accuracy')
        plt.xlabel('Training cycle')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

# 5. สร้างอินสแตนซ์ของ Perceptron และเริ่มฝึกโมเดล
Percep = Perceptron(m=4, eta=0.3)
Percep.learning(x_train, z_train, x_val, z_val, x_test, z_test, loop=5000)