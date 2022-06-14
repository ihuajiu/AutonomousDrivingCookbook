from keras.models import load_model
import sys
import numpy as np
import glob
import os

if ('../../PythonClient/' not in sys.path):
    sys.path.insert(0, '../../PythonClient/')
from AirSimClient import *

# 我们将使用我们在步骤1中训练的模型在AirSim中驾驶汽车

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('AirSimE2EDeepLearning/model/models/*.h5') 
    #models = glob.glob('D:\\220_StudyUpUp\\AI\\AIProject\\AirSim\\AutonomousDrivingCookbook\\AirSimE2EDeepLearning\\model\\models\\*.h5') 
    
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model
    
print('Using model {0} for testing.'.format(MODEL_PATH))

# 接下来，我们将加载模型并连接到Landscape环境中的AirSim模拟器。在启动此步骤**之前**，请确保模拟器正在不同的进程中运行。
model = load_model(MODEL_PATH)

client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
print('Connection established!')

# 我们将设置汽车的初始状态，以及一些用于存储模型输出的缓冲区
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

image_buf = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1,4))

# 我们将定义一个助手函数来从AirSim读取RGB图像，并准备好供模型使用
def get_image():
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    
    return image_rgba[76:135,0:255,0:3].astype(float)

from keras.preprocessing import image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import keras.backend as K

image_index = 0

def draw_image_with_label(img, label, prediction=None):
    global image_index

    theta = label * 0.69 #Steering range for the car is +- 40 degrees -> 0.69 radians
    line_length = 50
    line_thickness = 3
    label_line_color = (255, 0, 0)
    prediction_line_color = (0, 0, 255)
    pil_image = image.array_to_img(img, K.image_data_format(), scale=True)
    print('Actual Steering Angle = {0}'.format(label))
    draw_image = pil_image.copy()
    image_draw = ImageDraw.Draw(draw_image)
    first_point = (int(img.shape[1]/2),img.shape[0])
    second_point = (int((img.shape[1]/2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
    image_draw.line([first_point, second_point], fill=label_line_color, width=line_thickness)
    
    if (prediction is not None):
        print('Predicted Steering Angle = {0}'.format(prediction))
        print('L1 Error: {0}'.format(abs(prediction-label)))
        theta = prediction * 0.69
        second_point = (int((img.shape[1]/2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
        image_draw.line([first_point, second_point], fill=prediction_line_color, width=line_thickness)
    
    del image_draw
    # plt.imshow(draw_image)
    # plt.show()
    filename = os.path.join("AirSimE2EDeepLearning/model/imgs/", f"img_{image_index}.png")
    draw_image.save(filename, "PNG")
    image_index += 1

# 最后，一个控制块来运行汽车。因为我们的模型不预测速度，所以我们将尝试保持汽车以恒定的5米/秒的速度运行。运行下面的方块将导致模型驱动汽车!
while (True):
    car_state = client.getCarState()
    
    if (car_state.speed < 3):
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0
    
    image_buf[0] = get_image()
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
    
    t1 = time.perf_counter()
    model_output = model.predict([image_buf, state_buf])
    t2 = time.perf_counter()
    
    car_controls.steering = round(0.5 * float(model_output[0][0]), 2)
    
    print('Sending steering = {0}, throttle = {1}, diff:{2}'.format(car_controls.steering, car_controls.throttle, t2-t1))
    
    client.setCarControls(car_controls)

    # print(model_output[0])
    draw_image_with_label(image_buf[0], model_output[0][0])
