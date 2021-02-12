


from keras.models import load_model
import pickle
import cv2
import numpy as np
import cvlib as cv
from PIL import Image
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

loaded_model=load_model('FruitClassifier.h5') 

def predict_class(img):                     # this function will predict our class label
    img=cv2.resize(img, (100,100))
    img=img/255
    Img=img.reshape(1,100,100,3)
    prob= loaded_model.predict_proba(Img)
    for i in loaded_model.predict(Img):
        for j in i:
            if i >= 0.5:
                label= 'Rotten'
            else:
                label= 'Fresh'
    return(label, prob)


def final_model(img):
    
    """This function will classify fruit into fresh & rotten drawing boundary box and label on it"""
    
    # loaded_model=load_model('FruitClassifier.h5')      # this will load our pre-trained model
    
    # # def predict_class(img):                     # this function will predict our class label
    # img=cv2.resize(img, (100,100))
    # img=img/255
    # Img=img.reshape(1,100,100,3)
    # prob= loaded_model.predict_proba(Img)
    # for i in loaded_model.predict(Img):
    #     for j in i:
    #         if i >= 0.5:
    #             return 'Rotten'
    #         else:
    #             return 'Fresh'
            
    ###converting openCV to PIL for cropping
    Img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    PIL_img = Image.fromarray(Img)
    
    coor, label, conf = cv.detect_common_objects(img)    # detectimg fruit
    for i in coor:
        for j in label:
            if j == 'orange' or j == 'apple' or j == 'banana':      # checking for our 3 classes apple, banana, oranges
                x,y,w,h = i
                roi = PIL_img.crop((i))                            # cropping Image to find roi (region of interest)
                roi = np.asarray(roi)
                prediction, probability=predict_class(roi)
                # probability=loaded_model.predict_proba(roi)
                if prediction == 'Rotten':
                    cv2.rectangle(img, (x,y), (w,h), (255,0,0), 2)
                    cv2.putText(img, 'Rotten', (x+5, y+18), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 2)
                elif prediction == 'Fresh':
                    cv2.rectangle(img, (x,y), (w,h), (0,255,0), 2)
                    cv2.putText(img, 'Fresh', (x+5, y+18), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
                    
    return (img, probability)





# ### Model Deployment



import streamlit as st





def about():
    st.write('''
    This web app is for classification of fresh and rotten fruits.
    This app is for 3 fruit categories, i.e. Apples, Oranges and Banana.
    
    I used two basic algorithm for this project one is object detection using YOLO object detection,
    other is Convolutional Neural Network classifier for Image classification.''')





def main():
	st.title('Fresh and Rotten Fruit Classifier WebApp :apple::banana:🍊')
	st.write('Using YOLO object detection and CNN Classification')
    
	activities =['Home', 'About']
	choice = st.sidebar.selectbox('Pick your choice', activities)
    
	if choice == 'Home':
		st.write('Go to the About section on the sidebar to learn more about it.')
        
        # image type to be chosen
		image_file=st.file_uploader('Upload Image', type=['jpeg', 'png', 'jpg', 'webp'])
        
		if image_file is not None:
			image = Image.open(image_file)
			image = np.array(image)
            
			if st.button('Process'):
				st.image(image, use_column_width=True)
				st.info('Original Image')
				result_image, prob= final_model(image)
				st.image(result_image, use_column_width=True)
				st.info('Result Image')
                # st.image(image, use_column_width=True)
                # st.image(result_image, use_column_width=True)
                # for i in prob:
                # 	for j in i:
                # 		probability=1-j
                # st.info(f'{predict_class(result_image)[0]}')
                
                
	elif choice == 'About':
		about()





if __name__ == "__main__":
	main()







