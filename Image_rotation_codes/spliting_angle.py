import pandas as pd

## Rotation angle.txt file

angles= pd.read_csv(r"D:\Python Code\Image_rotation\Rotation Angles.txt",header=None)
angles.head()

angles_list=angles[0].tolist()
angles_list

f=[]
for i in angles_list:
    b=i.split('.jpg')
    f.append(b)

f

data=pd.DataFrame(f,columns=['Images','Angle'])
data
data['Images']=data['Images']+'.jpg'
data

data.to_csv(r"D:\Python Code\Image_rotation\Images_rotation_angle.txt",index=None)
data.to_csv(r'D:\Python Code\Image_rotation\Images_rotation_angle.csv',index=False)