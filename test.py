# s='/media/cv3/store1/postgraduate/y2023/WLF/datasets/wlasl_2000/keypoints_hrnet_dark_coco_wholebody.pkl'
s='/media/cv3/store1/postgraduate/y2023/WLF/datasets/wlasl_2000/keypoints_hrnet_dark_coco_wholebody/train_99.pkl'
import gzip,pickle
with open(s, 'rb') as f:
    ff=pickle.load(f)
    for key,value in ff.items():
        print(key)
        print(value)
    print(type(ff))
