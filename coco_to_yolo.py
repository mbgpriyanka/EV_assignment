import cv2
import json
import re

class ConvertCOCOToYOLO:

    """
    Takes in the path to COCO annotations and outputs YOLO annotations in multiple .txt files.
    COCO annotation are to be JSON formart as follows:
        "annotations":{
            "area":2304645,
            "id":1,
            "image_id":10,
            "category_id":4,
            "bbox":[
                0::704
                1:620
                2:1401
                3:1645
            ]
        }
        
    """

    def __init__(self, img_path, json_path):
        self.img_folder = img_path
        self.json_path = json_path
        

    def get_img_shape(self, img_path):
        img = cv2.imread(img_path)
        try:
            return img.shape
        except AttributeError:
            print('error!', img_path)
            return (None, None, None)

    def convert_labels(self, img_path, x1, y1, x2, y2):
        """
        Definition: Parses label files to extract label and bounding box
        coordinates. Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
        """

        def sorting(l1, l2):
            if l1 > l2:
                lmax, lmin = l1, l2
                return lmax, lmin
            else:
                lmax, lmin = l2, l1
                return lmax, lmin

        size = self.get_img_shape(img_path)
        print('size:',size)
        xmax, xmin = sorting(x1, x2)
        ymax, ymin = sorting(y1, y2)
        dw = 1./size[1]
        dh = 1./size[0]
        print('xx==',xmin,xmax,ymin,ymax)
        x = (float(xmin) + float(xmax))/2.0
        y = (float(ymin) + float(ymax))/2.0
        w = xmax - xmin
        h = ymax - ymin
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

    def convert(self,annotation_key='annotations',img_id='image_id',cat_id='category_id',bbox='bbox'):
        
        data = json.load(open(self.json_path))
        print('len(data[annotation_key]==',len(data[annotation_key]))
        
        check_set = set()

        # Retrieve data
        for i in range(len(data[annotation_key])):

            # Get required data
            print('i',i,data[annotation_key][i],data[annotation_key][i]['area'])
           # image_id = f'{data[annotation_key][i][img_id]}'
           # category_id = f'{data[annotation_key][i][cat_id]}'
            #bbox = f'{data[annotation_key][i][bbox]}'

            image_id = data[annotation_key][i][img_id]
            category_id = data[annotation_key][i][cat_id]

            bbox = data[annotation_key][i]['bbox']
            print('bbox',bbox)

            print('annotation dict---',image_id,category_id,bbox,type(bbox))

            image_id = 'image_'+ str(int(image_id)+1).zfill(9)
            print('image_id-----',image_id)
            

            # Retrieve image.
            if self.img_folder == None:
                
                image_path = f'{image_id}.jpg'
                
            else:
                image_path = f'./{self.img_folder}/{image_id}.jpg'
            print('imagepath:',image_path)    


            # Convert the data
            kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            #kitti_bbox = [re.sub("[^0-9]", "", x) for x in bbox.split(',')]
            print('kitti_bbox::',kitti_bbox)
            yolo_bbox = self.convert_labels(image_path, int(kitti_bbox[0]), int(kitti_bbox[1]), int(kitti_bbox[2]), int(kitti_bbox[3]))
            print('yolo_bbox::',yolo_bbox)
            
            # Prepare for export
            category_id = int(category_id) - 1 #classes start from 0
            filename = f'{image_id}.txt'
            content =f"{category_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}"
            print('content--',content)
            fn = 'Labels/train/'+filename

            # Export 
            if image_id in check_set:
                # Append to existing file as there can be more than one label in each image
                
                file = open(fn, "a")
                file.write("\n")
                file.write(content)
                file.close()

            elif image_id not in check_set:
                check_set.add(image_id)
                # Write files
                file = open(fn, "w")
                file.write(content)
                file.close()



# To run in as a class
ConvertCOCOToYOLO(img_path="Images/train",json_path='annotations/train.json').convert()