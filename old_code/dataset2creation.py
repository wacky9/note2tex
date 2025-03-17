# old file that will allow me to re-create dataset2 if necessary
from Line.preprocessing import *
import os
# Use toby_train files
def create_dataset2(Image):
    train_0 = ['upperW','lowerw','upperX','lowerx','upperY','lowery','upperZ','lowerz','0','1','2',
                '3','4','5','6','7','8','9','+','-','(',')']
    train_1 = ['upperL','lowerl','upperM','lowerm','upperN','lowern','upperO','lowero','upperP','lowerp','upperQ',
                   'lowerq','upperR','lowerr','upperS','lowers','upperT','lowert','upperU','loweru','upperV','lowerv']
    train_2 = ['dot','slash','less','greater','lessequal','greaterequal','=',
                   '≠',',','rarrow','larrow','biarrow',
                   'subset','real','integers','natural',
                   'rational','complex','pi','epsilon','theta','forall']
    train_3 = ['upperA','lowera','upperB','lowerb','upperC','lowerc','upperD','lowerd','upperE','lowere',
                   'upperF','lowerf','upperG','lowerg','upperH','lowerh','upperI','loweri','upperJ','lowerj',
                   'upperK','lowerk']
    train_4 = ['exists','arrow2']
    IM = Image
    bin = binarize(IM)>THRESHOLD
    lines = segment_lines(bin)
    line_num = 0
    for name in train_4:
        if not os.path.isdir(name):
            os.mkdir(name)
        components = get_components(lines[line_num])
        boxes = get_line_bounding_boxes(components)
        num = 0
        for box in boxes:
            Im = standardize(lines[line_num][box[0]:box[2],box[1]:box[3]],SIZE)
            io.imsave(name+'/'+ str(num)+'.png',Im,check_contrast=False)
            num+=1
        line_num+=1
    return []