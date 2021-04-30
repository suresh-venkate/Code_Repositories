#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from PIL import Image, ImageDraw, ImageFont


# In[ ]:


std_colors = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


# In[ ]:


def draw_boxes_and_labels(img_pil, scores, boxes, class_ind, ind_label_dict, score_threshold = 0.4, norm_coords = True, box_thickness = 4, font_size = 24):
    
    # Define font to use for displaying text
    try:
        font = ImageFont.truetype('arial.ttf', font_size)
    except IOError:
        font = ImageFont.load_default()

    # Extract image height and width
    img_w, img_h = img_pil.size

    # Define PIL draw object to draw BBoxes and text labels
    draw = ImageDraw.Draw(img_pil)   

    for box_ind in np.where(scores > score_threshold)[0]:
        # Extract score for current box
        score = scores[box_ind] 
        # Extract bbox co-ords of current box
        box = boxes[box_ind] 
        # Extract class label of current box
        class_lab = ind_label_dict[class_ind[box_ind]]
        # Generate text label for current box
        text_label = '{}: {:0.0f}%'.format(class_lab, (100 * score))
        # Define color to use for BBOX and text box
        color = std_colors[class_ind[box_ind] % len(std_colors)]
        # Obtain label size and label height
        label_size = draw.textsize(text_label, font)
        # Use margin of 5% on top and bottom for label height
        label_h = (1 + 2 * 0.05) * label_size[1]

        # Extract BBOX co-ords
        # Check if BBOX co-ords are normalized
        if (norm_coords):
            (box_ymin, box_xmin, box_ymax, box_xmax) = (int(box[0] * img_h), int(box[1] * img_w),                                                        int(box[2] * img_h), int(box[3] * img_w))
        else:
            (box_ymin, box_xmin, box_ymax, box_xmax) = (box[0], box[1], box[2], box[3]) 

        # Define upper y-coordinate of text box
        # If possible, place text box above BBOX
        # else place it inside BBOX
        if box_ymin > label_h: 
            text_top = np.ceil(box_ymin - label_h)
        else:
            text_top = box_ymin + 1   

        # Draw BBOX
        draw.rectangle([box_xmin, box_ymin, box_xmax, box_ymax], outline = color, width = box_thickness)
        # Draw text box
        draw.rectangle([box_xmin, text_top, box_xmin + label_size[0], text_top + label_size[1]],                   fill = color)
        # Draw text
        draw.text((box_xmin, text_top), text_label, fill = 'black', font = font)
    
    # Delete draw object
    del draw

    return img_pil

