"""Memory-bounded photo preprocessing for NOTA AI on a 512 MiB worker.
Original image and alternate view are estimates; do not erase notes from original.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SOURCE_PIXELS = 32_000_000
MAX_LONG_SIDE = 2000
MAX_WORK_PIXELS = 2_500_000
cv2.setNumThreads(1)

@dataclass
class PreparedPage:
    original: Image.Image
    enhanced: Image.Image
    ink_suppressed: Image.Image
    orientation_degrees: int
    staff_systems: list[tuple[int,int]]
    masks: dict[str,float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    rotation_confidence: str = 'low'

def bounded_size(w,h):
    scale=min(1.,MAX_LONG_SIDE/max(w,h),math.sqrt(MAX_WORK_PIXELS/max(1,w*h)))
    return max(1,int(w*scale)),max(1,int(h*scale))

def _to_rgb(img):
    if img.width*img.height>MAX_SOURCE_PIXELS:
        raise ValueError('image_pixel_limit')
    target=bounded_size(*img.size)
    if target!=img.size:
        if img.format=='JPEG':
            img.draft('RGB',target)
        target=bounded_size(*img.size)
        if target!=img.size:
            img=img.resize(target,Image.Resampling.LANCZOS)
    return np.asarray(ImageOps.exif_transpose(img).convert('RGB'),dtype=np.uint8)

def _gray(rgb):
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)

def _gray_enhanced(rgb):
    return cv2.createCLAHE(clipLimit=1.8,tileGridSize=(8,8)).apply(_gray(rgb))

def _line_mask(rgb):
    black=cv2.adaptiveThreshold(_gray_enhanced(rgb),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    return cv2.morphologyEx(black,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(35,rgb.shape[1]//25),1)))

def _horizontal_score(rgb):
    h,w=rgb.shape[:2]; mask=_line_mask(rgb); sub=mask[h//20:h*19//20,w//20:w*19//20]
    rows=(sub>0).sum(axis=1); threshold=max(22,int(sub.shape[1]*.23))
    return float(np.minimum(rows[rows>threshold],sub.shape[1]).sum())

def _turn(rgb,angle):
    if angle==90:return cv2.rotate(rgb,cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle==180:return cv2.rotate(rgb,cv2.ROTATE_180)
    if angle==270:return cv2.rotate(rgb,cv2.ROTATE_90_CLOCKWISE)
    return rgb

def _staff_line_rows(rgb):
    mask=_line_mask(rgb); h,w=mask.shape; sub=mask[:,w//20:w*19//20]; counts=(sub>0).sum(axis=1)
    positions=np.flatnonzero(counts>=max(20,int(sub.shape[1]*.23)))
    if not positions.size:return []
    groups=np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)
    return [int(np.median(g)) for g in groups if 1<=len(g)<=max(6,int(h*.012))]

def _staff_system_bounds(rgb):
    h=rgb.shape[0]; rows=_staff_line_rows(rgb)
    if len(rows)<5:return []
    valid=[int(d) for d in np.diff(rows) if 4<=d<=max(40,h//30)]
    if not valid:return []
    spacing=float(np.median(valid)); clusters=[]
    for y in rows:
        if not clusters or y-clusters[-1][-1]>spacing*3.2:clusters.append([y])
        else:clusters[-1].append(y)
    groups=[g for g in clusters if len(g)>=4]; bounds=[]
    for i,group in enumerate(groups):
        top=max(0,int(group[0]-spacing*9.)); bottom=min(h,int(group[-1]+spacing*4.5))
        if i>0:top=max(top,int((groups[i-1][-1]+group[0])/2))
        if i+1<len(groups):bottom=min(bottom,int((group[-1]+groups[i+1][0])/2))
        if bottom-top>=spacing*5:bounds.append((top,bottom))
    return bounds

def _deskew(rgb):
    # Detect on a bounded preview; apply the angle to the working image.
    h,w=rgb.shape[:2]; factor=min(1.,1000/max(h,w)); preview=rgb if factor==1 else cv2.resize(rgb,(max(1,int(w*factor)),max(1,int(h*factor))),interpolation=cv2.INTER_AREA)
    g=_gray_enhanced(preview); pw=preview.shape[1]
    lines=cv2.HoughLinesP(cv2.Canny(g,70,180),1,np.pi/1800,threshold=65,minLineLength=max(65,pw//8),maxLineGap=12)
    if lines is None:return rgb,0.
    candidates=[]
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        theta=math.degrees(math.atan2(float(y2-y1),float(x2-x1)))
        if abs(theta)<=6 and abs(x2-x1)>=max(65,pw//8):candidates.append(theta)
    if len(candidates)<4:return rgb,0.
    angle=float(np.median(candidates))
    if abs(angle)<.33 or abs(angle)>5:return rgb,0.
    matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
    return cv2.warpAffine(rgb,matrix,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255)),angle

def _annotation_mask(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,(93,35,38),(165,255,255)); del hsv
    # Exact equivalent of b-r >= 9 and b-g >= -15, with bounded temporaries.
    b=rgb[:,:,2].astype(np.int16)
    mask[(b-rgb[:,:,0])<9]=0
    mask[(b-rgb[:,:,1])<-15]=0
    return cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))

def prepare_photo(image,*,rotation_override=None):
    rgb=_to_rgb(image)
    if rotation_override is not None and rotation_override not in (0,90,180,270):raise ValueError('invalid_rotation')
    h,w=rgb.shape[:2]; factor=min(1.,900/max(h,w)); tiny=rgb if factor==1 else cv2.resize(rgb,(max(1,int(w*factor)),max(1,int(h*factor))),interpolation=cv2.INTER_AREA)
    horizontal=_horizontal_score(tiny); vertical=_horizontal_score(_turn(tiny,90)); warnings=[]
    if rotation_override is not None:rotation=rotation_override;confidence='override'
    elif vertical>horizontal*1.6:
        candidates=[]
        for angle in (90,270):
            preview=_turn(tiny,angle); rows=_staff_line_rows(preview); candidates.append((rows[0]/preview.shape[0] if rows else 1.,angle))
        rotation=min(candidates)[1]; confidence='heuristic'; warnings.append('rotation_180_ambiguity_review')
    elif horizontal>vertical*1.6:
        rotation=0;confidence='heuristic';warnings.append('upright_vs_upside_down_review')
    else:
        rotation=0;confidence='low';warnings.append('orientation_unreliable_review')
    oriented=_turn(rgb,rotation); del rgb,tiny
    leveled,skew=_deskew(oriented); del oriented
    if abs(skew)>.33:warnings.append('small_skew_corrected')
    bounds=_staff_system_bounds(leveled)
    if not bounds:warnings.append('staff_not_reliably_detected')
    if len(bounds)>12:warnings.append('many_staff_systems')
    mask=_annotation_mask(leveled); fraction=cv2.countNonZero(mask)/mask.size
    if fraction>.0003:warnings.append('colored_handwritten_annotations_detected')
    # No copy or inpainting unless there is colored ink. Never change original.
    suppressed=cv2.inpaint(leveled,mask,3,cv2.INPAINT_TELEA) if fraction>.0001 else leveled
    del mask
    # CLAHE alternate replaces memory-heavy detailEnhance.
    enhanced=Image.fromarray(_gray_enhanced(suppressed)).convert('RGB')
    sample=_gray(leveled); factor=min(1.,1000/max(sample.shape))
    if factor<1:sample=cv2.resize(sample,(max(1,int(sample.shape[1]*factor)),max(1,int(sample.shape[0]*factor))),interpolation=cv2.INTER_AREA)
    sharpness=float(cv2.Laplacian(sample,cv2.CV_32F).var())
    if sharpness<35:warnings.append('possibly_blurry')
    if min(leveled.shape[:2])<800:warnings.append('low_resolution')
    original=Image.fromarray(leveled); ink=original if suppressed is leveled else Image.fromarray(suppressed)
    return PreparedPage(original,enhanced,ink,rotation,bounds,{'annotation_fraction':fraction,'blur_laplacian':round(sharpness,1),'staff_horizontal_strength':horizontal,'staff_vertical_strength':vertical},list(dict.fromkeys(warnings)),confidence)

def prepare_pdf_page(page,*,rotation_override=None):
    import fitz
    rect=page.rect
    scale=min(2.25,MAX_LONG_SIDE/max(rect.width,rect.height),math.sqrt(MAX_WORK_PIXELS/max(1.,rect.width*rect.height)))
    pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False,colorspace=fitz.csRGB)
    img=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    del pix
    try:return prepare_photo(img,rotation_override=rotation_override)
    finally:img.close()

def crop_systems(prepared,view='original'):
    src=getattr(prepared,view)
    if not prepared.staff_systems:return [src]
    return [src.crop((0,top,src.width,bottom)) for top,bottom in prepared.staff_systems]
