"""NOTA AI photo preprocessing v3.

Offline, deterministic computer-vision layer before AI OMR.
Preserves the original image; enhancements are alternative views, never evidence.
Blue annotations are suppressed only in an optional review view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import math

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SOURCE_PIXELS = 32_000_000
MAX_LONG_SIDE = 3500


@dataclass
class PreparedPage:
    original: Image.Image
    enhanced: Image.Image
    ink_suppressed: Image.Image
    orientation_degrees: int
    staff_systems: list[tuple[int,int]]
    masks: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    rotation_confidence: str = "low"


def _to_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(img).convert("RGB"), dtype=np.uint8)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _gray_enhanced(rgb: np.ndarray) -> np.ndarray:
    # CLAHE improves uneven illumination; conservative so ink stays legible.
    g = _gray(rgb)
    return cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(g)


def _line_mask(rgb: np.ndarray) -> np.ndarray:
    g = _gray_enhanced(rgb)
    black = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,31,12)
    width = rgb.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(max(35,width//25),1))
    return cv2.morphologyEx(black,cv2.MORPH_OPEN,kernel)


def _horizontal_score(rgb: np.ndarray) -> float:
    h,w = rgb.shape[:2]
    # Horizontal note staff detection, excluding image frame and large title.
    mask = _line_mask(rgb)
    sub = mask[h//20:h*19//20,w//20:w*19//20]
    rows = (sub>0).sum(axis=1)
    threshold = max(22, int(sub.shape[1]*.23))
    return float(np.minimum(rows[rows > threshold],sub.shape[1]).sum())


def _turn(rgb: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(rgb,cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(rgb,cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(rgb,cv2.ROTATE_90_CLOCKWISE)
    return rgb.copy()


def _staff_line_rows(rgb: np.ndarray) -> list[int]:
    mask = _line_mask(rgb)
    h,w=mask.shape
    sub=mask[:,w//20:w*19//20]
    counts=(sub>0).sum(axis=1)
    cutoff=max(20,int(sub.shape[1]*.23))
    positions=np.flatnonzero(counts>=cutoff)
    if positions.size==0:return []
    groups=np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)
    return [int(np.median(g)) for g in groups if 1<=len(g)<=max(6,int(h*.012))]


def _staff_system_bounds(rgb: np.ndarray)->list[tuple[int,int]]:
    h=rgb.shape[0]
    rows=_staff_line_rows(rgb)
    if len(rows)<5:return []
    diffs=np.diff(rows)
    # Typical staff line spacing: 5..36px, with regular sets of five lines.
    valid=[int(d) for d in diffs if 4<=d<=max(40,h//30)]
    if not valid:return []
    spacing=float(np.median(valid))
    clusters:list[list[int]]=[]
    for y in rows:
        if not clusters or y-clusters[-1][-1]>spacing*3.2:
            clusters.append([y])
        else:clusters[-1].append(y)
    groups=[g for g in clusters if len(g)>=4]
    if not groups:return []
    # Crop whole systems, retaining fingering above and markings below.
    # Midpoint between neighbors prevents repeated reading at boundaries.
    bounds=[]
    for i,group in enumerate(groups):
        top=max(0,int(group[0]-spacing*9.0))
        bottom=min(h,int(group[-1]+spacing*4.5))
        if i>0:
            prev=groups[i-1][-1]
            top=max(top,int((prev+group[0])/2))
        if i+1<len(groups):
            nxt=groups[i+1][0]
            bottom=min(bottom,int((group[-1]+nxt)/2))
        if bottom-top>=spacing*5:
            bounds.append((top,bottom))
    return bounds


def _deskew(rgb:np.ndarray)->tuple[np.ndarray,float]:
    g=_gray_enhanced(rgb)
    lines=cv2.HoughLinesP(cv2.Canny(g,70,180),1,np.pi/1800,threshold=65,
                         minLineLength=max(65,rgb.shape[1]//8),maxLineGap=12)
    if lines is None:return rgb,0.0
    candidates=[]
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        theta=math.degrees(math.atan2(float(y2-y1),float(x2-x1)))
        if abs(theta)<=6 and abs(x2-x1)>=max(65,rgb.shape[1]//8):candidates.append(theta)
    if len(candidates)<4:return rgb,0.0
    angle=float(np.median(candidates))
    if abs(angle)<.33 or abs(angle)>5:return rgb,0.0
    h,w=rgb.shape[:2]
    matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    rotated=cv2.warpAffine(rgb,matrix,(w,h),flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    return rotated,angle


def _annotation_mask(rgb:np.ndarray)->np.ndarray:
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    # Detect blue/violet ink; do not remove a black note or staff line.
    hue=hsv[:,:,0]; sat=hsv[:,:,1]; val=hsv[:,:,2]
    r,g,b=rgb[:,:,0].astype(np.int16),rgb[:,:,1].astype(np.int16),rgb[:,:,2].astype(np.int16)
    blue=(hue>=93)&(hue<=165)&(sat>=35)&(val>=38)&(b-r>=9)&(b-g>=-15)
    m=(blue.astype(np.uint8)*255)
    return cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))


def prepare_photo(image:Image.Image, *, rotation_override:int|None=None)->PreparedPage:
    rgb=_to_rgb(image)
    if rgb.shape[0]*rgb.shape[1]>MAX_SOURCE_PIXELS:
        raise ValueError("image_pixel_limit")
    scale=min(1.,MAX_LONG_SIDE/max(rgb.shape[:2]))
    if scale<1:
        rgb=cv2.resize(rgb,(int(rgb.shape[1]*scale),int(rgb.shape[0]*scale)),
                       interpolation=cv2.INTER_AREA)
    if rotation_override is not None and rotation_override not in (0,90,180,270):
        raise ValueError("invalid_rotation")
    tiny=cv2.resize(rgb,(max(300,int(rgb.shape[1]*min(1.,900/max(rgb.shape[:2])))),
                         max(300,int(rgb.shape[0]*min(1.,900/max(rgb.shape[:2]))))),
                    interpolation=cv2.INTER_AREA) if max(rgb.shape[:2])>900 else rgb
    horizontal=_horizontal_score(tiny)
    vertical=_horizontal_score(_turn(tiny,90))
    warnings=[]
    if rotation_override is not None:
        rotation=rotation_override; confidence="override"
    elif vertical>horizontal*1.6:
        # Staff geometry cannot resolve 90 from 270. Use likely header location
        # as a hint, but always surface rotation ambiguity for AI/human review.
        candidates=[]
        for angle in (90,270):
            preview=_turn(tiny,angle)
            rows=_staff_line_rows(preview)
            top=rows[0]/preview.shape[0] if rows else 1.0
            candidates.append((top,angle))
        rotation=min(candidates)[1]
        confidence="heuristic"
        warnings.append("rotation_180_ambiguity_review")
    elif horizontal>vertical*1.6:
        rotation=0;confidence="heuristic"
        warnings.append("upright_vs_upside_down_review")
    else:
        rotation=0;confidence="low"
        warnings.append("orientation_unreliable_review")
    oriented=_turn(rgb,rotation)
    leveled,skew=_deskew(oriented)
    if abs(skew)>.33:warnings.append("small_skew_corrected")
    bounds=_staff_system_bounds(leveled)
    if not bounds:warnings.append("staff_not_reliably_detected")
    if len(bounds)>12:warnings.append("many_staff_systems")
    mask=_annotation_mask(leveled)
    fraction=float((mask>0).mean())
    if fraction>.0003:warnings.append("colored_handwritten_annotations_detected")
    # Optional alternative view; keep original for OMR to avoid losing notation.
    if fraction>.0001:
        suppressed=cv2.inpaint(leveled,mask,3,cv2.INPAINT_TELEA)
    else:suppressed=leveled.copy()
    gray=_gray_enhanced(suppressed)
    enhanced=cv2.cvtColor(cv2.detailEnhance(cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB),
                                             sigma_s=5,sigma_r=.18),cv2.COLOR_RGB2GRAY)
    enhanced=cv2.cvtColor(enhanced,cv2.COLOR_GRAY2RGB)
    sharpness=cv2.Laplacian(_gray(leveled),cv2.CV_64F).var()
    if sharpness<35:warnings.append("possibly_blurry")
    if min(leveled.shape[:2])<800:warnings.append("low_resolution")
    return PreparedPage(original=Image.fromarray(leveled),
                        enhanced=Image.fromarray(enhanced),
                        ink_suppressed=Image.fromarray(suppressed),
                        orientation_degrees=rotation,staff_systems=bounds,
                        masks={"annotation_fraction":fraction,"blur_laplacian":round(float(sharpness),1),
                               "staff_horizontal_strength":horizontal,
                               "staff_vertical_strength":vertical},
                        warnings=list(dict.fromkeys(warnings)),rotation_confidence=confidence)


def prepare_pdf_page(page,*,rotation_override=None)->PreparedPage:
    """Accepts a fitz.Page. Handles JPEG-derived and true PDF pages equally."""
    rect=page.rect
    scale=min(2.25,math.sqrt(8_000_000/max(1.,rect.width*rect.height)))
    pix=page.get_pixmap(matrix=__import__('fitz').Matrix(scale,scale),alpha=False,
                        colorspace=__import__('fitz').csRGB)
    img=Image.open(io.BytesIO(pix.tobytes("png")))
    return prepare_photo(img,rotation_override=rotation_override)


def crop_systems(prepared:PreparedPage, view:str="original")->list[Image.Image]:
    src=getattr(prepared,view)
    if not prepared.staff_systems:return [src]
    return [src.crop((0,top,src.width,bottom)) for top,bottom in prepared.staff_systems]
