from utils import iou,match,yolo_to_xyxy

def test_yolo():
    assert yolo_to_xyxy(.5,.5,.2,.4,100,100)==(40.,30.,60.,70.)

def test_iou():
    assert iou((0,0,10,10),(0,0,10,10))==1
    assert iou((0,0,10,10),(20,20,30,30))==0
    assert abs(iou((0,0,10,10),(5,0,15,10))-1/3)<1e-6

def test_match_one_to_one():
    assert match([(0,0,10,10)],[(0,0,10,10),(0,0,10,10)])==[(0,0)]
    assert match([(0,0,10,10)],[(20,20,30,30)])==[]
